from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from transformer import Transformer
import torch
from data_processor import LanguageData
import pathlib
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm
import json
import torch.nn as nn
import hyperparameters as hp
import numpy as np
import os
from misc_layers import download_data_config_tokenizers, causal_mask

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lang1_dataloader, lang2_dataloader, tokenizer1, tokenizer2 = download_data_config_tokenizers(hp.source_language, hp.target_language)

    # Creating the model
    model = Transformer(torch.tensor(tokenizer1.token_to_id("[PAD]")), 
                        torch.tensor(tokenizer2.token_to_id("[PAD]"))).to(device)
    
    # If we can use DP
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model = model.to(device)

    # Setup save path
    save_path = pathlib.Path(hp.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Creating the Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer1.token_to_id("[PAD]"), label_smoothing=hp.label_smoothing).to(device)

    saved_models=list(save_path.glob("*epoch_*.pth"))
    if saved_models:
        # Sort the saved models by epoch number
        saved_models.sort(key=lambda x: int(x.stem.split('_')[1]))
        latest_model_path = saved_models[-1]
        checkpoint = torch.load(latest_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        #logging.debug(f"Continuing training from epoch {start_epoch}")
    else:
        start_epoch = 0

    for epoch in range(start_epoch, hp.num_epochs):
        batch_iterator = tqdm(lang1_dataloader, desc=f"Epoch {epoch:02d}")
        count = 0
        runningloss = 0
        for batch in batch_iterator:
            count += 1
            model.train()

            encoder_input = batch["encoder_input"].to(device) # (batch_size, seq_len)
            decoder_input = batch["decoder_input"].to(device) # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch_size, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device) # (batch_size, 1, seq_len, seq_len)
            label = batch["label"].to(device) # (batch_size, seq_len)

            encoder_output = model.encoder(encoder_input, encoder_mask) # Run it through the encoder
            decoder_output = model.decoder(decoder_input, encoder_output, encoder_mask, decoder_mask) # Run it through the decoder
            output_logits = model.project(decoder_output) # Project it to the vocab size

            # Calculate Loss
            loss = loss_fn(output_logits.view(-1, tokenizer2.get_vocab_size()), label.view(-1))
            runningloss = runningloss + loss.item()
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            loss.backward()

            if isinstance(model, nn.DataParallel):
                clip_grad_norm_(model.module.parameters(), max_norm=hp.max_grad_norm)
            else:
                clip_grad_norm_(model.parameters(), max_norm=hp.max_grad_norm)

            # Step the optimizer
            optimizer.step()
            optimizer.zero_grad()

            # TODO: Validate to see how we are doing
            if count % hp.validation_cycle == 0:
                validation(model, lang2_dataloader, tokenizer1, tokenizer2, hp.max_tokens, device, lambda msg: batch_iterator.write(msg), num_examples=1)

        # Save the checkpoints
        if epoch % hp.save_cycle == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': runningloss
            }, save_path / f"model_epoch_{epoch}.pth")
        avg_epoch_loss = runningloss/count
        with open(hp.loss_track_file, 'a') as f:
            f.write(f'Epoch {epoch}: {avg_epoch_loss}\n')
    
    # Save the final model
    torch.save({model.state_dict()}, save_path / f"model.pth")

def validation(model: Transformer, validation_ds, tokenizer1, tokenizer2, max_len, device, print_msg, num_examples=1):
    # Some of this code is taken from https://github.com/hkproj/pytorch-transformer/tree/main
    model.eval()
    count = 0
    
    try:
        # get the console window width
        with os.open('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch_size, 1, 1, seq_len)

            # Check batch size is 1
            assert encoder_input.size(0) == 1, "Validation Batch Size is not 1"

            output_logits = greedy_decode(model, encoder_input, encoder_mask, tokenizer1, tokenizer2, max_len, device)

            source_text = batch["lang1"][0]
            target_text = batch["lang2"][0]
            predicted_text = tokenizer2.decode(output_logits.cpu().numpy())

            # Print it to console
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{predicted_text}")

def greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    SOS_ID = tokenizer_tgt.token_to_id("[SOS]")
    EOS_ID = tokenizer_tgt.token_to_id("[EOS]")
    model.eval()
    
    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encoder(encoder_input, encoder_mask)

    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(SOS_ID).type_as(encoder_input).to(device)
    while True:
        # Loop guard, breaks if we reach sequence length
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)

        # calculate output
        decoder_output = model.decoder(decoder_input, encoder_output, encoder_mask, decoder_mask)
        decoder_output_probs = model.project(decoder_output[:, -1])

        # get next token
        _, next_word = torch.max(decoder_output_probs, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == EOS_ID:
            break

if __name__ == "__main__":
    train()



















