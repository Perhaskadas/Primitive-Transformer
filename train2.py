from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from transformer import Transformer
import torch
from data_tensor import LanguageData
import pathlib
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm
import json
import torch.nn as nn
import hyperparameters as hp
import numpy as np
import logging
import sys
logging.basicConfig(filename = "train2.log", 
		    level=logging.DEBUG,
		    format="%(asctime)s:%(levelname)s:%(message)s")

def retrieve_sentences(dataset, language):
    for i in dataset:
        yield i["translation"][language]

def train_tokenizer(dataset, language):
    BPEtokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[EOS]", "[SOS]", "[PAD]"],
                        vocab_size=hp.vocab_size,
                        min_frequency=3,
                        show_progress=True)

    BPEtokenizer.pre_tokenizer = Metaspace() # Using Metaspace tokenizer since it is better than whitespace for subwords
    BPEtokenizer.train_from_iterator(retrieve_sentences(dataset, language), trainer = trainer)
    BPEtokenizer.save(f"{language}_tokenizer.json")
    return BPEtokenizer

def split_data(dataset, training_size: float):
    training_split = int(training_size*len(dataset))
    validation_split = int(len(dataset) - training_split)
    training , validation = random_split(dataset, [training_split, validation_split]) 
    return training, validation

def download_data_config_tokenizers(language1: str, language2: str):
    #logging.debug("Inside download_data_config_tokenizers. About to download dataset")
    dataset = load_from_disk("opus_books_de_en")
    #logging.debug("Dataset loaded")
    tokenizer1 = train_tokenizer(dataset, language1)
    tokenizer2 = train_tokenizer(dataset, language2)

    training_data , validation_data = split_data(dataset, hp.validation_split_ratio)

    training_data_processed = LanguageData(training_data, tokenizer1, tokenizer2, language1, language2, hp.max_tokens)
    validation_data_processed = LanguageData(validation_data, tokenizer1, tokenizer2, language1, language2, hp.max_tokens)

    # Figure out max length
    lang1_max = 0
    lang2_max = 0
    for item in dataset:
        lang1_ids = tokenizer1.encode(item["translation"][language1]).ids
        lang2_ids = tokenizer2.encode(item["translation"][language2]).ids
        lang1_max = max(lang1_max, len(lang1_ids))
        lang2_max = max(lang2_max, len(lang2_ids))
    print(f"Max length for {language1}: {lang1_max}")
    print(f"Max length for {language2}: {lang2_max}")

    lang1_dataloader = DataLoader(training_data_processed, batch_size=hp.batch_size, shuffle=True)
    lang2_dataloader = DataLoader(validation_data_processed, batch_size=hp.batch_size, shuffle=True)

    return lang1_dataloader, lang2_dataloader, tokenizer1, tokenizer2

def greedy_decode(model: Transformer, encoder_input, tokenizer1, tokenizer2, max_len, device, print_msg):
    #logging.debug("Inside Greedy Decoder-------------------------------------------------")
    SOS_ID = tokenizer2.token_to_id("[SOS]")
    EOS_ID = tokenizer2.token_to_id("[EOS]")
    model.eval()
    src_mask = model.encoder_padding_mask(encoder_input).to(device)
    memory = model.encode(encoder_input, src_mask)
    memory = memory.to(device)

    ys = torch.full((1, hp.max_tokens), tokenizer2.token_to_id("[PAD]"), dtype=torch.long, device=device)
    ys[0, 0] = SOS_ID
    ys = ys.to(device)
    
    for i in range(1, max_len):
        #logging.debug(i)
        tgt_mask = model.decoder_padding_causal_mask(ys).to(device)

        #print_msg(f"Encoder Output Size: {memory.size()}")
        #print_msg(f"Decoder Input Size: {ys.size()}")

        out = model.decode(ys, memory, tgt_mask, src_mask)
        out = model.linear(out)
        #logging.debug(f"Size of model after final projection layer: {out.size()}")
        prob = out[:, -1]
        # print_msg(f"Top 5 predictions:{torch.topk(prob, 5)}")
        #logging.debug(f"Size of tensor after 'prob = out[:, -1]': {prob.size()}")
        _, next_word = torch.max(prob, dim = 1)
        #logging.debug(f"Size of matrix after torch.argmax and torch.max: {next_word.size()}")
        #logging.debug(f"next_word.item(): {next_word.item()}")
        next_word = next_word.item()
        
        ys[0, i] = next_word
        #logging.debug(ys)
        if next_word == EOS_ID:
            #logging.debug("EOS Detected, breaking loop")
            break
    return ys.squeeze(0)
    


def validation(model: Transformer, validation_ds, tokenizer1, tokenizer2, max_len, device, print_msg, num_examples=1):
    model.eval()
    count = 0
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_input_single = encoder_input[0].unsqueeze(0)
            print_msg(f"Encoder input single pre decode: {encoder_input_single}")
            output = greedy_decode(model, encoder_input_single, tokenizer1, tokenizer2, max_len, device, print_msg)

            source_text = batch['lang1'][0]
            target_text = batch['lang2'][0]
            model_out = tokenizer2.decode(output.detach().cpu().numpy())
            # Print to console
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out}")

            if count == num_examples:
                break     


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create the model
    lang1_dataloader, lang2_dataloader, tokenizer1, tokenizer2 = download_data_config_tokenizers("de", "en")
    model = Transformer(
        heads=hp.num_heads,
        ffn_hidden=hp.n_hidden,
        layers=hp.layers,
        vocab_size1=tokenizer1.get_vocab_size(),
        vocab_size2=tokenizer2.get_vocab_size(),
        model=hp.model,
        enc_padding_tokID=tokenizer1.token_to_id("[PAD]"),
        dec_padding_tokID=tokenizer2.token_to_id("[PAD]"),
        use_rotary_embed=False,
        max_len=hp.max_tokens,
        dropout=hp.dropout
    )

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model = model.to(device)

    # Setup the device
    save_path = pathlib.Path("dpmodels4")

    #Create the directory if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, eps = 1e-9)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer2.token_to_id("[PAD]"), label_smoothing=hp.label_smoothing).to(device)
   
    saved_models= list(save_path.glob("epoch_*.pth"))
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
    max_grad_norm = hp.max_grad_norm
    for epoch in range(start_epoch, hp.num_epochs):
        #logging.debug("Epoch: %s", str(epoch))
        batch_iterator = tqdm(lang1_dataloader, desc=f"Epoch {epoch:02d}")
        count = 0
        runningloss = 0
        for batch in batch_iterator:
            count = count + 1
            model.train()
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            label = batch["label"].to(device)

            output = model(encoder_input, decoder_input)
            loss = loss_fn(output.view(-1, tokenizer2.get_vocab_size()), label.view(-1))
            runningloss = runningloss + float(loss.item())
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            loss.backward()

            if isinstance(model, nn.DataParallel):
                clip_grad_norm_(model.module.parameters(), max_norm=max_grad_norm)
            else:
                clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)


            optimizer.step()
            optimizer.zero_grad()
            if(count % 100 == 0):
                batch_iterator.write(f">>>>>>>>>>>>>>Epoch number {epoch} | Iteration {count}<<<<<<<<<<<<<<<")
                validation(model.module, lang2_dataloader, tokenizer1, tokenizer2, 80, device, lambda msg: batch_iterator.write(msg), num_examples=2)
                validation(model.module, lang1_dataloader, tokenizer1, tokenizer2, 80, device, lambda msg: batch_iterator.write(msg), num_examples=2)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path / (f"epoch_{epoch}.pth"))
        avg_epoch_loss = runningloss/count
        with open('epoch_loss_tracker.txt', 'a') as f:
            f.write(f'Epoch {epoch}: {avg_epoch_loss}\n')
        logging.debug(avg_epoch_loss)

    # Save the model
    torch.save(model.state_dict(), save_path / "model.pth")



if __name__ == "__main__":
    train()















