from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from torch.utils.data import DataLoader, random_split
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

def retrieve_sentences(dataset, language):
    for i in dataset:
        yield i["translation"][language]

def train_tokenizer(dataset, language):
    BPEtokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"],
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
    dataset = load_dataset("opus_books", f"{language1}-{language2}", split="train")
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

def greedy_decode(model: Transformer, encoder_input, tokenizer1, tokenizer2, max_len: hp.max_tokens, device, print_msg):
    SOS_ID = tokenizer2.token_to_id("[SOS]")
    EOS_ID = tokenizer2.token_to_id("[EOS]")
    model.eval()
    src_mask = model.encoder_padding_mask(encoder_input)
    memory = model.encode(encoder_input, src_mask)

    ys = torch.full((1, hp.max_tokens), tokenizer2.token_to_id("[PAD]"), dtype=torch.long, device=device)
    ys[0, 0] = SOS_ID
    
    for i in range(1, max_len):
        tgt_mask = model.decoder_padding_causal_mask(ys)

        #print_msg(f"Encoder Output Size: {memory.size()}")
        #print_msg(f"Decoder Input Size: {ys.size()}")

        out = model.decode(ys, memory, tgt_mask, src_mask)
        out = model.linear(out)
        prob = out[:, -1]
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()
        
        ys[0, i] = next_word
        if next_word == EOS_ID:
            break
    return ys.squeeze(0)
    


def validation(model: Transformer, validation_ds, tokenizer1, tokenizer2, max_len, device, print_msg, num_examples=2):
    model.eval()
    count = 0
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
    
            output = greedy_decode(model, encoder_input, tokenizer1, tokenizer2, max_len, device, print_msg)

            source_text = batch['lang1'][0]
            target_text = batch['lang2'][0]
            model_out = tokenizer2.decode(output.detach().cpu().numpy())
            # Print to console
            print_msg('-'*console_width)
            print_msg(f"Source: {source_text}")
            print_msg(f"Expected: {target_text}")
            print_msg(f"Predicted: {model_out}")

            if count == num_examples:
                break     


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create the model
    lang1_dataloader, lang2_dataloader, tokenizer1, tokenizer2 = download_data_config_tokenizers("de", "en")
    model = Transformer(hp.num_heads, hp.n_hidden, hp.layers, tokenizer1.get_vocab_size(), tokenizer2.get_vocab_size(), hp.model, tokenizer1.token_to_id("[PAD]"), tokenizer2.token_to_id("[PAD]"), hp.dropout).to(device)
    # Setup the device
    save_path = pathlib.Path("models", mkdir=True, exist_ok=True)

    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, eps = 1e-9)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer2.token_to_id("[PAD]"), label_smoothing=hp.label_smoothing).to(device)
    
    for epoch in range(hp.num_epochs):
        batch_iterator = tqdm(lang1_dataloader, desc=f"Epoch {epoch:02d}")

        for batch in batch_iterator:
            #print("Inside training loop----------------------")
            model.train()
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            label = batch["label"].to(device)

            output = model(encoder_input, decoder_input)
            output = output.view(-1, output.size(-1))
            label = label.view(-1)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Run validation
        validation(model, lang2_dataloader, tokenizer1, tokenizer2, 80, device, lambda msg: batch_iterator.write(msg), num_examples=2)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path / (f"epoch_{epoch}.pth"))
        

    # Save the model
    torch.save(model.state_dict(), save_path / "model.pth")



if __name__ == "__main__":
    de = "de"
    en = "en"
    dataset = load_dataset("opus_books", f"{de}-{en}", split="train")
    dataset.save_to_disk("opus_books_de_en")















