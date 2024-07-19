import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class LanguageData(Dataset):
    def __init__(self, dataset: Dataset, tokenizer1: Tokenizer, tokenizer2: Tokenizer, language1: str, language2: str, max_length: int = 350):
        super(LanguageData, self).__init__()
        self.dataset = dataset
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.language1 = language1
        self.language2 = language2
        self.max_length = max_length

        self.PAD = torch.tensor([tokenizer1.token_to_id("[PAD]")], dtype=torch.int64)
        self.SOS = torch.tensor([tokenizer1.token_to_id("[SOS]")], dtype=torch.int64)
        self.EOS = torch.tensor([tokenizer1.token_to_id("[EOS]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        lang1 = data["translation"][self.language1]
        lang2 = data["translation"][self.language2]

        lang1_tokens = self.tokenizer1.encode(lang1).ids
        lang2_tokens = self.tokenizer2.encode(lang2).ids

        num_enc_pad = self.max_length - len(lang1_tokens) - 2 # Subtract 2 for SOS and EOS tokens
        num_dec_pad = self.max_length - len(lang2_tokens) - 1 # Subtract 1 since dec only has SOS

        if num_dec_pad < 0 or num_enc_pad < 0:
            raise ValueError("Invalid padding length. The input sequence is too long.")
        
        # Creating the tensors for the input and output sequences
        # Adding SOS and EOS to encode input
        encoder_input = torch.cat([self.SOS, 
                                   torch.tensor(lang1_tokens, dtype=torch.int64), 
                                   self.EOS, 
                                   torch.tensor([self.PAD] * num_enc_pad, dtype=torch.int64)], 
                                   dim=0)
        # Adding SOS to decoder input
        decoder_input = torch.cat([self.SOS, 
                                   torch.tensor(lang2_tokens, dtype=torch.int64), 
                                   torch.tensor([self.PAD] * num_dec_pad, dtype=torch.int64)], 
                                   dim=0)
        # Adding EOS to the label
        label = torch.cat([torch.tensor(lang2_tokens, dtype=torch.int64), 
                           self.EOS, 
                           torch.tensor([self.PAD] * num_dec_pad, dtype=torch.int64)], dim=0)

        if encoder_input.size(0) != self.max_length or decoder_input.size(0) != self.max_length or label.size(0) != self.max_length:
            raise ValueError("Invalid tensor size. The tensor size is not equal to the max length.")
        
        input_dictionary = {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "lang1" : lang1,
            "lang2" : lang2
        }
        return input_dictionary