import torch.nn as nn
import hyperparameters as hp
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from torch.utils.data import DataLoader, random_split
import torch
from data_processor import LanguageData
from datasets import load_from_disk
import torch.nn as nn
import hyperparameters as hp

class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.L1 = nn.Linear(hp.embedding_dim, hp.n_hidden, bias=True)
        self.L2 = nn.Linear(hp.n_hidden, hp.embedding_dim, bias=True)
        self.dropout = nn.Dropout(p=hp.dropout)
    
    def forward(self, x):
        x = self.L1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.L2(x)
        return x
    
class ProjectionToVocab(nn.Module):
    def __init__(self):
        super(ProjectionToVocab, self).__init__()
        self.projection = nn.Linear(hp.embedding_dim, hp.vocab_size, bias=True)
    
    def forward(self, x):
        x = self.projection(x)
        return x
    

# BELOW IS ALL FOR TRAINING SETUP AND WHATNOT
def causal_mask(size):
    mask = torch.ones((1, size, size))
    mask = torch.tril(mask, diagonal=0).type(torch.int)
    return mask

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

    training_data_processed = LanguageData(training_data, tokenizer1, tokenizer2)
    validation_data_processed = LanguageData(validation_data, tokenizer1, tokenizer2)

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
    lang2_dataloader = DataLoader(validation_data_processed, batch_size=1, shuffle=True)

    return lang1_dataloader, lang2_dataloader, tokenizer1, tokenizer2