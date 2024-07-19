from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
import tokenizers.normalizers 
import tokenizers.decoders
import pathlib
import json


def create_tokenizer(vocab_size: int = 32000):
    # Only run this after you've run data_processing.py
    BPEtokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"],
                        vocab_size=vocab_size,
                        min_frequency=3,
                        show_progress=True)

    BPEtokenizer.pre_tokenizer = Metaspace() # Using Metaspace tokenizer since it is better than whitespace for subwords

    train_files = list (pathlib.Path("data/processed/train").glob("*"))
    val_files = list (pathlib.Path("data/processed/val").glob("*"))
    test_files = list (pathlib.Path("data/processed/test").glob("*"))

    tokenizer_train_files = list(map(str,train_files + val_files))
    tokenizer_test_files = list(map(str, test_files))

    BPEtokenizer.normalizer = tokenizers.normalizers.Sequence([tokenizers.normalizers.NFKC()]) # Convert German Umlauts into ASCII
    BPEtokenizer.decoder = tokenizers.decoders.Metaspace() # Use Metaspace again to decode the subwords

    BPEtokenizer.train(files=tokenizer_train_files, trainer=trainer)

    # Save the tokenizer
    BPEtokenizer.save("data/tokenizer.json")
    print("Tokenizer saved to data/tokenizer.json")

def tokenize_data():
    max_token_length = 25000
    train_files = list (pathlib.Path("data/processed/train").glob("*"))
    val_files = list (pathlib.Path("data/processed/val").glob("*"))
    test_files = list (pathlib.Path("data/processed/test").glob("*"))
    all_files = train_files + val_files + test_files
    all_files_dict = {"train": train_files, "val": val_files, "test": test_files}

    german_files_sorted = []
    english_files_sorted = []

    for f in all_files:
        if "de" in f.suffix or "de.sgm" in f.name:
            german_files_sorted.append(f)
        elif "en" in f.suffix or "en.sgm" in f.name:
            english_files_sorted.append(f)
    
    german_files_sorted.sort()
    english_files_sorted.sort()

    tokenized_dict = {"train": {"german": [], "english": []},
                      "val": {"german": [], "english": []},
                      "test": {"german": [], "english": []}}

    BPEtokenizer = Tokenizer.from_file("data/tokenizer.json")

    for de_en_tuple in zip(german_files_sorted, english_files_sorted):
        german_file = de_en_tuple[0]
        english_file = de_en_tuple[1]
        print(f"Processing files {german_file} and {english_file}")
        if german_file in all_files_dict["train"]:
            datatype = "train"
        elif german_file in all_files_dict["val"]:
            datatype = "val"
        else:
            datatype = "test"

        with open(german_file, 'r', encoding='utf-8') as german_f, open(english_file, 'r', encoding='utf-8') as english_f:
            for german_line, english_line in zip(german_f, english_f):
                german_line = german_line.strip()
                english_line = english_line.strip()
                if len(german_line) == 0 or len(english_line) == 0:
                    continue
                german_line = "[SOS] " + german_line + " [EOS]"
                english_line = "[SOS] " + english_line + " [EOS]"
                german_tokens = BPEtokenizer.encode(german_line).ids[:max_token_length]
                english_tokens = BPEtokenizer.encode(english_line).ids[:max_token_length]
                tokenized_dict[datatype]["german"].append(german_tokens)
                tokenized_dict[datatype]["english"].append(english_tokens)

    tokenized_dict = split_tokenized_data(tokenized_dict, split_ratio=0.1)
    # Save the tokenized data
    with open("data/tokenized/tokenized_data.json", "w") as outfile:
        json.dump(tokenized_dict, outfile)

    print("Tokenized data saved to data/tokenized.json")

def split_tokenized_data(tokenized_dict, split_ratio=0.1):
    """
    Splits the tokenized data into training and validation sets based on the specified split ratio.
    """
    # Calculate the number of items to take for validation from each language pair
    num_val_items = int(len(tokenized_dict["train"]["german"]) * split_ratio)

    # Split the data for both German and English
    val_german = tokenized_dict["train"]["german"][:num_val_items]
    train_german = tokenized_dict["train"]["german"][num_val_items:]

    val_english = tokenized_dict["train"]["english"][:num_val_items]
    train_english = tokenized_dict["train"]["english"][num_val_items:]

    # Update the tokenized_dict with the new splits
    tokenized_dict["train"]["german"] = train_german
    tokenized_dict["train"]["english"] = train_english

    tokenized_dict["val"]["german"] = val_german
    tokenized_dict["val"]["english"] = val_english

    return tokenized_dict
