from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
import tokenizers.normalizers
import tokenizers.decoders
import pathlib
import hyperparameters

# Only run this after you've run data_processing.py
BPEtokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"],
                     vocab_size=hyperparameters.vocab_size,
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

