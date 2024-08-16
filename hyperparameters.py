num_heads = 2 # Number of attention heads in encoder/decoder
embedding_dim = 512 # Dimension of the embedding aka dimension of the model
n_hidden = 1024 # Number of hidden units in the feed forward network
num_layers = 3 # Number of encoder/decoder layers
vocab_size = 35000 # Number of words in the english-german vocabulary
dropout = 0.1 # Dropout rate
seq_len = 600 # max number of tokens in a batch
batch_size = 128 # Batch size
label_smoothing = 0.1
num_epochs = 60 # Number of epochs
validation_split_ratio = 0.9
max_grad_norm = 1 # Gradient clipping
learning_rate = 0.001 # Learning rate

source_language = "de" # Source Language from Opus Books MUST BE STRING
target_language = "en" # Target Language from Opus Books MUST BE STRING
save_path = "dpmodels" # Path to save the model
loss_track_file = "loss_tracker.txt" # File to save the loss every epoch

validation_cycle = 1 # Validate model every validation_cycle steps
save_cycle = 1 # Save model every save_cycle epochs