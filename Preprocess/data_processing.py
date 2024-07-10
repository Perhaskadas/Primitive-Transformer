import pathlib
import utils as dps
import sys


# Checking to see if the raw data folder exists for processing
input_dir_path = pathlib.Path("data/raw")
if not input_dir_path.exists():
    print("No raw data available to process.")
    sys.exit()

# Setting up filepaths to the raw data
all_train_files = list( (input_dir_path / "train").glob("*"))
all_val_files = list( (input_dir_path / "val").glob("*"))
all_test_files = list( (input_dir_path / "test").glob("*"))
all_files_dict = {'train': all_train_files, 'val': all_val_files, 'test': all_test_files}

# Setting up Output Folders
output_dir_path = pathlib.Path("data/processed")
output_dir_path.mkdir(exist_ok=True, parents=True)

for folder_name in all_files_dict.keys():
    folder_path = output_dir_path / folder_name
    folder_path.mkdir(exist_ok=True)

# Printing all data that is going to be processed
print("The following files will be processed: ")
for file_type, files_list in all_files_dict.items():
    print(f"\nFiles inside {file_type}:")
    for file in files_list:
        print(file)

# Cleans texts and determine unique characters
unique_chars = set()
skip = False # Used to determine if the user wants to skip re-processing files that already exist

for file_type, files_list in all_files_dict.items():
    tmp_out_path = output_dir_path / file_type
    tmp_out_path.mkdir(exist_ok=True, parents=False)
    
    for file in files_list:
        print(f"Processing file {file}")
        with open(file, 'r', encoding='utf-8') as ff:
            full_text = ff.read()
            full_text = dps.cleanse_text(full_text)
            full_text = dps.cleans_test_text(full_text)
            with open(tmp_out_path / file.name, 'w', encoding='utf-8') as ff_out:
                ff_out.write(full_text)
                
            unique_chars.update(set(full_text))

num_unique = len(unique_chars)
print("Number of unique characters: ", num_unique)
