# # Reduce dataset size
# ---

# ## Importing the necessary packages

import os                                  # os handles directory/workspace changes
from subprocess import Popen               # Run shell commands
import pandas as pd                        # Pandas to load the data initially
from tqdm.auto import tqdm                 # Progress bar
import yaml                                # Save and load YAML files
import random                              # Randomization methods

# ## Initializing variables

# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# Path to the parquet dataset files
data_path = 'Datasets/Thesis/eICU/cleaned/normalized/ohe/'

# Random seed:

random.seed(42)

# Google Cloud settings:

bucket_name = input('Bucket name:')

# Training parameters:

test_train_ratio = 0.2                     # Percentage of the data which will be used as a test set
validation_ratio = 0.1                     # Percentage of the data from the training set which is used for validation purposes

# Reduction ratio:

reduct_ratio = 0.1

# ## Detecting which unit stays end up in death
#
# Go one by one in each unit stay's file and build a dictionary which indicates which contain positive samples (i.e. death happening at least 96h after the last sample in the unit stay data).

# Create a dictionary that indicates, for each unit stay file, if it's positive (i.e. death happening at least 96h after the last sample in the unit stay data) or negative:

ends_dead = dict()

for i in tqdm(range(8000)):
    # Download and read the file
    Popen(f'gsutil cp gs://{bucket_name}/eICU_{i}.ftr .', shell=True).wait()
    df = pd.read_feather(f'eICU_{i}.ftr')
    # Check if it's a positive or negative case (dead or alive)
    # 5760 minutes = 96 hours
    if df.death_ts.max() <= df.ts.max() + 5760:
        ends_dead[f'eICU_{i}.ftr'] = True
        print(f'File eICU_{i}.ftr is positive (df.death_ts.max() = df.ts.max() + {df.death_ts.max() - df.ts.max()})')
    else:
        ends_dead[f'eICU_{i}.ftr'] = False
        print(f'File eICU_{i}.ftr is false (df.death_ts.max() = df.ts.max() + {df.death_ts.max() - df.ts.max()})')
    # Delete the file from disk
    Popen(f'rm -rf eICU_{i}.ftr', shell=True).wait()

# Save the dictionary:

stream = open(f'{data_path}dead_or_alive.yml', 'w')
yaml.dump(ends_dead, stream, default_flow_style=False)

ends_dead

stream = open(f'{data_path}dead_or_alive.yml', 'r')
ends_dead = yaml.load(stream, Loader=yaml.FullLoader)
ends_dead

# ## Separating into train, validation and test sets
#
# Since I want to reduce the dataset size to 1/10 (i.e. to a total of 800 unit stays) and have each set contain approximately the same death ratio, I'll use the dictionary created above to get 576 training unit stays (0.9x0.8x800), 64 validation unit stays (0.1x0.8x800) and 160 test unit stays (0.2x800).

# Separate the positive files and the negative ones into two lists:

positive_files = list()
negative_files = list()
for file, is_positive in ends_dead.items():
    if is_positive:
        positive_files.append(file)
    else:
        negative_files.append(file)

positive_files

negative_files

n_positives = len(positive_files)
n_positives

n_negatives = len(negative_files)
n_negatives

pos_label_ratio = n_positives / (n_positives + n_negatives)
pos_label_ratio

# Get the required percentages for each set, maintaining the label ratio:

train_positive_files = random.sample(positive_files, 
                                     round(reduct_ratio * (1 - test_train_ratio) * (1 - validation_ratio) * n_positives))
train_negative_files = random.sample(negative_files, 
                                     round(reduct_ratio * (1 - test_train_ratio) * (1 - validation_ratio) * n_negatives))
train_files = train_positive_files + train_negative_files
train_files

len(train_files)

[positive_files.remove(file) for file in train_positive_files]
[negative_files.remove(file) for file in train_negative_files]

val_positive_files = random.sample(positive_files, 
                                   round(reduct_ratio * (1 - test_train_ratio) * validation_ratio * n_positives))
val_negative_files = random.sample(negative_files, 
                                   round(reduct_ratio * (1 - test_train_ratio) * validation_ratio * n_negatives))
val_files = val_positive_files + val_negative_files
val_files

len(val_files)

[positive_files.remove(file) for file in val_positive_files]
[negative_files.remove(file) for file in val_negative_files]

test_positive_files = random.sample(positive_files, 
                                    round(reduct_ratio * test_train_ratio * n_positives))
test_negative_files = random.sample(negative_files, 
                                    round(reduct_ratio * test_train_ratio * n_negatives))
test_files = test_positive_files + test_negative_files
test_files

len(test_files)

[positive_files.remove(file) for file in test_positive_files]
[negative_files.remove(file) for file in test_negative_files]

len(set(train_files) & set(val_files) & set(test_files))

# Get only the files' numbers, instead of their full names:

train_indeces = [int(file_name.split('eICU_')[1].split('.ftr')[0]) for file_name in train_files]
train_indeces

val_indeces = [int(file_name.split('eICU_')[1].split('.ftr')[0]) for file_name in val_files]
val_indeces

test_indeces = [int(file_name.split('eICU_')[1].split('.ftr')[0]) for file_name in test_files]
test_indeces

# Save the train, validation and test indeces (i.e. the lists of file numbers):

eICU_tvt_sets = dict(train_indeces=train_indeces, 
                     val_indeces=val_indeces, 
                     test_indeces=test_indeces)
eICU_tvt_sets

stream = open(f'{data_path}eICU_tvt_sets.yml', 'w')
yaml.dump(eICU_tvt_sets, stream, default_flow_style=False)


