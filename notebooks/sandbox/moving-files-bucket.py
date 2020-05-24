# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: eICU-mortality-prediction
#     language: python
#     name: eicu-mortality-prediction
# ---

# # Move data in a GCP bucket

# ## Importing data and setting initial variables

import os                                  # os handles directory/workspace changes
from subprocess import Popen               # Run shell commands
import yaml                                # Save and load YAML files
from tqdm.auto import tqdm                 # Progress bar
from google.cloud import storage           # Google cloud storage python package
import warnings                            # Print warnings for bad practices

bucket_name = input('Bucket name:')

# Change to parent directory (presumably "Documents")
os.chdir("../../../..")

# Path to the dataset files
data_path = 'Datasets/Thesis/eICU/cleaned/'

stream_tvt_sets = open(f'{data_path}eICU_tvt_sets.yml', 'r')
eICU_tvt_sets = yaml.load(stream_tvt_sets, Loader=yaml.FullLoader)
eICU_tvt_sets

# ## Moving data to a subfolder in the bucket

all_subset_files = [val for set_list in eICU_tvt_sets.values() for val in set_list]
all_subset_files

for file_num in tqdm(all_subset_files):
    Popen(f'gsutil mv gs://{bucket_name}/eICU_{file_num}.ftr gs://{bucket_name}/subset', shell=True).wait()

# ## Getting a list of the file indeces in the subfolder

client = storage.Client()

subset_files = list()
for blob in client.list_blobs(bucket_name, prefix=f'subset/eICU_'):
    subset_files.append(str(blob))

subset_files

len(subset_files)

subset_files = [file.split(', ')[1].replace('subset/', '') for file in subset_files]
subset_files

eICU_tvt_sets_indices = eICU_tvt_sets.copy()

for tvt_set in eICU_tvt_sets.keys():
    # Create a list of subset indeces for each set (train, validation and test)
    tvt_set_indices = list()
    for file_num in eICU_tvt_sets[tvt_set]:
        file = f'eICU_{file_num}.ftr'
        if file not in subset_files:
            warnings.warn(f'File {file} isn\'t in the subset folder.')
        else:
            tvt_set_indices.append(subset_files.index(file))
    eICU_tvt_sets_indices[tvt_set] = tvt_set_indices

eICU_tvt_sets_indices

max_val = 0
for tvt_set in eICU_tvt_sets.keys():
    if max(eICU_tvt_sets[tvt_set]) > max_val:
        max_val = max(eICU_tvt_sets[tvt_set])
print(max_val)

max_val = 0
for tvt_set in eICU_tvt_sets_indices.keys():
    if max(eICU_tvt_sets_indices[tvt_set]) > max_val:
        max_val = max(eICU_tvt_sets_indices[tvt_set])
print(max_val)

stream_tvt_sets_indices = open(f'{data_path}eICU_tvt_sets_indices.yml', 'w')
yaml.dump(eICU_tvt_sets_indices, stream_tvt_sets_indices, default_flow_style=False)


