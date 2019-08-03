# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: eicu-mortality-prediction
#     language: python
#     name: eicu-mortality-prediction
# ---

# # eICU Data Joining
# ---
#
# Reading and joining all parts of the eICU dataset from MIT with the data from over 139k patients collected in the US.
#
# The main goal of this notebook is to prepare a single CSV document that contains all the relevant data to be used when training a machine learning model that predicts mortality, joining tables, filtering useless columns and performing imputation.

# + {"colab_type": "text", "id": "KOdmFzXqF7nq", "cell_type": "markdown"}
# ## Importing the necessary packages

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl"}
import dask.dataframe as dd                # Dask to handle big data in dataframes
from dask.diagnostics import ProgressBar   # Dask progress bar
import re                                  # re to do regex searches in string data
import os                                  # os handles directory/workspace changes
from tqdm import tqdm_notebook             # tqdm allows to track code execution progress
import numbers                             # numbers allows to check if data is numeric
import utils                               # Contains auxiliary functions

# +
# Change to parent directory (presumably "Documents")
os.chdir("../..")

# Path to the CSV dataset files
data_path = 'Datasets/Thesis/eICU/uncompressed/'
# -

# Activate the progress bar for all dask computations
pbar = ProgressBar()
pbar.register()

# ## Patient data

# ### Read the data

patient_df = dd.read_csv(f'{data_path}patient.csv')
patient_df.head()

patient_df.columns

patient_df.npartitions

# ### Remove unneeded features

patient_df.drop(columns=['patienthealthsystemstayid', 'hospitalid', 'wardid', 'apacheadmissiondx',
                         ''])

# ### Create a discharge instance and the timestamp feature



# ### Discretize categorical features
#
# Convert binary categorical features into simple numberings, one hot encode features with a low number of categories and enumerate sparse categorical features that will be embedded.



# ### Create mortality label
#
# Combine info from discharge location and discharge status.



# ### Normalize data



# ## Vital signs periodic data





# ### Join dataframes and save to a parquet file


