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
import pandas as pd                        # Pandas to load the data initially
from dask.distributed import Client        # Dask scheduler
from dask.diagnostics import ProgressBar   # Dask progress bar
import re                                  # re to do regex searches in string data
import os                                  # os handles directory/workspace changes
import numpy as np                         # NumPy to handle numeric and NaN operations
from tqdm import tqdm_notebook             # tqdm allows to track code execution progress
import numbers                             # numbers allows to check if data is numeric
import utils                               # Contains auxiliary functions
# -

# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# +
# Change to parent directory (presumably "Documents")
os.chdir("../..")

# Path to the CSV dataset files
data_path = 'Datasets/Thesis/eICU/uncompressed/'
# -

# Activate the progress bar for all dask computations
pbar = ProgressBar()
pbar.register()

# +
# Set up local cluster
# client = Client("tcp://127.0.0.1:57608")
# client

# +
# Upload the utils.py file, so that the Dask cluster has access to relevant auxiliary functions
# client.upload_file('GitHub/eICU-mortality-prediction/NeuralNetwork.py')
# client.upload_file('utils.py')
# -

# ## Initialize variables

cat_feat = []                              # List of categorical features
cat_embed_feat = []                        # List of categorical features that will be embedded
cat_embed_feat_enum = dict()               # Dictionary of the enumerations of the categorical features that will be embedded

# ## Patient data

# ### Read the data

patient_df = pd.read_csv(f'{data_path}patient.csv')
patient_df = dd.from_pandas(patient_df, npartitions=8)
patient_df.head()

# +
# patient_df = dd.read_csv(f'{data_path}patient.csv')
# patient_df.head()

# +
# patient_df = patient_df.repartition(npartitions=4)
# -

patient_df.visualize()

patient_df.columns

patient_df.dtypes

patient_df.npartitions

# ### Remove unneeded features

patient_df = patient_df[['patientunitstayid', 'gender', 'age', 'ethnicity', 'apacheadmissiondx',  'admissionheight', 
                         'hospitaldischargeoffset', 'hospitaldischargelocation', 'hospitaldischargestatus', 
                         'admissionweight', 'dischargeweight', 'unitdischargeoffset']]
patient_df.head()

# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}}
utils.dataframe_missing_values(patient_df)
# -

patient_df.visualize()

# ### Make the age feature numeric
#
# In the eICU dataset, ages above 89 years old are not specified. Instead, we just receive the indication "> 89". In order to be able to work with the age feature numerically, we'll just replace the "> 89" values with "90", as if the patient is 90 years old. It might not always be the case, but it shouldn't be very different and it probably doesn't affect too much the model's logic.

patient_df.age.value_counts().head()

# Replace the "> 89" years old indication with 90 years
patient_df.age = patient_df.age.replace(to_replace='> 89', value=90)

patient_df.age.value_counts().head()

# Make the age feature numeric
patient_df.age = patient_df.age.astype(float)

patient_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
# patient_df = client.persist(patient_df)
patient_df = patient_df.persist()

patient_df.visualize()

# + {"toc-hr-collapsed": false, "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into simple numberings, one hot encode features with a low number of categories (in this case, 5) and enumerate sparse categorical features that will be embedded.
# -

# #### Convert binary categorical features into numeric

patient_df.gender.value_counts().compute()

patient_df.gender = patient_df.gender.map(lambda x: 1 if x == 'Male' else 0 if x == 'Female' else np.nan)

patient_df.gender.value_counts().compute()

# #### Separate and prepare features for embedding
#
# Identify categorical features that have more than 5 unique categories, which will go through an embedding layer afterwards, and enumerate them.
#
# [TODO] Only enumerate the `apacheadmissiondx` feature after joining it with all the remaining diagnosis features

# Update list of categorical features
cat_feat = ['ethnicity', 'apacheadmissiondx']

cat_feat_nunique = [patient_df[feature].nunique().compute() for feature in cat_feat]
cat_feat_nunique

patient_df[cat_feat].head()

for i in range(len(cat_feat)):
    feature = cat_feat[i]
    if cat_feat_nunique[i] > 5 and feature is not 'apacheadmissiondx':
        # Prepare for embedding, i.e. enumerate categories
        patient_df[feature], cat_embed_feat_enum[feature] = utils.enum_categorical_feature(patient_df, feature)

patient_df[cat_feat].head()

cat_embed_feat_enum

patient_df[cat_feat].dtypes

patient_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
# patient_df = client.persist(patient_df)
patient_df = patient_df.persist()

patient_df.visualize()

# ### Create mortality label
#
# Combine info from discharge location and discharge status. Using the hospital discharge data, instead of the unit, as it has a longer perspective on the patient's status. I then save a feature called "deathOffset", which has a number if the patient is dead on hospital discharge or is NaN if the patient is still alive/unknown (presumed alive if unknown). Based on this, a label can be made later on, when all the tables are combined in a single dataframe, indicating if a patient dies in the following X time, according to how faraway we want to predict.

patient_df.hospitaldischargestatus.value_counts().compute()

patient_df.hospitaldischargelocation.value_counts().compute()

patient_df['deathoffset'] = patient_df.apply(lambda df: df['hospitaldischargeoffset'] 
                                                        if df['hospitaldischargestatus'] == 'Expired' or
                                                        df['hospitaldischargelocation'] == 'Death' else np.nan, axis=1, 
                                                        meta=('x', float))

patient_df.head()

# Remove the now unneeded hospital discharge features:

patient_df = patient_df.drop(['hospitaldischargeoffset', 'hospitaldischargestatus', 'hospitaldischargelocation'], axis=1)
patient_df.head(6)

patient_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
# patient_df = client.persist(patient_df)
patient_df = patient_df.persist()

patient_df.visualize()

# ### Create a discharge instance and the timestamp feature

# Create the timestamp (`ts`) feature:

patient_df['ts'] = 0
patient_df.head()

patient_df.patientunitstayid.value_counts().compute()

# Duplicate every row, so as to create a discharge event:

patient_df = patient_df.append(patient_df)
patient_df.patientunitstayid.value_counts().compute()

# Sort by `patientunitstayid` so as to keep the timestamps of the same patient together:

patient_df = patient_df.compute().sort_values(by='patientunitstayid')
patient_df.head(6)

# Create a weight feature:

# Create feature weight and assign the initial weight that the patient has on admission
patient_df['weight'] = patient_df['admissionweight']
patient_df.head()


# Set the `weight` and `ts` features to initially have the value on admission and, on the second timestamp, have the value on discharge:

def set_weight(row):
    global patient_first_row
    if not patient_first_row:
        row['weight'] = row['dischargeweight']
        patient_first_row = True
    else:
        patient_first_row = False
    return row


patient_first_row = False
patient_df = patient_df.apply(lambda row: set_weight(row), axis=1)
patient_df.head(6)


def set_ts(row):
    global patient_first_row
    if not patient_first_row:
        row['ts'] = row['unitdischargeoffset']
        patient_first_row = True
    else:
        patient_first_row = False
    return row


patient_first_row = False
patient_df = dd.from_pandas(patient_df.apply(lambda row: set_ts(row), axis=1), npartitions=8)
patient_df.head(6)

# Remove the remaining, now unneeded, weight and timestamp features:

patient_df = patient_df.drop(['admissionweight', 'dischargeweight', 'unitdischargeoffset'], axis=1)
patient_df.head(6)

patient_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
# patient_df = client.persist(patient_df)
patient_df = patient_df.persist()

patient_df.visualize()

# ### Normalize data

# Save the dataframe before normalizing:

patient_df.to_parquet(f'{data_path}/cleaned/patient.parquet')

patient_df_norm = utils.normalize_data(patient_df)
patient_df_norm

# ### Rename columns



# ## Vital signs periodic data





# ### Join dataframes and save to a parquet file


