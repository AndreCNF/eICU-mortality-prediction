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

# + {"toc-hr-collapsed": false, "cell_type": "markdown"}
# # eICU Data Joining
# ---
#
# Reading and joining all parts of the eICU dataset from MIT with the data from over 139k patients collected in the US.
#
# The main goal of this notebook is to prepare a single CSV document that contains all the relevant data to be used when training a machine learning model that predicts mortality, joining tables, filtering useless columns and performing imputation.

# + {"colab_type": "text", "id": "KOdmFzXqF7nq", "toc-hr-collapsed": true, "cell_type": "markdown"}
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
os.chdir("../../..")

# Path to the CSV dataset files
data_path = 'Documents/Datasets/Thesis/eICU/uncompressed/'
project_path = 'Documents/GitHub/eICU-mortality-prediction/'
# -

# Set up local cluster
client = Client("tcp://127.0.0.1:58996")
client

# Upload the utils.py file, so that the Dask cluster has access to relevant auxiliary functions
client.upload_file(f'{project_path}NeuralNetwork.py')
client.upload_file(f'{project_path}utils.py')

client.run(os.getcwd)

# ## Initialize variables

cat_feat = []                              # List of categorical features
cat_embed_feat = []                        # List of categorical features that will be embedded
cat_embed_feat_enum = dict()               # Dictionary of the enumerations of the categorical features that will be embedded

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ## Patient data
# -

# ### Read the data

patient_df = dd.read_csv(f'{data_path}original/patient.csv')
patient_df.head()

patient_df = patient_df.repartition(npartitions=30)

patient_df.npartitions

# Get an overview of the dataframe through the `describe` method:

patient_df.describe().compute().transpose()

patient_df.visualize()

patient_df.columns

patient_df.dtypes

# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}}
utils.dataframe_missing_values(patient_df)
# -

# ### Remove unneeded features

patient_df = patient_df[['patientunitstayid', 'gender', 'age', 'ethnicity', 'apacheadmissiondx',  'admissionheight', 
                         'hospitaldischargeoffset', 'hospitaldischargelocation', 'hospitaldischargestatus', 
                         'admissionweight', 'dischargeweight', 'unitdischargeoffset']]
patient_df.head()

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
patient_df = client.persist(patient_df)

patient_df.visualize()

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
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

# Update list of categorical features and add those that will need embedding (features with more than 5 unique values):

new_cat_feat = ['ethnicity', 'apacheadmissiondx']
[cat_feat.append(col) for col in new_cat_feat]

cat_feat_nunique = [patient_df[feature].nunique().compute() for feature in new_cat_feat]
cat_feat_nunique

new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        # Add feature to the list of the new ones (from the current table) that will be embedded
        new_cat_embed_feat.append(new_cat_feat[i])

patient_df[new_cat_feat].head()

for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    patient_df[feature], cat_embed_feat_enum[feature] = utils.enum_categorical_feature(patient_df, feature)

patient_df[new_cat_feat].head()

cat_embed_feat_enum

patient_df[cat_feat].dtypes

patient_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
patient_df = client.persist(patient_df)

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
patient_df = client.persist(patient_df)

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
patient_df = patient_df.apply(lambda row: set_ts(row), axis=1)
patient_df.head(6)

# Remove the remaining, now unneeded, weight and timestamp features:

patient_df = patient_df.drop(['admissionweight', 'dischargeweight', 'unitdischargeoffset'], axis=1)
patient_df.head(6)

# Create a `diagnosis` feature:

patient_df['diagnosis'] = patient_df['apacheadmissiondx']
patient_df.head()

# Add to the list of categorical and to be embedded features:

cat_feat.remove('apacheadmissiondx')
cat_embed_feat.remove('apacheadmissiondx')
new_cat_feat.remove('apacheadmissiondx')
new_cat_embed_feat.remove('apacheadmissiondx')
cat_feat.append('diagnosis')
cat_embed_feat.append('diagnosis')
new_cat_feat.append('diagnosis')
new_cat_embed_feat.append('diagnosis')


# Similarly, only set the `diagnosis` to the admission instance, as the current table only has diagnosis on admission:

def set_diagnosis(row):
    global patient_first_row
    if not patient_first_row:
        row['diagnosis'] = np.nan
        patient_first_row = True
    else:
        patient_first_row = False
    return row


patient_first_row = False
patient_df = patient_df.apply(lambda row: set_diagnosis(row), axis=1)
patient_df.head(6)

# Remove the admission diagnosis feature `apacheadmissiondx`:

patient_df = patient_df.drop('apacheadmissiondx', axis=1)
patient_df.head(6)

# Sort by `ts` so as to be easier to merge with other dataframes later:

patient_df = dd.from_pandas(patient_df.set_index('ts'), npartitions=30, sort=False)
patient_df.head(6)

patient_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
patient_df = client.persist(patient_df)

patient_df.visualize()

# ### Normalize data

# Save the dataframe before normalizing:

patient_df.to_parquet(f'{data_path}cleaned/unnormalized/patient.parquet')

new_cat_feat

patient_df.head()

# + {"pixiedust": {"displayParams": {}}}
patient_df_norm = utils.normalize_data(patient_df, embed_columns=new_cat_feat, 
                                       id_columns=['patientunitstayid', 'deathoffset'])
patient_df_norm.head(6)
# -

patient_df_norm.to_parquet(f'{data_path}cleaned/normalized/patient.parquet')

# Confirm that everything is ok through the `describe` method:

patient_df_norm.describe().compute().transpose()

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ## Vital signs periodic data
# -

# ### Read the data

vital_prdc_df = dd.read_csv(f'{data_path}original/vitalPeriodic.csv')
vital_prdc_df.head()

vital_prdc_df.npartitions

vital_prdc_df = vital_prdc_df.repartition(npartitions=30)

# Get an overview of the dataframe through the `describe` method:

vital_prdc_df.describe().compute().transpose()

vital_prdc_df.visualize()

vital_prdc_df.columns

vital_prdc_df.dtypes

# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}}
utils.dataframe_missing_values(patient_df)
# -

# ### Remove unneeded features

patient_df = patient_df[['patientunitstayid', 'gender', 'age', 'ethnicity', 'apacheadmissiondx',  'admissionheight', 
                         'hospitaldischargeoffset', 'hospitaldischargelocation', 'hospitaldischargestatus', 
                         'admissionweight', 'dischargeweight', 'unitdischargeoffset']]
patient_df.head()

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
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

# Update list of categorical features and add those that will need embedding (features with more than 5 unique values):

new_cat_feat = ['ethnicity', 'apacheadmissiondx']
[cat_feat.append(col) for col in new_cat_feat]

cat_feat_nunique = [patient_df[feature].nunique().compute() for feature in new_cat_feat]
cat_feat_nunique

new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        # Add feature to the list of the new ones (from the current table) that will be embedded
        new_cat_embed_feat.append(new_cat_feat[i])

patient_df[new_cat_feat].head()

for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    patient_df[feature], cat_embed_feat_enum[feature] = utils.enum_categorical_feature(patient_df, feature)

patient_df[new_cat_feat].head()

cat_embed_feat_enum

patient_df[cat_feat].dtypes

patient_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
patient_df = client.persist(patient_df)

patient_df.visualize()

# ### Create the timestamp feature and sort

# Create the timestamp (`ts`) feature:

patient_df['ts'] = 0
vital_aprdc_df = vital_aprdc_df.drop('observationoffset', axis=1)
patient_df.head()

patient_df.patientunitstayid.value_counts().compute()

# Remove duplicate rows:

len(patient_df)

patient_df = patient_df.drop_duplicates()
patient_df.head()

len(patient_df)

patient_df = patient_df.repartition(npartitions=30)

# Sort by `ts` so as to be easier to merge with other dataframes later:

vital_prdc_df = vital_prdc_df.set_index('ts')
vital_prdc_df.head(6)

patient_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
patient_df = client.persist(patient_df)

patient_df.visualize()

# Check for possible multiple rows with the same unit stay ID and timestamp:

micro_df.reset_index().head()

micro_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='culturesite').head()

micro_df[micro_df.patientunitstayid == 3069495].compute().head(20)

# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}}
micro_df = utils.join_categorical_enum(micro_df, new_cat_embed_feat)
micro_df.head()
# -

micro_df.dtypes

micro_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='culturesite').head()

micro_df[micro_df.patientunitstayid == 3069495].compute().head(20)

# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

micro_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
micro_df = client.persist(micro_df)

micro_df.visualize()

# ### Normalize data

# Save the dataframe before normalizing:

patient_df.to_parquet(f'{data_path}cleaned/unnormalized/patient.parquet')

# + {"pixiedust": {"displayParams": {}}}
patient_df_norm = utils.normalize_data(patient_df, embed_columns=new_cat_feat, 
                                       id_columns=['patientunitstayid', 'ts', 'deathoffset'])
patient_df_norm.head(6)
# -

patient_df_norm.to_parquet(f'{data_path}cleaned/normalized/patient.parquet')

# Confirm that everything is ok through the `describe` method:

patient_df_norm.describe().compute().transpose()

# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

patient_df = dd.read_parquet(f'{data_path}cleaned/normalized/patient.parquet')
patient_df.head()

vital_prdc_df = dd.read_parquet(f'{data_path}cleaned/normalized/vitalPeriodic.parquet')
vital_prdc_df.head()

eICU_df = dd.merge_asof(patient_df, vital_aprdc_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ## Vital signs aperiodic data
# -

# ### Read the data

vital_aprdc_df = dd.read_csv(f'{data_path}original/vitalAperiodic.csv')
vital_aprdc_df.head()

vital_aprdc_df = vital_aprdc_df.repartition(npartitions=30)

vital_aprdc_df.npartitions

# Get an overview of the dataframe through the `describe` method:

vital_aprdc_df.describe().compute().transpose()

vital_aprdc_df.visualize()

vital_aprdc_df.columns

vital_aprdc_df.dtypes

# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}}
utils.dataframe_missing_values(vital_aprdc_df)
# -

# ### Remove unneeded features

vital_aprdc_df = vital_aprdc_df.drop('vitalaperiodicid', axis=1)
vital_aprdc_df.head()

# ### Create the timestamp feature and sort

# Create the timestamp (`ts`) feature:

vital_aprdc_df['ts'] = vital_aprdc_df['observationoffset']
vital_aprdc_df = vital_aprdc_df.drop('observationoffset', axis=1)
vital_aprdc_df.head()

# Remove duplicate rows:

len(vital_aprdc_df)

vital_aprdc_df = vital_aprdc_df.drop_duplicates()
vital_aprdc_df.head()

len(vital_aprdc_df)

vital_aprdc_df = vital_aprdc_df.repartition(npartitions=30)

# Sort by `ts` so as to be easier to merge with other dataframes later:

vital_aprdc_df = vital_aprdc_df.set_index('ts')
vital_aprdc_df.head(6)

vital_aprdc_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
vital_aprdc_df = client.persist(vital_aprdc_df)

vital_aprdc_df.visualize()

# Check for possible multiple rows with the same unit stay ID and timestamp:

vital_aprdc_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='noninvasivemean').head()

vital_aprdc_df[micro_df.patientunitstayid == 3069495].compute().head(20)

# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}}
micro_df = utils.join_categorical_enum(micro_df, new_cat_embed_feat)
micro_df.head()
# -

micro_df.dtypes

micro_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='culturesite').head()

micro_df[micro_df.patientunitstayid == 3069495].compute().head(20)

# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

micro_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
micro_df = client.persist(micro_df)

micro_df.visualize()

# ### Normalize data

# Save the dataframe before normalizing:

vital_aprdc_df.to_parquet(f'{data_path}cleaned/unnormalized/vitalAperiodic.parquet')

# + {"pixiedust": {"displayParams": {}}}
vital_aprdc_df_norm = utils.normalize_data(vital_aprdc_df, 
                                           id_columns=['patientunitstayid', 'ts'])
vital_aprdc_df_norm.head(6)
# -

vital_aprdc_df_norm.to_parquet(f'{data_path}cleaned/normalized/vitalAperiodic.parquet')

# Confirm that everything is ok through the `describe` method:

vital_aprdc_df_norm.describe().compute().transpose()

# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

patient_df = dd.read_parquet(f'{data_path}cleaned/normalized/patient.parquet')
patient_df.head()

vital_aprdc_df = dd.read_parquet(f'{data_path}cleaned/normalized/vitalAperiodic.parquet')
vital_aprdc_df.head()

eICU_df = dd.merge_asof(patient_df, vital_aprdc_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ## Infectious disease data
# -

# ### Read the data

infect_df = dd.read_csv(f'{data_path}original/carePlanInfectiousDisease.csv')
infect_df.head()

infect_df.npartitions

infect_df = infect_df.repartition(npartitions=30)

infect_df.infectdiseasesite.value_counts().head(10)

infect_df.infectdiseaseassessment.value_counts().head(10)

infect_df.responsetotherapy.value_counts().head(10)

infect_df.treatment.value_counts().head(10)

# Most features in this table either don't add much information or they have a lot of missing values. The truly relevant one seems to be `infectdiseasesite`. Even `activeupondischarge` doesn't seem very practical as we don't have complete information as to when infections end, might as well just register when they are first verified.

# Get an overview of the dataframe through the `describe` method:

infect_df.describe().compute().transpose()

infect_df.visualize()

infect_df.columns

infect_df.dtypes

# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}}
utils.dataframe_missing_values(infect_df)
# -

# ### Remove unneeded features

infect_df = infect_df[['patientunitstayid', 'cplinfectdiseaseoffset', 'infectdiseasesite']]
infect_df.head()

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into simple numberings, one hot encode features with a low number of categories (in this case, 5) and enumerate sparse categorical features that will be embedded.
# -

# #### Separate and prepare features for embedding
#
# Identify categorical features that have more than 5 unique categories, which will go through an embedding layer afterwards, and enumerate them.

# Update list of categorical features and add those that will need embedding (features with more than 5 unique values):

new_cat_feat = ['infectdiseasesite']
[cat_feat.append(col) for col in new_cat_feat]

cat_feat_nunique = [infect_df[feature].nunique().compute() for feature in new_cat_feat]
cat_feat_nunique

new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        # Add feature to the list of the new ones (from the current table) that will be embedded
        new_cat_embed_feat.append(new_cat_feat[i])

infect_df[new_cat_feat].head()

for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    infect_df[feature], cat_embed_feat_enum[feature] = utils.enum_categorical_feature(infect_df, feature)

infect_df[new_cat_feat].head()

cat_embed_feat_enum

infect_df[cat_feat].dtypes

infect_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
infect_df = client.persist(infect_df)

infect_df.visualize()

# ### Create the timestamp feature and sort

# Create the timestamp (`ts`) feature:

infect_df['ts'] = infect_df['cplinfectdiseaseoffset']
infect_df = infect_df.drop('cplinfectdiseaseoffset', axis=1)
infect_df.head()

infect_df.patientunitstayid.value_counts().compute()

# Only 3620 unit stays have infection data. Might not be useful to include them.

# Remove duplicate rows:

len(infect_df)

infect_df = infect_df.drop_duplicates()
infect_df.head()

len(infect_df)

infect_df = infect_df.repartition(npartitions=30)

# Sort by `ts` so as to be easier to merge with other dataframes later:

infect_df = infect_df.set_index('ts')
infect_df.head(6)

infect_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
infect_df = client.persist(infect_df)

infect_df.visualize()

# Check for possible multiple rows with the same unit stay ID and timestamp:

infect_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='infectdiseasesite').head()

infect_df[infect_df.patientunitstayid == 3049689].compute().head(20)

# We can see that there are up to 6 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}}
infect_df = utils.join_categorical_enum(infect_df, new_cat_embed_feat)
infect_df.head()
# -

infect_df.dtypes

infect_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='infectdiseasesite').head()

infect_df[infect_df.patientunitstayid == 3049689].compute().head(20)

# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

infect_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
infect_df = client.persist(infect_df)

infect_df.visualize()

# ### Normalize data

# Save the dataframe before normalizing:

infect_df.to_parquet(f'{data_path}cleaned/unnormalized/carePlanInfectiousDisease.parquet')

# + {"pixiedust": {"displayParams": {}}}
infect_df_norm = utils.normalize_data(infect_df, embed_columns=new_cat_feat, 
                                      id_columns=['patientunitstayid'])
infect_df_norm.head(6)
# -

infect_df_norm.to_parquet(f'{data_path}cleaned/normalized/carePlanInfectiousDisease.parquet')

# Confirm that everything is ok through the `describe` method:

infect_df_norm.describe().compute().transpose()

# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

infect_df = dd.read_parquet(f'{data_path}cleaned/normalized/carePlanInfectiousDisease.parquet')
infect_df.head()

eICU_df = dd.merge_asof(eICU_df, infect_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ## Microbiology data
# -

# ### Read the data

micro_df = dd.read_csv(f'{data_path}original/microLab.csv')
micro_df.head()

len(micro_df)

micro_df.patientunitstayid.nunique().compute()

# Only 2923 unit stays have microbiology data. Might not be useful to include them.

micro_df.npartitions

micro_df = micro_df.repartition(npartitions=30)

# Get an overview of the dataframe through the `describe` method:

micro_df.describe().compute().transpose()

micro_df.visualize()

micro_df.columns

micro_df.dtypes

# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}}
utils.dataframe_missing_values(micro_df)
# -

# ### Remove unneeded features

micro_df.culturesite.value_counts().compute()

micro_df.organism.value_counts().compute()

micro_df.antibiotic.value_counts().compute()

micro_df.sensitivitylevel.value_counts().compute()

# All features appear to be relevant, except the unique identifier of the table.

micro_df = micro_df.drop('microlabid', axis=1)
micro_df.head()

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into simple numberings, one hot encode features with a low number of categories (in this case, 5) and enumerate sparse categorical features that will be embedded.
# -

# #### Separate and prepare features for embedding
#
# Identify categorical features that have more than 5 unique categories, which will go through an embedding layer afterwards, and enumerate them.
#
# In the case of microbiology data, we're also going to embed the antibiotic `sensitivitylevel`, not because it has many categories, but because there can be several rows of data per timestamp (which would be impractical on one hot encoded data).

# Update list of categorical features and add those that will need embedding (features with more than 5 unique values):

new_cat_feat = ['culturesite', 'organism', 'antibiotic', 'sensitivitylevel']
[cat_feat.append(col) for col in new_cat_feat]

cat_feat_nunique = [micro_df[feature].nunique().compute() for feature in new_cat_feat]
cat_feat_nunique

new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5 or new_cat_feat[i] == 'sensitivitylevel':
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        new_cat_embed_feat.append(new_cat_feat[i])

micro_df[new_cat_feat].head()

# + {"pixiedust": {"displayParams": {}}}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    micro_df[feature], cat_embed_feat_enum[feature] = utils.enum_categorical_feature(micro_df, feature)
# -

micro_df[new_cat_feat].head()

cat_embed_feat_enum

micro_df[new_cat_feat].dtypes

micro_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
micro_df = client.persist(micro_df)

micro_df.visualize()

# ### Create the timestamp feature and sort

# Create the timestamp (`ts`) feature:

micro_df['ts'] = micro_df['culturetakenoffset']
micro_df = micro_df.drop('culturetakenoffset', axis=1)
micro_df.head()

# Remove duplicate rows:

len(micro_df)

micro_df = micro_df.drop_duplicates()
micro_df.head()

len(micro_df)

micro_df = micro_df.repartition(npartitions=30)

# Sort by `ts` so as to be easier to merge with other dataframes later:

micro_df = micro_df.set_index('ts')
micro_df.head()

micro_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
micro_df = client.persist(micro_df)

micro_df.visualize()

# Check for possible multiple rows with the same unit stay ID and timestamp:

micro_df.reset_index().head()

micro_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='culturesite').head()

micro_df[micro_df.patientunitstayid == 3069495].compute().head(20)

# We can see that there are up to 120 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}}
micro_df = utils.join_categorical_enum(micro_df, new_cat_embed_feat)
micro_df.head()
# -

micro_df.dtypes

micro_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='culturesite').head()

micro_df[micro_df.patientunitstayid == 3069495].compute().head(20)

# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

micro_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
micro_df = client.persist(micro_df)

micro_df.visualize()

# ### Normalize data

# Save the dataframe before normalizing:

micro_df.to_parquet(f'{data_path}cleaned/unnormalized/microLab.parquet')

# + {"pixiedust": {"displayParams": {}}}
micro_df_norm = utils.normalize_data(micro_df, embed_columns=new_cat_feat, 
                                     id_columns=['patientunitstayid'])
micro_df_norm.head(6)
# -

micro_df_norm.to_parquet(f'{data_path}cleaned/normalized/microLab.parquet')

# Confirm that everything is ok through the `describe` method:

micro_df_norm.describe().compute().transpose()

# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

micro_df = dd.read_parquet(f'{data_path}cleaned/normalized/microLab.parquet')
micro_df.head()

eICU_df = dd.merge_asof(eICU_df, micro_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ## Respiratory care data
# -

# ### Read the data

resp_care_df = dd.read_csv(f'{data_path}original/respiratoryCare.csv', dtype={'airwayposition': 'object',
                                                                              'airwaysize': 'object',
                                                                              'apneaparms': 'object',
                                                                              'setapneafio2': 'object',
                                                                              'setapneaie': 'object',
                                                                              'setapneainsptime': 'object',
                                                                              'setapneainterval': 'object',
                                                                              'setapneaippeephigh': 'object',
                                                                              'setapneapeakflow': 'object',
                                                                              'setapneatv': 'object'})
resp_care_df.head()

len(resp_care_df)

resp_care_df.patientunitstayid.nunique().compute()

resp_care_df.npartitions

resp_care_df = resp_care_df.repartition(npartitions=30)

# Get an overview of the dataframe through the `describe` method:

resp_care_df.describe().compute().transpose()

resp_care_df.visualize()

resp_care_df.columns

resp_care_df.dtypes

# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}}
utils.dataframe_missing_values(resp_care_df)
# -

# ### Remove unneeded features

# For the respiratoryCare table, I'm not going to use any of the several features that detail what the vent in the hospital is like. Besides not appearing to be very relevant for the patient, they have a lot of missing values (>67%). Instead, I'm going to set a ventilation label (between the start and the end), and a previous ventilation label.

resp_care_df = resp_care_df[['patientunitstayid', 'ventstartoffset',
                             'ventendoffset', 'priorventstartoffset']]
resp_care_df.head()

# ### Create the timestamp feature and sort

# Create the timestamp (`ts`) feature:

resp_care_df['ts'] = resp_care_df['ventstartoffset']
resp_care_df = resp_care_df.drop('ventstartoffset', axis=1)
resp_care_df.head()

# Remove duplicate rows:

len(resp_care_df)

resp_care_df = resp_care_df.drop_duplicates()
resp_care_df.head()

len(resp_care_df)

resp_care_df = resp_care_df.repartition(npartitions=30)

# Sort by `ts` so as to be easier to merge with other dataframes later:

resp_care_df = resp_care_df.set_index('ts')
resp_care_df.head()

resp_care_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
resp_care_df = client.persist(resp_care_df)

resp_care_df.visualize()

# Check for possible multiple rows with the same unit stay ID and timestamp:

resp_care_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='ventendoffset').head()

resp_care_df[resp_care_df.patientunitstayid == 3348331].compute().head(20)

# We can see that there are up to 5283 duplicate rows per set of `patientunitstayid` and `ts`. As such, we must join them.

# ### Join rows that have the same IDs

# Even after removing duplicates rows, there are still some that have different information for the same ID and timestamp. We have to apply a groupby function, selecting the minimum value for each of the offset features, as the larger values don't make sense (in the `priorventstartoffset`).

((resp_care_df.index > resp_care_df.ventendoffset) & resp_care_df.ventendoffset != 0).compute().value_counts()

# There are no errors of having the start vent timestamp later than the end vent timestamp.

# + {"pixiedust": {"displayParams": {}}}
resp_care_df = utils.join_categorical_enum(resp_care_df, cont_join_method='min')
resp_care_df.head()
# -

resp_care_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='ventendoffset').head()

resp_care_df[resp_care_df.patientunitstayid == 1113084].compute().head(10)

# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

resp_care_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
resp_care_df = client.persist(resp_care_df)

resp_care_df.visualize()

# Only keep the first instance of each patient, as we're only keeping track of when they are on ventilation:

resp_care_df = resp_care_df.reset_index().groupby('patientunitstayid').first().reset_index().set_index('ts')
resp_care_df.head(20)

# ### Create prior ventilation label
#
# Make a feature `priorvent` that indicates if the patient has been on ventilation before.

# Convert to pandas:

resp_care_df = resp_care_df.compute()

# Create the prior ventilation column:

resp_care_df['priorvent'] = (resp_care_df.priorventstartoffset < resp_care_df.index).astype(int)
resp_care_df.head()

# Revert to Dask:

resp_care_df = dd.from_pandas(resp_care_df, npartitions=30)
resp_care_df.head()

# Remove the now unneeded `priorventstartoffset` column:

resp_care_df = resp_care_df.drop('priorventstartoffset', axis=1)
resp_care_df.head()

# ### Create current ventilation label
#
# Make a feature `onvent` that indicates if the patient is currently on ventilation.

# Duplicate every row, so as to create a discharge event:

resp_care_df = resp_care_df.append(resp_care_df)
resp_care_df.head()

# Sort by `ts` so as to keep the order of timestamps:

resp_care_df = resp_care_df.reset_index()
resp_care_df.head()

resp_care_df = resp_care_df.compute().sort_values(by='ts')
resp_care_df.head(6)

# Create a `onvent` feature:

resp_care_df['onvent'] = 1
resp_care_df.head(6)


# Set the `onvent` and `ts` features to initially have the value on ventilation start and, on the second timestamp, have the value on ventilation end:

def set_onvent(row):
    global first_row
    if not first_row:
        row['onvent'] = 0
        first_row = True
    else:
        first_row = False
    return row


first_row = False
resp_care_df = resp_care_df.apply(lambda row: set_onvent(row), axis=1)
resp_care_df.head(6)


def set_ts_vent(row):
    global first_row
    if not first_row:
        row['ts'] = row['ventendoffset']
        first_row = True
    else:
        first_row = False
    return row


first_row = False
resp_care_df = resp_care_df.apply(lambda row: set_ts_vent(row), axis=1)
resp_care_df.head(6)

# Remove the now unneeded ventilation end column:

resp_care_df = resp_care_df.drop('ventendoffset', axis=1)
resp_care_df.head(6)

# Sort by `ts` so as to be easier to merge with other dataframes later:

resp_care_df = dd.from_pandas(resp_care_df.set_index('ts'), npartitions=30, sort=True)
resp_care_df.head(6)

resp_care_df.tail(6)

resp_care_df[resp_care_df.patientunitstayid == 1557538].compute()

# Save the dataframe:

resp_care_df.to_parquet(f'{data_path}cleaned/unnormalized/respiratoryCare.parquet')

resp_care_df.to_parquet(f'{data_path}cleaned/normalized/respiratoryCare.parquet')

# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

resp_care_df = dd.read_parquet(f'{data_path}cleaned/normalized/respiratoryCare.parquet')
resp_care_df.head()

eICU_df = dd.merge_asof(eICU_df, resp_care_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ## Allergy data
# -

# ### Read the data

alrg_df = dd.read_csv(f'{data_path}original/allergy.csv')
alrg_df.head()

len(alrg_df)

alrg_df.patientunitstayid.nunique().compute()

alrg_df.npartitions

alrg_df = alrg_df.repartition(npartitions=30)

# Get an overview of the dataframe through the `describe` method:

alrg_df.describe().compute().transpose()

alrg_df.visualize()

alrg_df.columns

alrg_df.dtypes

# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}}
utils.dataframe_missing_values(alrg_df)
# -

# ### Remove unneeded features

alrg_df[alrg_df.allergytype == 'Non Drug'].drughiclseqno.value_counts().compute()

alrg_df[alrg_df.allergytype == 'Drug'].drughiclseqno.value_counts().compute()

# As we can see, the drug features in this table only have data if the allergy derives from using the drug. As such, we don't need the `allergytype` feature. Also ignoring hospital staff related information and using just the drug codes instead of their names, as they're independent of the drug brand.

alrg_df.allergynotetype.value_counts().compute()

# Feature `allergynotetype` also doesn't seem very relevant, discarding it.

alrg_df = alrg_df[['patientunitstayid', 'allergyoffset', 
                   'allergyname', 'drughiclseqno']]
alrg_df.head()

# + {"toc-hr-collapsed": false, "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into simple numberings, one hot encode features with a low number of categories (in this case, 5) and enumerate sparse categorical features that will be embedded.
# -

# #### Separate and prepare features for embedding
#
# Identify categorical features that have more than 5 unique categories, which will go through an embedding layer afterwards, and enumerate them.
#
# In the case of microbiology data, we're also going to embed the antibiotic `sensitivitylevel`, not because it has many categories, but because there can be several rows of data per timestamp (which would be impractical on one hot encoded data).

# Update list of categorical features and add those that will need embedding (features with more than 5 unique values):

new_cat_feat = ['allergyname', 'drughiclseqno']
[cat_feat.append(col) for col in new_cat_feat]

cat_feat_nunique = [alrg_df[feature].nunique().compute() for feature in new_cat_feat]
cat_feat_nunique

new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        new_cat_embed_feat.append(new_cat_feat[i])

alrg_df[new_cat_feat].head()

# + {"pixiedust": {"displayParams": {}}}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Skip the 'drughiclseqno' from enumeration encoding
    if feature == 'drughiclseqno':
        continue
    # Prepare for embedding, i.e. enumerate categories
    alrg_df[feature], cat_embed_feat_enum[feature] = utils.enum_categorical_feature(alrg_df, feature)
# -

# Fill missing values of the drug data with 0, so as to prepare for embedding:

alrg_df.drughiclseqno = alrg_df.drughiclseqno.fillna(0).astype(int)

alrg_df[new_cat_feat].head()

cat_embed_feat_enum

alrg_df[new_cat_feat].dtypes

alrg_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
alrg_df = client.persist(alrg_df)

alrg_df.visualize()

# ### Create the timestamp feature and sort

# Create the timestamp (`ts`) feature:

alrg_df['ts'] = alrg_df['allergyoffset']
alrg_df = alrg_df.drop('allergyoffset', axis=1)
alrg_df.head()

# Remove duplicate rows:

len(alrg_df)

alrg_df = alrg_df.drop_duplicates()
alrg_df.head()

len(alrg_df)

alrg_df = alrg_df.repartition(npartitions=30)

# Sort by `ts` so as to be easier to merge with other dataframes later:

alrg_df = alrg_df.set_index('ts')
alrg_df.head()

alrg_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
alrg_df = client.persist(alrg_df)

alrg_df.visualize()

# Check for possible multiple rows with the same unit stay ID and timestamp:

alrg_df.reset_index().head()

alrg_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='allergyname').head()

alrg_df[alrg_df.patientunitstayid == 3197554].compute().head(10)

# We can see that there are up to 47 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# ### Join rows that have the same IDs

# Even after removing duplicates rows, there are still some that have different information for the same ID and timestamp. We have to concatenate the categorical enumerations.

# + {"pixiedust": {"displayParams": {}}}
alrg_df = utils.join_categorical_enum(alrg_df, new_cat_embed_feat)
alrg_df.head()
# -

alrg_df.dtypes

alrg_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='allergyname').head()

alrg_df[alrg_df.patientunitstayid == 3197554].compute().head(10)

# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

alrg_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
alrg_df = client.persist(alrg_df)

alrg_df.visualize()

# ### Renaming columns

alrg_df = alrg_df.rename(columns={'drughiclseqno':'drugallergyhiclseqno'})
alrg_df.head()

# Save the dataframe:

alrg_df = alrg_df.repartition(npartitions=30)

alrg_df.to_parquet(f'{data_path}cleaned/unnormalized/allergy.parquet')

alrg_df.to_parquet(f'{data_path}cleaned/normalized/allergy.parquet')

# Confirm that everything is ok through the `describe` method:

alrg_df.describe().compute().transpose()

# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

alrg_df = dd.read_parquet(f'{data_path}cleaned/normalized/allergy.parquet')
alrg_df.head()

alrg_df.npartitions

eICU_df = dd.merge_asof(eICU_df, alrg_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# +
# [TODO] Check if careplangeneral table could be useful. It seems to have mostly subjective data.

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ## General care plan data
# -

# ### Read the data

careplangen_df = dd.read_csv(f'{data_path}original/carePlanGeneral.csv')
careplangen_df.head()

len(careplangen_df)

careplangen_df.patientunitstayid.nunique().compute()

careplangen_df.npartitions

careplangen_df = careplangen_df.repartition(npartitions=30)

# Get an overview of the dataframe through the `describe` method:

careplangen_df.describe().compute().transpose()

careplangen_df.visualize()

careplangen_df.columns

careplangen_df.dtypes

# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}}
utils.dataframe_missing_values(careplangen_df)
# -

# ### Remove unneeded features

careplangen_df.cplgroup.value_counts().compute()

careplangen_df.cplitemvalue.value_counts().compute()

careplangen_df[careplangen_df.cplgroup == 'Activity'].cplitemvalue.value_counts().compute()

careplangen_df[careplangen_df.cplgroup == 'Care Limitation'].cplitemvalue.value_counts().compute()

careplangen_df[careplangen_df.cplgroup == 'Route-Status'].cplitemvalue.value_counts().compute()

careplangen_df[careplangen_df.cplgroup == 'Critical Care Discharge/Transfer Planning'].cplitemvalue.value_counts().compute()

careplangen_df[careplangen_df.cplgroup == 'Safety/Restraints'].cplitemvalue.value_counts().compute()

careplangen_df[careplangen_df.cplgroup == 'Sedation'].cplitemvalue.value_counts().compute()

careplangen_df[careplangen_df.cplgroup == 'Analgesia'].cplitemvalue.value_counts().compute()

careplangen_df[careplangen_df.cplgroup == 'Ordered Protocols'].cplitemvalue.value_counts().compute()

careplangen_df[careplangen_df.cplgroup == 'Volume Status'].cplitemvalue.value_counts().compute()

careplangen_df[careplangen_df.cplgroup == 'Psychosocial Status'].cplitemvalue.value_counts().compute()

careplangen_df[careplangen_df.cplgroup == 'Current Rate'].cplitemvalue.value_counts().compute()

careplangen_df[careplangen_df.cplgroup == 'Baseline Status'].cplitemvalue.value_counts().compute()

careplangen_df[careplangen_df.cplgroup == 'Protein'].cplitemvalue.value_counts().compute()

careplangen_df[careplangen_df.cplgroup == 'Calories'].cplitemvalue.value_counts().compute()

# In this case, there aren't entire columns to remove. However, some specific types of care plan categories seem to be less relevant (e.g. activity, critical care discharge/transfer planning) or redundant (e.g. ventilation, infectious diseases). So, we're going to remove rows that have those categories.

careplangen_df = careplangen_df.drop('cplgeneralid', axis=1)
careplangen_df.head()

categories_to_remove = ['Ventilation', 'Airway', 'Activity', 'Care Limitation', 
                        'Route-Status', 'Critical Care Discharge/Transfer Planning', 
                        'Ordered Protocols', 'Acuity', 'Volume Status', 'Prognosis', 
                        'Care Providers', 'Family/Health Care Proxy/Contact Info', 'Current Rate', 
                        'Daily Goals/Safety Risks/Discharge Requirements', 'Goal Rate', 
                        'Planned Procedures', 'Infectious Disease', 
                        'Care Plan Reviewed with Patient/Family', 'Protein', 'Calories']

~(careplangen_df.cplgroup.isin(categories_to_remove)).head()

careplangen_df = careplangen_df[~(careplangen_df.cplgroup.isin(categories_to_remove))]
careplangen_df.head()

len(careplangen_df)

careplangen_df.patientunitstayid.nunique().compute()

# There's still plenty of data left, affecting around 92.48% of the unit stays, even after removing several categories.

# + {"toc-hr-collapsed": false, "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into simple numberings, one hot encode features with a low number of categories (in this case, 5) and enumerate sparse categorical features that will be embedded.
# -

# #### Separate and prepare features for embedding
#
# Identify categorical features that have more than 5 unique categories, which will go through an embedding layer afterwards, and enumerate them.

# Update list of categorical features and add those that will need embedding (features with more than 5 unique values):

new_cat_feat = ['cplgroup', 'cplitemvalue']
[cat_feat.append(col) for col in new_cat_feat]

cat_feat_nunique = [careplangen_df[feature].nunique().compute() for feature in new_cat_feat]
cat_feat_nunique

new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        new_cat_embed_feat.append(new_cat_feat[i])

careplangen_df[new_cat_feat].head()

# + {"pixiedust": {"displayParams": {}}}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    careplangen_df[feature], cat_embed_feat_enum[feature] = utils.enum_categorical_feature(careplangen_df, feature)
# -

careplangen_df[new_cat_feat].head()

cat_embed_feat_enum

careplangen_df[new_cat_feat].dtypes

careplangen_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
careplangen_df = client.persist(careplangen_df)

careplangen_df.visualize()

# ### Create the timestamp feature and sort

# Create the timestamp (`ts`) feature:

careplangen_df['ts'] = careplangen_df['cplitemoffset']
careplangen_df = careplangen_df.drop('cplitemoffset', axis=1)
careplangen_df.head()

# Remove duplicate rows:

len(careplangen_df)

careplangen_df = careplangen_df.drop_duplicates()
careplangen_df.head()

len(careplangen_df)

careplangen_df = careplangen_df.repartition(npartitions=30)

# Sort by `ts` so as to be easier to merge with other dataframes later:

careplangen_df = careplangen_df.set_index('ts')
careplangen_df.head()

careplangen_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
careplangen_df = client.persist(careplangen_df)

careplangen_df.visualize()

# Check for possible multiple rows with the same unit stay ID and timestamp:

careplangen_df.reset_index().head()

careplangen_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='cplgroup').head()

careplangen_df[careplangen_df.patientunitstayid == 3138123].compute().head(10)

# We can see that there are up to 32 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}}
careplangen_df = utils.join_categorical_enum(careplangen_df, new_cat_embed_feat)
careplangen_df.head()
# -

careplangen_df.dtypes

careplangen_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='cplgroup').head()

careplangen_df[careplangen_df.patientunitstayid == 3138123].compute().head(10)

# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

careplangen_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
careplangen_df = client.persist(careplangen_df)

careplangen_df.visualize()

# ### Renaming columns
#
# Keeping the `activeupondischarge` feature so as to decide if forward fill or leave at NaN each general care plan value, when we have the full dataframe. However, we need to identify this feature's original table, general care plan, so as to not confound with other data.

careplangen_df = careplangen_df.rename(columns={'activeupondischarge':'cpl_activeupondischarge'})
careplangen_df.head()

# Save the dataframe:

careplangen_df = careplangen_df.repartition(npartitions=30)

careplangen_df.to_parquet(f'{data_path}cleaned/unnormalized/carePlanGeneral.parquet')

careplangen_df.to_parquet(f'{data_path}cleaned/normalized/carePlanGeneral.parquet')

# Confirm that everything is ok through the `describe` method:

careplangen_df.describe().compute().transpose()

# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

careplangen_df = dd.read_parquet(f'{data_path}cleaned/normalized/carePlanGeneral.parquet')
careplangen_df.head()

careplangen_df.npartitions

eICU_df = dd.merge_asof(eICU_df, careplangen_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "cell_type": "markdown"}
# ## Past history data
# -

# ### Read the data

pasthist_df = dd.read_csv(f'{data_path}original/pastHistory.csv')
pasthist_df.head()

len(pasthist_df)

pasthist_df.patientunitstayid.nunique().compute()

pasthist_df.npartitions

pasthist_df = pasthist_df.repartition(npartitions=30)

# Get an overview of the dataframe through the `describe` method:

pasthist_df.describe().compute().transpose()

pasthist_df.visualize()

pasthist_df.columns

pasthist_df.dtypes

# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}}
utils.dataframe_missing_values(pasthist_df)
# -

# ### Remove unneeded features

pasthist_df.pasthistorypath.value_counts().head(20)

pasthist_df.pasthistorypath.value_counts().tail(20)

pasthist_df.pasthistoryvalue.value_counts().compute()

pasthist_df.pasthistorynotetype.value_counts().compute()

pasthist_df[pasthist_df.pasthistorypath == 'notes/Progress Notes/Past History/Past History Obtain Options/Performed'].pasthistoryvalue.value_counts().compute()

# In this case, considering that it regards past diagnosis of the patients, the timestamp when that was observed probably isn't very reliable nor useful. As such, I'm going to remove the offset variables. Furthermore, `pasthistoryvaluetext` is redundant with `pasthistoryvalue`, while `pasthistorynotetype` and the past history path 'notes/Progress Notes/Past History/Past History Obtain Options/Performed' seem to be irrelevant.

pasthist_df = pasthist_df.drop(['pasthistoryid', 'pasthistoryoffset', 'pasthistoryenteredoffset',
                                'pasthistorynotetype', 'pasthistoryvaluetext'], axis=1)
pasthist_df.head()

categories_to_remove = ['notes/Progress Notes/Past History/Past History Obtain Options/Performed']

~(pasthist_df.pasthistorypath.isin(categories_to_remove)).head()

pasthist_df = pasthist_df[~(pasthist_df.pasthistorypath.isin(categories_to_remove))]
pasthist_df.head()

len(pasthist_df)

pasthist_df.patientunitstayid.nunique().compute()

# There's still plenty of data left, affecting around 81.87% of the unit stays, even after removing several categories.

# + {"toc-hr-collapsed": false, "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into simple numberings, one hot encode features with a low number of categories (in this case, 5) and enumerate sparse categorical features that will be embedded.
# -

# #### Separate and prepare features for embedding
#
# Identify categorical features that have more than 5 unique categories, which will go through an embedding layer afterwards, and enumerate them.

# Update list of categorical features and add those that will need embedding (features with more than 5 unique values):

new_cat_feat = ['pasthistorypath', 'pasthistoryvalue']
[cat_feat.append(col) for col in new_cat_feat]

cat_feat_nunique = [pasthist_df[feature].nunique().compute() for feature in new_cat_feat]
cat_feat_nunique

new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        new_cat_embed_feat.append(new_cat_feat[i])

pasthist_df[new_cat_feat].head()

# + {"pixiedust": {"displayParams": {}}}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    pasthist_df[feature], cat_embed_feat_enum[feature] = utils.enum_categorical_feature(pasthist_df, feature)
# -

pasthist_df[new_cat_feat].head()

cat_embed_feat_enum

pasthist_df[new_cat_feat].dtypes

pasthist_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
pasthist_df = client.persist(pasthist_df)

pasthist_df.visualize()

# ### Remove duplicate rows

# Remove duplicate rows:

len(pasthist_df)

pasthist_df = pasthist_df.drop_duplicates()
pasthist_df.head()

len(pasthist_df)

pasthist_df = pasthist_df.repartition(npartitions=30)

pasthist_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
pasthist_df = client.persist(pasthist_df)

pasthist_df.visualize()

# Check for possible multiple rows with the same unit stay ID and timestamp:

pasthist_df.groupby(['patientunitstayid']).count().nlargest(columns='pasthistorypath').head()

pasthist_df[pasthist_df.patientunitstayid == 1558102].compute().head(10)

# We can see that there are up to 20 categories per `patientunitstayid`. As such, we must join them.

# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}}
pasthist_df = utils.join_categorical_enum(pasthist_df, new_cat_embed_feat, id_columns=['patientunitstayid'])
pasthist_df.head()
# -

pasthist_df.dtypes

pasthist_df.groupby(['patientunitstayid']).count().nlargest(columns='pasthistorypath').head()

pasthist_df[pasthist_df.patientunitstayid == 1558102].compute().head(10)

# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

pasthist_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
pasthist_df = client.persist(pasthist_df)

pasthist_df.visualize()

# ### Save the dataframe

pasthist_df = pasthist_df.repartition(npartitions=30)

pasthist_df.to_parquet(f'{data_path}cleaned/unnormalized/pastHistory.parquet')

pasthist_df.to_parquet(f'{data_path}cleaned/normalized/pastHistory.parquet')

# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

pasthist_df = dd.read_parquet(f'{data_path}cleaned/normalized/pastHistory.parquet')
pasthist_df.head()

pasthist_df.npartitions

eICU_df = dd.merge_asof(eICU_df, pasthist_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()
