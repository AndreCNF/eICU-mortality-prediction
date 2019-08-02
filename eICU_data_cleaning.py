# -*- coding: utf-8 -*-
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
#     display_name: fcul-als-python
#     language: python
#     name: fcul-als-python
# ---

# # eICU Data Cleaning
# ---
#
# Exploring the eICU dataset from MIT with the data from over 139k patients collected in the US.
#
# The main goal of this notebook is to prepare a single CSV document that contains all the relevant data to be used when training a machine learning model that predicts disease progression, filtering useless columns and performing imputation.

# + {"colab_type": "text", "id": "KOdmFzXqF7nq", "cell_type": "markdown"}
# ## Importing the necessary packages

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl"}
import pandas as pd              # Pandas to handle the data in dataframes
import re                        # re to do regex searches in string data
import plotly                    # Plotly for interactive and pretty plots
import plotly.graph_objs as go
from datetime import datetime    # datetime to use proper date and time formats
import os                        # os handles directory/workspace changes
import numpy as np               # NumPy to handle numeric and NaN operations
from tqdm import tqdm_notebook   # tqdm allows to track code execution progress
import numbers                   # numbers allows to check if data is numeric
import torch                     # PyTorch to create and apply deep learning models
from torch.utils.data.sampler import SubsetRandomSampler
import utils                     # Contains auxiliary functions

# +
# Change to parent directory (presumably "Documents")
os.chdir("../..")

# Path to the CSV dataset files
data_path = 'Datasets/Thesis/FCUL_ALS/'

# + {"colab_type": "text", "id": "bEqFkmlYCGOz", "cell_type": "markdown"}
# **Important:** Use the following two lines to be able to do plotly plots offline:

# + {"colab": {}, "colab_type": "code", "id": "fZCUmUOzCPeI"}
import plotly.offline as py
plotly.offline.init_notebook_mode(connected=True)


# + {"colab_type": "text", "id": "Yrzi8YbzDVTH", "cell_type": "markdown"}
# **Important:** The following function is needed in every Google Colab cell that contains a Plotly chart:

# + {"colab": {}, "colab_type": "code", "id": "wxyGCedgC6bX"}
def configure_plotly_browser_state():
    import IPython
    display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
            },
          });
        </script>
        '''))


# -

# ## Reading the data

ALS_proc_df = pd.read_csv(f'{data_path}dataWithoutDunnoNIV.csv')
ALS_proc_df.head()

# ## Renaming columns

ALS_proc_df.rename(columns={'REF': 'subject_id'}, inplace=True)
ALS_proc_df.head()

# ## Deleting unused columns
#
# Removing kind of useless columns ('NIV_DATE', 'firstDate', 'lastDate', 'medianDate'), ones with too many missing values ('SNIP', 'CervicalFlex', 'CervicalExt') and ones that would give away the labels ('ALS-FRS', 'ALS-FRS-R', 'ALS-FRSb', 'ALS-FRSsUL', 'ALS-FRSsLL', 'ALS-FRSr').

ALS_proc_df.columns

ALS_proc_df.drop(columns=['NIV_DATE', 'firstDate', 'lastDate', 'medianDate', 
                          'SNIP', 'CervicalFlex', 'CervicalExt', 'ALS-FRS', 
                          'ALS-FRS-R', 'ALS-FRSb', 'ALS-FRSsUL', 'ALS-FRSsLL', 
                          'ALS-FRSr'], inplace=True)
ALS_proc_df.head()

# ## Getting discrete timestamps
#
# Creating a index for each patient that serves as a discrete timestamp, starting at 0 in their first clinical visit and ending at the length of their time series (-1).

ALS_proc_df['ts'] = ALS_proc_df.groupby('subject_id').cumcount()
ALS_proc_df.head(10)

# ## Removing patients with only one clinical visit
#
# Since we want to predict the use of NIV in the next clinical visit, it doesn't make any sense to include patients with only one data point.

ALS_proc_df.subject_id.nunique()

ALS_proc_df.groupby('subject_id').ts.count().min()

for patient in ALS_proc_df.subject_id.unique():
    # Check if the current patient only has one clinical visit
    if len(ALS_proc_df[ALS_proc_df.subject_id == patient]) == 1:
        # Remove patient's data from the dataframe
        ALS_proc_df = ALS_proc_df[ALS_proc_df.subject_id != patient]

ALS_proc_df.subject_id.nunique()

ALS_proc_df.groupby('subject_id').ts.count().min()

# ## Cleaning categorical columns
#
# Combining redundant values and one hot encoding categorical features.

# Making "Gender" a proper one hot encoded column
ALS_proc_df['Gender'] = ALS_proc_df['Gender'] - 1

# Fixing a bug in the "1R" column
ALS_proc_df['1R'] = ALS_proc_df['1R'].replace(to_replace='\\1', value=1).astype('float64')

ALS_proc_df = utils.one_hot_encoding_dataframe(ALS_proc_df, columns=['El Escorial reviewed criteria',
                                                                     'Onset form',
                                                                     'UMN vs LMN',
                                                                     'C9orf72'])
ALS_proc_df.head()

# Reduxing the UMN vs LMN columns into just 2 clear columns:

ALS_proc_df.rename(columns={'UMN vs LMN_lmn': 'LMN',
                            'UMN vs LMN_umn': 'UMN',
                            'UMN vs LMN_nan': 'UMN_vs_LMN_unknown'}, inplace=True)
ALS_proc_df.head()

# Activate both UMN and LMN features if the "both" value is 1
ALS_proc_df.LMN = ALS_proc_df.apply(lambda df: 1 if df['UMN vs LMN_both'] == 1 or df['LMN'] == 1 else 0, axis=1)
ALS_proc_df.UMN = ALS_proc_df.apply(lambda df: 1 if df['UMN vs LMN_both'] == 1 or df['UMN'] == 1 else 0, axis=1)

# Drop the "both" column as it's redundant
ALS_proc_df.drop(columns='UMN vs LMN_both', inplace=True)

ALS_proc_df.head()

len(ALS_proc_df[(ALS_proc_df.UMN == 1) & (ALS_proc_df.LMN == 1)])

# **Comment:** The previous length matches the number found on the value counts of the original dataframe, corresponding to the value "both".

# ## Standardize all column names to be lower case and without spaces

ALS_proc_df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in ALS_proc_df.columns]

ALS_proc_df.head()

# ## NIV label
#
# In order to predict the use of NIV in the next 3 months, we need to create a shifted version of the "niv" column.

ALS_proc_df['niv_label'] = ALS_proc_df['niv']

ALS_proc_df[['subject_id', 'ts', 'niv', 'niv_label']].head(20)

ALS_proc_df['niv_label'] = ALS_proc_df.groupby('subject_id')['niv_label'].shift(-1)

ALS_proc_df[['subject_id', 'ts', 'niv', 'niv_label']].head(20)

# Save a version of the dataframe without normalization
ALS_proc_df.to_csv(f'{data_path}cleaned/FCUL_ALS_cleaned_denorm.csv')

ALS_proc_df.describe().transpose()

# ## Normalizing continuous values
#
# Continuous data is normalized into z-scores, where 0 represents the mean and an absolute value of 1 corresponds to the standard deviation.

ALS_proc_df = utils.normalize_data(ALS_proc_df)
ALS_proc_df.head()


ALS_proc_df.describe().transpose()

# ## Imputation and removal of incomplete data
#
# Starting from a last information carried forward technique, the data is initially forward filled. Next, a backward fill is done, as current data of the patient should still be a good indicator of the recent past. Finally, the remaining missing values are filled with zeroes, as it represents the average value of each given feature.

ALS_proc_df[['subject_id', 'ts', 'r', 'p1', 'p2', 'bmi', 'fvc', 'vc', 'mip', 'niv_label']].head(20)

# Forward fill each patient's data
ALS_proc_df = ALS_proc_df.set_index('subject_id', append=True).groupby('subject_id').fillna(method='ffill').reset_index(level=1)

ALS_proc_df[['subject_id', 'ts', 'r', 'p1', 'p2', 'bmi', 'fvc', 'vc', 'mip', 'niv_label']].head(20)

# Backward fill each patient's data
ALS_proc_df = ALS_proc_df.set_index('subject_id', append=True).groupby('subject_id').fillna(method='bfill').reset_index(level=1)

ALS_proc_df[['subject_id', 'ts', 'r', 'p1', 'p2', 'bmi', 'fvc', 'vc', 'mip', 'niv_label']].head(20)

# Fill remaining missing values with 0, as they represent that feature's average value
ALS_proc_df = ALS_proc_df.fillna(value=0)

ALS_proc_df[['subject_id', 'ts', 'r', 'p1', 'p2', 'bmi', 'fvc', 'vc', 'mip', 'niv_label']].head(20)

ALS_proc_df.to_csv(f'{data_path}cleaned/FCUL_ALS_cleaned.csv')

ALS_proc_df.head()

ALS_proc_df.columns
