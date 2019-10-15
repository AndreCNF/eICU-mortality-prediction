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

# # eICU Data Exploration
# ---
#
# Exploring the eICU dataset from MIT with the data from over 139k patients collected in the US.
#
# The eICU Collaborative Research Database is a multi-center intensive care unit (ICU) database with high granularity data for over 200,000 admissions to ICUs monitored by eICU Programs across the United States. The database is deidentified, and includes vital sign measurements, care plan documentation, severity of illness measures, diagnosis information, treatment information, and more.

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
# -

# ## Exploring the preprocessed dataset

# ### Basic stats

ALS_proc_df = pd.read_csv(f'{data_path}dataWithoutDunnoNIV.csv')
ALS_proc_df.head()

ALS_proc_df.dtypes

ALS_proc_df.nunique()

utils.dataframe_missing_values(ALS_proc_df)

# **Comment:** Many relevant features (timestamps, NIV, age, ALSFRS, etc) have zero or low missing values percentage (bellow 10%), much better than in the PRO-ACT dataset. However, there other interesting ones with more than half missing values (FVC, VC, etc).

ALS_proc_df.describe().transpose()

ALS_proc_df['El Escorial reviewed criteria'].value_counts()

ALS_proc_df['Onset form'].value_counts()

ALS_proc_df['UMN vs LMN'].value_counts()

ALS_proc_df['C9orf72'].value_counts()

ALS_proc_df['SNIP'].value_counts()

ALS_proc_df['1R'].value_counts()

# ### Plots

# +
configure_plotly_browser_state()

ALS_proc_gender_count = ALS_proc_df.Gender.value_counts().to_frame()
data = [go.Pie(labels=ALS_proc_gender_count.index, values=ALS_proc_gender_count.Gender)]
layout = go.Layout(title='Patients Gender Demographics')
fig = go.Figure(data, layout)
py.iplot(fig)

# +
configure_plotly_browser_state()

ALS_proc_niv_count = ALS_proc_df.NIV.value_counts().to_frame()
data = [go.Pie(labels=ALS_proc_niv_count.index, values=ALS_proc_niv_count.NIV)]
layout = go.Layout(title='Visits where the patient is using NIV')
fig = go.Figure(data, layout)
py.iplot(fig)

# +
configure_plotly_browser_state()

data = [go.Histogram(x = ALS_proc_df.NIV)]
layout = go.Layout(title='Number of visits where the patient is using NIV.')
fig = go.Figure(data, layout)
py.iplot(fig)

# +
configure_plotly_browser_state()

data = [go.Scatter(
                    x = ALS_proc_df.FVC,
                    y = ALS_proc_df.NIV,
                    mode = 'markers'
                  )]
layout = go.Layout(
                    title='Relation between NIV use and FVC values',
                    xaxis=dict(title='FVC'),
                    yaxis=dict(title='NIV')
                  )
fig = go.Figure(data, layout)
py.iplot(fig)
# -

# Average FVC value when NIV is used:
ALS_proc_df[ALS_proc_df.NIV == 1].FVC.mean()

# **Comments:** The average FVC when NIV is 1 is lower than average, but the scatter plot doesn't show a very clear dependence between the variables.

# +
configure_plotly_browser_state()

data = [go.Scatter(
                    x = ALS_proc_df['Disease duration'],
                    y = ALS_proc_df.NIV,
                    mode = 'markers'
                  )]
layout = go.Layout(
                    title='Relation between NIV use and disease duration',
                    xaxis=dict(title='Disease duration'),
                    yaxis=dict(title='NIV')
                  )
fig = go.Figure(data, layout)
py.iplot(fig)
# -

# Average disease duration when NIV is used:
ALS_proc_df[ALS_proc_df.NIV == 1]['Disease duration'].mean()

# +
configure_plotly_browser_state()

data = [go.Scatter(
                    x = ALS_proc_df['Age at onset'],
                    y = ALS_proc_df.NIV,
                    mode = 'markers'
                  )]
layout = go.Layout(
                    title='Relation between NIV use and age',
                    xaxis=dict(title='Age at onset'),
                    yaxis=dict(title='NIV')
                  )
fig = go.Figure(data, layout)
py.iplot(fig)
# -

# Average age at onset when NIV is used:
ALS_proc_df[ALS_proc_df.NIV == 1]['Age at onset'].mean()

# +
configure_plotly_browser_state()

ALS_proc_NIV_3R = ALS_proc_df.groupby(['3R', 'NIV']).REF.count().to_frame().reset_index()
data = [go.Bar(
                    x=ALS_proc_NIV_3R[ALS_proc_NIV_3R.NIV == 0]['3R'],
                    y=ALS_proc_NIV_3R[ALS_proc_NIV_3R.NIV == 0]['REF'],
                    name='Not used'
              ),
        go.Bar(
                    x=ALS_proc_NIV_3R[ALS_proc_NIV_3R.NIV == 1]['3R'],
                    y=ALS_proc_NIV_3R[ALS_proc_NIV_3R.NIV == 1]['REF'],
                    name='Using NIV'
        )]
layout = go.Layout(barmode='group')
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')
# -

# Average 3R value when NIV is used:
ALS_proc_df[ALS_proc_df.NIV == 1]['3R'].mean()

# **Comments:** Clearly, there's a big dependence of the use of NIV with the respiratory symptoms indicated by 3R, as expected.

# ## Exploring the raw dataset

ALS_raw_df = pd.read_excel(f'{data_path}TabelaGeralnew_21012019_sem.xlsx')
ALS_raw_df.head()
