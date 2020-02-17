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

# + {"toc-hr-collapsed": false, "Collapsed": "false", "cell_type": "markdown"}
# # Vital Data Preprocessing
# ---
#
# Reading and preprocessing vital signals data of the eICU dataset from MIT with the data from over 139k patients collected in the US.
#
# This notebook addresses the preprocessing of the following eICU tables:
# * vitalAperiodic
# * vitalPeriodic

# + {"colab_type": "text", "id": "KOdmFzXqF7nq", "toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Importing the necessary packages

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl", "Collapsed": "false", "persistent_id": "522745b5-b5bf-479f-b697-5c7e9e12fc33", "last_executed_text": "import os                                  # os handles directory/workspace changes\nimport numpy as np                         # NumPy to handle numeric and NaN operations\nimport yaml                                # Save and load YAML files", "execution_event_id": "deb57b39-6a79-4b3a-95ed-02f8089ff593"}
import os                                  # os handles directory/workspace changes
import numpy as np                         # NumPy to handle numeric and NaN operations
import yaml                                # Save and load YAML files

# + {"Collapsed": "false", "persistent_id": "02accdbf-be7e-415c-ba11-165906e66c50", "last_executed_text": "# Debugging packages\nimport pixiedust                           # Debugging in Jupyter Notebook cells", "execution_event_id": "fa33a2f7-7127-49c6-bbe9-f89555b1f2be"}
# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# + {"Collapsed": "false", "persistent_id": "a1f6ee7f-36d4-489d-b2dd-ec2a38f15d11", "last_executed_text": "# Change to parent directory (presumably \"Documents\")\nos.chdir(\"../../..\")\n\n# Path to the CSV dataset files\ndata_path = 'Documents/Datasets/Thesis/eICU/uncompressed/'\n\n# Path to the code files\nproject_path = 'Documents/GitHub/eICU-mortality-prediction/'", "execution_event_id": "baeb346a-1c34-42d1-a501-7ae37369255e"}
# Change to parent directory (presumably "Documents")
os.chdir("../../../..")
# Path to the CSV dataset files
data_path = 'Datasets/Thesis/eICU/uncompressed/'
# Path to the code files
project_path = 'GitHub/eICU-mortality-prediction/'

# + {"Collapsed": "false", "persistent_id": "c0c2e356-d4f4-4a9d-bec2-88bdf9eb6a38", "last_executed_text": "import modin.pandas as pd                  # Optimized distributed version of Pandas\nimport data_utils as du                    # Data science and machine learning relevant methods", "execution_event_id": "82ef68be-443a-4bb8-8abd-7457a7005b4d"}
# import modin.pandas as pd                  # Optimized distributed version of Pandas
import pandas as pd
import data_utils as du                    # Data science and machine learning relevant methods

# + {"Collapsed": "false", "cell_type": "markdown"}
# Set the random seed for reproducibility

# + {"Collapsed": "false", "persistent_id": "39b552cd-6948-4ec8-ac04-42f850c1e05a", "last_executed_text": "du.set_random_seed(42)", "execution_event_id": "29ab85ce-b7fd-4c5a-a110-5841e741c369"}
du.set_random_seed(42)

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Vital signs aperiodic data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "51197f67-95e2-4184-a73f-7885cb975084", "last_executed_text": "vital_aprdc_df = pd.read_csv(f'{data_path}original/vitalAperiodic.csv')\nvital_aprdc_df.head()", "execution_event_id": "bf6e34cd-7ae9-4636-a9b9-27d404b49610"}
vital_aprdc_df = pd.read_csv(f'{data_path}original/vitalAperiodic.csv')
vital_aprdc_df.head()

# + {"Collapsed": "false", "persistent_id": "c3d45ba7-91dd-41d7-8699-c70390556018", "last_executed_text": "len(vital_aprdc_df)", "execution_event_id": "d2a30b69-ae10-4a5e-be98-56280de75b37"}
len(vital_aprdc_df)

# + {"Collapsed": "false", "persistent_id": "17c46962-79a6-48a6-b67a-4863028ed897", "last_executed_text": "vital_aprdc_df.patientunitstayid.nunique()", "execution_event_id": "e7d22357-65a7-4573-bd2e-42fc3794e931"}
vital_aprdc_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "6e745ae7-a0ea-4169-96de-d49e9f510ed9", "last_executed_text": "vital_aprdc_df.describe().transpose()", "execution_event_id": "546023f0-9a90-47da-a108-5f59f57fa5af"}
vital_aprdc_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "ac1da815-dcd6-44df-8164-a987f9623255", "last_executed_text": "vital_aprdc_df.columns", "execution_event_id": "dac11dab-f33a-4ab5-bf3e-c78447f9cdc4"}
vital_aprdc_df.info()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "c4e967db-7ace-41df-8678-7cd11d1e002b", "last_executed_text": "du.search_explore.dataframe_missing_values(vital_aprdc_df)", "execution_event_id": "a04c5f38-289a-4569-8c57-89eddcda2b0d"}
du.search_explore.dataframe_missing_values(vital_aprdc_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "9f3b85d0-8480-4f30-8455-fde084ba7c69", "last_executed_text": "vital_aprdc_df.notetype.value_counts().head(20)", "execution_event_id": "39bf557e-d53a-4ecd-8118-a9719aa49188"}
vital_aprdc_df.noninvasivesystolic.value_counts()

# + {"Collapsed": "false", "persistent_id": "e41f6a9d-c5ea-45e3-b38b-89ba70de1b87", "last_executed_text": "vital_aprdc_df.notepath.value_counts().head(40)", "execution_event_id": "0df9919f-4d46-4c34-a326-df8e8cf7f18c"}
vital_aprdc_df.noninvasivediastolic.value_counts()

# + {"Collapsed": "false", "persistent_id": "b12dcddf-9703-4aff-8a20-df1d147ef554", "last_executed_text": "vital_aprdc_df.notevalue.value_counts().head(20)", "execution_event_id": "04986ee5-3cc1-453c-8127-05f513370b5d"}
vital_aprdc_df.noninvasivemean.value_counts()

# + {"Collapsed": "false", "persistent_id": "9f3b85d0-8480-4f30-8455-fde084ba7c69", "last_executed_text": "vital_aprdc_df.notetype.value_counts().head(20)", "execution_event_id": "39bf557e-d53a-4ecd-8118-a9719aa49188"}
vital_aprdc_df.paop.value_counts()

# + {"Collapsed": "false", "persistent_id": "e41f6a9d-c5ea-45e3-b38b-89ba70de1b87", "last_executed_text": "vital_aprdc_df.notepath.value_counts().head(40)", "execution_event_id": "0df9919f-4d46-4c34-a326-df8e8cf7f18c"}
vital_aprdc_df.cardiacoutput.value_counts()

# + {"Collapsed": "false", "persistent_id": "b12dcddf-9703-4aff-8a20-df1d147ef554", "last_executed_text": "vital_aprdc_df.notevalue.value_counts().head(20)", "execution_event_id": "04986ee5-3cc1-453c-8127-05f513370b5d"}
vital_aprdc_df.cardiacinput.value_counts()

# + {"Collapsed": "false", "persistent_id": "9f3b85d0-8480-4f30-8455-fde084ba7c69", "last_executed_text": "vital_aprdc_df.notetype.value_counts().head(20)", "execution_event_id": "39bf557e-d53a-4ecd-8118-a9719aa49188"}
vital_aprdc_df.svr.value_counts()

# + {"Collapsed": "false", "persistent_id": "e41f6a9d-c5ea-45e3-b38b-89ba70de1b87", "last_executed_text": "vital_aprdc_df.notepath.value_counts().head(40)", "execution_event_id": "0df9919f-4d46-4c34-a326-df8e8cf7f18c"}
vital_aprdc_df.svri.value_counts()

# + {"Collapsed": "false", "persistent_id": "b12dcddf-9703-4aff-8a20-df1d147ef554", "last_executed_text": "vital_aprdc_df.notevalue.value_counts().head(20)", "execution_event_id": "04986ee5-3cc1-453c-8127-05f513370b5d"}
vital_aprdc_df.pvr.value_counts()

# + {"Collapsed": "false", "persistent_id": "b12dcddf-9703-4aff-8a20-df1d147ef554", "last_executed_text": "vital_aprdc_df.notevalue.value_counts().head(20)", "execution_event_id": "04986ee5-3cc1-453c-8127-05f513370b5d"}
vital_aprdc_df.pvri.value_counts()

# + {"Collapsed": "false", "persistent_id": "5beb1b97-7a5b-446c-934d-74b99556151f", "last_executed_text": "vital_aprdc_df = vital_aprdc_df.drop(['noteid', 'noteenteredoffset', 'notetype', 'notetext'], axis=1)\nvital_aprdc_df.head()", "execution_event_id": "2e499a1a-e9a0-42cf-b351-98273b224c15"}
vital_aprdc_df = vital_aprdc_df.drop(columns=['vitalaperiodicid'])
vital_aprdc_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "dfab2799-af6c-4475-8341-ec3f40546ed1"}
vital_aprdc_df = vital_aprdc_df.rename(columns={'observationoffset': 'ts'})
vital_aprdc_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "8af4dd26-9eb8-4edf-8bcf-361b10c94979"}
len(vital_aprdc_df)

# + {"Collapsed": "false", "persistent_id": "de85aa8f-0d02-4c35-868c-16116a83cf7f"}
vital_aprdc_df = vital_aprdc_df.drop_duplicates()
vital_aprdc_df.head()

# + {"Collapsed": "false", "persistent_id": "bb6efd0a-aa95-40d6-84b2-8916705a4cf4", "last_executed_text": "len(vital_aprdc_df)", "execution_event_id": "0f6fb1fb-5d50-4f2c-acd1-804100222250"}
len(vital_aprdc_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "a03573d9-f345-4ff4-84b9-2b2a3f73ce27"}
vital_aprdc_df = vital_aprdc_df.sort_values('ts')
vital_aprdc_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "c94e7b7b-dc34-478b-842b-c34c926c934d"}
vital_aprdc_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='noninvasivemean', n=5).head()

# + {"Collapsed": "false", "persistent_id": "b8bcf17b-3d52-4cc9-bfbb-7f7d8fe83b3b"}
vital_aprdc_df[(vital_aprdc_df.patientunitstayid == 625065) & (vital_aprdc_df.ts == 1515)].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 4 rows per set of `patientunitstayid` and `ts`. As such, we must join them. However, this is a different scenario than in the other cases. Since all features are numeric, we just need to average the features.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "591b2ccd-fa5c-4eb2-bec1-8ac21de1c890", "last_executed_text": "vital_aprdc_df = du.embedding.join_categorical_enum(vital_aprdc_df, cont_join_method='max')\nvital_aprdc_df.head()", "execution_event_id": "c6c89c91-ec15-4636-99d0-6ed07bcc921c"}
vital_aprdc_df = du.embedding.join_categorical_enum(vital_aprdc_df, cont_join_method='mean', inplace=True)
vital_aprdc_df.head()

# + {"Collapsed": "false", "persistent_id": "c94e7b7b-dc34-478b-842b-c34c926c934d"}
vital_aprdc_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='noninvasivemean', n=5).head()

# + {"Collapsed": "false", "persistent_id": "b8bcf17b-3d52-4cc9-bfbb-7f7d8fe83b3b"}
vital_aprdc_df[(vital_aprdc_df.patientunitstayid == 625065) & (vital_aprdc_df.ts == 1515)].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "32450572-639e-4539-b35a-181078ed3335"}
vital_aprdc_df.columns = du.data_processing.clean_naming(vital_aprdc_df.columns)
vital_aprdc_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Normalize data

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "e42f577a-db00-4ecf-9e3c-433007a3bdaf"}
vital_aprdc_df.to_csv(f'{data_path}cleaned/unnormalized/vitalAperiodic.csv')

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "d5ad6017-ad4a-419c-badb-9454add7752d", "last_executed_text": "vital_aprdc_df = du.data_processing.normalize_data(vital_aprdc_df, categ_columns=new_cat_feat,\n                                                    id_columns=['patientunitstayid', 'ts', 'death_ts'])\nvital_aprdc_df.head(6)", "execution_event_id": "3d6d0a5c-9160-4ffc-87d4-85632a968a1d"}
vital_aprdc_df = du.data_processing.normalize_data(vital_aprdc_df, inplace=True)
vital_aprdc_df.head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "812e7eb1-ff92-4a26-a970-2f40fc5bbdb1"}
vital_aprdc_df.to_csv(f'{data_path}cleaned/normalized/vitalAperiodic.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "eebc356f-507e-4872-be9d-a1d774f2fd7a"}
vital_aprdc_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "ac1da815-dcd6-44df-8164-a987f9623255", "last_executed_text": "vital_aprdc_df.columns", "execution_event_id": "dac11dab-f33a-4ab5-bf3e-c78447f9cdc4"}
vital_aprdc_df.info()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Vital signs periodic data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "51197f67-95e2-4184-a73f-7885cb975084", "last_executed_text": "vital_aprdc_df = pd.read_csv(f'{data_path}original/vitalAperiodic.csv')\nvital_aprdc_df.head()", "execution_event_id": "bf6e34cd-7ae9-4636-a9b9-27d404b49610"}
vital_prdc_df = pd.read_csv(f'{data_path}original/vitalPeriodic.csv')
vital_prdc_df.head()

# + {"Collapsed": "false", "persistent_id": "c3d45ba7-91dd-41d7-8699-c70390556018", "last_executed_text": "len(vital_prdc_df)", "execution_event_id": "d2a30b69-ae10-4a5e-be98-56280de75b37"}
len(vital_prdc_df)

# + {"Collapsed": "false", "persistent_id": "17c46962-79a6-48a6-b67a-4863028ed897", "last_executed_text": "vital_prdc_df.patientunitstayid.nunique()", "execution_event_id": "e7d22357-65a7-4573-bd2e-42fc3794e931"}
vital_prdc_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "6e745ae7-a0ea-4169-96de-d49e9f510ed9", "last_executed_text": "vital_prdc_df.describe().transpose()", "execution_event_id": "546023f0-9a90-47da-a108-5f59f57fa5af"}
vital_prdc_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "ac1da815-dcd6-44df-8164-a987f9623255", "last_executed_text": "vital_prdc_df.columns", "execution_event_id": "dac11dab-f33a-4ab5-bf3e-c78447f9cdc4"}
vital_prdc_df.info()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "c4e967db-7ace-41df-8678-7cd11d1e002b", "last_executed_text": "du.search_explore.dataframe_missing_values(vital_prdc_df)", "execution_event_id": "a04c5f38-289a-4569-8c57-89eddcda2b0d"}
du.search_explore.dataframe_missing_values(vital_prdc_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "9f3b85d0-8480-4f30-8455-fde084ba7c69", "last_executed_text": "vital_prdc_df.notetype.value_counts().head(20)", "execution_event_id": "39bf557e-d53a-4ecd-8118-a9719aa49188"}
vital_prdc_df.temperature.value_counts()

# + {"Collapsed": "false", "persistent_id": "e41f6a9d-c5ea-45e3-b38b-89ba70de1b87", "last_executed_text": "vital_prdc_df.notepath.value_counts().head(40)", "execution_event_id": "0df9919f-4d46-4c34-a326-df8e8cf7f18c"}
vital_prdc_df.sao2.value_counts()

# + {"Collapsed": "false", "persistent_id": "b12dcddf-9703-4aff-8a20-df1d147ef554", "last_executed_text": "vital_prdc_df.notevalue.value_counts().head(20)", "execution_event_id": "04986ee5-3cc1-453c-8127-05f513370b5d"}
vital_prdc_df.heartrate.value_counts()

# + {"Collapsed": "false", "persistent_id": "9f3b85d0-8480-4f30-8455-fde084ba7c69", "last_executed_text": "vital_prdc_df.notetype.value_counts().head(20)", "execution_event_id": "39bf557e-d53a-4ecd-8118-a9719aa49188"}
vital_prdc_df.respiration.value_counts()

# + {"Collapsed": "false", "persistent_id": "e41f6a9d-c5ea-45e3-b38b-89ba70de1b87", "last_executed_text": "vital_prdc_df.notepath.value_counts().head(40)", "execution_event_id": "0df9919f-4d46-4c34-a326-df8e8cf7f18c"}
vital_prdc_df.cvp.value_counts()

# + {"Collapsed": "false", "persistent_id": "b12dcddf-9703-4aff-8a20-df1d147ef554", "last_executed_text": "vital_prdc_df.notevalue.value_counts().head(20)", "execution_event_id": "04986ee5-3cc1-453c-8127-05f513370b5d"}
vital_prdc_df.systemicsystolic.value_counts()

# + {"Collapsed": "false", "persistent_id": "9f3b85d0-8480-4f30-8455-fde084ba7c69", "last_executed_text": "vital_prdc_df.notetype.value_counts().head(20)", "execution_event_id": "39bf557e-d53a-4ecd-8118-a9719aa49188"}
vital_prdc_df.systemicdiastolic.value_counts()

# + {"Collapsed": "false", "persistent_id": "e41f6a9d-c5ea-45e3-b38b-89ba70de1b87", "last_executed_text": "vital_prdc_df.notepath.value_counts().head(40)", "execution_event_id": "0df9919f-4d46-4c34-a326-df8e8cf7f18c"}
vital_prdc_df.systemicmean.value_counts()

# + {"Collapsed": "false", "persistent_id": "b12dcddf-9703-4aff-8a20-df1d147ef554", "last_executed_text": "vital_prdc_df.notevalue.value_counts().head(20)", "execution_event_id": "04986ee5-3cc1-453c-8127-05f513370b5d"}
vital_prdc_df.pasystolic.value_counts()

# + {"Collapsed": "false", "persistent_id": "b12dcddf-9703-4aff-8a20-df1d147ef554", "last_executed_text": "vital_prdc_df.notevalue.value_counts().head(20)", "execution_event_id": "04986ee5-3cc1-453c-8127-05f513370b5d"}
vital_prdc_df.padiastolic.value_counts()

# + {"Collapsed": "false", "persistent_id": "9f3b85d0-8480-4f30-8455-fde084ba7c69", "last_executed_text": "vital_prdc_df.notetype.value_counts().head(20)", "execution_event_id": "39bf557e-d53a-4ecd-8118-a9719aa49188"}
vital_prdc_df.pamean.value_counts()

# + {"Collapsed": "false", "persistent_id": "e41f6a9d-c5ea-45e3-b38b-89ba70de1b87", "last_executed_text": "vital_prdc_df.notepath.value_counts().head(40)", "execution_event_id": "0df9919f-4d46-4c34-a326-df8e8cf7f18c"}
vital_prdc_df.st1.value_counts()

# + {"Collapsed": "false", "persistent_id": "b12dcddf-9703-4aff-8a20-df1d147ef554", "last_executed_text": "vital_prdc_df.notevalue.value_counts().head(20)", "execution_event_id": "04986ee5-3cc1-453c-8127-05f513370b5d"}
vital_prdc_df.st2.value_counts()

# + {"Collapsed": "false", "persistent_id": "9f3b85d0-8480-4f30-8455-fde084ba7c69", "last_executed_text": "vital_prdc_df.notetype.value_counts().head(20)", "execution_event_id": "39bf557e-d53a-4ecd-8118-a9719aa49188"}
vital_prdc_df.st3.value_counts()

# + {"Collapsed": "false", "persistent_id": "9f3b85d0-8480-4f30-8455-fde084ba7c69", "last_executed_text": "vital_prdc_df.notetype.value_counts().head(20)", "execution_event_id": "39bf557e-d53a-4ecd-8118-a9719aa49188"}
vital_prdc_df.icp.value_counts()

# + {"Collapsed": "false", "persistent_id": "5beb1b97-7a5b-446c-934d-74b99556151f", "last_executed_text": "vital_prdc_df = vital_prdc_df.drop(['noteid', 'noteenteredoffset', 'notetype', 'notetext'], axis=1)\nvital_prdc_df.head()", "execution_event_id": "2e499a1a-e9a0-42cf-b351-98273b224c15"}
vital_prdc_df = vital_prdc_df.drop(columns=['vitalperiodicid'])
vital_prdc_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "dfab2799-af6c-4475-8341-ec3f40546ed1"}
vital_prdc_df = vital_prdc_df.rename(columns={'observationoffset': 'ts'})
vital_prdc_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "8af4dd26-9eb8-4edf-8bcf-361b10c94979"}
len(vital_prdc_df)

# + {"Collapsed": "false", "persistent_id": "de85aa8f-0d02-4c35-868c-16116a83cf7f"}
vital_prdc_df = vital_prdc_df.drop_duplicates()
vital_prdc_df.head()

# + {"Collapsed": "false", "persistent_id": "bb6efd0a-aa95-40d6-84b2-8916705a4cf4", "last_executed_text": "len(vital_prdc_df)", "execution_event_id": "0f6fb1fb-5d50-4f2c-acd1-804100222250"}
len(vital_prdc_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "a03573d9-f345-4ff4-84b9-2b2a3f73ce27"}
vital_prdc_df = vital_prdc_df.sort_values('ts')
vital_prdc_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "c94e7b7b-dc34-478b-842b-c34c926c934d"}
vital_prdc_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='heartrate', n=5).head()

# + {"Collapsed": "false", "persistent_id": "b8bcf17b-3d52-4cc9-bfbb-7f7d8fe83b3b"}
vital_prdc_df[(vital_prdc_df.patientunitstayid == 625065) & (vital_prdc_df.ts == 1515)].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 4 rows per set of `patientunitstayid` and `ts`. As such, we must join them. However, this is a different scenario than in the other cases. Since all features are numeric, we just need to average the features.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "591b2ccd-fa5c-4eb2-bec1-8ac21de1c890", "last_executed_text": "vital_prdc_df = du.embedding.join_categorical_enum(vital_prdc_df, cont_join_method='max')\nvital_prdc_df.head()", "execution_event_id": "c6c89c91-ec15-4636-99d0-6ed07bcc921c"}
vital_prdc_df = du.embedding.join_categorical_enum(vital_prdc_df, cont_join_method='mean', inplace=True)
vital_prdc_df.head()

# + {"Collapsed": "false", "persistent_id": "c94e7b7b-dc34-478b-842b-c34c926c934d"}
vital_prdc_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='heartrate', n=5).head()

# + {"Collapsed": "false", "persistent_id": "b8bcf17b-3d52-4cc9-bfbb-7f7d8fe83b3b"}
vital_prdc_df[(vital_prdc_df.patientunitstayid == 625065) & (vital_prdc_df.ts == 1515)].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "32450572-639e-4539-b35a-181078ed3335"}
vital_prdc_df.columns = du.data_processing.clean_naming(vital_prdc_df.columns)
vital_prdc_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Normalize data

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "e42f577a-db00-4ecf-9e3c-433007a3bdaf"}
vital_prdc_df.to_csv(f'{data_path}cleaned/unnormalized/vitalPeriodic.csv')

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "d5ad6017-ad4a-419c-badb-9454add7752d", "last_executed_text": "vital_prdc_df = du.data_processing.normalize_data(vital_prdc_df, categ_columns=new_cat_feat,\n                                                    id_columns=['patientunitstayid', 'ts', 'death_ts'])\nvital_prdc_df.head(6)", "execution_event_id": "3d6d0a5c-9160-4ffc-87d4-85632a968a1d"}
vital_prdc_df = du.data_processing.normalize_data(vital_prdc_df, inplace=True)
vital_prdc_df.head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "812e7eb1-ff92-4a26-a970-2f40fc5bbdb1"}
vital_prdc_df.to_csv(f'{data_path}cleaned/normalized/vitalPeriodic.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "eebc356f-507e-4872-be9d-a1d774f2fd7a"}
vital_prdc_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "ac1da815-dcd6-44df-8164-a987f9623255", "last_executed_text": "vital_prdc_df.columns", "execution_event_id": "dac11dab-f33a-4ab5-bf3e-c78447f9cdc4"}
vital_prdc_df.info()
