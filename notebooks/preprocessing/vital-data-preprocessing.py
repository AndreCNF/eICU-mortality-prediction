# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] {"toc-hr-collapsed": false, "Collapsed": "false"}
# # Vital Data Preprocessing
# ---
#
# Reading and preprocessing vital signals data of the eICU dataset from MIT with the data from over 139k patients collected in the US.
#
# This notebook addresses the preprocessing of the following eICU tables:
# * vitalAperiodic
# * vitalPeriodic

# + [markdown] {"colab_type": "text", "id": "KOdmFzXqF7nq", "toc-hr-collapsed": true, "Collapsed": "false"}
# ## Importing the necessary packages

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl", "Collapsed": "false", "persistent_id": "522745b5-b5bf-479f-b697-5c7e9e12fc33", "last_executed_text": "import os                                  # os handles directory/workspace changes\nimport numpy as np                         # NumPy to handle numeric and NaN operations\nimport yaml                                # Save and load YAML files", "execution_event_id": "deb57b39-6a79-4b3a-95ed-02f8089ff593", "execution": {"iopub.status.busy": "2020-02-24T00:04:03.471469Z", "iopub.execute_input": "2020-02-24T00:04:03.471742Z", "iopub.status.idle": "2020-02-24T00:04:03.494293Z", "shell.execute_reply.started": "2020-02-24T00:04:03.471700Z", "shell.execute_reply": "2020-02-24T00:04:03.493582Z"}}
import os                                  # os handles directory/workspace changes
import numpy as np                         # NumPy to handle numeric and NaN operations
import yaml                                # Save and load YAML files

# + {"Collapsed": "false", "persistent_id": "02accdbf-be7e-415c-ba11-165906e66c50", "last_executed_text": "# Debugging packages\nimport pixiedust                           # Debugging in Jupyter Notebook cells", "execution_event_id": "fa33a2f7-7127-49c6-bbe9-f89555b1f2be", "execution": {"iopub.status.busy": "2020-02-24T00:04:03.782110Z", "iopub.execute_input": "2020-02-24T00:04:03.782388Z", "iopub.status.idle": "2020-02-24T00:04:06.120050Z", "shell.execute_reply.started": "2020-02-24T00:04:03.782347Z", "shell.execute_reply": "2020-02-24T00:04:06.119185Z"}}
# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# + {"Collapsed": "false", "execution_event_id": "baeb346a-1c34-42d1-a501-7ae37369255e", "last_executed_text": "# Change to parent directory (presumably \"Documents\")\nos.chdir(\"../../..\")\n\n# Path to the CSV dataset files\ndata_path = 'Documents/Datasets/Thesis/eICU/uncompressed/'\n\n# Path to the code files\nproject_path = 'Documents/GitHub/eICU-mortality-prediction/'", "persistent_id": "a1f6ee7f-36d4-489d-b2dd-ec2a38f15d11", "execution": {"iopub.status.busy": "2020-03-11T02:55:21.503003Z", "iopub.execute_input": "2020-03-11T02:55:21.503271Z", "iopub.status.idle": "2020-03-11T02:55:21.507321Z", "shell.execute_reply.started": "2020-03-11T02:55:21.503228Z", "shell.execute_reply": "2020-03-11T02:55:21.506626Z"}}
# Change to parent directory (presumably "Documents")
os.chdir("../../../..")
# Path to the CSV dataset files
data_path = 'data/eICU/uncompressed/'
# Path to the code files
project_path = 'code/eICU-mortality-prediction/'

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-06T02:39:14.255803Z", "iopub.execute_input": "2020-03-06T02:39:14.256033Z", "iopub.status.idle": "2020-03-06T02:39:14.509317Z", "shell.execute_reply.started": "2020-03-06T02:39:14.255994Z", "shell.execute_reply": "2020-03-06T02:39:14.508277Z"}}
# Make sure that every large operation can be handled, by using the disk as an overflow for the memory
# !export MODIN_OUT_OF_CORE=true
# Another trick to do with Pandas so as to be able to allocate bigger objects to memory
# !sudo bash -c 'echo 1 > /proc/sys/vm/overcommit_memory'

# + {"Collapsed": "false", "execution_event_id": "82ef68be-443a-4bb8-8abd-7457a7005b4d", "last_executed_text": "import modin.pandas as pd                  # Optimized distributed version of Pandas\nimport data_utils as du                    # Data science and machine learning relevant methods", "persistent_id": "c0c2e356-d4f4-4a9d-bec2-88bdf9eb6a38", "execution": {"iopub.status.busy": "2020-03-11T02:55:21.508636Z", "iopub.execute_input": "2020-03-11T02:55:21.508881Z", "iopub.status.idle": "2020-03-11T02:55:23.475317Z", "shell.execute_reply.started": "2020-03-11T02:55:21.508828Z", "shell.execute_reply": "2020-03-11T02:55:23.473880Z"}}
import modin.pandas as pd                  # Optimized distributed version of Pandas
# import pandas as pd
import data_utils as du                    # Data science and machine learning relevant methods
# -

# Allow pandas to show more columns:

# + {"execution": {"iopub.status.busy": "2020-03-06T02:40:18.427780Z", "iopub.execute_input": "2020-03-06T02:40:18.428128Z", "iopub.status.idle": "2020-03-06T02:40:18.437422Z", "shell.execute_reply.started": "2020-03-06T02:40:18.428078Z", "shell.execute_reply": "2020-03-06T02:40:18.436390Z"}}
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

# + [markdown] {"Collapsed": "false"}
# Set the random seed for reproducibility

# + {"Collapsed": "false", "persistent_id": "39b552cd-6948-4ec8-ac04-42f850c1e05a", "last_executed_text": "du.set_random_seed(42)", "execution_event_id": "29ab85ce-b7fd-4c5a-a110-5841e741c369", "execution": {"iopub.status.busy": "2020-02-24T00:04:18.851797Z", "iopub.execute_input": "2020-02-24T00:04:18.852355Z", "iopub.status.idle": "2020-02-24T00:04:18.865734Z", "shell.execute_reply.started": "2020-02-24T00:04:18.852253Z", "shell.execute_reply": "2020-02-24T00:04:18.864418Z"}}
du.set_random_seed(42)

# + [markdown] {"toc-hr-collapsed": true, "Collapsed": "false"}
# ## Vital signs aperiodic data

# + [markdown] {"Collapsed": "false"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "51197f67-95e2-4184-a73f-7885cb975084", "last_executed_text": "vital_aprdc_df = pd.read_csv(f'{data_path}original/vitalAperiodic.csv')\nvital_aprdc_df.head()", "execution_event_id": "bf6e34cd-7ae9-4636-a9b9-27d404b49610"}
vital_aprdc_df = pd.read_csv(f'{data_path}original/vitalAperiodic.csv')
vital_aprdc_df.head()

# + {"Collapsed": "false", "persistent_id": "c3d45ba7-91dd-41d7-8699-c70390556018", "last_executed_text": "len(vital_aprdc_df)", "execution_event_id": "d2a30b69-ae10-4a5e-be98-56280de75b37"}
len(vital_aprdc_df)

# + {"Collapsed": "false", "persistent_id": "17c46962-79a6-48a6-b67a-4863028ed897", "last_executed_text": "vital_aprdc_df.patientunitstayid.nunique()", "execution_event_id": "e7d22357-65a7-4573-bd2e-42fc3794e931"}
vital_aprdc_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "6e745ae7-a0ea-4169-96de-d49e9f510ed9", "last_executed_text": "vital_aprdc_df.describe().transpose()", "execution_event_id": "546023f0-9a90-47da-a108-5f59f57fa5af"}
vital_aprdc_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "ac1da815-dcd6-44df-8164-a987f9623255", "last_executed_text": "vital_aprdc_df.columns", "execution_event_id": "dac11dab-f33a-4ab5-bf3e-c78447f9cdc4"}
vital_aprdc_df.info()

# + [markdown] {"Collapsed": "false"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "c4e967db-7ace-41df-8678-7cd11d1e002b", "last_executed_text": "du.search_explore.dataframe_missing_values(vital_aprdc_df)", "execution_event_id": "a04c5f38-289a-4569-8c57-89eddcda2b0d"}
du.search_explore.dataframe_missing_values(vital_aprdc_df)

# + [markdown] {"Collapsed": "false"}
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

# + [markdown] {"Collapsed": "false"}
# ### Create the timestamp feature and sort

# + [markdown] {"Collapsed": "false"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "dfab2799-af6c-4475-8341-ec3f40546ed1"}
vital_aprdc_df = vital_aprdc_df.rename(columns={'observationoffset': 'ts'})
vital_aprdc_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "8af4dd26-9eb8-4edf-8bcf-361b10c94979"}
len(vital_aprdc_df)

# + {"Collapsed": "false", "persistent_id": "de85aa8f-0d02-4c35-868c-16116a83cf7f"}
vital_aprdc_df = vital_aprdc_df.drop_duplicates()
vital_aprdc_df.head()

# + {"Collapsed": "false", "persistent_id": "bb6efd0a-aa95-40d6-84b2-8916705a4cf4", "last_executed_text": "len(vital_aprdc_df)", "execution_event_id": "0f6fb1fb-5d50-4f2c-acd1-804100222250"}
len(vital_aprdc_df)

# + [markdown] {"Collapsed": "false"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "a03573d9-f345-4ff4-84b9-2b2a3f73ce27"}
vital_aprdc_df = vital_aprdc_df.sort_values('ts')
vital_aprdc_df.head()

# + [markdown] {"Collapsed": "false"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "c94e7b7b-dc34-478b-842b-c34c926c934d"}
vital_aprdc_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='noninvasivemean', n=5).head()

# + {"Collapsed": "false", "persistent_id": "b8bcf17b-3d52-4cc9-bfbb-7f7d8fe83b3b"}
vital_aprdc_df[(vital_aprdc_df.patientunitstayid == 625065) & (vital_aprdc_df.ts == 1515)].head(10)

# + [markdown] {"Collapsed": "false"}
# We can see that there are up to 4 rows per set of `patientunitstayid` and `ts`. As such, we must join them. However, this is a different scenario than in the other cases. Since all features are numeric, we just need to average the features.

# + [markdown] {"Collapsed": "false"}
# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "591b2ccd-fa5c-4eb2-bec1-8ac21de1c890", "last_executed_text": "vital_aprdc_df = du.embedding.join_repeated_rows(vital_aprdc_df, cont_join_method='max')\nvital_aprdc_df.head()", "execution_event_id": "c6c89c91-ec15-4636-99d0-6ed07bcc921c"}
vital_aprdc_df = du.embedding.join_repeated_rows(vital_aprdc_df, cont_join_method='mean', inplace=True)
vital_aprdc_df.head()

# + {"Collapsed": "false", "persistent_id": "c94e7b7b-dc34-478b-842b-c34c926c934d"}
vital_aprdc_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='noninvasivemean', n=5).head()

# + {"Collapsed": "false", "persistent_id": "b8bcf17b-3d52-4cc9-bfbb-7f7d8fe83b3b"}
vital_aprdc_df[(vital_aprdc_df.patientunitstayid == 625065) & (vital_aprdc_df.ts == 1515)].head(10)

# + [markdown] {"Collapsed": "false"}
# Comparing the output from the two previous cells with what we had before the `join_repeated_rows` method, we can see that all rows with duplicate IDs have been successfully joined.

# + [markdown] {"Collapsed": "false"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "32450572-639e-4539-b35a-181078ed3335"}
vital_aprdc_df.columns = du.data_processing.clean_naming(vital_aprdc_df.columns)
vital_aprdc_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Normalize data

# + [markdown] {"Collapsed": "false"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "e42f577a-db00-4ecf-9e3c-433007a3bdaf"}
vital_aprdc_df.to_csv(f'{data_path}cleaned/unnormalized/ohe/vitalAperiodic.csv')

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "d5ad6017-ad4a-419c-badb-9454add7752d", "last_executed_text": "vital_aprdc_df = du.data_processing.normalize_data(vital_aprdc_df, categ_columns=cat_feat,\n                                                    id_columns=['patientunitstayid', 'ts', 'death_ts'])\nvital_aprdc_df.head(6)", "execution_event_id": "3d6d0a5c-9160-4ffc-87d4-85632a968a1d"}
vital_aprdc_df, mean, std = du.data_processing.normalize_data(vital_aprdc_df, get_stats=True, inplace=True)
vital_aprdc_df.head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save a dictionary with the mean and standard deviation values of each column that was normalized:

# + {"Collapsed": "false"}
norm_stats = dict()
for key, _ in mean.items():
    norm_stats[key] = dict()
    norm_stats[key]['mean'] = mean[key]
    norm_stats[key]['std'] = std[key]
norm_stats

# + {"Collapsed": "false"}
stream = open(f'{data_path}/cleaned/vitalAperiodic_norm_stats.yaml', 'w')
yaml.dump(norm_stats, stream, default_flow_style=False)

# + [markdown] {"Collapsed": "false"}
# ### Save the dataframe

# + [markdown] {"Collapsed": "false"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "812e7eb1-ff92-4a26-a970-2f40fc5bbdb1"}
vital_aprdc_df.to_csv(f'{data_path}cleaned/normalized/ohe/vitalAperiodic.csv')

# + [markdown] {"Collapsed": "false"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "eebc356f-507e-4872-be9d-a1d774f2fd7a"}
vital_aprdc_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "ac1da815-dcd6-44df-8164-a987f9623255", "last_executed_text": "vital_aprdc_df.columns", "execution_event_id": "dac11dab-f33a-4ab5-bf3e-c78447f9cdc4"}
vital_aprdc_df.info()

# + [markdown] {"toc-hr-collapsed": false, "Collapsed": "false"}
# ## Vital signs periodic data

# + [markdown] {"Collapsed": "false"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "51197f67-95e2-4184-a73f-7885cb975084", "last_executed_text": "vital_aprdc_df = pd.read_csv(f'{data_path}original/vitalAperiodic.csv')\nvital_aprdc_df.head()", "execution_event_id": "bf6e34cd-7ae9-4636-a9b9-27d404b49610", "execution": {"iopub.status.busy": "2020-02-23T22:37:55.744773Z", "iopub.execute_input": "2020-02-23T22:37:55.744974Z", "iopub.status.idle": "2020-02-23T22:41:55.561721Z", "shell.execute_reply.started": "2020-02-23T22:37:55.744939Z", "shell.execute_reply": "2020-02-23T22:41:55.561014Z"}}
vital_prdc_df = pd.read_csv(f'{data_path}original/vitalPeriodic.csv')
vital_prdc_df.head()

# + {"Collapsed": "false", "persistent_id": "c3d45ba7-91dd-41d7-8699-c70390556018", "last_executed_text": "len(vital_prdc_df)", "execution_event_id": "d2a30b69-ae10-4a5e-be98-56280de75b37", "execution": {"iopub.status.busy": "2020-02-17T14:30:24.096873Z", "iopub.execute_input": "2020-02-17T14:30:24.097152Z", "iopub.status.idle": "2020-02-17T14:30:24.102000Z", "shell.execute_reply.started": "2020-02-17T14:30:24.097105Z", "shell.execute_reply": "2020-02-17T14:30:24.101053Z"}}
len(vital_prdc_df)

# + {"Collapsed": "false", "persistent_id": "17c46962-79a6-48a6-b67a-4863028ed897", "last_executed_text": "vital_prdc_df.patientunitstayid.nunique()", "execution_event_id": "e7d22357-65a7-4573-bd2e-42fc3794e931", "execution": {"iopub.status.busy": "2020-02-17T14:30:24.103966Z", "iopub.execute_input": "2020-02-17T14:30:24.104270Z", "iopub.status.idle": "2020-02-17T14:30:28.311978Z", "shell.execute_reply.started": "2020-02-17T14:30:24.104208Z", "shell.execute_reply": "2020-02-17T14:30:28.311020Z"}}
vital_prdc_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "6e745ae7-a0ea-4169-96de-d49e9f510ed9", "last_executed_text": "vital_prdc_df.describe().transpose()", "execution_event_id": "546023f0-9a90-47da-a108-5f59f57fa5af", "execution": {"iopub.status.busy": "2020-02-17T14:30:28.313108Z", "iopub.execute_input": "2020-02-17T14:30:28.313347Z", "iopub.status.idle": "2020-02-17T14:10:04.117778Z", "shell.execute_reply.started": "2020-02-17T14:08:52.753139Z", "shell.execute_reply": "2020-02-17T14:10:04.117082Z"}}
vital_prdc_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "ac1da815-dcd6-44df-8164-a987f9623255", "last_executed_text": "vital_prdc_df.columns", "execution_event_id": "dac11dab-f33a-4ab5-bf3e-c78447f9cdc4", "execution": {"iopub.status.busy": "2020-02-17T18:35:11.301956Z", "iopub.execute_input": "2020-02-17T18:35:11.302222Z", "iopub.status.idle": "2020-02-17T18:36:54.665731Z", "shell.execute_reply.started": "2020-02-17T18:35:11.302178Z", "shell.execute_reply": "2020-02-17T18:36:54.663014Z"}}
vital_prdc_df.info()

# + [markdown] {"Collapsed": "false"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "c4e967db-7ace-41df-8678-7cd11d1e002b", "last_executed_text": "du.search_explore.dataframe_missing_values(vital_prdc_df)", "execution_event_id": "a04c5f38-289a-4569-8c57-89eddcda2b0d", "execution": {"iopub.status.busy": "2020-02-17T18:42:29.139689Z", "iopub.execute_input": "2020-02-17T18:42:29.139909Z", "iopub.status.idle": "2020-02-17T18:46:43.254351Z", "shell.execute_reply.started": "2020-02-17T18:42:29.139872Z", "shell.execute_reply": "2020-02-17T18:46:43.253741Z"}}
du.search_explore.dataframe_missing_values(vital_prdc_df)

# + [markdown] {"Collapsed": "false"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "9f3b85d0-8480-4f30-8455-fde084ba7c69", "last_executed_text": "vital_prdc_df.notetype.value_counts().head(20)", "execution_event_id": "39bf557e-d53a-4ecd-8118-a9719aa49188", "execution": {"iopub.status.busy": "2020-02-17T18:46:43.256036Z", "iopub.execute_input": "2020-02-17T18:46:43.256267Z", "iopub.status.idle": "2020-02-17T18:59:43.105854Z", "shell.execute_reply.started": "2020-02-17T18:46:43.256228Z", "shell.execute_reply": "2020-02-17T18:59:43.104968Z"}}
vital_prdc_df.temperature.value_counts()

# + {"Collapsed": "false", "persistent_id": "e41f6a9d-c5ea-45e3-b38b-89ba70de1b87", "last_executed_text": "vital_prdc_df.notepath.value_counts().head(40)", "execution_event_id": "0df9919f-4d46-4c34-a326-df8e8cf7f18c", "execution": {"iopub.status.busy": "2020-02-17T18:59:43.107272Z", "iopub.execute_input": "2020-02-17T18:59:43.107500Z", "iopub.status.idle": "2020-02-17T18:59:55.984762Z", "shell.execute_reply.started": "2020-02-17T18:59:43.107463Z", "shell.execute_reply": "2020-02-17T18:59:55.983629Z"}}
vital_prdc_df.sao2.value_counts()

# + {"Collapsed": "false", "persistent_id": "b12dcddf-9703-4aff-8a20-df1d147ef554", "last_executed_text": "vital_prdc_df.notevalue.value_counts().head(20)", "execution_event_id": "04986ee5-3cc1-453c-8127-05f513370b5d", "execution": {"iopub.status.busy": "2020-02-17T18:59:55.985825Z", "iopub.execute_input": "2020-02-17T18:59:55.986211Z", "iopub.status.idle": "2020-02-17T19:00:08.569192Z", "shell.execute_reply.started": "2020-02-17T18:59:55.986157Z", "shell.execute_reply": "2020-02-17T19:00:08.568375Z"}}
vital_prdc_df.heartrate.value_counts()

# + {"Collapsed": "false", "persistent_id": "9f3b85d0-8480-4f30-8455-fde084ba7c69", "last_executed_text": "vital_prdc_df.notetype.value_counts().head(20)", "execution_event_id": "39bf557e-d53a-4ecd-8118-a9719aa49188", "execution": {"iopub.status.busy": "2020-02-17T19:00:08.570202Z", "iopub.execute_input": "2020-02-17T19:00:08.570408Z", "iopub.status.idle": "2020-02-17T19:00:23.592624Z", "shell.execute_reply.started": "2020-02-17T19:00:08.570373Z", "shell.execute_reply": "2020-02-17T19:00:23.591739Z"}}
vital_prdc_df.respiration.value_counts()

# + {"Collapsed": "false", "persistent_id": "e41f6a9d-c5ea-45e3-b38b-89ba70de1b87", "last_executed_text": "vital_prdc_df.notepath.value_counts().head(40)", "execution_event_id": "0df9919f-4d46-4c34-a326-df8e8cf7f18c", "execution": {"iopub.status.busy": "2020-02-17T19:00:23.594259Z", "iopub.execute_input": "2020-02-17T19:00:23.594521Z", "iopub.status.idle": "2020-02-17T19:03:47.440612Z", "shell.execute_reply.started": "2020-02-17T19:00:23.594471Z", "shell.execute_reply": "2020-02-17T19:03:47.438521Z"}}
vital_prdc_df.cvp.value_counts()

# + {"Collapsed": "false", "persistent_id": "b12dcddf-9703-4aff-8a20-df1d147ef554", "last_executed_text": "vital_prdc_df.notevalue.value_counts().head(20)", "execution_event_id": "04986ee5-3cc1-453c-8127-05f513370b5d", "execution": {"iopub.status.busy": "2020-02-17T19:03:47.441573Z", "iopub.execute_input": "2020-02-17T14:10:18.845830Z", "iopub.status.idle": "2020-02-17T19:03:47.442262Z", "shell.execute_reply.started": "2020-02-17T14:10:18.845792Z", "shell.execute_reply": "2020-02-17T14:10:19.917841Z"}}
vital_prdc_df.systemicsystolic.value_counts()

# + {"Collapsed": "false", "persistent_id": "9f3b85d0-8480-4f30-8455-fde084ba7c69", "last_executed_text": "vital_prdc_df.notetype.value_counts().head(20)", "execution_event_id": "39bf557e-d53a-4ecd-8118-a9719aa49188", "execution": {"iopub.status.busy": "2020-02-17T19:03:47.443156Z", "iopub.execute_input": "2020-02-17T14:10:19.919965Z", "iopub.status.idle": "2020-02-17T19:03:47.443460Z", "shell.execute_reply.started": "2020-02-17T14:10:19.919932Z", "shell.execute_reply": "2020-02-17T14:10:20.970693Z"}}
vital_prdc_df.systemicdiastolic.value_counts()

# + {"Collapsed": "false", "persistent_id": "e41f6a9d-c5ea-45e3-b38b-89ba70de1b87", "last_executed_text": "vital_prdc_df.notepath.value_counts().head(40)", "execution_event_id": "0df9919f-4d46-4c34-a326-df8e8cf7f18c", "execution": {"iopub.status.busy": "2020-02-17T19:03:47.444222Z", "iopub.execute_input": "2020-02-17T14:10:20.972781Z", "iopub.status.idle": "2020-02-17T19:03:47.444504Z", "shell.execute_reply.started": "2020-02-17T14:10:20.972742Z", "shell.execute_reply": "2020-02-17T14:10:22.040051Z"}}
vital_prdc_df.systemicmean.value_counts()

# + {"Collapsed": "false", "persistent_id": "b12dcddf-9703-4aff-8a20-df1d147ef554", "last_executed_text": "vital_prdc_df.notevalue.value_counts().head(20)", "execution_event_id": "04986ee5-3cc1-453c-8127-05f513370b5d", "execution": {"iopub.status.busy": "2020-02-17T19:03:47.445382Z", "iopub.execute_input": "2020-02-17T14:10:22.042102Z", "iopub.status.idle": "2020-02-17T19:03:47.445805Z", "shell.execute_reply.started": "2020-02-17T14:10:22.042062Z", "shell.execute_reply": "2020-02-17T14:10:22.817829Z"}}
vital_prdc_df.pasystolic.value_counts()

# + {"Collapsed": "false", "persistent_id": "b12dcddf-9703-4aff-8a20-df1d147ef554", "last_executed_text": "vital_prdc_df.notevalue.value_counts().head(20)", "execution_event_id": "04986ee5-3cc1-453c-8127-05f513370b5d", "execution": {"iopub.status.busy": "2020-02-17T19:03:47.446859Z", "iopub.execute_input": "2020-02-17T14:10:22.819988Z", "iopub.status.idle": "2020-02-17T19:03:47.447361Z", "shell.execute_reply.started": "2020-02-17T14:10:22.819945Z", "shell.execute_reply": "2020-02-17T14:10:23.601859Z"}}
vital_prdc_df.padiastolic.value_counts()

# + {"Collapsed": "false", "persistent_id": "9f3b85d0-8480-4f30-8455-fde084ba7c69", "last_executed_text": "vital_prdc_df.notetype.value_counts().head(20)", "execution_event_id": "39bf557e-d53a-4ecd-8118-a9719aa49188", "execution": {"iopub.status.busy": "2020-02-17T19:03:47.448278Z", "iopub.execute_input": "2020-02-17T14:10:23.604435Z", "iopub.status.idle": "2020-02-17T19:03:47.448666Z", "shell.execute_reply.started": "2020-02-17T14:10:23.604395Z", "shell.execute_reply": "2020-02-17T14:10:24.407641Z"}}
vital_prdc_df.pamean.value_counts()

# + {"Collapsed": "false", "persistent_id": "e41f6a9d-c5ea-45e3-b38b-89ba70de1b87", "last_executed_text": "vital_prdc_df.notepath.value_counts().head(40)", "execution_event_id": "0df9919f-4d46-4c34-a326-df8e8cf7f18c", "execution": {"iopub.status.busy": "2020-02-17T19:03:47.449637Z", "iopub.execute_input": "2020-02-17T14:10:24.409772Z", "iopub.status.idle": "2020-02-17T19:03:47.450023Z", "shell.execute_reply.started": "2020-02-17T14:10:24.409730Z", "shell.execute_reply": "2020-02-17T14:10:25.899399Z"}}
vital_prdc_df.st1.value_counts()

# + {"Collapsed": "false", "persistent_id": "b12dcddf-9703-4aff-8a20-df1d147ef554", "last_executed_text": "vital_prdc_df.notevalue.value_counts().head(20)", "execution_event_id": "04986ee5-3cc1-453c-8127-05f513370b5d", "execution": {"iopub.status.busy": "2020-02-17T19:03:47.450753Z", "iopub.execute_input": "2020-02-17T14:10:25.902367Z", "iopub.status.idle": "2020-02-17T19:03:47.451063Z", "shell.execute_reply.started": "2020-02-17T14:10:25.902277Z", "shell.execute_reply": "2020-02-17T14:10:27.334721Z"}}
vital_prdc_df.st2.value_counts()

# + {"Collapsed": "false", "persistent_id": "9f3b85d0-8480-4f30-8455-fde084ba7c69", "last_executed_text": "vital_prdc_df.notetype.value_counts().head(20)", "execution_event_id": "39bf557e-d53a-4ecd-8118-a9719aa49188", "execution": {"iopub.status.busy": "2020-02-17T19:03:47.451855Z", "iopub.execute_input": "2020-02-17T14:10:27.336773Z", "iopub.status.idle": "2020-02-17T19:03:47.452156Z", "shell.execute_reply.started": "2020-02-17T14:10:27.336730Z", "shell.execute_reply": "2020-02-17T14:10:28.732544Z"}}
vital_prdc_df.st3.value_counts()

# + {"Collapsed": "false", "persistent_id": "9f3b85d0-8480-4f30-8455-fde084ba7c69", "last_executed_text": "vital_prdc_df.notetype.value_counts().head(20)", "execution_event_id": "39bf557e-d53a-4ecd-8118-a9719aa49188", "execution": {"iopub.status.busy": "2020-02-17T19:03:47.452941Z", "iopub.execute_input": "2020-02-17T14:10:28.735069Z", "iopub.status.idle": "2020-02-17T19:03:47.453271Z", "shell.execute_reply.started": "2020-02-17T14:10:28.735023Z", "shell.execute_reply": "2020-02-17T14:10:29.516359Z"}}
vital_prdc_df.icp.value_counts()

# + {"Collapsed": "false", "persistent_id": "5beb1b97-7a5b-446c-934d-74b99556151f", "last_executed_text": "vital_prdc_df = vital_prdc_df.drop(['noteid', 'noteenteredoffset', 'notetype', 'notetext'], axis=1)\nvital_prdc_df.head()", "execution_event_id": "2e499a1a-e9a0-42cf-b351-98273b224c15", "execution": {"iopub.status.busy": "2020-02-23T22:41:55.562925Z", "iopub.execute_input": "2020-02-23T22:41:55.563132Z", "iopub.status.idle": "2020-02-23T22:41:56.926921Z", "shell.execute_reply.started": "2020-02-23T22:41:55.563096Z", "shell.execute_reply": "2020-02-23T22:41:56.926320Z"}}
vital_prdc_df = vital_prdc_df.drop(columns=['vitalperiodicid'])
vital_prdc_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Create the timestamp feature and sort

# + [markdown] {"Collapsed": "false"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "dfab2799-af6c-4475-8341-ec3f40546ed1", "execution": {"iopub.status.busy": "2020-02-23T22:41:56.927845Z", "iopub.execute_input": "2020-02-23T22:41:56.928042Z", "iopub.status.idle": "2020-02-23T22:41:57.354887Z", "shell.execute_reply.started": "2020-02-23T22:41:56.928006Z", "shell.execute_reply": "2020-02-23T22:41:57.354287Z"}}
vital_prdc_df = vital_prdc_df.rename(columns={'observationoffset': 'ts'})
vital_prdc_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "8af4dd26-9eb8-4edf-8bcf-361b10c94979", "execution": {"iopub.status.busy": "2020-02-23T22:41:57.356387Z", "iopub.execute_input": "2020-02-23T22:41:57.356601Z", "iopub.status.idle": "2020-02-23T22:41:57.848645Z", "shell.execute_reply.started": "2020-02-23T22:41:57.356563Z", "shell.execute_reply": "2020-02-23T22:41:57.847851Z"}}
len(vital_prdc_df)

# + {"Collapsed": "false", "persistent_id": "de85aa8f-0d02-4c35-868c-16116a83cf7f", "execution": {"iopub.status.busy": "2020-02-23T22:41:57.849937Z", "iopub.execute_input": "2020-02-23T22:41:57.850180Z", "iopub.status.idle": "2020-02-23T22:45:50.613431Z", "shell.execute_reply.started": "2020-02-23T22:41:57.850132Z", "shell.execute_reply": "2020-02-23T22:45:50.612809Z"}}
vital_prdc_df = vital_prdc_df.drop_duplicates()
vital_prdc_df.head()

# + {"Collapsed": "false", "persistent_id": "bb6efd0a-aa95-40d6-84b2-8916705a4cf4", "last_executed_text": "len(vital_prdc_df)", "execution_event_id": "0f6fb1fb-5d50-4f2c-acd1-804100222250", "execution": {"iopub.status.busy": "2020-02-23T22:45:50.614650Z", "iopub.status.idle": "2020-02-23T22:45:50.619036Z", "iopub.execute_input": "2020-02-23T22:45:50.614900Z", "shell.execute_reply.started": "2020-02-23T22:45:50.614857Z", "shell.execute_reply": "2020-02-23T22:45:50.618471Z"}}
len(vital_prdc_df)

# + [markdown] {"Collapsed": "false"}
# Convert dataframe to Pandas, as the next cells aren't working properly with Modin:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-23T22:45:50.619836Z", "iopub.execute_input": "2020-02-23T22:45:50.620222Z", "iopub.status.idle": "2020-02-23T22:46:19.041074Z", "shell.execute_reply.started": "2020-02-23T22:45:50.620173Z", "shell.execute_reply": "2020-02-23T22:46:19.040127Z"}}
vital_prdc_df, pd = du.utils.convert_dataframe(vital_prdc_df, to='pandas', dtypes=dict(vital_prdc_df.dtypes))

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-23T22:46:19.042045Z", "iopub.execute_input": "2020-02-23T22:46:19.042245Z", "iopub.status.idle": "2020-02-23T22:46:19.046129Z", "shell.execute_reply.started": "2020-02-23T22:46:19.042209Z", "shell.execute_reply": "2020-02-23T22:46:19.045596Z"}}
type(vital_prdc_df)

# + [markdown] {"Collapsed": "false"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "a03573d9-f345-4ff4-84b9-2b2a3f73ce27", "execution": {"iopub.status.busy": "2020-02-23T22:46:19.046974Z", "iopub.status.idle": "2020-02-23T22:47:46.942677Z", "iopub.execute_input": "2020-02-23T22:46:19.047160Z", "shell.execute_reply.started": "2020-02-23T22:46:19.047127Z", "shell.execute_reply": "2020-02-23T22:47:46.941955Z"}}
vital_prdc_df = vital_prdc_df.sort_values('ts')
vital_prdc_df.head()

# + [markdown] {"Collapsed": "false"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "c94e7b7b-dc34-478b-842b-c34c926c934d", "execution": {"iopub.status.busy": "2020-02-23T03:35:32.930352Z", "iopub.status.idle": "2020-02-23T03:40:16.470755Z", "iopub.execute_input": "2020-02-23T03:35:32.930560Z", "shell.execute_reply.started": "2020-02-23T03:35:32.930523Z", "shell.execute_reply": "2020-02-23T03:40:16.469968Z"}}
vital_prdc_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='heartrate', n=5).head()

# + {"Collapsed": "false", "persistent_id": "b8bcf17b-3d52-4cc9-bfbb-7f7d8fe83b3b", "execution": {"iopub.status.busy": "2020-02-23T03:40:16.472027Z", "iopub.status.idle": "2020-02-23T03:40:17.303600Z", "iopub.execute_input": "2020-02-23T03:40:16.472299Z", "shell.execute_reply.started": "2020-02-23T03:40:16.472252Z", "shell.execute_reply": "2020-02-23T03:40:17.302791Z"}}
vital_prdc_df[(vital_prdc_df.patientunitstayid == 2290828.0) & (vital_prdc_df.ts == 13842.0)].head(10)

# + [markdown] {"Collapsed": "false"}
# We can see that there are up to 4 rows per set of `patientunitstayid` and `ts`. As such, we must join them. However, this is a different scenario than in the other cases. Since all features are numeric, we just need to average the features.

# + [markdown] {"Collapsed": "false"}
# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "591b2ccd-fa5c-4eb2-bec1-8ac21de1c890", "last_executed_text": "vital_prdc_df = du.embedding.join_repeated_rows(vital_prdc_df, cont_join_method='max')\nvital_prdc_df.head()", "execution_event_id": "c6c89c91-ec15-4636-99d0-6ed07bcc921c", "execution": {"iopub.status.busy": "2020-02-23T22:47:46.943632Z", "iopub.status.idle": "2020-02-23T22:52:23.371793Z", "iopub.execute_input": "2020-02-23T22:47:46.943835Z", "shell.execute_reply.started": "2020-02-23T22:47:46.943801Z", "shell.execute_reply": "2020-02-23T22:52:23.371039Z"}}
vital_prdc_df = du.embedding.join_repeated_rows(vital_prdc_df, cont_join_method='mean', inplace=True)
vital_prdc_df.head()

# + {"Collapsed": "false", "persistent_id": "c94e7b7b-dc34-478b-842b-c34c926c934d", "execution": {"iopub.status.busy": "2020-02-23T03:46:33.714909Z", "iopub.status.idle": "2020-02-23T03:51:11.690597Z", "iopub.execute_input": "2020-02-23T03:46:33.715114Z", "shell.execute_reply.started": "2020-02-23T03:46:33.715078Z", "shell.execute_reply": "2020-02-23T03:51:11.689987Z"}}
vital_prdc_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='heartrate', n=5).head()

# + {"Collapsed": "false", "persistent_id": "b8bcf17b-3d52-4cc9-bfbb-7f7d8fe83b3b", "execution": {"iopub.status.busy": "2020-02-23T03:51:11.691515Z", "iopub.status.idle": "2020-02-23T03:51:12.521392Z", "iopub.execute_input": "2020-02-23T03:51:11.691714Z", "shell.execute_reply.started": "2020-02-23T03:51:11.691679Z", "shell.execute_reply": "2020-02-23T03:51:12.520696Z"}}
vital_prdc_df[(vital_prdc_df.patientunitstayid == 2290828.0) & (vital_prdc_df.ts == 13842.0)].head(10)

# + [markdown] {"Collapsed": "false"}
# Comparing the output from the two previous cells with what we had before the `join_repeated_rows` method, we can see that all rows with duplicate IDs have been successfully joined.

# + [markdown] {"Collapsed": "false"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "32450572-639e-4539-b35a-181078ed3335", "execution": {"iopub.status.busy": "2020-02-23T22:52:23.372900Z", "iopub.status.idle": "2020-02-23T22:52:23.390619Z", "iopub.execute_input": "2020-02-23T22:52:23.373103Z", "shell.execute_reply.started": "2020-02-23T22:52:23.373066Z", "shell.execute_reply": "2020-02-23T22:52:23.389782Z"}}
vital_prdc_df.columns = du.data_processing.clean_naming(vital_prdc_df.columns)
vital_prdc_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Normalize data

# + [markdown] {"Collapsed": "false"}
# Reconvert dataframe to Modin:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-23T23:25:21.865035Z", "iopub.execute_input": "2020-02-23T23:25:21.865255Z", "iopub.status.idle": "2020-02-23T23:43:55.957105Z", "shell.execute_reply.started": "2020-02-23T23:25:21.865215Z", "shell.execute_reply": "2020-02-23T23:43:55.956209Z"}}
vital_prdc_df, pd = du.utils.convert_dataframe(vital_prdc_df, to='modin')

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-23T23:43:55.958341Z", "iopub.execute_input": "2020-02-23T23:43:55.958608Z", "iopub.status.idle": "2020-02-23T23:43:55.963310Z", "shell.execute_reply.started": "2020-02-23T23:43:55.958550Z", "shell.execute_reply": "2020-02-23T23:43:55.962240Z"}}
type(vital_prdc_df)

# + [markdown] {"Collapsed": "false"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "e42f577a-db00-4ecf-9e3c-433007a3bdaf", "execution": {"iopub.status.busy": "2020-02-23T22:52:23.392632Z", "iopub.status.idle": "2020-02-23T23:25:21.863695Z", "iopub.execute_input": "2020-02-23T22:52:23.392853Z", "shell.execute_reply.started": "2020-02-23T22:52:23.392816Z", "shell.execute_reply": "2020-02-23T23:25:21.862696Z"}}
vital_prdc_df.to_csv(f'{data_path}cleaned/unnormalized/ohe/vitalPeriodic.csv')

# + {"Collapsed": "false", "persistent_id": "e42f577a-db00-4ecf-9e3c-433007a3bdaf", "execution": {"iopub.status.busy": "2020-02-24T00:05:23.843116Z", "iopub.status.idle": "2020-02-24T00:08:16.755770Z", "iopub.execute_input": "2020-02-24T00:05:23.843348Z", "shell.execute_reply.started": "2020-02-24T00:05:23.843307Z", "shell.execute_reply": "2020-02-24T00:08:16.754743Z"}}
vital_prdc_df = pd.read_csv(f'{data_path}cleaned/unnormalized/ohe/vitalPeriodic.csv')
vital_prdc_df = vital_prdc_df.drop(columns=['Unnamed: 0'])
vital_prdc_df.head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "d5ad6017-ad4a-419c-badb-9454add7752d", "last_executed_text": "vital_prdc_df = du.data_processing.normalize_data(vital_prdc_df, categ_columns=cat_feat,\n                                                    id_columns=['patientunitstayid', 'ts', 'death_ts'])\nvital_prdc_df.head(6)", "execution_event_id": "3d6d0a5c-9160-4ffc-87d4-85632a968a1d", "execution": {"iopub.status.busy": "2020-02-24T00:10:03.127013Z", "iopub.status.idle": "2020-02-24T00:11:16.615136Z", "iopub.execute_input": "2020-02-24T00:10:03.127293Z", "shell.execute_reply.started": "2020-02-24T00:10:03.127252Z", "shell.execute_reply": "2020-02-24T00:11:16.614331Z"}}
vital_prdc_df, mean, std = du.data_processing.normalize_data(vital_prdc_df, get_stats=True, inplace=True)
vital_prdc_df.head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save a dictionary with the mean and standard deviation values of each column that was normalized:

# + {"Collapsed": "false"}
norm_stats = dict()
for key, _ in mean.items():
    norm_stats[key] = dict()
    norm_stats[key]['mean'] = mean[key]
    norm_stats[key]['std'] = std[key]
norm_stats

# + {"Collapsed": "false"}
stream = open(f'{data_path}/cleaned/vitalPeriodic_norm_stats.yaml', 'w')
yaml.dump(norm_stats, stream, default_flow_style=False)

# + [markdown] {"Collapsed": "false"}
# ### Save the dataframe

# + [markdown] {"Collapsed": "false"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "812e7eb1-ff92-4a26-a970-2f40fc5bbdb1", "execution": {"iopub.status.busy": "2020-02-24T00:11:16.616663Z", "iopub.status.idle": "2020-02-24T00:56:10.413312Z", "iopub.execute_input": "2020-02-24T00:11:16.616895Z", "shell.execute_reply.started": "2020-02-24T00:11:16.616855Z", "shell.execute_reply": "2020-02-24T00:56:10.412488Z"}}
vital_prdc_df.to_csv(f'{data_path}cleaned/normalized/ohe/vitalPeriodic.csv')

# + [markdown] {"Collapsed": "false"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "ac1da815-dcd6-44df-8164-a987f9623255", "last_executed_text": "vital_prdc_df.columns", "execution_event_id": "dac11dab-f33a-4ab5-bf3e-c78447f9cdc4", "execution": {"iopub.status.busy": "2020-02-24T00:56:10.415390Z", "iopub.status.idle": "2020-02-24T00:56:10.425372Z", "iopub.execute_input": "2020-02-24T00:56:10.415722Z", "shell.execute_reply.started": "2020-02-24T00:56:10.415661Z", "shell.execute_reply": "2020-02-24T00:56:10.424529Z"}}
vital_prdc_df.info()

# + {"Collapsed": "false", "persistent_id": "eebc356f-507e-4872-be9d-a1d774f2fd7a", "execution": {"iopub.status.busy": "2020-02-24T00:56:10.426913Z", "iopub.status.idle": "2020-02-24T00:57:28.727797Z", "iopub.execute_input": "2020-02-24T00:56:10.427229Z", "shell.execute_reply.started": "2020-02-24T00:56:10.427169Z", "shell.execute_reply": "2020-02-24T00:57:28.726832Z"}}
vital_prdc_df.describe().transpose()

# + {"Collapsed": "false"}
