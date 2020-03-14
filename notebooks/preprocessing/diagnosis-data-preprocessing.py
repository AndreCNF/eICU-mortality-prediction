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
# # Diagnosis Data Preprocessing
# ---
#
# Reading and preprocessing diagnosis data of the eICU dataset from MIT with the data from over 139k patients collected in the US.
#
# This notebook addresses the preprocessing of the following eICU tables:
# * allergy
# * diagnosis
# * pastHistory

# + [markdown] {"colab_type": "text", "id": "KOdmFzXqF7nq", "toc-hr-collapsed": true, "Collapsed": "false"}
# ## Importing the necessary packages

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl", "Collapsed": "false", "persistent_id": "522745b5-b5bf-479f-b697-5c7e9e12fc33", "last_executed_text": "import os                                  # os handles directory/workspace changes\nimport numpy as np                         # NumPy to handle numeric and NaN operations\nimport yaml                                # Save and load YAML files", "execution_event_id": "deb57b39-6a79-4b3a-95ed-02f8089ff593", "execution": {"iopub.status.busy": "2020-03-13T01:32:55.903945Z", "iopub.execute_input": "2020-03-13T01:32:55.904252Z", "iopub.status.idle": "2020-03-13T01:32:55.929378Z", "shell.execute_reply.started": "2020-03-13T01:32:55.904212Z", "shell.execute_reply": "2020-03-13T01:32:55.928404Z"}}
import os                                  # os handles directory/workspace changes
import numpy as np                         # NumPy to handle numeric and NaN operations
import yaml                                # Save and load YAML files

# + {"Collapsed": "false", "persistent_id": "02accdbf-be7e-415c-ba11-165906e66c50", "last_executed_text": "# Debugging packages\nimport pixiedust                           # Debugging in Jupyter Notebook cells", "execution_event_id": "fa33a2f7-7127-49c6-bbe9-f89555b1f2be", "execution": {"iopub.status.busy": "2020-03-13T01:32:56.466671Z", "iopub.execute_input": "2020-03-13T01:32:56.466949Z", "iopub.status.idle": "2020-03-13T01:32:57.500877Z", "shell.execute_reply.started": "2020-03-13T01:32:56.466908Z", "shell.execute_reply": "2020-03-13T01:32:57.500072Z"}}
# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# + {"Collapsed": "false", "execution_event_id": "baeb346a-1c34-42d1-a501-7ae37369255e", "last_executed_text": "# Change to parent directory (presumably \"Documents\")\nos.chdir(\"../../..\")\n\n# Path to the CSV dataset files\ndata_path = 'Documents/Datasets/Thesis/eICU/uncompressed/'\n\n# Path to the code files\nproject_path = 'Documents/GitHub/eICU-mortality-prediction/'", "persistent_id": "a1f6ee7f-36d4-489d-b2dd-ec2a38f15d11", "execution": {"iopub.status.busy": "2020-03-13T01:32:57.502260Z", "iopub.execute_input": "2020-03-13T01:32:57.502494Z", "iopub.status.idle": "2020-03-13T01:32:57.506416Z", "shell.execute_reply.started": "2020-03-13T01:32:57.502453Z", "shell.execute_reply": "2020-03-13T01:32:57.505607Z"}}
# Change to parent directory (presumably "Documents")
os.chdir("../../../..")
# Path to the CSV dataset files
data_path = 'data/eICU/uncompressed/'
# Path to the code files
project_path = 'code/eICU-mortality-prediction/'

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-13T01:32:58.499959Z", "iopub.execute_input": "2020-03-13T01:32:58.500259Z", "iopub.status.idle": "2020-03-13T01:32:58.744257Z", "shell.execute_reply.started": "2020-03-13T01:32:58.500215Z", "shell.execute_reply": "2020-03-13T01:32:58.743301Z"}}
# Make sure that every large operation can be handled, by using the disk as an overflow for the memory
# !export MODIN_OUT_OF_CORE=true
# Another trick to do with Pandas so as to be able to allocate bigger objects to memory
# !sudo bash -c 'echo 1 > /proc/sys/vm/overcommit_memory'

# + {"Collapsed": "false", "execution_event_id": "82ef68be-443a-4bb8-8abd-7457a7005b4d", "last_executed_text": "import modin.pandas as pd                  # Optimized distributed version of Pandas\nimport data_utils as du                    # Data science and machine learning relevant methods", "persistent_id": "c0c2e356-d4f4-4a9d-bec2-88bdf9eb6a38", "execution": {"iopub.status.busy": "2020-03-13T01:33:09.660334Z", "iopub.execute_input": "2020-03-13T01:33:09.660651Z", "iopub.status.idle": "2020-03-13T01:33:11.423861Z", "shell.execute_reply.started": "2020-03-13T01:33:09.660603Z", "shell.execute_reply": "2020-03-13T01:33:11.422423Z"}}
import modin.pandas as pd                  # Optimized distributed version of Pandas
# import pandas as pd
import data_utils as du                    # Data science and machine learning relevant methods
# -

# Allow pandas to show more columns:

# + {"execution": {"iopub.status.busy": "2020-03-13T01:33:13.348131Z", "iopub.execute_input": "2020-03-13T01:33:13.348949Z", "iopub.status.idle": "2020-03-13T01:33:13.353428Z", "shell.execute_reply.started": "2020-03-13T01:33:13.348859Z", "shell.execute_reply": "2020-03-13T01:33:13.352409Z"}}
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

# + [markdown] {"Collapsed": "false"}
# Set the random seed for reproducibility

# + {"Collapsed": "false", "persistent_id": "39b552cd-6948-4ec8-ac04-42f850c1e05a", "last_executed_text": "du.set_random_seed(42)", "execution_event_id": "29ab85ce-b7fd-4c5a-a110-5841e741c369", "execution": {"iopub.status.busy": "2020-03-13T01:33:13.901532Z", "iopub.execute_input": "2020-03-13T01:33:13.901810Z", "iopub.status.idle": "2020-03-13T01:33:13.907114Z", "shell.execute_reply.started": "2020-03-13T01:33:13.901769Z", "shell.execute_reply": "2020-03-13T01:33:13.906132Z"}}
du.set_random_seed(42)

# + [markdown] {"Collapsed": "false"}
# Set the maximum number of categories

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-13T01:33:14.537823Z", "iopub.execute_input": "2020-03-13T01:33:14.538131Z", "iopub.status.idle": "2020-03-13T01:33:14.542494Z", "shell.execute_reply.started": "2020-03-13T01:33:14.538084Z", "shell.execute_reply": "2020-03-13T01:33:14.541091Z"}}
MAX_CATEGORIES = 250

# + [markdown] {"toc-hr-collapsed": true, "Collapsed": "false"}
# ## Allergy data

# + [markdown] {"Collapsed": "false"}
# ### Initialize variables

# + {"Collapsed": "false", "persistent_id": "754a96f8-d389-4968-8c13-52e5e9d0bf82"}
# List of categorical features
cat_feat = []
# Dictionary of the one hot encoded columns originary from each categorical feature, that will be embedded
cat_feat_ohe = dict()

# + [markdown] {"Collapsed": "false"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "9383001f-c733-4cf4-8b59-abfd57cf49e5"}
alrg_df = pd.read_csv(f'{data_path}original/allergy.csv')
alrg_df.head()

# + {"Collapsed": "false", "persistent_id": "cbd138f8-8c1e-4c99-91ca-68a1244be654"}
len(alrg_df)

# + {"Collapsed": "false", "persistent_id": "5cdf0cb6-3179-459c-844d-46edb8a71619"}
alrg_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "e0483a75-628e-48bd-97e8-0350d7c4f889"}
alrg_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "c2a187d6-fb24-4a21-8f2e-26b6473c257a"}
alrg_df.columns

# + {"Collapsed": "false", "persistent_id": "df596865-c169-4738-a491-1a0694c8144a"}
alrg_df.dtypes

# + [markdown] {"Collapsed": "false"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "c309a388-c464-4fa2-85b3-9f6aab6bbb1d"}
du.search_explore.dataframe_missing_values(alrg_df)

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "c309a388-c464-4fa2-85b3-9f6aab6bbb1d"}
du.search_explore.dataframe_missing_values(alrg_df, 'allergyname')

# + {"Collapsed": "false"}
alrg_df.allergyname.isnull().sum()

# + [markdown] {"Collapsed": "false"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "e8352cc0-cec5-4fb3-a5dd-d05e05586a96"}
alrg_df[alrg_df.allergytype == 'Non Drug'].drughiclseqno.value_counts()

# + {"Collapsed": "false", "persistent_id": "32fd94b7-8ce1-4b2e-9309-4f3653be654d"}
alrg_df[alrg_df.allergytype == 'Drug'].drughiclseqno.value_counts()

# + [markdown] {"Collapsed": "false"}
# As we can see, the drug features in this table only have data if the allergy derives from using the drug. As such, we don't need the `allergytype` feature. Also ignoring hospital staff related information and using just the drug codes instead of their names, as they're independent of the drug brand.

# + {"Collapsed": "false", "persistent_id": "fce01df2-fdd7-42c3-8fa8-22794989df1a"}
alrg_df.allergynotetype.value_counts()

# + {"Collapsed": "false", "persistent_id": "fce01df2-fdd7-42c3-8fa8-22794989df1a"}
alrg_df.allergyname.value_counts()

# + [markdown] {"Collapsed": "false"}
# Feature `allergynotetype` also doesn't seem very relevant, discarding it.

# + {"Collapsed": "false", "persistent_id": "a090232b-049c-4386-90eb-6a68f6487a34"}
alrg_df = alrg_df[['patientunitstayid', 'allergyoffset',
                   'allergyname', 'drughiclseqno']]
alrg_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Rename columns

# + {"Collapsed": "false", "persistent_id": "d804e12b-79df-4a29-87ef-89c26d57b8e9"}
alrg_df = alrg_df.rename(columns={'drughiclseqno': 'drugallergyhiclseqno'})
alrg_df.head()

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": false}
# ### Discretize categorical features
#
# Convert binary categorical features into one hot encode columns, which can later be embedded or used as is.

# + [markdown] {"Collapsed": "false"}
# #### One hot encode features

# + [markdown] {"Collapsed": "false"}
# Update list of categorical features:

# + {"Collapsed": "false", "persistent_id": "621e1414-6268-4641-818b-5f8b5d54a446"}
cat_feat = ['allergyname', 'drugallergyhiclseqno']

# + {"Collapsed": "false", "persistent_id": "ce58accd-9f73-407c-b441-6da299604bb1"}
alrg_df[cat_feat].head()

# + {"Collapsed": "false"}
alrg_df[alrg_df.allergyname.str.contains('other')].allergyname.value_counts()

# + {"Collapsed": "false"}
alrg_df[alrg_df.allergyname.str.contains('unknown')].allergyname.value_counts()

# + [markdown] {"Collapsed": "false"}
# Filter just to the most common categories:

# + {"Collapsed": "false"}
for col in cat_feat:
    most_common_cat = list(alrg_df[col].value_counts().nlargest(MAX_CATEGORIES).index)
    alrg_df = alrg_df[alrg_df[col].isin(most_common_cat)]

# + [markdown] {"Collapsed": "false"}
# Apply one hot encoding:

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "9c0217cf-66d8-467b-b0df-b75441b1c0dc"}
alrg_df, new_columns = du.data_processing.one_hot_encoding_dataframe(alrg_df, columns=cat_feat,
                                                                     join_rows=False,
                                                                     get_new_column_names=True,
                                                                     inplace=True)
alrg_df

# + [markdown] {"Collapsed": "false"}
# Fill missing values of the drug and allergies data with 0, so as to prepare for embedding:

# + {"Collapsed": "false", "persistent_id": "50a95412-7211-4780-b0da-aad5e166191e"}
alrg_df.drugallergyhiclseqno = alrg_df.drugallergyhiclseqno.fillna(0).astype(int)

# + {"Collapsed": "false", "persistent_id": "50a95412-7211-4780-b0da-aad5e166191e"}
alrg_df.allergyname = alrg_df.allergyname.fillna(0).astype(int)

# + {"Collapsed": "false", "persistent_id": "e7ebceaf-e35b-4143-8fea-b5b14016e7f3"}
alrg_df.head()

# + {"Collapsed": "false", "persistent_id": "1b6e81c6-87ba-44dc-9c8d-73e168e946a6"}
alrg_df.dtypes

# + [markdown] {"Collapsed": "false"}
# Save the association between the original categorical features and the new one hot encoded columns:

# + {"Collapsed": "false", "persistent_id": "26eac7f3-9081-4a96-ae4a-40054c223fd7", "execution": {"iopub.status.busy": "2020-03-09T16:37:35.157248Z", "iopub.execute_input": "2020-03-09T16:37:35.157526Z", "iopub.status.idle": "2020-03-09T16:37:35.164656Z", "shell.execute_reply.started": "2020-03-09T16:37:35.157493Z", "shell.execute_reply": "2020-03-09T16:37:35.163771Z"}}
for orig_col in cat_feat:
    cat_feat_ohe[orig_col] = [ohe_col for ohe_col in new_columns
                              if ohe_col.startswith(orig_col)]

# + {"execution": {"iopub.status.busy": "2020-03-09T16:37:35.165864Z", "iopub.execute_input": "2020-03-09T16:37:35.166280Z", "iopub.status.idle": "2020-03-09T16:37:35.190294Z", "shell.execute_reply.started": "2020-03-09T16:37:35.166256Z", "shell.execute_reply": "2020-03-09T16:37:35.189358Z"}, "Collapsed": "false"}
cat_feat_ohe

# + [markdown] {"Collapsed": "false"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "45fdd1e4-00cd-49f8-b498-f50fb291e89a"}
stream = open(f'{data_path}/cleaned/cat_feat_ohe_alrg.yaml', 'w')
yaml.dump(cat_feat_ohe, stream, default_flow_style=False)

# + [markdown] {"Collapsed": "false"}
# ### Create the timestamp feature and sort

# + [markdown] {"Collapsed": "false"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "2f698d25-5a2b-44be-9d82-2c7790ee489f"}
alrg_df = alrg_df.rename(columns={'allergyoffset': 'ts'})
alrg_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "815f11a7-7b0d-44eb-b0f1-9061157864ca"}
len(alrg_df)

# + {"Collapsed": "false", "persistent_id": "96c4e771-2db5-461f-bf53-6b098feb26b4"}
alrg_df = alrg_df.drop_duplicates()
alrg_df.head()

# + {"Collapsed": "false", "persistent_id": "4baae0fb-9777-4abe-bcfe-f7c254e7bfc7"}
len(alrg_df)

# + [markdown] {"Collapsed": "false"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "173d1236-0aad-49a4-a8fd-b25c99bc30bc"}
alrg_df = alrg_df.sort_values('ts')
alrg_df.head()

# + [markdown] {"Collapsed": "false"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "d2660024-2f7e-4d37-b312-2a69aea35f0a"}
alrg_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='allergyname', n=5).head()

# + {"Collapsed": "false", "persistent_id": "a49094b0-6f71-4f70-ae6c-223373294b50"}
alrg_df[alrg_df.patientunitstayid == 3197554].head(10)

# + [markdown] {"Collapsed": "false"}
# We can see that there are up to 47 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# + [markdown] {"Collapsed": "false"}
# ### Join rows that have the same IDs

# + [markdown] {"Collapsed": "false"}
# Even after removing duplicates rows, there are still some that have different information for the same ID and timestamp. We have to concatenate the categorical enumerations.

# + [markdown] {"Collapsed": "false"}
# Convert dataframe to Pandas, as the multi-column groupby operation in `join_repeated_rows` isn't working with Modin:

# + {"Collapsed": "false"}
alrg_df, pd = du.utils.convert_dataframe(alrg_df, to='pandas', dtypes=dict(alrg_df.dtypes))

# + {"Collapsed": "false"}
type(alrg_df)

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "bf671749-9886-44f0-923f-24fd31d7d371"}
alrg_df = du.embedding.join_repeated_rows(alrg_df, inplace=True)
alrg_df.head()

# + [markdown] {"Collapsed": "false"}
# Reconvert dataframe to Modin:

# + {"Collapsed": "false"}
alrg_df, pd = du.utils.convert_dataframe(alrg_df, to='modin')

# + {"Collapsed": "false"}
type(alrg_df)

# + {"Collapsed": "false", "persistent_id": "96d8f554-5a5f-4b82-b5d5-47e5ce4c0f75"}
alrg_df.dtypes

# + {"Collapsed": "false", "persistent_id": "4024c555-2938-4f9a-9105-e4d77569267a"}
alrg_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='allergyname', n=5).head()

# + {"Collapsed": "false", "persistent_id": "7bb3f0b5-8f04-42f8-96aa-716762f65e5a"}
alrg_df[alrg_df.patientunitstayid == 3197554].head(10)

# + [markdown] {"Collapsed": "false"}
# Comparing the output from the two previous cells with what we had before the `join_repeated_rows` method, we can see that all rows with duplicate IDs have been successfully joined.

# + [markdown] {"Collapsed": "false"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "d6152c1e-f76f-4148-8c54-aec40a2636b1"}
alrg_df.columns = du.data_processing.clean_naming(alrg_df.columns)
alrg_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Save the dataframe

# + [markdown] {"Collapsed": "false"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "58eeead0-9cdc-4095-9643-2009af44a9a3"}
alrg_df.to_csv(f'{data_path}cleaned/unnormalized/ohe/allergy.csv')

# + [markdown] {"Collapsed": "false"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "b89fd87e-1200-41ef-83dc-09e0e3309f09"}
alrg_df.to_csv(f'{data_path}cleaned/normalized/ohe/allergy.csv')

# + [markdown] {"Collapsed": "false"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "2297202e-d250-430b-9ecd-23efc756cb25"}
alrg_df.describe().transpose()

# + [markdown] {"toc-hr-collapsed": true, "Collapsed": "false"}
# ## Past history data

# + [markdown] {"Collapsed": "false"}
# ### Initialize variables

# + {"Collapsed": "false", "persistent_id": "754a96f8-d389-4968-8c13-52e5e9d0bf82"}
# List of categorical features
cat_feat = []
# Dictionary of the one hot encoded columns originary from each categorical feature, that will be embedded
cat_feat_ohe = dict()

# + [markdown] {"Collapsed": "false"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "db086782-764c-4f63-b32f-6246f7c49a9b"}
past_hist_df = pd.read_csv(f'{data_path}original/pastHistory.csv')
past_hist_df.head()

# + {"Collapsed": "false", "persistent_id": "3a3c896b-98cf-4d4c-b35b-48f2bb46645d"}
len(past_hist_df)

# + {"Collapsed": "false", "persistent_id": "c670ebb3-032d-496a-8973-f213e5b04b2d"}
past_hist_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "f65b8d4c-4832-4346-bf23-b87b2ec5c16f"}
past_hist_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "fae1abf5-445a-4d27-8f17-6379aee5fa72"}
past_hist_df.columns

# + {"Collapsed": "false", "persistent_id": "60672930-c5c9-4482-a5de-919c9dff3f75"}
past_hist_df.dtypes

# + [markdown] {"Collapsed": "false"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "a417ded5-02be-4fa6-9f63-3c0a79d7f512"}
du.search_explore.dataframe_missing_values(past_hist_df)

# + [markdown] {"Collapsed": "false"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "156f2c4b-029d-483f-ae2f-efc88ee80b88"}
past_hist_df.pasthistorypath.value_counts().head(20)

# + {"Collapsed": "false", "persistent_id": "a4e04c0a-863a-4b60-8028-5f6a384dc057"}
past_hist_df.pasthistorypath.value_counts().tail(20)

# + {"Collapsed": "false", "persistent_id": "be0eea31-5880-4233-b223-47401d6ac827"}
past_hist_df.pasthistoryvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "acb0333c-2b3f-4aca-88b2-2b1cf7c479e4"}
past_hist_df.pasthistorynotetype.value_counts()

# + {"Collapsed": "false", "persistent_id": "c141a8e0-2af2-4d2c-aee8-60700ddc5301"}
past_hist_df[past_hist_df.pasthistorypath == 'notes/Progress Notes/Past History/Past History Obtain Options/Performed'].pasthistoryvalue.value_counts()

# + [markdown] {"Collapsed": "false"}
# In this case, considering that it regards past diagnosis of the patients, the timestamp when that was observed probably isn't very reliable nor useful. As such, I'm going to remove the offset variables. Furthermore, `past_historyvaluetext` is redundant with `past_historyvalue`, while `past_historynotetype` and the past history path 'notes/Progress Notes/Past History/Past History Obtain Options/Performed' seem to be irrelevant.

# + {"Collapsed": "false", "persistent_id": "cce8b969-b435-45e5-a3aa-2f646f816491"}
past_hist_df = past_hist_df.drop(['pasthistoryid', 'pasthistoryoffset', 'pasthistoryenteredoffset',
                                  'pasthistorynotetype', 'pasthistoryvaluetext'], axis=1)
past_hist_df.head()

# + {"Collapsed": "false", "persistent_id": "787315bf-b470-4538-9ad1-adcc6ef93c65"}
categories_to_remove = ['notes/Progress Notes/Past History/Past History Obtain Options/Performed']

# + {"Collapsed": "false", "persistent_id": "ce524b85-8006-4c1a-af08-aa1c6094b152"}
~(past_hist_df.pasthistorypath.isin(categories_to_remove)).head()

# + {"Collapsed": "false", "persistent_id": "23926113-127a-442a-8c33-33deb5efa772"}
past_hist_df = past_hist_df[~(past_hist_df.pasthistorypath.isin(categories_to_remove))]
past_hist_df.head()

# + {"Collapsed": "false", "persistent_id": "e9374a3c-b2be-428c-a853-0ded647a6c70"}
len(past_hist_df)

# + {"Collapsed": "false", "persistent_id": "b08768b7-7942-4cff-b4a6-8ee10322c4f1"}
past_hist_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "ac4e7984-b356-4696-9ba7-ef5763aeac89"}
past_hist_df.pasthistorypath.value_counts().head(20)

# + {"Collapsed": "false", "persistent_id": "b8a9f87a-0259-4094-8f03-eb2d64aeea8b"}
past_hist_df.pasthistorypath.value_counts().tail(20)

# + {"Collapsed": "false", "persistent_id": "4f93d732-2641-4ded-83b0-fbb8eb7f2421"}
past_hist_df.pasthistoryvalue.value_counts()

# + [markdown] {"Collapsed": "false"}
# There's still plenty of data left, affecting around 81.87% of the unit stays, even after removing several categories.

# + [markdown] {"Collapsed": "false"}
# ### Separate high level notes

# + {"Collapsed": "false", "persistent_id": "e6f07ded-5aa3-4efc-9fc8-3c7a35eadb25"}
past_hist_df.pasthistorypath.map(lambda x: x.split('/')).head().values

# + {"Collapsed": "false", "persistent_id": "d4d5ecbb-5b79-4a15-a997-7863b3facb38"}
past_hist_df.pasthistorypath.map(lambda x: len(x.split('/'))).min()

# + {"Collapsed": "false", "persistent_id": "26a98b9f-0972-482d-956f-57c3b7eac41a"}
past_hist_df.pasthistorypath.map(lambda x: len(x.split('/'))).max()

# + {"Collapsed": "false", "persistent_id": "ae522e1e-8465-4e53-8299-c0fc1f3757c1"}
past_hist_df.pasthistorypath.apply(lambda x: du.search_explore.get_element_from_split(x, 0, separator='/')).value_counts()

# + {"Collapsed": "false", "persistent_id": "e72babaa-63b4-4804-87b5-f9ee67fd7118"}
past_hist_df.pasthistorypath.apply(lambda x: du.search_explore.get_element_from_split(x, 1, separator='/')).value_counts()

# + {"Collapsed": "false", "persistent_id": "bf504856-1e69-40a8-9677-1878144e00f7"}
past_hist_df.pasthistorypath.apply(lambda x: du.search_explore.get_element_from_split(x, 2, separator='/')).value_counts()

# + {"Collapsed": "false", "persistent_id": "5f79fcc3-0dfa-40fa-b2f9-4e2f1211feab"}
past_hist_df.pasthistorypath.apply(lambda x: du.search_explore.get_element_from_split(x, 3, separator='/')).value_counts()

# + {"Collapsed": "false", "persistent_id": "271ee812-3434-47b0-9dd0-6edfea59c5fe"}
past_hist_df.pasthistorypath.apply(lambda x: du.search_explore.get_element_from_split(x, 4, separator='/')).value_counts()

# + {"Collapsed": "false", "persistent_id": "4b03a6bd-ea02-456e-9d28-095f2b10fea0"}
past_hist_df.pasthistorypath.apply(lambda x: du.search_explore.get_element_from_split(x, 5, separator='/')).value_counts()

# + {"Collapsed": "false", "persistent_id": "954a240a-a4b1-4e5a-b2d0-1f1864040aac"}
past_hist_df.pasthistorypath.apply(lambda x: du.search_explore.get_element_from_split(x, 6, separator='/')).value_counts()

# + [markdown] {"Collapsed": "false"}
# There are always at least 5 levels of the notes. As the first 4 ones are essentially always the same ("notes/Progress Notes/Past History/Organ Systems/") and the 5th one tends to not be very specific (only indicates which organ system it affected, when it isn't just a case of no health problems detected), it's best to preserve the 5th and isolate the remaining string as a new feature. This way, the split provides further insight to the model on similar notes.

# + {"Collapsed": "false", "persistent_id": "abfe7998-c744-4653-96d4-752c3c7c62a8"}
past_hist_df['pasthistorytype'] = past_hist_df.pasthistorypath.apply(lambda x: du.search_explore.get_element_from_split(x, 4, separator='/'))
past_hist_df['pasthistorydetails'] = past_hist_df.pasthistorypath.apply(lambda x: du.search_explore.get_element_from_split(x, 5, separator='/', till_the_end=True))
past_hist_df.head()

# + [markdown] {"Collapsed": "false"}
# `pasthistoryvalue` seems to correspond to the last element of `pasthistorydetails`. Let's confirm it:

# + {"Collapsed": "false", "persistent_id": "d299e5c1-9355-4c3d-9af4-c54e24f289ad"}
past_hist_df['pasthistorydetails_last'] = past_hist_df.pasthistorydetails.map(lambda x: x.split('/')[-1])
past_hist_df.head()

# + [markdown] {"Collapsed": "false"}
# Compare columns `past_historyvalue` and `past_historydetails`'s last element:

# + {"Collapsed": "false", "persistent_id": "f62af377-9237-4005-80b8-47aa0c83570a"}
past_hist_df[past_hist_df.pasthistoryvalue != past_hist_df.pasthistorydetails_last]

# + [markdown] {"Collapsed": "false"}
# The previous output confirms that the newly created `pasthistorydetails` feature's last elememt (last string in the symbol separated lists) is almost exactly equal to the already existing `pasthistoryvalue` feature, with the differences that `pasthistoryvalue` takes into account the scenarios of no health problems detected and behaves correctly in strings that contain the separator symbol in them. So, we should remove `pasthistorydetails`'s last element:

# + {"Collapsed": "false", "persistent_id": "40418862-5680-4dc5-9348-62e98599a638"}
past_hist_df = past_hist_df.drop('pasthistorydetails_last', axis=1)
past_hist_df.head()

# + {"Collapsed": "false", "persistent_id": "5022385b-8935-436e-a82b-8c402c0808f5"}
past_hist_df['pasthistorydetails'] = past_hist_df.pasthistorydetails.apply(lambda x: '/'.join(x.split('/')[:-1]))
past_hist_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove irrelevant `Not Obtainable` and `Not Performed` values:

# + {"Collapsed": "false", "persistent_id": "634e9588-7d76-46d5-a152-c1b73660d558"}
past_hist_df[past_hist_df.pasthistoryvalue == 'Not Obtainable'].pasthistorydetails.value_counts()

# + {"Collapsed": "false", "persistent_id": "490b10c9-202d-4d5e-830d-cbb84272486a"}
past_hist_df[past_hist_df.pasthistoryvalue == 'Not Performed'].pasthistorydetails.value_counts()

# + {"Collapsed": "false", "persistent_id": "20746db6-d74f-49eb-bceb-d46fc0c981c0"}
past_hist_df = past_hist_df[~((past_hist_df.pasthistoryvalue == 'Not Obtainable') | (past_hist_df.pasthistoryvalue == 'Not Performed'))]
past_hist_df.head()

# + {"Collapsed": "false", "persistent_id": "aa099682-9266-4567-8c7c-11043ca3d932"}
past_hist_df.pasthistorytype.unique()

# + [markdown] {"Collapsed": "false"}
# Replace blank `pasthistorydetails` values:

# + {"Collapsed": "false", "persistent_id": "8a7f891f-2f84-45ac-a2d5-856531eba2bb"}
past_hist_df[past_hist_df.pasthistoryvalue == 'No Health Problems'].pasthistorydetails.value_counts()

# + {"Collapsed": "false", "persistent_id": "9fb3ddad-84bb-44c9-b76f-ff2d7475b38a"}
past_hist_df[past_hist_df.pasthistoryvalue == 'No Health Problems'].pasthistorydetails.value_counts().index

# + {"Collapsed": "false", "persistent_id": "781296cb-c8fa-49e5-826d-c9e297553c0e"}
past_hist_df[past_hist_df.pasthistorydetails == ''].head()

# + {"Collapsed": "false", "persistent_id": "23607e71-135a-4281-baaa-ccff0f9765ad"}
past_hist_df['pasthistorydetails'] = past_hist_df.apply(lambda df: 'No Health Problems' if df['pasthistorytype'] == 'No Health Problems'
                                                                 else df['pasthistorydetails'], axis=1)
past_hist_df.head()

# + {"Collapsed": "false", "persistent_id": "6a32a636-2c60-45c4-b20f-8b82c9921cb4"}
past_hist_df[past_hist_df.pasthistorydetails == '']

# + {"Collapsed": "false"}
past_hist_df.pasthistoryvalue.value_counts()

# + {"Collapsed": "false"}
past_hist_df.pasthistorydetails.value_counts()

# + {"Collapsed": "false"}
past_hist_df.pasthistoryvalue.nunique()

# + {"Collapsed": "false"}
past_hist_df.pasthistorydetails.nunique()

# + [markdown] {"Collapsed": "false"}
# Remove the now redundant `pasthistorypath` column:

# + {"Collapsed": "false", "persistent_id": "9268a24d-f3e7-407c-9c99-65020d7c17f0"}
past_hist_df = past_hist_df.drop('pasthistorypath', axis=1)
past_hist_df.head()

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": false}
# ### Discretize categorical features
#
# Convert binary categorical features into one hot encode columns, which can later be embedded or used as is.

# + [markdown] {"Collapsed": "false"}
# #### One hot encode features

# + [markdown] {"Collapsed": "false"}
# Update list of categorical features:

# + {"Collapsed": "false", "persistent_id": "e6083094-1f99-408d-9b12-dfe4d88ee39a"}
cat_feat = ['pasthistoryvalue', 'pasthistorytype', 'pasthistorydetails']

# + {"Collapsed": "false", "persistent_id": "aa911443-7f86-44ea-ab90-997fd38ba074"}
past_hist_df[cat_feat].head()

# + [markdown] {"Collapsed": "false"}
# Filter just to the most common categories:

# + {"Collapsed": "false"}
for col in cat_feat:
    most_common_cat = list(past_hist_df[col].value_counts().nlargest(MAX_CATEGORIES).index)
    past_hist_df = past_hist_df[past_hist_df[col].isin(most_common_cat)]

# + [markdown] {"Collapsed": "false"}
# Apply one hot encoding:

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "58e7c624-cfc6-4df1-83dc-a8a22fe7ffc0"}
past_hist_df, new_columns = du.data_processing.one_hot_encoding_dataframe(past_hist_df, columns=cat_feat,
                                                                          join_rows=False,
                                                                          get_new_column_names=True,
                                                                          inplace=True)
past_hist_df

# + {"Collapsed": "false", "persistent_id": "7ee06a1a-cb99-4a94-9271-6f67948fd2a6"}
past_hist_df.head()

# + {"Collapsed": "false", "persistent_id": "2b7dde92-fe4b-42ce-9705-9d505878a696"}
past_hist_df.dtypes

# + [markdown] {"Collapsed": "false"}
# Save the association between the original categorical features and the new one hot encoded columns:

# + {"Collapsed": "false", "persistent_id": "26eac7f3-9081-4a96-ae4a-40054c223fd7", "execution": {"iopub.status.busy": "2020-03-09T16:37:35.157248Z", "iopub.execute_input": "2020-03-09T16:37:35.157526Z", "iopub.status.idle": "2020-03-09T16:37:35.164656Z", "shell.execute_reply.started": "2020-03-09T16:37:35.157493Z", "shell.execute_reply": "2020-03-09T16:37:35.163771Z"}}
for orig_col in cat_feat:
    cat_feat_ohe[orig_col] = [ohe_col for ohe_col in new_columns
                              if ohe_col.startswith(orig_col)]

# + {"execution": {"iopub.status.busy": "2020-03-09T16:37:35.165864Z", "iopub.execute_input": "2020-03-09T16:37:35.166280Z", "iopub.status.idle": "2020-03-09T16:37:35.190294Z", "shell.execute_reply.started": "2020-03-09T16:37:35.166256Z", "shell.execute_reply": "2020-03-09T16:37:35.189358Z"}, "Collapsed": "false"}
cat_feat_ohe

# + [markdown] {"Collapsed": "false"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "534d4a78-6d3e-4e3b-b318-5f353835d53a"}
stream = open(f'{data_path}/cleaned/cat_feat_ohe_past_hist.yaml', 'w')
yaml.dump(cat_feat_ohe, stream, default_flow_style=False)

# + [markdown] {"Collapsed": "false"}
# ### Remove duplicate rows

# + [markdown] {"Collapsed": "false"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "1a99e9da-39a1-46ae-9d0d-68c3dd063d8f"}
len(past_hist_df)

# + {"Collapsed": "false", "persistent_id": "5296709d-b5e3-44f7-bdcc-45d17955a2b6"}
past_hist_df = past_hist_df.drop_duplicates()
past_hist_df.head()

# + {"Collapsed": "false", "persistent_id": "b3483b1e-cced-4e2b-acc0-e95870a628b2"}
len(past_hist_df)

# + [markdown] {"Collapsed": "false"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "acae2295-6f7e-4290-b8a7-12d13042b65d"}
past_hist_df.groupby('patientunitstayid').count().nlargest(columns='pasthistoryvalue', n=5).head()

# + {"Collapsed": "false", "persistent_id": "1c71ea45-4026-43ac-8433-bd70d567bee9"}
past_hist_df[past_hist_df.patientunitstayid == 1558102].head(10)

# + [markdown] {"Collapsed": "false"}
# We can see that there are up to 20 categories per `patientunitstayid`. As such, we must join them.

# + [markdown] {"Collapsed": "false"}
# ### Join rows that have the same IDs

# + [markdown] {"Collapsed": "false"}
# Convert dataframe to Pandas, as the groupby operation in `join_repeated_rows` isn't working properly with Modin:

# + {"Collapsed": "false"}
past_hist_df, pd = du.utils.convert_dataframe(past_hist_df, to='pandas', dtypes=dict(past_hist_df.dtypes))

# + {"Collapsed": "false"}
type(past_hist_df)

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "589931b8-fe11-439a-8b14-4857c168c023"}
past_hist_df = du.embedding.join_repeated_rows(past_hist_df, id_columns=['patientunitstayid'], inplace=True)
past_hist_df.head()

# + {"Collapsed": "false"}
type(past_hist_df)

# + {"Collapsed": "false", "persistent_id": "6edc2eea-1139-4f7f-a314-db03cd785128"}
past_hist_df.dtypes

# + {"Collapsed": "false", "persistent_id": "61f2c4df-d3b6-459b-9632-194e2736ff27"}
past_hist_df.groupby(['patientunitstayid']).count().nlargest(columns='pasthistoryvalue', n=5).head()

# + {"Collapsed": "false", "persistent_id": "aa5f247a-8e4b-4527-a265-2af71b0f8e06"}
past_hist_df[past_hist_df.patientunitstayid == 1558102].head(10)

# + [markdown] {"Collapsed": "false"}
# Comparing the output from the two previous cells with what we had before the `join_repeated_rows` method, we can see that all rows with duplicate IDs have been successfully joined.

# + [markdown] {"Collapsed": "false"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "0cab3fc4-cb1f-4845-a0f9-7e9758df6f28"}
past_hist_df.columns = du.data_processing.clean_naming(past_hist_df.columns)
past_hist_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Save the dataframe

# + [markdown] {"Collapsed": "false"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "01cb3dd4-7afa-454b-b8c4-e473cb305367"}
past_hist_df.to_csv(f'{data_path}cleaned/unnormalized/ohe/pastHistory.csv')

# + [markdown] {"Collapsed": "false"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "5924e842-7b4f-4b3a-bddf-9b89952dfe26"}
past_hist_df.to_csv(f'{data_path}cleaned/normalized/ohe/pastHistory.csv')

# + [markdown] {"Collapsed": "false"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "ee91beb5-1415-4960-9e84-8cbfbde07e15"}
past_hist_df.describe().transpose()

# + [markdown] {"toc-hr-collapsed": false, "Collapsed": "false"}
# ## Diagnosis data

# + [markdown] {"Collapsed": "false"}
# ### Initialize variables

# + {"Collapsed": "false", "persistent_id": "754a96f8-d389-4968-8c13-52e5e9d0bf82", "execution": {"iopub.status.busy": "2020-03-13T01:33:21.105845Z", "iopub.execute_input": "2020-03-13T01:33:21.106162Z", "iopub.status.idle": "2020-03-13T01:33:21.110329Z", "shell.execute_reply.started": "2020-03-13T01:33:21.106113Z", "shell.execute_reply": "2020-03-13T01:33:21.109207Z"}}
# List of categorical features
cat_feat = []
# Dictionary of the one hot encoded columns originary from each categorical feature, that will be embedded
cat_feat_ohe = dict()

# + [markdown] {"Collapsed": "false"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "8f98afc2-5613-4042-94e7-98e0b72867a1", "execution": {"iopub.status.busy": "2020-03-13T01:33:21.879737Z", "iopub.execute_input": "2020-03-13T01:33:21.880100Z", "iopub.status.idle": "2020-03-13T01:33:24.722820Z", "shell.execute_reply.started": "2020-03-13T01:33:21.880047Z", "shell.execute_reply": "2020-03-13T01:33:24.721854Z"}}
diagn_df = pd.read_csv(f'{data_path}original/diagnosis.csv')
diagn_df.head()

# + [markdown] {"Collapsed": "false"}
# Convert dataframe to Pandas, as the `one_hot_encoding_dataframe` isn't working properly with Modin:

# + {"Collapsed": "false"}
diagn_df, pd = du.utils.convert_dataframe(diagn_df, to='pandas', dtypes=dict(diagn_df.dtypes))

# + {"Collapsed": "false"}
type(diagn_df)

# + {"Collapsed": "false", "persistent_id": "f2bd9f00-de58-48d9-a304-5e96ba6b392d"}
len(diagn_df)

# + {"Collapsed": "false", "persistent_id": "a0b999fb-9767-43de-8ed7-59dc72f68635"}
diagn_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "3f42e4cb-0064-4b01-9f6e-496caafb08dd"}
diagn_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "aaa6012c-e776-4335-9c5b-0101f4dc153d"}
diagn_df.columns

# + {"Collapsed": "false", "persistent_id": "81262e48-301a-4230-aae0-b94bf3f584d8"}
diagn_df.dtypes

# + [markdown] {"Collapsed": "false"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "a494abea-0f81-4ea8-8745-51ebfc306125"}
du.search_explore.dataframe_missing_values(diagn_df)

# + [markdown] {"Collapsed": "false"}
# ### Remove unneeded features

# + [markdown] {"Collapsed": "false"}
# Besides the usual removal of row identifier, `diagnosisid`, I'm also removing apparently irrelevant (and subjective) `diagnosispriority`, redundant, with missing values and other issues `icd9code`, and `activeupondischarge`, as we don't have complete information as to when diagnosis end.

# + {"Collapsed": "false", "persistent_id": "9073b0ba-7bc9-4b9c-aac7-ac1fd4802e63", "execution": {"iopub.status.busy": "2020-03-13T01:33:25.374128Z", "iopub.execute_input": "2020-03-13T01:33:25.374400Z", "iopub.status.idle": "2020-03-13T01:33:25.422219Z", "shell.execute_reply.started": "2020-03-13T01:33:25.374360Z", "shell.execute_reply": "2020-03-13T01:33:25.421327Z"}}
diagn_df = diagn_df.drop(['diagnosisid', 'diagnosispriority', 'icd9code', 'activeupondischarge'], axis=1)
diagn_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Separate high level diagnosis

# + {"Collapsed": "false", "persistent_id": "2318f39a-d909-49af-a13b-c4b1095bf161"}
diagn_df.diagnosisstring.value_counts()

# + {"Collapsed": "false", "persistent_id": "af82bc05-a5d8-4a40-a61a-d1f160c69b3d"}
diagn_df.diagnosisstring.map(lambda x: x.split('|')).head()

# + {"Collapsed": "false", "persistent_id": "08936a84-6640-4b15-bb1b-0192922d6daf"}
diagn_df.diagnosisstring.map(lambda x: len(x.split('|'))).min()

# + {"Collapsed": "false", "persistent_id": "08936a84-6640-4b15-bb1b-0192922d6daf"}
diagn_df.diagnosisstring.map(lambda x: len(x.split('|'))).max()

# + [markdown] {"Collapsed": "false"}
# There are always at least 2 higher level diagnosis. It could be beneficial to extract those first 2 levels to separate features, so as to avoid the need for the model to learn similarities that are already known.

# + {"Collapsed": "false", "persistent_id": "38475e5c-1260-4068-af50-5072366282ce", "execution": {"iopub.status.busy": "2020-03-13T01:33:28.234828Z", "iopub.execute_input": "2020-03-13T01:33:28.235237Z", "iopub.status.idle": "2020-03-13T01:33:37.304380Z", "shell.execute_reply.started": "2020-03-13T01:33:28.235188Z", "shell.execute_reply": "2020-03-13T01:33:37.303399Z"}}
diagn_df['diagnosis_type_1'] = diagn_df.diagnosisstring.apply(lambda x: du.search_explore.get_element_from_split(x, 0, separator='|'))
diagn_df['diagnosis_disorder_2'] = diagn_df.diagnosisstring.apply(lambda x: du.search_explore.get_element_from_split(x, 1, separator='|'))
diagn_df['diagnosis_detailed_3'] = diagn_df.diagnosisstring.apply(lambda x: du.search_explore.get_element_from_split(x, 2, separator='|', till_the_end=True))
# Remove now redundant `diagnosisstring` feature
diagn_df = diagn_df.drop('diagnosisstring', axis=1)
diagn_df.head()

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": false}
# ### Discretize categorical features
#
# Convert binary categorical features into one hot encode columns, which can later be embedded or used as is.

# + [markdown] {"Collapsed": "false"}
# #### One hot encode features

# + [markdown] {"Collapsed": "false"}
# Update list of categorical features:

# + {"Collapsed": "false", "persistent_id": "67f7e344-f016-4d85-a051-df96d07b5274", "execution": {"iopub.status.busy": "2020-03-13T01:33:37.305974Z", "iopub.execute_input": "2020-03-13T01:33:37.306217Z", "iopub.status.idle": "2020-03-13T01:33:37.309986Z", "shell.execute_reply.started": "2020-03-13T01:33:37.306172Z", "shell.execute_reply": "2020-03-13T01:33:37.309166Z"}}
cat_feat = ['diagnosis_type_1', 'diagnosis_disorder_2', 'diagnosis_detailed_3']

# + {"Collapsed": "false", "persistent_id": "72f3e710-08ef-4c9d-9876-c9d8cbaaf5f0", "execution": {"iopub.status.busy": "2020-03-13T00:47:30.594493Z", "iopub.execute_input": "2020-03-13T00:47:30.595456Z", "iopub.status.idle": "2020-03-13T00:47:30.743517Z", "shell.execute_reply.started": "2020-03-13T00:47:30.595295Z", "shell.execute_reply": "2020-03-13T00:47:30.742797Z"}}
diagn_df[cat_feat].head()

# + [markdown] {"Collapsed": "false"}
# Filter just to the most common categories:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-13T01:33:37.311818Z", "iopub.execute_input": "2020-03-13T01:33:37.312165Z", "iopub.status.idle": "2020-03-13T01:33:39.512953Z", "shell.execute_reply.started": "2020-03-13T01:33:37.312113Z", "shell.execute_reply": "2020-03-13T01:33:39.512118Z"}}
for col in cat_feat:
    most_common_cat = list(diagn_df[col].value_counts().nlargest(MAX_CATEGORIES).index)
    diagn_df = diagn_df[diagn_df[col].isin(most_common_cat)]

# + [markdown] {"Collapsed": "false"}
# Apply one hot encoding:

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "bba23ddd-c1e5-49b7-9b7f-8fe5819ee7f9", "execution": {"iopub.status.busy": "2020-03-13T01:33:39.514439Z", "iopub.execute_input": "2020-03-13T01:33:39.514664Z", "iopub.status.idle": "2020-03-13T01:34:35.578835Z", "shell.execute_reply.started": "2020-03-13T01:33:39.514625Z", "shell.execute_reply": "2020-03-13T01:34:35.577856Z"}}
diagn_df, new_columns = du.data_processing.one_hot_encoding_dataframe(diagn_df, columns=cat_feat,
                                                                      join_rows=False,
                                                                      get_new_column_names=True,
                                                                      inplace=True)
diagn_df

# + {"Collapsed": "false", "persistent_id": "cbe8c721-69d6-4af1-ba27-ef8a6c166b19"}
diagn_df.dtypes

# + [markdown] {"Collapsed": "false"}
# Save the association between the original categorical features and the new one hot encoded columns:

# + {"Collapsed": "false", "persistent_id": "26eac7f3-9081-4a96-ae4a-40054c223fd7", "execution": {"iopub.status.busy": "2020-03-13T01:34:35.580225Z", "iopub.execute_input": "2020-03-13T01:34:35.580474Z", "iopub.status.idle": "2020-03-13T01:34:35.584799Z", "shell.execute_reply.started": "2020-03-13T01:34:35.580426Z", "shell.execute_reply": "2020-03-13T01:34:35.583824Z"}}
for orig_col in cat_feat:
    cat_feat_ohe[orig_col] = [ohe_col for ohe_col in new_columns
                              if ohe_col.startswith(orig_col)]

# + {"execution": {"iopub.status.busy": "2020-03-13T01:34:35.587342Z", "iopub.execute_input": "2020-03-13T01:34:35.587876Z", "iopub.status.idle": "2020-03-13T01:34:35.615729Z", "shell.execute_reply.started": "2020-03-13T01:34:35.587578Z", "shell.execute_reply": "2020-03-13T01:34:35.614673Z"}, "Collapsed": "false"}
cat_feat_ohe

# + [markdown] {"Collapsed": "false"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "95ac0351-6a18-41b0-9937-37d255fa34ca", "execution": {"iopub.status.busy": "2020-03-13T01:34:35.617214Z", "iopub.execute_input": "2020-03-13T01:34:35.617453Z", "iopub.status.idle": "2020-03-13T01:34:35.654211Z", "shell.execute_reply.started": "2020-03-13T01:34:35.617408Z", "shell.execute_reply": "2020-03-13T01:34:35.653133Z"}}
stream = open(f'{data_path}/cleaned/cat_feat_ohe_diag.yaml', 'w')
yaml.dump(cat_feat_ohe, stream, default_flow_style=False)

# + [markdown] {"Collapsed": "false"}
# ### Create the timestamp feature and sort

# + [markdown] {"Collapsed": "false"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "8011a320-6066-416e-bb3e-280d97088afe", "execution": {"iopub.status.busy": "2020-03-13T01:34:35.655365Z", "iopub.execute_input": "2020-03-13T01:34:35.655587Z", "iopub.status.idle": "2020-03-13T01:34:36.370946Z", "shell.execute_reply.started": "2020-03-13T01:34:35.655549Z", "shell.execute_reply": "2020-03-13T01:34:36.369920Z"}}
diagn_df = diagn_df.rename(columns={'diagnosisoffset': 'ts'})
diagn_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "2dfd1e82-de35-40b7-a67b-f57b14389787", "execution": {"iopub.status.busy": "2020-03-13T01:34:36.372033Z", "iopub.execute_input": "2020-03-13T01:34:36.372365Z", "iopub.status.idle": "2020-03-13T01:34:36.378401Z", "shell.execute_reply.started": "2020-03-13T01:34:36.372317Z", "shell.execute_reply": "2020-03-13T01:34:36.377469Z"}}
len(diagn_df)

# + {"Collapsed": "false", "persistent_id": "6c733c3e-26e8-4618-a849-cfab5e4065d8", "execution": {"iopub.status.busy": "2020-03-13T01:34:36.379773Z", "iopub.execute_input": "2020-03-13T01:34:36.380125Z", "iopub.status.idle": "2020-03-13T01:34:56.614532Z", "shell.execute_reply.started": "2020-03-13T01:34:36.380054Z", "shell.execute_reply": "2020-03-13T01:34:56.613639Z"}}
diagn_df = diagn_df.drop_duplicates()
diagn_df.head()

# + {"Collapsed": "false", "persistent_id": "710c6b32-3d96-427d-9d7f-e1ecc453ba7e", "execution": {"iopub.status.busy": "2020-03-13T01:34:56.615686Z", "iopub.execute_input": "2020-03-13T01:34:56.615938Z", "iopub.status.idle": "2020-03-13T01:34:56.622001Z", "shell.execute_reply.started": "2020-03-13T01:34:56.615896Z", "shell.execute_reply": "2020-03-13T01:34:56.621020Z"}}
len(diagn_df)

# + [markdown] {"Collapsed": "false"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "e82d40c2-9e0c-411a-a2e2-acb1dd4f1d96", "execution": {"iopub.status.busy": "2020-03-13T01:34:56.623232Z", "iopub.execute_input": "2020-03-13T01:34:56.623496Z", "iopub.status.idle": "2020-03-13T01:34:59.885673Z", "shell.execute_reply.started": "2020-03-13T01:34:56.623446Z", "shell.execute_reply": "2020-03-13T01:34:59.884569Z"}}
diagn_df = diagn_df.sort_values('ts')
diagn_df.head()

# + [markdown] {"Collapsed": "false"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "f28206fd-f1d0-4ca2-b006-653f60d05782"}
diagn_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='diagnosis_type_1', n=5).head()

# + {"Collapsed": "false", "persistent_id": "cac394e5-a6fa-4bc7-a496-1c97b49de381"}
diagn_df[diagn_df.patientunitstayid == 3089982].head(10)

# + [markdown] {"Collapsed": "false"}
# We can see that there are up to 69 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# + [markdown] {"Collapsed": "false"}
# ### Join rows that have the same IDs

# + [markdown] {"Collapsed": "false"}
# Convert dataframe to Pandas, as the groupby operation in `join_repeated_rows` isn't working properly with Modin:

# + {"Collapsed": "false"}
diagn_df, pd = du.utils.convert_dataframe(diagn_df, to='pandas', dtypes=dict(diagn_df.dtypes))

# + {"Collapsed": "false"}
type(diagn_df)

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "589931b8-fe11-439a-8b14-4857c168c023", "execution": {"iopub.status.busy": "2020-03-13T01:34:59.887004Z", "iopub.execute_input": "2020-03-13T01:34:59.887277Z", "iopub.status.idle": "2020-03-13T01:35:39.622442Z", "shell.execute_reply.started": "2020-03-13T01:34:59.887227Z", "shell.execute_reply": "2020-03-13T01:35:39.621552Z"}}
diagn_df = du.embedding.join_repeated_rows(diagn_df, inplace=True)
diagn_df.head()

# + [markdown] {"Collapsed": "false"}
# Reconvert dataframe to Modin:

# + {"Collapsed": "false"}
# diagn_df, pd = du.utils.convert_dataframe(diagn_df, to='modin')

# + {"Collapsed": "false"}
type(diagn_df)

# + {"Collapsed": "false", "persistent_id": "e854db5e-06f3-45ee-ad27-f5214f3f6ea7"}
diagn_df.dtypes

# + {"Collapsed": "false", "persistent_id": "21a955e7-fdc3-42d8-97e0-47f4401b7c8e"}
diagn_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='diagnosis_type_1', n=5).head()

# + {"Collapsed": "false", "persistent_id": "4dddda8f-daf1-4d2b-8782-8de20201ea7f"}
diagn_df[diagn_df.patientunitstayid == 3089982].head(10)

# + [markdown] {"Collapsed": "false"}
# Comparing the output from the two previous cells with what we had before the `join_repeated_rows` method, we can see that all rows with duplicate IDs have been successfully joined.

# + [markdown] {"Collapsed": "false"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "0f06cc9f-f14f-4c96-afdd-030ac8975f0b", "execution": {"iopub.status.busy": "2020-03-13T01:35:39.624193Z", "iopub.execute_input": "2020-03-13T01:35:39.624427Z", "iopub.status.idle": "2020-03-13T01:35:39.907047Z", "shell.execute_reply.started": "2020-03-13T01:35:39.624389Z", "shell.execute_reply": "2020-03-13T01:35:39.906081Z"}}
diagn_df.columns = du.data_processing.clean_naming(diagn_df.columns)
diagn_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Save the dataframe

# + [markdown] {"Collapsed": "false"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "0c0cc6d4-56a9-40d4-a213-288b7080fb72"}
# diagn_df.to_csv(f'{data_path}cleaned/unnormalized/ohe/diagnosis.csv')

# + [markdown] {"Collapsed": "false"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "4c0def92-984f-4dd3-a807-be7188be38e8", "execution": {"iopub.status.busy": "2020-03-13T01:35:39.908529Z", "iopub.execute_input": "2020-03-13T01:35:39.908823Z", "iopub.status.idle": "2020-03-13T01:37:14.758421Z", "shell.execute_reply.started": "2020-03-13T01:35:39.908774Z", "shell.execute_reply": "2020-03-13T01:37:14.757582Z"}}
diagn_df.to_csv(f'{data_path}cleaned/normalized/ohe/diagnosis.csv')

# + [markdown] {"Collapsed": "false"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "0906c011-5d47-49e4-b8d0-bfb97b575f66", "execution": {"iopub.status.busy": "2020-03-13T01:37:14.759579Z", "iopub.execute_input": "2020-03-13T01:37:14.759807Z"}}
diagn_df.describe().transpose()

# + {"Collapsed": "false"}
