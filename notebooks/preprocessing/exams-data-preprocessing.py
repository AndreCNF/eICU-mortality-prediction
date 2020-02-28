# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
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

# + {"Collapsed": "false", "cell_type": "markdown"}
# # Exams Data Preprocessing
# ---
#
# Reading and preprocessing exams data of the eICU dataset from MIT with the data from over 139k patients collected in the US.
#
# This notebook addresses the preprocessing of the following eICU tables:
# * lab

# + {"Collapsed": "false", "cell_type": "markdown"}
# ## Importing the necessary packages

# + {"Collapsed": "false", "colab": {}, "colab_type": "code", "execution": {"iopub.execute_input": "2020-02-21T03:03:42.335049Z", "iopub.status.busy": "2020-02-21T03:03:42.334719Z", "iopub.status.idle": "2020-02-21T03:03:42.358919Z", "shell.execute_reply": "2020-02-21T03:03:42.358083Z", "shell.execute_reply.started": "2020-02-21T03:03:42.334993Z"}, "execution_event_id": "deb57b39-6a79-4b3a-95ed-02f8089ff593", "id": "G5RrWE9R_Nkl", "last_executed_text": "import os                                  # os handles directory/workspace changes\nimport numpy as np                         # NumPy to handle numeric and NaN operations\nimport yaml                                # Save and load YAML files", "persistent_id": "522745b5-b5bf-479f-b697-5c7e9e12fc33"}
import os                                  # os handles directory/workspace changes
import numpy as np                         # NumPy to handle numeric and NaN operations
import yaml                                # Save and load YAML files

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-21T03:03:42.544114Z", "iopub.status.busy": "2020-02-21T03:03:42.543761Z", "iopub.status.idle": "2020-02-21T03:03:43.601472Z", "shell.execute_reply": "2020-02-21T03:03:43.600682Z", "shell.execute_reply.started": "2020-02-21T03:03:42.544061Z"}, "execution_event_id": "fa33a2f7-7127-49c6-bbe9-f89555b1f2be", "last_executed_text": "# Debugging packages\nimport pixiedust                           # Debugging in Jupyter Notebook cells", "persistent_id": "02accdbf-be7e-415c-ba11-165906e66c50"}
# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-21T03:03:43.603065Z", "iopub.status.busy": "2020-02-21T03:03:43.602844Z", "iopub.status.idle": "2020-02-21T03:03:43.606614Z", "shell.execute_reply": "2020-02-21T03:03:43.605832Z", "shell.execute_reply.started": "2020-02-21T03:03:43.603026Z"}, "execution_event_id": "baeb346a-1c34-42d1-a501-7ae37369255e", "last_executed_text": "# Change to parent directory (presumably \"Documents\")\nos.chdir(\"../../..\")\n\n# Path to the CSV dataset files\ndata_path = 'Documents/Datasets/Thesis/eICU/uncompressed/'\n\n# Path to the code files\nproject_path = 'Documents/GitHub/eICU-mortality-prediction/'", "persistent_id": "a1f6ee7f-36d4-489d-b2dd-ec2a38f15d11"}
# Change to parent directory (presumably "Documents")
os.chdir("../../../..")
# Path to the CSV dataset files
data_path = 'Datasets/Thesis/eICU/uncompressed/'
# Path to the code files
project_path = 'GitHub/eICU-mortality-prediction/'

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-21T03:03:43.608132Z", "iopub.status.busy": "2020-02-21T03:03:43.607878Z", "iopub.status.idle": "2020-02-21T03:03:43.615592Z", "shell.execute_reply": "2020-02-21T03:03:43.614885Z", "shell.execute_reply.started": "2020-02-21T03:03:43.608048Z"}}
# Make sure that every large operation can be handled, by using the disk as an overflow for the memory
# !export MODIN_OUT_OF_CORE=true

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-21T03:03:43.616826Z", "iopub.status.busy": "2020-02-21T03:03:43.616614Z", "iopub.status.idle": "2020-02-21T03:03:45.258306Z", "shell.execute_reply": "2020-02-21T03:03:45.257139Z", "shell.execute_reply.started": "2020-02-21T03:03:43.616789Z"}, "execution_event_id": "82ef68be-443a-4bb8-8abd-7457a7005b4d", "last_executed_text": "import modin.pandas as pd                  # Optimized distributed version of Pandas\nimport data_utils as du                    # Data science and machine learning relevant methods", "persistent_id": "c0c2e356-d4f4-4a9d-bec2-88bdf9eb6a38"}
import modin.pandas as pd                  # Optimized distributed version of Pandas
import data_utils as du                    # Data science and machine learning relevant methods

# + {"Collapsed": "false", "cell_type": "markdown"}
# Set the random seed for reproducibility:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-21T03:03:45.269907Z", "iopub.status.busy": "2020-02-21T03:03:45.266546Z", "iopub.status.idle": "2020-02-21T03:03:45.286373Z", "shell.execute_reply": "2020-02-21T03:03:45.285275Z", "shell.execute_reply.started": "2020-02-21T03:03:45.269833Z"}, "execution_event_id": "29ab85ce-b7fd-4c5a-a110-5841e741c369", "last_executed_text": "du.set_random_seed(42)", "persistent_id": "39b552cd-6948-4ec8-ac04-42f850c1e05a"}
du.set_random_seed(42)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ## Laboratory data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Initialize variables

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-21T03:03:47.279007Z", "iopub.status.busy": "2020-02-21T03:03:47.278690Z", "iopub.status.idle": "2020-02-21T03:03:47.282928Z", "shell.execute_reply": "2020-02-21T03:03:47.282076Z", "shell.execute_reply.started": "2020-02-21T03:03:47.278963Z"}}
cat_feat = []                              # List of categorical features
cat_embed_feat = []                        # List of categorical features that will be embedded
cat_embed_feat_enum = dict()               # Dictionary of the enumerations of the categorical features that will be embedded

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:15:48.682256Z", "iopub.status.busy": "2020-02-19T02:15:48.682053Z", "iopub.status.idle": "2020-02-19T02:16:16.651391Z", "shell.execute_reply": "2020-02-19T02:16:16.650436Z", "shell.execute_reply.started": "2020-02-19T02:15:48.682216Z"}, "persistent_id": "37d12a8d-d08c-41cd-b904-f005b1497fe1"}
lab_df = pd.read_csv(f'{data_path}original/lab.csv')
lab_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-18T14:38:34.295318Z", "iopub.status.busy": "2020-02-18T14:38:34.295101Z", "iopub.status.idle": "2020-02-18T14:38:34.299332Z", "shell.execute_reply": "2020-02-18T14:38:34.298760Z", "shell.execute_reply.started": "2020-02-18T14:38:34.295278Z"}, "persistent_id": "c1bc04e9-c48c-443b-9208-8ec4618a0e2d"}
len(lab_df)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-18T14:38:34.300680Z", "iopub.status.busy": "2020-02-18T14:38:34.300489Z", "iopub.status.idle": "2020-02-18T14:38:36.587129Z", "shell.execute_reply": "2020-02-18T14:38:36.586199Z", "shell.execute_reply.started": "2020-02-18T14:38:34.300645Z"}, "persistent_id": "675a744b-8308-4a4d-89fb-3f8ba150343f"}
lab_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "2eda81ce-69df-4328-96f0-4f770bd683d3"}
lab_df.describe().transpose()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-16T18:58:50.265277Z", "iopub.status.busy": "2020-02-16T18:58:50.264964Z"}}
lab_df.info()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-18T14:38:36.588409Z", "iopub.status.busy": "2020-02-18T14:38:36.588191Z", "iopub.status.idle": "2020-02-18T14:38:39.073587Z", "shell.execute_reply": "2020-02-18T14:38:39.072797Z", "shell.execute_reply.started": "2020-02-18T14:38:36.588367Z"}, "persistent_id": "095dbcb4-a4c4-4b9a-baca-41b954126f19", "pixiedust": {"displayParams": {}}}
du.search_explore.dataframe_missing_values(lab_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Merge similar columns

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-18T14:38:39.075093Z", "iopub.status.busy": "2020-02-18T14:38:39.074822Z", "iopub.status.idle": "2020-02-18T14:38:52.606523Z", "shell.execute_reply": "2020-02-18T14:38:52.605268Z", "shell.execute_reply.started": "2020-02-18T14:38:39.075050Z"}, "persistent_id": "68ac1b53-5c7f-4bae-9d20-7685d398bd04"}
lab_df.labresult.value_counts()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-18T14:38:52.607746Z", "iopub.status.busy": "2020-02-18T14:38:52.607509Z", "iopub.status.idle": "2020-02-18T14:39:27.545809Z", "shell.execute_reply": "2020-02-18T14:39:27.545110Z", "shell.execute_reply.started": "2020-02-18T14:38:52.607683Z"}, "persistent_id": "522ff137-54bb-47ef-a5c9-f3017c4665d3"}
lab_df.labresulttext.value_counts()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-18T14:39:27.546923Z", "iopub.status.busy": "2020-02-18T14:39:27.546700Z", "iopub.status.idle": "2020-02-18T14:39:50.969863Z", "shell.execute_reply": "2020-02-18T14:39:50.969155Z", "shell.execute_reply.started": "2020-02-18T14:39:27.546884Z"}, "persistent_id": "522ff137-54bb-47ef-a5c9-f3017c4665d3"}
lab_df.labresulttext.value_counts().tail(30)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-18T14:39:50.971019Z", "iopub.status.busy": "2020-02-18T14:39:50.970818Z", "iopub.status.idle": "2020-02-18T14:40:10.098142Z", "shell.execute_reply": "2020-02-18T14:40:10.097421Z", "shell.execute_reply.started": "2020-02-18T14:39:50.970982Z"}, "persistent_id": "24ae1f2a-bfa7-432f-bda5-0efcf48f110b"}
lab_df.labmeasurenamesystem.value_counts()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-18T14:40:10.100708Z", "iopub.status.busy": "2020-02-18T14:40:10.100389Z", "iopub.status.idle": "2020-02-18T14:40:31.163079Z", "shell.execute_reply": "2020-02-18T14:40:31.162257Z", "shell.execute_reply.started": "2020-02-18T14:40:10.100657Z"}}
lab_df.labmeasurenameinterface.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ~Merge the result columns:~
#
# I will not merge the result columns, so as to make sure that we only have numeric result values:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:16:16.653008Z", "iopub.status.busy": "2020-02-19T02:16:16.652771Z", "iopub.status.idle": "2020-02-19T02:16:16.666646Z", "shell.execute_reply": "2020-02-19T02:16:16.665846Z", "shell.execute_reply.started": "2020-02-19T02:16:16.652962Z"}}
# lab_df['lab_result'] = lab_df.apply(lambda df: du.data_processing.merge_values(df['labresult'],
#                                                                                df['labresulttext'],
#                                                                                str_over_num=False, join_strings=False),
#                                     axis=1)
# lab_df.head(10)
# Just renaming the lab results feature:
lab_df = lab_df.rename(columns={'labresult': 'lab_result'})

# + {"Collapsed": "false", "cell_type": "markdown"}
# ~Drop the now redundant `labresult` and `labresulttext` columns:~

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-18T14:40:31.176198Z", "iopub.status.busy": "2020-02-18T14:40:31.175884Z", "iopub.status.idle": "2020-02-18T14:40:31.185188Z", "shell.execute_reply": "2020-02-18T14:40:31.184575Z", "shell.execute_reply.started": "2020-02-18T14:40:31.176140Z"}, "persistent_id": "b31b8605-5a88-4821-a65c-5337128f4264"}
# lab_df = lab_df.drop(columns=['labresult', 'labresulttext'])
# lab_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Merge the measurement unit columns:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:16:16.668275Z", "iopub.status.busy": "2020-02-19T02:16:16.668013Z", "iopub.status.idle": "2020-02-19T02:18:47.264547Z", "shell.execute_reply": "2020-02-19T02:18:47.263652Z", "shell.execute_reply.started": "2020-02-19T02:16:16.668233Z"}}
lab_df['lab_units'] = lab_df.apply(lambda df: du.data_processing.merge_values(df['labmeasurenamesystem'],
                                                                              df['labmeasurenameinterface'],
                                                                              str_over_num=True, join_strings=False),
                                   axis=1)
lab_df.head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Drop the now redundant `labresult` and `labresulttext` columns:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:18:47.265795Z", "iopub.status.busy": "2020-02-19T02:18:47.265591Z", "iopub.status.idle": "2020-02-19T02:18:47.790396Z", "shell.execute_reply": "2020-02-19T02:18:47.789562Z", "shell.execute_reply.started": "2020-02-19T02:18:47.265759Z"}, "persistent_id": "b31b8605-5a88-4821-a65c-5337128f4264"}
lab_df = lab_df.drop(columns=['labmeasurenamesystem', 'labmeasurenameinterface'])
lab_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-18T14:43:57.481560Z", "iopub.status.busy": "2020-02-18T14:43:57.481339Z", "iopub.status.idle": "2020-02-18T14:44:15.543316Z", "shell.execute_reply": "2020-02-18T14:44:15.542698Z", "shell.execute_reply.started": "2020-02-18T14:43:57.481519Z"}, "persistent_id": "5cb3ea02-b35b-43dc-9f72-d040f53def73"}
lab_df.labtypeid.value_counts()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-18T14:44:15.544494Z", "iopub.status.busy": "2020-02-18T14:44:15.544285Z", "iopub.status.idle": "2020-02-18T14:44:50.212443Z", "shell.execute_reply": "2020-02-18T14:44:50.211617Z", "shell.execute_reply.started": "2020-02-18T14:44:15.544458Z"}, "persistent_id": "f803016b-728d-4200-8ffd-ed1992bec091"}
lab_df.labname.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Besides removing the row ID `labid` and the time when data was entered `labresultrevisedoffset`, I'm also removing `labresulttext` as it's redundant with `labresult` and has a string format instead of a numeric one.

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:18:47.791760Z", "iopub.status.busy": "2020-02-19T02:18:47.791532Z", "iopub.status.idle": "2020-02-19T02:18:48.408063Z", "shell.execute_reply": "2020-02-19T02:18:48.407380Z", "shell.execute_reply.started": "2020-02-19T02:18:47.791714Z"}, "persistent_id": "b31b8605-5a88-4821-a65c-5337128f4264"}
lab_df = lab_df.drop(columns=['labid', 'labresultrevisedoffset', 'labresulttext'])
lab_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-18T14:44:51.427540Z", "iopub.status.busy": "2020-02-18T14:44:51.427324Z", "iopub.status.idle": "2020-02-18T14:44:54.415864Z", "shell.execute_reply": "2020-02-18T14:44:54.414974Z", "shell.execute_reply.started": "2020-02-18T14:44:51.427503Z"}, "persistent_id": "095dbcb4-a4c4-4b9a-baca-41b954126f19", "pixiedust": {"displayParams": {}}}
du.search_explore.dataframe_missing_values(lab_df)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-18T14:44:54.417206Z", "iopub.status.busy": "2020-02-18T14:44:54.416971Z", "iopub.status.idle": "2020-02-18T14:45:59.818614Z", "shell.execute_reply": "2020-02-18T14:45:59.817843Z", "shell.execute_reply.started": "2020-02-18T14:44:54.417164Z"}}
lab_df.info()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into simple numberings, one hot encode features with a low number of categories (in this case, 5) and enumerate sparse categorical features that will be embedded.

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Separate and prepare features for embedding
#
# Identify categorical features that have more than 5 unique categories, which will go through an embedding layer afterwards, and enumerate them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Update list of categorical features and add those that will need embedding (features with more than 5 unique values):

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:18:48.409879Z", "iopub.status.busy": "2020-02-19T02:18:48.409662Z", "iopub.status.idle": "2020-02-19T02:18:48.414709Z", "shell.execute_reply": "2020-02-19T02:18:48.414127Z", "shell.execute_reply.started": "2020-02-19T02:18:48.409838Z"}, "persistent_id": "d714ff24-c50b-4dff-9b21-832d030d050f"}
new_cat_feat = ['labtypeid', 'labname', 'lab_units']
[cat_feat.append(col) for col in new_cat_feat]

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-18T14:45:59.825916Z", "iopub.status.busy": "2020-02-18T14:45:59.825691Z"}, "persistent_id": "d7e8e94d-e178-4fe6-96ff-e828eba8dc62"}
# Skipping this step here as it's very slow for this large dataframe and we already
# know that all of these features are going to be embedded
# cat_feat_nunique = [lab_df[feature].nunique() for feature in du.utils.iterations_loop(new_cat_feat)]
# cat_feat_nunique

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:18:48.416082Z", "iopub.status.busy": "2020-02-19T02:18:48.415886Z", "iopub.status.idle": "2020-02-19T02:18:48.420006Z", "shell.execute_reply": "2020-02-19T02:18:48.419355Z", "shell.execute_reply.started": "2020-02-19T02:18:48.416046Z"}, "persistent_id": "f3f370c1-79e0-4df7-ad6b-1c59ef8bcec6"}
new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
#     if cat_feat_nunique[i] > 5:
    # Add feature to the list of those that will be embedded
    cat_embed_feat.append(new_cat_feat[i])
    new_cat_embed_feat.append(new_cat_feat[i])

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:18:48.421015Z", "iopub.status.busy": "2020-02-19T02:18:48.420827Z", "iopub.status.idle": "2020-02-19T02:18:48.988729Z", "shell.execute_reply": "2020-02-19T02:18:48.988096Z", "shell.execute_reply.started": "2020-02-19T02:18:48.420980Z"}, "persistent_id": "ed09c3dd-50b3-48cf-aa04-76534feaf767"}
lab_df[new_cat_feat].head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:18:48.990117Z", "iopub.status.busy": "2020-02-19T02:18:48.989706Z", "iopub.status.idle": "2020-02-19T02:21:21.325944Z", "shell.execute_reply": "2020-02-19T02:21:21.325070Z", "shell.execute_reply.started": "2020-02-19T02:18:48.990072Z"}, "persistent_id": "51ac8fd1-cbd2-4f59-a737-f0fcc13043fd", "pixiedust": {"displayParams": {}}}
for i in du.utils.iterations_loop(range(len(new_cat_embed_feat))):
    feature = new_cat_embed_feat[i]
    if feature == 'labtypeid':
        # Skip the `labtypeid` feature as it already has a good numeric format
        continue
    # Prepare for embedding, i.e. enumerate categories
    lab_df[feature], cat_embed_feat_enum[feature] = du.embedding.enum_categorical_feature(lab_df, feature, nan_value=0,
                                                                                          forbidden_digit=0)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:21:21.328295Z", "iopub.status.busy": "2020-02-19T02:21:21.327725Z", "iopub.status.idle": "2020-02-19T02:21:29.995641Z", "shell.execute_reply": "2020-02-19T02:21:29.994965Z", "shell.execute_reply.started": "2020-02-19T02:21:21.328236Z"}, "persistent_id": "e38470e5-73f7-4d35-b91d-ce0793b7f6f6"}
lab_df[new_cat_feat].head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:21:29.996738Z", "iopub.status.busy": "2020-02-19T02:21:29.996536Z", "iopub.status.idle": "2020-02-19T02:21:30.004860Z", "shell.execute_reply": "2020-02-19T02:21:30.004303Z", "shell.execute_reply.started": "2020-02-19T02:21:29.996701Z"}, "persistent_id": "00b3ce19-756a-4de4-8b05-762de386aa29"}
cat_embed_feat_enum

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:21:30.005873Z", "iopub.status.busy": "2020-02-19T02:21:30.005679Z", "iopub.status.idle": "2020-02-19T02:21:31.845484Z", "shell.execute_reply": "2020-02-19T02:21:31.844781Z", "shell.execute_reply.started": "2020-02-19T02:21:30.005837Z"}, "persistent_id": "e5615265-4372-4117-a368-ec539c871763"}
lab_df[new_cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:21:31.846912Z", "iopub.status.busy": "2020-02-19T02:21:31.846680Z", "iopub.status.idle": "2020-02-19T02:21:31.951476Z", "shell.execute_reply": "2020-02-19T02:21:31.950669Z", "shell.execute_reply.started": "2020-02-19T02:21:31.846872Z"}, "persistent_id": "e51cc2e0-b598-484f-a3f8-8c764950777f"}
stream = open(f'{data_path}/cleaned/cat_embed_feat_enum_lab.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False) 

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:21:31.952745Z", "iopub.status.busy": "2020-02-19T02:21:31.952539Z", "iopub.status.idle": "2020-02-19T02:21:32.562030Z", "shell.execute_reply": "2020-02-19T02:21:32.561437Z", "shell.execute_reply.started": "2020-02-19T02:21:31.952709Z"}, "persistent_id": "88ab8d50-b556-4c76-bb0b-33e76900018f"}
lab_df = lab_df.rename(columns={'labresultoffset': 'ts'})
lab_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:21:32.563147Z", "iopub.status.busy": "2020-02-19T02:21:32.562939Z", "iopub.status.idle": "2020-02-19T02:21:32.567376Z", "shell.execute_reply": "2020-02-19T02:21:32.566837Z", "shell.execute_reply.started": "2020-02-19T02:21:32.563110Z"}, "persistent_id": "74b9d214-8083-4d37-acbe-6ce26d6b1629"}
len(lab_df)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:21:32.568346Z", "iopub.status.busy": "2020-02-19T02:21:32.568156Z", "iopub.status.idle": "2020-02-19T02:22:41.280877Z", "shell.execute_reply": "2020-02-19T02:22:41.280034Z", "shell.execute_reply.started": "2020-02-19T02:21:32.568311Z"}, "persistent_id": "a2529353-bf59-464a-a32b-a940dd66007a"}
lab_df = lab_df.drop_duplicates()
lab_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:22:41.282526Z", "iopub.status.busy": "2020-02-19T02:22:41.282195Z", "iopub.status.idle": "2020-02-19T02:22:41.287757Z", "shell.execute_reply": "2020-02-19T02:22:41.286844Z", "shell.execute_reply.started": "2020-02-19T02:22:41.282465Z"}, "persistent_id": "be199b11-006c-4619-ac80-b3d86fd10f3b"}
len(lab_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:22:41.290582Z", "iopub.status.busy": "2020-02-19T02:22:41.290137Z", "iopub.status.idle": "2020-02-19T02:23:32.085475Z", "shell.execute_reply": "2020-02-19T02:23:32.084658Z", "shell.execute_reply.started": "2020-02-19T02:22:41.290320Z"}, "persistent_id": "81712e88-3b96-4f10-a536-a80268bfe805"}
lab_df = lab_df.sort_values('ts')
lab_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:23:32.086987Z", "iopub.status.busy": "2020-02-19T02:23:32.086611Z", "iopub.status.idle": "2020-02-19T02:23:49.556502Z", "shell.execute_reply": "2020-02-19T02:23:49.555419Z", "shell.execute_reply.started": "2020-02-19T02:23:32.086914Z"}, "persistent_id": "ac2b0e4b-d2bd-4eb5-a629-637361a85457"}
lab_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='lab_result', n=5).head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-19T02:23:49.557907Z", "iopub.status.busy": "2020-02-19T02:23:49.557646Z", "iopub.status.idle": "2020-02-19T02:24:54.280715Z", "shell.execute_reply": "2020-02-19T02:24:54.279517Z", "shell.execute_reply.started": "2020-02-19T02:23:49.557863Z"}, "persistent_id": "da5e70d7-0514-4bdb-a5e2-12b6e8a1b197"}
lab_df[(lab_df.patientunitstayid == 3240757) & (lab_df.ts == 162)].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to ___ categories per set of `patientunitstayid` and `ts`. As such, we must join them. But first, we need to normalize the results by the respective sets of exam name and units, so as to avoid mixing different absolute values.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Normalize data

# + {"Collapsed": "false", "cell_type": "markdown"}
# Convert dataframe to Pandas, as the next cells aren't working properly with Modin:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-18T18:49:48.993997Z", "iopub.status.busy": "2020-02-18T18:49:48.993702Z", "iopub.status.idle": "2020-02-18T18:49:52.947587Z", "shell.execute_reply": "2020-02-18T18:49:52.946673Z", "shell.execute_reply.started": "2020-02-18T18:49:48.993943Z"}}
lab_df, pd = du.utils.convert_dataframe(lab_df, to='pandas')

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-18T18:49:52.948969Z", "iopub.status.busy": "2020-02-18T18:49:52.948722Z", "iopub.status.idle": "2020-02-18T18:49:52.954356Z", "shell.execute_reply": "2020-02-18T18:49:52.953463Z", "shell.execute_reply.started": "2020-02-18T18:49:52.948928Z"}}
type(lab_df)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-21T03:04:09.662593Z", "iopub.status.busy": "2020-02-21T03:04:09.662343Z", "iopub.status.idle": "2020-02-21T03:17:19.416462Z", "shell.execute_reply": "2020-02-21T03:17:19.415653Z", "shell.execute_reply.started": "2020-02-21T03:04:09.662547Z"}, "persistent_id": "a4cd949b-e561-485d-bcb6-10fccc343352", "pixiedust": {"displayParams": {}}}
lab_df_norm = du.data_processing.normalize_data(lab_df, columns_to_normalize=False,
                                                columns_to_normalize_categ=[(['labname', 'lab_units'], 'lab_result')],
                                                inplace=True)
lab_df_norm.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs
#
# Even after removing duplicates rows, there are still some that have different information for the same ID and timestamp. We have to concatenate the categorical enumerations.

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-21T03:25:32.896217Z", "iopub.status.busy": "2020-02-21T03:25:32.895919Z", "iopub.status.idle": "2020-02-21T03:25:32.903303Z", "shell.execute_reply": "2020-02-21T03:25:32.902344Z", "shell.execute_reply.started": "2020-02-21T03:25:32.896162Z"}, "persistent_id": "ed86d5a7-eeb3-44c4-9a4e-6dd67af307f2"}
list(set(lab_df_norm.columns) - set(new_cat_embed_feat) - set(['patientunitstayid', 'ts']))

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-21T03:25:32.905157Z", "iopub.status.busy": "2020-02-21T03:25:32.904887Z", "iopub.status.idle": "2020-02-18T18:09:22.840381Z"}, "persistent_id": "589931b8-fe11-439a-8b14-4857c168c023", "pixiedust": {"displayParams": {}}}
lab_df_norm = du.embedding.join_categorical_enum(lab_df_norm, new_cat_embed_feat, inplace=True)
lab_df_norm.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Reconvert dataframe to Modin:

# + {"Collapsed": "false"}
lab_df_norm, pd = du.utils.convert_dataframe(lab_df_norm, to='modin')

# + {"Collapsed": "false"}
type(lab_df_norm)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-18T18:09:22.841192Z", "iopub.status.idle": "2020-02-18T18:09:22.841498Z"}, "persistent_id": "db6b5624-e600-4d90-bc5a-ffa5a876d8dd"}
lab_df_norm.dtypes

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-18T18:09:22.842441Z", "iopub.status.idle": "2020-02-18T18:09:22.842781Z"}, "persistent_id": "954d2c26-4ef4-42ec-b0f4-a73febb5115d"}
lab_df_norm.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='lab_result', n=5).head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-18T18:09:22.843763Z", "iopub.status.idle": "2020-02-18T18:09:22.844076Z"}, "persistent_id": "85536a51-d31a-4b25-aaee-9c9d4ec392f6"}
lab_df[(lab_df.patientunitstayid == 3240757) & (lab_df.ts == 162)].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-18T18:09:22.844914Z", "iopub.status.idle": "2020-02-18T18:09:22.845188Z"}, "persistent_id": "0f255c7d-1d1a-4dd3-94d7-d7fd54e13da0"}
lab_df.columns = du.data_processing.clean_naming(lab_df.columns)
lab_df_norm.columns = du.data_processing.clean_naming(lab_df_norm.columns)
lab_df_norm.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe
#
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-18T18:09:22.845809Z", "iopub.status.idle": "2020-02-18T18:09:22.846162Z"}, "persistent_id": "7c95b423-0fd8-4e65-ac07-a0117f0c36bd"}
lab_df.to_csv(f'{data_path}cleaned/unnormalized/lab.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-18T18:09:22.846850Z", "iopub.status.idle": "2020-02-18T18:09:22.847192Z"}, "persistent_id": "eae5d63f-5635-4fa0-8c42-ff6081336e18"}
lab_df_norm.to_csv(f'{data_path}cleaned/normalized/lab.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-18T18:09:22.848003Z", "iopub.status.idle": "2020-02-18T18:09:22.848313Z"}, "persistent_id": "2cdabf5e-7df3-441b-b8ed-a06c404df27e"}
lab_df_norm.describe().transpose()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-18T18:09:22.849113Z", "iopub.status.idle": "2020-02-18T18:09:22.849429Z"}, "persistent_id": "9255fb38-ba28-4bc2-8f97-7596e8acbc5a"}
lab_df.nlargest(columns='lab_result', n=5)

# + {"Collapsed": "false"}
lab_df = pd.read_csv(f'{data_path}cleaned/normalized/lab.csv')
lab_df.head()

# + {"Collapsed": "false"}
lab_df = lab_df.drop(columns='Unnamed: 0')

# + {"Collapsed": "false"}
lab_df.info()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Replace the dot representation with the semicolon one

# + {"Collapsed": "false"}
for col in du.utils.iterations_loop(['labtypeid', 'labname', 'lab_units']):
    # Remove the '.0' from the end of the strings
    lab_df[col] = lab_df[col].str.replace('.0', '')
    # Replace the dots with semicolons
    lab_df[col] = lab_df[col].str.replace('.', ';')

# + {"Collapsed": "false"}
lab_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after fixing the categories list representation:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-18T18:09:22.846850Z", "iopub.status.idle": "2020-02-18T18:09:22.847192Z"}, "persistent_id": "eae5d63f-5635-4fa0-8c42-ff6081336e18"}
lab_df.to_csv(f'{data_path}cleaned/normalized/lab.csv')

# + {"Collapsed": "false"}

