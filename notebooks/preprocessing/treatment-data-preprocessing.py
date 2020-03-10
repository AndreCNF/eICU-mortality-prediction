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

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": false}
# # Treatment Data Preprocessing
# ---
#
# Reading and preprocessing treatment data of the eICU dataset from MIT with the data from over 139k patients collected in the US.
#
# This notebook addresses the preprocessing of the following eICU tables:
# * admissionDrug
# * infusionDrug
# * medication
# * treatment
# * intakeOutput

# + [markdown] {"Collapsed": "false", "colab_type": "text", "id": "KOdmFzXqF7nq", "toc-hr-collapsed": true}
# ## Importing the necessary packages

# + {"Collapsed": "false", "colab": {}, "colab_type": "code", "execution_event_id": "deb57b39-6a79-4b3a-95ed-02f8089ff593", "id": "G5RrWE9R_Nkl", "last_executed_text": "import os                                  # os handles directory/workspace changes\nimport numpy as np                         # NumPy to handle numeric and NaN operations\nimport yaml                                # Save and load YAML files", "persistent_id": "522745b5-b5bf-479f-b697-5c7e9e12fc33", "execution": {"iopub.status.busy": "2020-03-09T16:35:00.843458Z", "iopub.execute_input": "2020-03-09T16:35:00.843852Z", "iopub.status.idle": "2020-03-09T16:35:01.054945Z", "shell.execute_reply.started": "2020-03-09T16:35:00.843784Z", "shell.execute_reply": "2020-03-09T16:35:01.054346Z"}}
import os                                  # os handles directory/workspace changes
import numpy as np                         # NumPy to handle numeric and NaN operations
import yaml                                # Save and load YAML files

# + {"Collapsed": "false", "execution_event_id": "fa33a2f7-7127-49c6-bbe9-f89555b1f2be", "last_executed_text": "# Debugging packages\nimport pixiedust                           # Debugging in Jupyter Notebook cells", "persistent_id": "02accdbf-be7e-415c-ba11-165906e66c50", "execution": {"iopub.status.busy": "2020-03-09T16:35:01.123814Z", "iopub.execute_input": "2020-03-09T16:35:01.124069Z", "iopub.status.idle": "2020-03-09T16:35:05.726479Z", "shell.execute_reply.started": "2020-03-09T16:35:01.124041Z", "shell.execute_reply": "2020-03-09T16:35:05.725772Z"}}
# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# + {"Collapsed": "false", "execution_event_id": "baeb346a-1c34-42d1-a501-7ae37369255e", "last_executed_text": "# Change to parent directory (presumably \"Documents\")\nos.chdir(\"../../..\")\n\n# Path to the CSV dataset files\ndata_path = 'Documents/Datasets/Thesis/eICU/uncompressed/'\n\n# Path to the code files\nproject_path = 'Documents/GitHub/eICU-mortality-prediction/'", "persistent_id": "a1f6ee7f-36d4-489d-b2dd-ec2a38f15d11", "execution": {"iopub.status.busy": "2020-03-09T16:35:05.728469Z", "iopub.execute_input": "2020-03-09T16:35:05.728770Z", "iopub.status.idle": "2020-03-09T16:35:05.736189Z", "shell.execute_reply.started": "2020-03-09T16:35:05.728729Z", "shell.execute_reply": "2020-03-09T16:35:05.735097Z"}}
# Change to parent directory (presumably "Documents")
os.chdir("../../../..")
# Path to the CSV dataset files
data_path = 'data/eICU/uncompressed/'
# Path to the code files
project_path = 'code/eICU-mortality-prediction/'

# + {"Collapsed": "false", "execution_event_id": "82ef68be-443a-4bb8-8abd-7457a7005b4d", "last_executed_text": "import modin.pandas as pd                  # Optimized distributed version of Pandas\nimport data_utils as du                    # Data science and machine learning relevant methods", "persistent_id": "c0c2e356-d4f4-4a9d-bec2-88bdf9eb6a38", "execution": {"iopub.status.busy": "2020-03-09T16:35:05.738126Z", "iopub.execute_input": "2020-03-09T16:35:05.738351Z", "iopub.status.idle": "2020-03-09T16:35:09.084739Z", "shell.execute_reply.started": "2020-03-09T16:35:05.738325Z", "shell.execute_reply": "2020-03-09T16:35:09.083209Z"}}
# import modin.pandas as pd                  # Optimized distributed version of Pandas
import pandas as pd
import data_utils as du                    # Data science and machine learning relevant methods

# + [markdown] {"Collapsed": "false"}
# Set the random seed for reproducibility

# + {"Collapsed": "false", "execution_event_id": "29ab85ce-b7fd-4c5a-a110-5841e741c369", "last_executed_text": "du.set_random_seed(42)", "persistent_id": "39b552cd-6948-4ec8-ac04-42f850c1e05a", "execution": {"iopub.status.busy": "2020-03-09T16:35:09.088126Z", "iopub.execute_input": "2020-03-09T16:35:09.088720Z", "iopub.status.idle": "2020-03-09T16:35:09.095409Z", "shell.execute_reply.started": "2020-03-09T16:35:09.088663Z", "shell.execute_reply": "2020-03-09T16:35:09.094473Z"}}
du.set_random_seed(42)

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": true}
# ## Infusion drug data

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": true}
# ### Initialize variables

# + {"Collapsed": "false", "persistent_id": "754a96f8-d389-4968-8c13-52e5e9d0bf82", "execution": {"iopub.status.busy": "2020-02-17T02:24:13.243501Z", "iopub.execute_input": "2020-02-17T02:24:13.243729Z", "iopub.status.idle": "2020-02-17T02:24:13.247974Z", "shell.execute_reply.started": "2020-02-17T02:24:13.243694Z", "shell.execute_reply": "2020-02-17T02:24:13.246579Z"}}
# List of categorical features
cat_feat = []
# Dictionary of the one hot encoded columns originary from each categorical feature, that will be embedded
cat_feat_ohe = dict()

# + [markdown] {"Collapsed": "false"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "36c79435-0530-4459-8832-cb924012b62e"}
inf_drug_df = pd.read_csv(f'{data_path}original/infusionDrug.csv')
inf_drug_df.head()

# + {"Collapsed": "false", "persistent_id": "cf1205df-b87e-42cf-8740-f5663955860b"}
len(inf_drug_df)

# + {"Collapsed": "false", "persistent_id": "fe500a2c-f9b0-41ff-a833-b61de0e87728"}
inf_drug_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "08b8557e-0837-45a2-a462-3e05528756f1"}
inf_drug_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "5e14aba4-9f24-4ecb-a406-5d1590c4f538"}
inf_drug_df.columns

# + {"Collapsed": "false", "persistent_id": "77be4fb5-821f-4ab0-b1ee-3e4288565439"}
inf_drug_df.dtypes

# + [markdown] {"Collapsed": "false"}
# ### Check for missing values

# + {"Collapsed": "false", "persistent_id": "7b3530da-0b79-4bed-935b-3a48af63d92e", "pixiedust": {"displayParams": {}}}
du.search_explore.dataframe_missing_values(inf_drug_df)

# + [markdown] {"Collapsed": "false"}
# ### Remove unneeded features
#
# Besides removing the row ID `infusiondrugid`, I'm also removing `infusionrate`, `volumeoffluid` and `drugamount` as they seem redundant with `drugrate` although with a lot more missing values.

# + {"Collapsed": "false", "persistent_id": "90324c6d-0d62-432c-be74-b8f5cf41de65"}
inf_drug_df = inf_drug_df.drop(['infusiondrugid', 'infusionrate', 'volumeoffluid',
                              'drugamount', 'patientweight'], axis=1)
inf_drug_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Remove string drug rate values

# + {"Collapsed": "false", "persistent_id": "a786da73-2e65-437b-a215-7f2ed3964df5"}
inf_drug_df[inf_drug_df.drugrate.map(du.utils.is_definitely_string)].head()

# + {"Collapsed": "false", "persistent_id": "4f985943-20dd-49a3-8df4-54bacc5a2863"}
inf_drug_df[inf_drug_df.drugrate.map(du.utils.is_definitely_string)].drugrate.value_counts()

# + {"Collapsed": "false", "persistent_id": "0182647e-311c-471f-91d8-d91fcc0ed92a"}
inf_drug_df.drugrate = inf_drug_df.drugrate.map(lambda x: np.nan if du.utils.is_definitely_string(x) else x)
inf_drug_df.head()

# + {"Collapsed": "false", "persistent_id": "19f22442-547a-48af-8c9a-a0e89d2264b3"}
inf_drug_df.patientunitstayid = inf_drug_df.patientunitstayid.astype(int)
inf_drug_df.infusionoffset = inf_drug_df.infusionoffset.astype(int)
inf_drug_df.drugname = inf_drug_df.drugname.astype(str)
inf_drug_df.drugrate = inf_drug_df.drugrate.astype(float)
inf_drug_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Rename columns

# + {"Collapsed": "false", "persistent_id": "9a27b7be-7a8a-435b-acdb-9407a325ac53"}
inf_drug_df = inf_drug_df.rename(columns={'drugname': 'infusion_drugname',
                                          'drugrate': 'infusion_drugrate'})
inf_drug_df.head()

# + [markdown] {"Collapsed": "false"}
# We can see that there are up to 17 categories per set of `patientunitstayid` and `ts`. As such, we must join them. But first, as we shouldn't mix absolute values of drug rates from different drugs, we better normalize it first.

# + [markdown] {"Collapsed": "false"}
# ### Normalize data

# + [markdown] {"Collapsed": "false"}
# Update list of categorical features:

# + {"Collapsed": "false", "persistent_id": "d080567b-0609-4069-aecc-293e98c3277b"}
cat_feat = ['infusion_drugname']

# + [markdown] {"Collapsed": "false"}
# Filter just to the most common categories:

# + {"Collapsed": "false"}
for col in cat_feat:
    most_common_cat = list(inf_drug_df[col].value_counts().nlargest(500).index)
    inf_drug_df = inf_drug_df[inf_drug_df[col].isin(most_common_cat)]

# + {"Collapsed": "false"}
inf_drug_df.infusion_drugrate = inf_drug_df.infusion_drugrate.astype(float)

# + {"Collapsed": "false", "persistent_id": "5d512225-ad7e-40b4-a091-b18df3f38c4c", "pixiedust": {"displayParams": {}}}
inf_drug_df, mean, std = du.data_processing.normalize_data(inf_drug_df, columns_to_normalize=False,
                                                           columns_to_normalize_categ=[('infusion_drugname', 'infusion_drugrate')],
                                                           get_stats=True, inplace=True)
inf_drug_df.head()

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
stream = open(f'{data_path}/cleaned/infusionDrug_norm_stats.yaml', 'w')
yaml.dump(norm_stats, stream, default_flow_style=False)

# + [markdown] {"Collapsed": "false"}
# Prevent infinite drug rate values:

# + {"Collapsed": "false"}
inf_drug_df = inf_drug_df.replace(to_replace=np.inf, value=0)

# + {"Collapsed": "false"}
inf_drug_df.drugrate.max()

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": false}
# ### Discretize categorical features
#
# Convert binary categorical features into one hot encode columns, which can later be embedded or used as is.

# + [markdown] {"Collapsed": "false"}
# #### One hot encode features

# + {"Collapsed": "false", "persistent_id": "82bf9aa5-c5de-433a-97d3-01d6af81e2e4"}
inf_drug_df[cat_feat].head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-09T16:36:38.933807Z", "iopub.execute_input": "2020-03-09T16:36:38.934065Z", "iopub.status.idle": "2020-03-09T16:36:38.937828Z", "shell.execute_reply.started": "2020-03-09T16:36:38.934035Z", "shell.execute_reply": "2020-03-09T16:36:38.936941Z"}}
old_columns = inf_drug_df.columns
# -

inf_drug_df.drugname.nunique()

# + [markdown] {"Collapsed": "false"}
# Apply one hot encoding:

# + {"Collapsed": "false", "persistent_id": "8b0f065b-fb2b-4330-b155-d86769ac1635", "pixiedust": {"displayParams": {}}}
inf_drug_df = du.data_processing.one_hot_encoding_dataframe(inf_drug_df, columns=cat_feat, join_rows=False)
inf_drug_df

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-09T16:37:27.392359Z", "iopub.status.idle": "2020-03-09T16:37:27.399976Z", "iopub.execute_input": "2020-03-09T16:37:27.392616Z", "shell.execute_reply.started": "2020-03-09T16:37:27.392582Z", "shell.execute_reply": "2020-03-09T16:37:27.399076Z"}}
new_columns = set(inf_drug_df.columns) - set(old_columns)

# + {"Collapsed": "false", "persistent_id": "b0e7a06d-f451-470f-8cf5-d154f76e83a2"}
inf_drug_df.dtypes

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

# + {"Collapsed": "false", "persistent_id": "b40f8861-dae3-49ca-8649-d937ba3cfcf0"}
stream = open(f'{data_path}/cleaned/cat_feat_ohe_inf_drug.yaml', 'w')
yaml.dump(cat_feat_ohe, stream, default_flow_style=False)

# + [markdown] {"Collapsed": "false"}
# ### Create the timestamp feature and sort

# + [markdown] {"Collapsed": "false"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "b1ea5e2a-d7eb-41e6-9cad-4dcbf5997ca7"}
inf_drug_df = inf_drug_df.rename(columns={'infusionoffset': 'ts'})
inf_drug_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "edfbcec8-4ca6-430a-8caf-940a115f6cac"}
len(inf_drug_df)

# + {"Collapsed": "false", "persistent_id": "e6e2fef9-878b-449c-bf08-d42dc6e4f9da"}
inf_drug_df = inf_drug_df.drop_duplicates()
inf_drug_df.head()

# + {"Collapsed": "false", "persistent_id": "1cd6a490-f63f-458f-a274-30170c70fc66"}
len(inf_drug_df)

# + [markdown] {"Collapsed": "false"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "b3eb5c69-d034-45d7-ab48-0217169a48fb"}
inf_drug_df = inf_drug_df.sort_values('ts')
inf_drug_df.head(6)

# + [markdown] {"Collapsed": "false"}
# Convert dataframe to Pandas, as the next cells aren't working properly with Modin:

# + {"Collapsed": "false"}
inf_drug_df.dtypes

# + {"Collapsed": "false"}
inf_drug_df, pd = du.utils.convert_dataframe(inf_drug_df, to='pandas')

# + {"Collapsed": "false"}
type(inf_drug_df)

# + {"Collapsed": "false"}
inf_drug_df.dtypes

# + [markdown] {"Collapsed": "false"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "049a3fd1-0ae4-454e-a5b5-5ce8fa94d3e1"}
inf_drug_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='infusion_drugname_', n=5).head()

# + {"Collapsed": "false", "persistent_id": "29b5843e-679e-4c4c-941f-e39a92965d1f"}
inf_drug_df[inf_drug_df.patientunitstayid == 1785711].head(20)

# + [markdown] {"Collapsed": "false"}
# ### Join rows that have the same IDs

# + {"Collapsed": "false", "persistent_id": "589931b8-fe11-439a-8b14-4857c168c023", "pixiedust": {"displayParams": {}}}
inf_drug_df = du.embedding.join_repeated_rows(inf_drug_df, inplace=True)
inf_drug_df.head()

# + [markdown] {"Collapsed": "false"}
# Reconvert dataframe to Modin:

# + {"Collapsed": "false"}
inf_drug_df, pd = du.utils.convert_dataframe(inf_drug_df, to='modin')

# + {"Collapsed": "false"}
type(inf_drug_df)

# + {"Collapsed": "false", "persistent_id": "e8c3544d-cf75-45d1-8af2-e6feb8a8d587"}
inf_drug_df.dtypes

# + {"Collapsed": "false", "persistent_id": "bbc89593-5b2b-42b5-83f2-8bab9c1d5ef6"}
inf_drug_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='infusion_drugname_', n=5).head()

# + {"Collapsed": "false", "persistent_id": "b64df6bc-254f-40b8-97cf-15737ce27db1"}
inf_drug_df[inf_drug_df.patientunitstayid == 1785711].head(20)

# + [markdown] {"Collapsed": "false"}
# Comparing the output from the two previous cells with what we had before the `join_repeated_rows` method, we can see that all rows with duplicate IDs have been successfully joined.

# + [markdown] {"Collapsed": "false"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "258e5bdf-3880-4f14-8a03-7394c55c2c2e"}
inf_drug_df.columns = du.data_processing.clean_naming(inf_drug_df.columns)
inf_drug_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Save the dataframe

# + [markdown] {"Collapsed": "false"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "767d2077-112b-483a-bd2c-4b578d61ba1a"}
# inf_drug_df.to_csv(f'{data_path}cleaned/unnormalized/ohe/infusionDrug.csv')

# + [markdown] {"Collapsed": "false"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "615e3df8-d467-4042-801f-a296a528b77a"}
inf_drug_df.to_csv(f'{data_path}cleaned/normalized/ohe/infusionDrug.csv')

# + [markdown] {"Collapsed": "false"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "3fe6821a-5324-4b36-94cd-7d8073c5262f"}
inf_drug_df.describe().transpose()

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": true}
# ## Admission drug data

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": true}
# ### Initialize variables

# + {"Collapsed": "false", "persistent_id": "754a96f8-d389-4968-8c13-52e5e9d0bf82", "execution": {"iopub.status.busy": "2020-03-09T16:35:09.097413Z", "iopub.execute_input": "2020-03-09T16:35:09.097647Z", "iopub.status.idle": "2020-03-09T16:35:09.101351Z", "shell.execute_reply.started": "2020-03-09T16:35:09.097620Z", "shell.execute_reply": "2020-03-09T16:35:09.100590Z"}}
# List of categorical features
cat_feat = []
# Dictionary of the one hot encoded columns originary from each categorical feature, that will be embedded
cat_feat_ohe = dict()

# + [markdown] {"Collapsed": "false"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "d75bab34-d386-49fb-b6b7-273035226f86", "execution": {"iopub.status.busy": "2020-03-09T16:35:10.239211Z", "iopub.execute_input": "2020-03-09T16:35:10.239514Z", "iopub.status.idle": "2020-03-09T16:35:12.532761Z", "shell.execute_reply.started": "2020-03-09T16:35:10.239484Z", "shell.execute_reply": "2020-03-09T16:35:12.531958Z"}}
adms_drug_df = pd.read_csv(f'{data_path}original/admissionDrug.csv')
adms_drug_df.head()

# + {"Collapsed": "false", "persistent_id": "0b16b7ea-653f-45d6-a923-ea5538cfa1a8"}
len(adms_drug_df)

# + {"Collapsed": "false", "persistent_id": "65855dd5-c78f-4596-8b10-4ad9ca706403"}
adms_drug_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# There's not much admission drug data (only around 20% of the unit stays have this data). However, it might be useful, considering also that it complements the medication table.

# + [markdown] {"Collapsed": "false"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "cdca1bbc-5fe1-4823-8828-f2adcc14d9b5"}
adms_drug_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "7d6bb51f-8e11-44af-8427-f50a90117bd4"}
adms_drug_df.columns

# + {"Collapsed": "false", "persistent_id": "7d0197fe-9d63-4ffb-b0ce-8c15f5231b52"}
adms_drug_df.dtypes

# + {"execution": {"iopub.status.busy": "2020-03-09T16:35:19.568638Z", "iopub.execute_input": "2020-03-09T16:35:19.568926Z", "iopub.status.idle": "2020-03-09T16:35:19.818847Z", "shell.execute_reply.started": "2020-03-09T16:35:19.568898Z", "shell.execute_reply": "2020-03-09T16:35:19.817817Z"}, "Collapsed": "false"}
adms_drug_df.info()

# + [markdown] {"Collapsed": "false"}
# ### Check for missing values

# + {"Collapsed": "false", "persistent_id": "e6b1e527-f089-4418-98cc-497cc63f2454", "pixiedust": {"displayParams": {}}}
du.search_explore.dataframe_missing_values(adms_drug_df)

# + [markdown] {"Collapsed": "false"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "a5e3fd48-5fa3-4d9b-aa52-e1a29d7c6523"}
adms_drug_df.drugname.value_counts()

# + {"Collapsed": "false"}
adms_drug_df.drugname.nunique()

# + {"Collapsed": "false", "persistent_id": "0c6824dc-8949-4d55-96dc-932cd84a63e6"}
adms_drug_df.drughiclseqno.value_counts()

# + {"Collapsed": "false"}
adms_drug_df.drughiclseqno.nunique()

# + {"Collapsed": "false", "persistent_id": "67466a3f-3bdc-4f91-b83a-7746b15345b4"}
adms_drug_df.drugnotetype.value_counts()

# + {"Collapsed": "false", "persistent_id": "c11d4f43-468c-4175-a455-0473f5ad2807"}
adms_drug_df.drugdosage.value_counts()

# + {"Collapsed": "false", "persistent_id": "d77aa18d-8ce7-4e2b-ac94-25e5074bf371"}
adms_drug_df.drugunit.value_counts()

# + {"Collapsed": "false", "persistent_id": "12d3606e-1dec-4173-aa42-1c59370ea262"}
adms_drug_df.drugadmitfrequency.value_counts()

# + {"Collapsed": "false", "persistent_id": "09e4ec8d-8ebc-4503-a2ef-e1f20c83f3cb"}
adms_drug_df[adms_drug_df.drugdosage == 0].head(20)

# + {"Collapsed": "false", "persistent_id": "5d29249a-b8ba-4f8c-81a7-754418232261"}
adms_drug_df[adms_drug_df.drugdosage == 0].drugunit.value_counts()

# + {"Collapsed": "false", "persistent_id": "9c1a6253-713e-4a1f-be38-add4f687973d"}
adms_drug_df[adms_drug_df.drugdosage == 0].drugadmitfrequency.value_counts()

# + {"Collapsed": "false", "persistent_id": "d2eba66b-33c6-4427-a88b-c4ec28174653"}
adms_drug_df[adms_drug_df.drugunit == ' '].drugdosage.value_counts()

# + [markdown] {"Collapsed": "false"}
# Oddly, `drugunit` and `drugadmitfrequency` have several blank values. At the same time, when this happens, `drugdosage` tends to be 0 (which is also an unrealistic value). Considering that no NaNs are reported, these blanks and zeros probably represent missing values.

# + [markdown] {"Collapsed": "false"}
# Besides removing irrelevant or hospital staff related data (e.g. `usertype`), I'm also removing the `drugname` column, which is redundant with the codes `drughiclseqno`, while also being brand dependant.

# + {"Collapsed": "false", "persistent_id": "69129157-8c81-42bc-bc88-00df3249bc86", "execution": {"iopub.status.busy": "2020-03-09T16:36:18.606537Z", "iopub.execute_input": "2020-03-09T16:36:18.606789Z", "iopub.status.idle": "2020-03-09T16:36:18.660609Z", "shell.execute_reply.started": "2020-03-09T16:36:18.606765Z", "shell.execute_reply": "2020-03-09T16:36:18.659898Z"}}
adms_drug_df = adms_drug_df[['patientunitstayid', 'drugoffset', 'drugdosage',
                             'drugunit', 'drugadmitfrequency', 'drughiclseqno']]
adms_drug_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Fix missing values representation
#
# Replace blank and unrealistic zero values with NaNs.

# + [markdown] {"Collapsed": "false"}
# Convert dataframe to Pandas, as the next cells aren't working properly with Modin:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-09T16:36:30.341328Z", "iopub.execute_input": "2020-03-09T16:36:30.341672Z", "iopub.status.idle": "2020-03-09T16:36:30.690720Z", "shell.execute_reply.started": "2020-03-09T16:36:30.341629Z", "shell.execute_reply": "2020-03-09T16:36:30.690131Z"}}
adms_drug_df, pd = du.utils.convert_dataframe(adms_drug_df, to='pandas')

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-09T05:25:03.330537Z", "iopub.execute_input": "2020-03-09T05:25:03.330748Z", "iopub.status.idle": "2020-03-09T05:25:03.335984Z", "shell.execute_reply.started": "2020-03-09T05:25:03.330725Z", "shell.execute_reply": "2020-03-09T05:25:03.335446Z"}}
type(adms_drug_df)

# + {"Collapsed": "false", "persistent_id": "23d0168e-e5f3-4630-954e-3becc23307ae", "execution": {"iopub.status.busy": "2020-03-09T16:36:31.729380Z", "iopub.execute_input": "2020-03-09T16:36:31.729676Z", "iopub.status.idle": "2020-03-09T16:36:31.940311Z", "shell.execute_reply.started": "2020-03-09T16:36:31.729639Z", "shell.execute_reply": "2020-03-09T16:36:31.939669Z"}}
adms_drug_df.drugdosage = adms_drug_df.drugdosage.replace(to_replace=0, value=np.nan)
adms_drug_df.drugunit = adms_drug_df.drugunit.replace(to_replace=' ', value=np.nan)
adms_drug_df.drugadmitfrequency = adms_drug_df.drugadmitfrequency.replace(to_replace=' ', value=np.nan)
adms_drug_df.head()

# + {"Collapsed": "false", "persistent_id": "10eab8e7-1ef2-46cb-8d41-7953dd66ef15", "pixiedust": {"displayParams": {}}, "execution": {"iopub.status.busy": "2020-03-09T16:43:40.487379Z", "iopub.execute_input": "2020-03-09T16:43:40.489541Z", "iopub.status.idle": "2020-03-09T16:43:50.667139Z", "shell.execute_reply.started": "2020-03-09T16:43:40.489450Z", "shell.execute_reply": "2020-03-09T16:43:50.665719Z"}}
du.search_explore.dataframe_missing_values(adms_drug_df)

# + [markdown] {"Collapsed": "false"}
# ### Normalize data

# + [markdown] {"Collapsed": "false"}
# Update list of categorical features:

# + {"Collapsed": "false", "persistent_id": "96f4f32e-9442-4705-acbc-37060bfba492", "execution": {"iopub.status.busy": "2020-03-09T16:36:38.060782Z", "iopub.execute_input": "2020-03-09T16:36:38.061222Z", "iopub.status.idle": "2020-03-09T16:36:38.065901Z", "shell.execute_reply.started": "2020-03-09T16:36:38.061170Z", "shell.execute_reply": "2020-03-09T16:36:38.064846Z"}}
cat_feat = ['drugunit', 'drugadmitfrequency', 'drughiclseqno']

# + [markdown] {"Collapsed": "false"}
# Filter just to the most common categories:

# + {"Collapsed": "false"}
for col in cat_feat:
    most_common_cat = list(adms_drug_df[col].value_counts().nlargest(500).index)
    adms_drug_df = adms_drug_df[adms_drug_df[col].isin(most_common_cat)]

# + {"Collapsed": "false", "persistent_id": "417dd68c-7c54-4eaf-856c-9f6a44ee1d26", "pixiedust": {"displayParams": {}}, "execution": {"iopub.status.busy": "2020-03-09T18:09:29.409923Z", "iopub.status.idle": "2020-03-09T16:39:57.582416Z", "iopub.execute_input": "2020-03-09T18:09:29.410401Z"}}
adms_drug_df_norm, mean, std = du.data_processing.normalize_data(adms_drug_df, columns_to_normalize=False,
                                                                 columns_to_normalize_categ=[(['drughiclseqno', 'drugunit'], 'drugdosage')],
                                                                 get_stats=True, inplace=True)
adms_drug_df_norm.head()

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
stream = open(f'{data_path}/cleaned/admissionDrug_norm_stats.yaml', 'w')
yaml.dump(norm_stats, stream, default_flow_style=False)

# + {"Collapsed": "false", "persistent_id": "d8f2c0a4-ba81-4958-90b7-7b5165ff1363", "execution": {"iopub.status.busy": "2020-03-09T16:39:57.583568Z", "iopub.status.idle": "2020-03-09T16:39:57.583851Z"}}
adms_drug_df_norm = adms_drug_df_norm.sort_values('ts')
adms_drug_df_norm.head()

# + [markdown] {"Collapsed": "false"}
# Prevent infinite values:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-09T16:39:57.585474Z", "iopub.status.idle": "2020-03-09T16:39:57.585858Z"}}
adms_drug_df_norm = adms_drug_df_norm.replace(to_replace=np.inf, value=0)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-09T16:39:57.587757Z", "iopub.status.idle": "2020-03-09T16:39:57.588271Z"}}
adms_drug_df_norm = adms_drug_df_norm.replace(to_replace=-np.inf, value=0)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-09T16:39:57.590669Z", "iopub.status.idle": "2020-03-09T16:39:57.591218Z"}}
adms_drug_df_norm.drugdosage.max()

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": false}
# ### Discretize categorical features
#
# Convert binary categorical features into one hot encode columns, which can later be embedded or used as is.

# + [markdown] {"Collapsed": "false"}
# #### One hot encode features

# + {"Collapsed": "false", "persistent_id": "a33e788b-d6ba-4d79-9fee-54a32959d453", "execution": {"iopub.status.busy": "2020-03-09T16:36:38.520331Z", "iopub.execute_input": "2020-03-09T16:36:38.520674Z", "iopub.status.idle": "2020-03-09T16:36:38.547357Z", "shell.execute_reply.started": "2020-03-09T16:36:38.520605Z", "shell.execute_reply": "2020-03-09T16:36:38.546646Z"}}
adms_drug_df[cat_feat].head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-09T16:36:38.933807Z", "iopub.execute_input": "2020-03-09T16:36:38.934065Z", "iopub.status.idle": "2020-03-09T16:36:38.937828Z", "shell.execute_reply.started": "2020-03-09T16:36:38.934035Z", "shell.execute_reply": "2020-03-09T16:36:38.936941Z"}}
old_columns = adms_drug_df.columns

# + [markdown] {"Collapsed": "false"}
# Apply one hot encoding:

# + {"Collapsed": "false", "persistent_id": "8f69d437-6fec-4e7e-ae76-226b742b03a7", "pixiedust": {"displayParams": {}}, "execution": {"iopub.status.busy": "2020-03-09T16:36:42.370906Z", "iopub.execute_input": "2020-03-09T16:36:42.371306Z", "iopub.status.idle": "2020-03-09T16:37:27.388470Z", "shell.execute_reply.started": "2020-03-09T16:36:42.371258Z", "shell.execute_reply": "2020-03-09T16:37:27.387133Z"}}
adms_drug_df = du.data_processing.one_hot_encoding_dataframe(adms_drug_df, columns=cat_feat, join_rows=False)
adms_drug_df

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-09T16:37:27.392359Z", "iopub.status.idle": "2020-03-09T16:37:27.399976Z", "iopub.execute_input": "2020-03-09T16:37:27.392616Z", "shell.execute_reply.started": "2020-03-09T16:37:27.392582Z", "shell.execute_reply": "2020-03-09T16:37:27.399076Z"}}
new_columns = set(adms_drug_df.columns) - set(old_columns)

# + {"Collapsed": "false", "persistent_id": "b54a0213-dfda-46d3-aef5-a7a5ed8c2810", "execution": {"iopub.status.busy": "2020-03-09T05:26:40.696643Z", "iopub.status.idle": "2020-03-09T05:26:40.707182Z", "iopub.execute_input": "2020-03-09T05:26:40.696987Z", "shell.execute_reply.started": "2020-03-09T05:26:40.696953Z", "shell.execute_reply": "2020-03-09T05:26:40.706286Z"}}
adms_drug_df.dtypes

# + {"execution": {"iopub.status.busy": "2020-03-09T16:37:34.790942Z", "iopub.execute_input": "2020-03-09T16:37:34.791840Z", "iopub.status.idle": "2020-03-09T16:37:34.943668Z", "shell.execute_reply.started": "2020-03-09T16:37:34.791806Z", "shell.execute_reply": "2020-03-09T16:37:34.942531Z"}, "Collapsed": "false"}
adms_drug_df.patientunitstayid = adms_drug_df.patientunitstayid.astype('uint')
adms_drug_df.drugoffset = adms_drug_df.drugoffset.astype('int')

# + {"execution": {"iopub.status.busy": "2020-03-09T16:37:34.945240Z", "iopub.execute_input": "2020-03-09T16:37:34.945449Z", "iopub.status.idle": "2020-03-09T16:37:35.155724Z", "shell.execute_reply.started": "2020-03-09T16:37:34.945424Z", "shell.execute_reply": "2020-03-09T16:37:35.154732Z"}, "Collapsed": "false"}
adms_drug_df.info()

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

# + {"Collapsed": "false", "persistent_id": "fb54b948-abf1-412d-9fb3-4edd22500e97", "execution": {"iopub.status.busy": "2020-03-09T16:37:35.194288Z", "iopub.execute_input": "2020-03-09T16:37:35.194644Z", "iopub.status.idle": "2020-03-09T16:37:35.338301Z", "shell.execute_reply.started": "2020-03-09T16:37:35.194590Z", "shell.execute_reply": "2020-03-09T16:37:35.336950Z"}}
stream = open(f'{data_path}/cleaned/cat_feat_ohe_adms_drug.yaml', 'w')
yaml.dump(cat_feat_ohe, stream, default_flow_style=False)

# + [markdown] {"Collapsed": "false"}
# ### Create the timestamp feature and sort

# + [markdown] {"Collapsed": "false"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "ceab2ec2-9b2b-4439-9c3c-7674dd5b2445", "execution": {"iopub.status.busy": "2020-03-09T16:37:35.340834Z", "iopub.execute_input": "2020-03-09T16:37:35.341712Z", "iopub.status.idle": "2020-03-09T16:37:41.652459Z", "shell.execute_reply.started": "2020-03-09T16:37:35.341681Z", "shell.execute_reply": "2020-03-09T16:37:41.651090Z"}}
adms_drug_df = adms_drug_df.rename(columns={'drugoffset': 'ts'})
adms_drug_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "387be9a1-51bf-43fd-9a7c-df32b5f6bfc6", "execution": {"iopub.status.busy": "2020-03-09T16:37:41.655773Z", "iopub.execute_input": "2020-03-09T16:37:41.656129Z", "iopub.status.idle": "2020-03-09T16:37:41.664141Z", "shell.execute_reply.started": "2020-03-09T16:37:41.656090Z", "shell.execute_reply": "2020-03-09T16:37:41.663580Z"}}
len(adms_drug_df)

# + {"Collapsed": "false", "persistent_id": "a7fb23bc-4735-4f8a-92d0-f7e359653e5f", "execution": {"iopub.status.busy": "2020-03-09T16:37:41.665826Z", "iopub.execute_input": "2020-03-09T16:37:41.666015Z", "iopub.status.idle": "2020-03-09T16:39:13.907282Z", "shell.execute_reply.started": "2020-03-09T16:37:41.665992Z", "shell.execute_reply": "2020-03-09T16:39:13.904458Z"}}
adms_drug_df = adms_drug_df.drop_duplicates()
adms_drug_df.head()

# + {"Collapsed": "false", "persistent_id": "abf09d5e-b24e-46cd-968b-4a1051ee8504", "execution": {"iopub.status.busy": "2020-03-09T16:39:13.914636Z", "iopub.execute_input": "2020-03-09T16:39:13.915160Z", "iopub.status.idle": "2020-03-09T16:39:13.922209Z", "shell.execute_reply.started": "2020-03-09T16:39:13.915120Z", "shell.execute_reply": "2020-03-09T16:39:13.921516Z"}}
len(adms_drug_df)

# + [markdown] {"Collapsed": "false"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "d35d4953-51aa-46ec-8107-1d4d7f3651a8", "execution": {"iopub.status.busy": "2020-03-09T16:39:13.924732Z", "iopub.execute_input": "2020-03-09T16:39:13.924949Z", "iopub.status.idle": "2020-03-09T16:39:26.038713Z", "shell.execute_reply.started": "2020-03-09T16:39:13.924922Z", "shell.execute_reply": "2020-03-09T16:39:26.037049Z"}}
adms_drug_df = adms_drug_df.sort_values('ts')
adms_drug_df.head()

# + [markdown] {"Collapsed": "false"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "b656a52b-271c-42fd-b8ff-2c4c01a7d2dc", "execution": {"iopub.status.busy": "2020-03-09T16:44:27.829641Z", "iopub.execute_input": "2020-03-09T16:44:27.830184Z", "iopub.status.idle": "2020-03-09T16:45:01.496782Z", "shell.execute_reply.started": "2020-03-09T16:44:27.830120Z", "shell.execute_reply": "2020-03-09T16:45:01.495338Z"}}
adms_drug_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='drughiclseqno_37589', n=5).head()

# + {"Collapsed": "false", "persistent_id": "84c0cc0a-72eb-4fab-93e3-4c9cb83b4fc4", "execution": {"iopub.status.busy": "2020-03-09T16:45:01.500522Z", "iopub.status.idle": "2020-03-09T16:45:01.621409Z", "iopub.execute_input": "2020-03-09T16:45:01.501320Z", "shell.execute_reply.started": "2020-03-09T16:45:01.501284Z", "shell.execute_reply": "2020-03-09T16:45:01.620177Z"}}
adms_drug_df[adms_drug_df.patientunitstayid == 2346930].head(10)

# + [markdown] {"Collapsed": "false"}
# We can see that there are up to 48 categories per set of `patientunitstayid` and `ts`. As such, we must join them. But first, we need to normalize the dosage by the respective sets of drug code and units, so as to avoid mixing different absolute values.

# + [markdown] {"Collapsed": "false"}
# ### Join rows that have the same IDs

# + [markdown] {"Collapsed": "false"}
# Even after removing duplicates rows, there are still some that have different information for the same ID and timestamp. We have to concatenate the categorical enumerations.

# + {"Collapsed": "false", "persistent_id": "589931b8-fe11-439a-8b14-4857c168c023", "pixiedust": {"displayParams": {}}, "execution": {"iopub.status.busy": "2020-03-09T16:39:57.592923Z", "iopub.status.idle": "2020-03-09T16:39:57.593442Z"}}
adms_drug_df_norm = du.embedding.join_repeated_rows(adms_drug_df_norm, inplace=True)
adms_drug_df_norm.head()

# + [markdown] {"Collapsed": "false"}
# Reconvert dataframe to Modin:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-09T16:39:57.595770Z", "iopub.status.idle": "2020-03-09T16:39:57.596200Z"}}
adms_drug_df_norm, pd = du.utils.convert_dataframe(adms_drug_df_norm, to='modin')

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-09T16:39:57.597658Z", "iopub.status.idle": "2020-03-09T16:39:57.598035Z"}}
type(adms_drug_df_norm)

# + {"Collapsed": "false", "persistent_id": "004138b3-74c4-4ca8-b8b4-3409648e00d0", "execution": {"iopub.status.busy": "2020-03-09T16:39:57.599006Z", "iopub.status.idle": "2020-03-09T16:39:57.599414Z"}}
adms_drug_df_norm.dtypes

# + {"Collapsed": "false", "persistent_id": "efd11709-c95e-4245-a662-188556d66680", "execution": {"iopub.status.busy": "2020-03-09T16:39:57.600624Z", "iopub.status.idle": "2020-03-09T16:39:57.601424Z"}}
adms_drug_df_norm.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='drughiclseqno_37589', n=5).head()

# + {"Collapsed": "false", "persistent_id": "d6663be1-591b-4c35-b446-978dbc205444", "execution": {"iopub.status.busy": "2020-03-09T16:39:57.602565Z", "iopub.status.idle": "2020-03-09T16:39:57.602911Z"}}
adms_drug_df_norm[adms_drug_df_norm.patientunitstayid == 2346930].head(10)

# + [markdown] {"Collapsed": "false"}
# Comparing the output from the two previous cells with what we had before the `join_repeated_rows` method, we can see that all rows with duplicate IDs have been successfully joined.

# + [markdown] {"Collapsed": "false"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "938421fc-57b7-4c40-b7a9-315b5619cbb0", "execution": {"iopub.status.busy": "2020-03-09T16:39:57.603913Z", "iopub.status.idle": "2020-03-09T16:39:57.604426Z"}}
adms_drug_df.columns = du.data_processing.clean_naming(adms_drug_df.columns)
adms_drug_df_norm.columns = du.data_processing.clean_naming(adms_drug_df_norm.columns)
adms_drug_df_norm.head()

# + [markdown] {"Collapsed": "false"}
# ### Save the dataframe

# + [markdown] {"Collapsed": "false"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "15ca1a3f-614e-49da-aa1d-7ac42bceee73", "execution": {"iopub.status.busy": "2020-03-09T16:39:57.605450Z", "iopub.status.idle": "2020-03-09T16:39:57.605735Z"}}
adms_drug_df.to_csv(f'{data_path}cleaned/unnormalized/ohe/ohe/admissionDrug.csv')

# + [markdown] {"Collapsed": "false"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "9a20e4a3-a8d6-4842-8470-e6bfccd03267", "execution": {"iopub.status.busy": "2020-03-09T16:39:57.607380Z", "iopub.status.idle": "2020-03-09T16:39:57.607712Z"}}
adms_drug_df_norm.to_csv(f'{data_path}cleaned/normalized/ohe/ohe/admissionDrug.csv')

# + [markdown] {"Collapsed": "false"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "bf406619-1133-4314-95ca-808f9fe81aee", "execution": {"iopub.status.busy": "2020-03-09T16:39:57.608840Z", "iopub.status.idle": "2020-03-09T16:39:57.609134Z"}}
adms_drug_df_norm.describe().transpose()

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": true}
# ## Medication data

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": true}
# ### Initialize variables

# + {"Collapsed": "false", "persistent_id": "754a96f8-d389-4968-8c13-52e5e9d0bf82", "execution": {"iopub.status.busy": "2020-02-17T02:24:13.243501Z", "iopub.execute_input": "2020-02-17T02:24:13.243729Z", "iopub.status.idle": "2020-02-17T02:24:13.247974Z", "shell.execute_reply.started": "2020-02-17T02:24:13.243694Z", "shell.execute_reply": "2020-02-17T02:24:13.246579Z"}}
# List of categorical features
cat_feat = []
# Dictionary of the one hot encoded columns originary from each categorical feature, that will be embedded
cat_feat_ohe = dict()

# + [markdown] {"Collapsed": "false"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "37d12a8d-d08c-41cd-b904-f005b1497fe1"}
med_df = pd.read_csv(f'{data_path}original/medication.csv', dtype={'loadingdose': 'object'})
med_df.head()

# + {"Collapsed": "false", "persistent_id": "c1bc04e9-c48c-443b-9208-8ec4618a0e2d"}
len(med_df)

# + {"Collapsed": "false", "persistent_id": "675a744b-8308-4a4d-89fb-3f8ba150343f"}
med_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# There's not much admission drug data (only around 20% of the unit stays have this data). However, it might be useful, considering also that it complements the medication table.

# + [markdown] {"Collapsed": "false"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "2eda81ce-69df-4328-96f0-4f770bd683d3"}
med_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "c3118567-0513-47f1-a5d0-920683fc8273"}
med_df.columns

# + {"Collapsed": "false", "persistent_id": "86c4c435-d002-41ac-a635-78c7973d2aa9"}
med_df.dtypes

# + [markdown] {"Collapsed": "false"}
# ### Check for missing values

# + {"Collapsed": "false", "persistent_id": "095dbcb4-a4c4-4b9a-baca-41b954126f19", "pixiedust": {"displayParams": {}}}
du.search_explore.dataframe_missing_values(med_df)

# + [markdown] {"Collapsed": "false"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "5cb3ea02-b35b-43dc-9f72-d040f53def73"}
med_df.drugname.value_counts()

# + {"Collapsed": "false", "persistent_id": "f803016b-728d-4200-8ffd-ed1992bec091"}
med_df.drughiclseqno.value_counts()

# + {"Collapsed": "false", "persistent_id": "68ac1b53-5c7f-4bae-9d20-7685d398bd04"}
med_df.dosage.value_counts()

# + {"Collapsed": "false", "persistent_id": "522ff137-54bb-47ef-a5c9-f3017c4665d3"}
med_df.frequency.value_counts()

# + {"Collapsed": "false", "persistent_id": "24ae1f2a-bfa7-432f-bda5-0efcf48f110b"}
med_df.drugstartoffset.value_counts()

# + {"Collapsed": "false", "persistent_id": "8427b730-79a9-4c87-b44b-ab784cee15a1"}
med_df[med_df.drugstartoffset == 0].head()

# + [markdown] {"Collapsed": "false"}
# Besides removing less interesting data (e.g. `drugivadmixture`), I'm also removing the `drugname` column, which is redundant with the codes `drughiclseqno`, while also being brand dependant.

# + {"Collapsed": "false", "persistent_id": "b31b8605-5a88-4821-a65c-5337128f4264"}
med_df = med_df[['patientunitstayid', 'drugstartoffset', 'drugstopoffset',
                 'drugordercancelled', 'dosage', 'frequency', 'drughiclseqno']]
med_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Remove rows of which the drug has been cancelled or not specified

# + {"Collapsed": "false", "persistent_id": "4ffb498c-d6b5-46f3-9863-d74f672dd9be"}
med_df.drugordercancelled.value_counts()

# + {"Collapsed": "false", "persistent_id": "93a3e2f3-3632-4c5d-a25b-4e9195335348"}
med_df = med_df[~((med_df.drugordercancelled == 'Yes') | (med_df.drughiclseqno.isnull()))]
med_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the now unneeded `drugordercancelled` column:

# + {"Collapsed": "false", "persistent_id": "8c832678-d7eb-46d9-9521-a8a389cacc06"}
med_df = med_df.drop('drugordercancelled', axis=1)
med_df.head()

# + {"Collapsed": "false", "persistent_id": "413613d9-1166-414d-a0c5-c51b1824d93e", "pixiedust": {"displayParams": {}}}
du.search_explore.dataframe_missing_values(med_df)

# + [markdown] {"Collapsed": "false"}
# ### Separating units from dosage
#
# In order to properly take into account the dosage quantities, as well as to standardize according to other tables like admission drugs, we should take the original `dosage` column and separate it to just the `drugdosage` values and the `drugunit`.

# + [markdown] {"Collapsed": "false"}
# No need to create a separate `pyxis` feature, which would indicate the use of the popular automated medications manager, as the frequency embedding will have that into account.

# + [markdown] {"Collapsed": "false"}
# Create dosage and unit features:

# + {"Collapsed": "false", "persistent_id": "68f9ce27-4230-4f52-b7f9-51c67ef4afea"}
med_df['drugdosage'] = np.nan
med_df['drugunit'] = np.nan
med_df.head()

# + [markdown] {"Collapsed": "false"}
# Convert dataframe to Pandas, as the next cells aren't working properly with Modin:

# + {"Collapsed": "false"}
med_df, pd = du.utils.convert_dataframe(med_df, to='pandas')

# + {"Collapsed": "false"}
type(med_df)

# + [markdown] {"Collapsed": "false"}
# Get the dosage and unit values for each row:

# + {"Collapsed": "false", "persistent_id": "014b4c4e-31cb-487d-8bd9-b8f7048e981e"}
med_df[['drugdosage', 'drugunit']] = med_df.apply(du.data_processing.set_dosage_and_units, axis=1, result_type='expand')
med_df.head()

# + {"Collapsed": "false"}
med_df.drugunit.value_counts()

# + [markdown] {"Collapsed": "false"}
# Remove the now unneeded `dosage` column:

# + {"Collapsed": "false", "persistent_id": "61c02af1-fa9e-49d7-81e1-abd56d133510"}
med_df = med_df.drop('dosage', axis=1)
med_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Rename columns

# + {"Collapsed": "false", "persistent_id": "356775df-9204-4fef-b7a0-b82edf4ba85f"}
med_df_norm = med_df_norm.rename(columns={'frequency': 'drugadmitfrequency'})
med_df_norm.head()

# + [markdown] {"Collapsed": "false"}
# ### Normalize data

# + [markdown] {"Collapsed": "false"}
# Update list of categorical features:

# + {"Collapsed": "false", "persistent_id": "d714ff24-c50b-4dff-9b21-832d030d050f"}
cat_feat = ['drugunit', 'drugadmitfrequency', 'drughiclseqno']

# + [markdown] {"Collapsed": "false"}
# Filter just to the most common categories:

# + {"Collapsed": "false"}
for col in cat_feat:
    most_common_cat = list(med_df[col].value_counts().nlargest(500).index)
    med_df = med_df[med_df[col].isin(most_common_cat)]

# + {"Collapsed": "false", "persistent_id": "a4cd949b-e561-485d-bcb6-10fccc343352", "pixiedust": {"displayParams": {}}}
med_df_norm, mean, std = du.data_processing.normalize_data(med_df, columns_to_normalize=False,
                                                           columns_to_normalize_categ=[(['drughiclseqno', 'drugunit'], 'drugdosage')],
                                                           get_stats=True, inplace=True)
med_df_norm.head()

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
stream = open(f'{data_path}/cleaned/medication_norm_stats.yaml', 'w')
yaml.dump(norm_stats, stream, default_flow_style=False)

# + {"Collapsed": "false", "persistent_id": "93145ae0-2204-4553-a7eb-5d235552dd82"}
med_df_norm = med_df_norm.sort_values('ts')
med_df_norm.head()

# + [markdown] {"Collapsed": "false"}
# Prevent infinite values:

# + {"Collapsed": "false"}
med_df_norm = med_df_norm.replace(to_replace=np.inf, value=0)

# + {"Collapsed": "false"}
med_df_norm = med_df_norm.replace(to_replace=-np.inf, value=0)

# + {"Collapsed": "false"}
med_df_norm.drugdosage.max()

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": false}
# ### Discretize categorical features
#
# Convert binary categorical features into one hot encode columns, which can later be embedded or used as is.

# + [markdown] {"Collapsed": "false"}
# #### One hot encode features

# + {"Collapsed": "false", "persistent_id": "ed09c3dd-50b3-48cf-aa04-76534feaf767"}
med_df[cat_feat].head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-09T16:36:38.933807Z", "iopub.execute_input": "2020-03-09T16:36:38.934065Z", "iopub.status.idle": "2020-03-09T16:36:38.937828Z", "shell.execute_reply.started": "2020-03-09T16:36:38.934035Z", "shell.execute_reply": "2020-03-09T16:36:38.936941Z"}}
old_columns = med_df.columns

# + [markdown] {"Collapsed": "false"}
# Apply one hot encoding:

# + {"Collapsed": "false", "persistent_id": "51ac8fd1-cbd2-4f59-a737-f0fcc13043fd", "pixiedust": {"displayParams": {}}}
med_df = du.data_processing.one_hot_encoding_dataframe(med_df, columns=cat_feat, join_rows=False)
med_df

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-09T16:37:27.392359Z", "iopub.status.idle": "2020-03-09T16:37:27.399976Z", "iopub.execute_input": "2020-03-09T16:37:27.392616Z", "shell.execute_reply.started": "2020-03-09T16:37:27.392582Z", "shell.execute_reply": "2020-03-09T16:37:27.399076Z"}}
new_columns = set(med_df.columns) - set(old_columns)

# + {"Collapsed": "false", "persistent_id": "e5615265-4372-4117-a368-ec539c871763"}
med_df.dtypes

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

# + {"Collapsed": "false", "persistent_id": "e51cc2e0-b598-484f-a3f8-8c764950777f"}
stream = open(f'{data_path}/cleaned/cat_feat_ohe_med.yaml', 'w')
yaml.dump(cat_feat_ohe, stream, default_flow_style=False)

# + [markdown] {"Collapsed": "false"}
# ### Create drug stop event
#
# Add a timestamp corresponding to when each patient stops taking each medication.

# + [markdown] {"Collapsed": "false"}
# Duplicate every row, so as to create a discharge event:

# + {"Collapsed": "false", "persistent_id": "db8e91ff-cfd7-4ddd-8873-20c20e1f9e46"}
new_df = med_df.copy()
new_df.head()

# + [markdown] {"Collapsed": "false"}
# Set the new dataframe's rows to have the drug stop timestamp, with no more information on the drug that was being used:

# + {"Collapsed": "false", "persistent_id": "6d2bf14a-b75b-4b4c-ae33-403cde6781d7"}
new_df.drugstartoffset = new_df.drugstopoffset
new_df.drugunit = 0
new_df.drugdosage = np.nan
new_df.frequency = 0
new_df.drughiclseqno = 0
new_df.head()

# + [markdown] {"Collapsed": "false"}
# Join the new rows to the remaining dataframe:

# + {"Collapsed": "false", "persistent_id": "321fe743-eae2-4b2c-a4ef-5e1c3350a231"}
med_df = med_df.append(new_df)
med_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the now unneeded medication stop column:

# + {"Collapsed": "false", "persistent_id": "6b1a48ca-fecc-467f-8071-534fbd4944bb"}
med_df = med_df.drop('drugstopoffset', axis=1)
med_df.head(6)

# + [markdown] {"Collapsed": "false"}
# ### Create the timestamp feature and sort

# + [markdown] {"Collapsed": "false"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "88ab8d50-b556-4c76-bb0b-33e76900018f"}
med_df = med_df.rename(columns={'drugstartoffset': 'ts'})
med_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "74b9d214-8083-4d37-acbe-6ce26d6b1629"}
len(med_df)

# + {"Collapsed": "false", "persistent_id": "a2529353-bf59-464a-a32b-a940dd66007a"}
med_df = med_df.drop_duplicates()
med_df.head()

# + {"Collapsed": "false", "persistent_id": "be199b11-006c-4619-ac80-b3d86fd10f3b"}
len(med_df)

# + [markdown] {"Collapsed": "false"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "81712e88-3b96-4f10-a536-a80268bfe805"}
med_df = med_df.sort_values('ts')
med_df.head()

# + [markdown] {"Collapsed": "false"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "ac2b0e4b-d2bd-4eb5-a629-637361a85457"}
med_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='drughiclseqno', n=5).head()

# + {"Collapsed": "false", "persistent_id": "da5e70d7-0514-4bdb-a5e2-12b6e8a1b197"}
med_df[med_df.patientunitstayid == 979183].head(10)

# + [markdown] {"Collapsed": "false"}
# We can see that there are up to 41 categories per set of `patientunitstayid` and `ts`. As such, we must join them. But first, we need to normalize the dosage by the respective sets of drug code and units, so as to avoid mixing different absolute values.

# + [markdown] {"Collapsed": "false"}
# ### Join rows that have the same IDs

# + [markdown] {"Collapsed": "false"}
# Even after removing duplicates rows, there are still some that have different information for the same ID and timestamp. We have to concatenate the categorical enumerations.

# + {"Collapsed": "false", "persistent_id": "ed86d5a7-eeb3-44c4-9a4e-6dd67af307f2"}
list(set(med_df_norm.columns) - set(cat_feat) - set(['patientunitstayid', 'ts']))

# + {"Collapsed": "false", "persistent_id": "589931b8-fe11-439a-8b14-4857c168c023", "pixiedust": {"displayParams": {}}}
med_df_norm = du.embedding.join_repeated_rows(med_df_norm, inplace=True)
med_df_norm.head()

# + [markdown] {"Collapsed": "false"}
# Reconvert dataframe to Modin:

# + {"Collapsed": "false"}
med_df_norm, pd = du.utils.convert_dataframe(med_df, to='modin')

# + {"Collapsed": "false"}
type(med_df_norm)

# + {"Collapsed": "false", "persistent_id": "db6b5624-e600-4d90-bc5a-ffa5a876d8dd"}
med_df_norm.dtypes

# + {"Collapsed": "false", "persistent_id": "954d2c26-4ef4-42ec-b0f4-a73febb5115d"}
med_df_norm.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='drughiclseqno', n=5).head()

# + {"Collapsed": "false", "persistent_id": "85536a51-d31a-4b25-aaee-9c9d4ec392f6"}
med_df_norm[med_df_norm.patientunitstayid == 979183].head(10)

# + [markdown] {"Collapsed": "false"}
# Comparing the output from the two previous cells with what we had before the `join_repeated_rows` method, we can see that all rows with duplicate IDs have been successfully joined.

# + [markdown] {"Collapsed": "false"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "0f255c7d-1d1a-4dd3-94d7-d7fd54e13da0"}
med_df.columns = du.data_processing.clean_naming(med_df.columns)
med_df_norm.columns = du.data_processing.clean_naming(med_df_norm.columns)
med_df_norm.head()

# + [markdown] {"Collapsed": "false"}
# ### Save the dataframe

# + [markdown] {"Collapsed": "false"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "7c95b423-0fd8-4e65-ac07-a0117f0c36bd"}
med_df.to_csv(f'{data_path}cleaned/unnormalized/ohe/medication.csv')

# + [markdown] {"Collapsed": "false"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "eae5d63f-5635-4fa0-8c42-ff6081336e18"}
med_df_norm.to_csv(f'{data_path}cleaned/normalized/ohe/medication.csv')

# + [markdown] {"Collapsed": "false"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "2cdabf5e-7df3-441b-b8ed-a06c404df27e"}
med_df_norm.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "9255fb38-ba28-4bc2-8f97-7596e8acbc5a"}
med_df.nlargest(columns='drugdosage', n=5)

# + [markdown] {"Collapsed": "false"}
# Although the `drugdosage` looks good on mean (close to 0) and standard deviation (close to 1), it has very large magnitude minimum (-88.9) and maximum (174.1) values. Furthermore, these don't seem to be because of NaN values, whose groupby normalization could have been unideal. As such, it's hard to say if these are outliers or realistic values.

# + [markdown] {"Collapsed": "false"}
# [TODO] Check if these very large extreme dosage values make sense and, if not, try to fix them.

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": true}
# ## Treatment data

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": true}
# ### Initialize variables

# + {"Collapsed": "false", "persistent_id": "754a96f8-d389-4968-8c13-52e5e9d0bf82", "execution": {"iopub.status.busy": "2020-02-17T02:24:13.243501Z", "iopub.execute_input": "2020-02-17T02:24:13.243729Z", "iopub.status.idle": "2020-02-17T02:24:13.247974Z", "shell.execute_reply.started": "2020-02-17T02:24:13.243694Z", "shell.execute_reply": "2020-02-17T02:24:13.246579Z"}}
# List of categorical features
cat_feat = []
# Dictionary of the one hot encoded columns originary from each categorical feature, that will be embedded
cat_feat_ohe = dict()

# + [markdown] {"Collapsed": "false"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "d0a4e172-9a38-4bef-a052-6310b7fb5214"}
treat_df = pd.read_csv(f'{data_path}original/treatment.csv')
treat_df.head()

# + {"Collapsed": "false", "persistent_id": "bb5ccda2-5a79-46ed-9be2-8bb21b995771"}
len(treat_df)

# + {"Collapsed": "false", "persistent_id": "ad5c7291-d2ba-4e05-baea-5a887de8f535"}
treat_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "594cd2ff-12e5-4c90-83e7-722818eaa39e"}
treat_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "3cca7683-fa6c-414f-854f-cd1f36cc6e42"}
treat_df.columns

# + {"Collapsed": "false", "persistent_id": "ce410a02-9154-40e5-bf5f-ddbdf1724bcb"}
treat_df.dtypes

# + [markdown] {"Collapsed": "false"}
# ### Check for missing values

# + {"Collapsed": "false", "persistent_id": "ba586d2f-b03e-4a8d-942f-a2440be92101", "pixiedust": {"displayParams": {}}}
du.search_explore.dataframe_missing_values(treat_df)

# + [markdown] {"Collapsed": "false"}
# ### Remove unneeded features

# + [markdown] {"Collapsed": "false"}
# Besides the usual removal of row identifier, `treatmentid`, I'm also removing `activeupondischarge`, as we don't have complete information as to when diagnosis end.

# + {"Collapsed": "false", "persistent_id": "64f644db-fb9b-45c2-9c93-e4878850ed08"}
treat_df = treat_df.drop(['treatmentid', 'activeupondischarge'], axis=1)
treat_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Separate high level diagnosis

# + {"Collapsed": "false", "persistent_id": "99292fb1-2fff-46c0-9f57-e686a52bce75"}
treat_df.treatmentstring.value_counts()

# + {"Collapsed": "false", "persistent_id": "56252187-e07d-4042-a7c2-9bad34e8da22"}
treat_df.treatmentstring.map(lambda x: x.split('|')).head()

# + {"Collapsed": "false", "persistent_id": "d3c5a8e5-b60d-4b04-ad9f-9df30e1b63c3"}
treat_df.treatmentstring.map(lambda x: len(x.split('|'))).min()

# + {"Collapsed": "false", "persistent_id": "0fb9adae-17a9-433d-aa4a-9367511846c1"}
treat_df.treatmentstring.map(lambda x: len(x.split('|'))).max()

# + [markdown] {"Collapsed": "false"}
# There are always at least 3 higher level diagnosis. It could be beneficial to extract those first 3 levels to separate features, with the last one getting values until the end of the string, so as to avoid the need for the model to learn similarities that are already known.

# + {"Collapsed": "false", "persistent_id": "aee50693-322a-4a01-9ef2-fac27c51e45f"}
treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 0, separator='|')).value_counts()

# + {"Collapsed": "false", "persistent_id": "6ffd154f-ab1e-4614-b990-3414c2e8abf5"}
treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 1, separator='|')).value_counts()

# + {"Collapsed": "false", "persistent_id": "add7fc79-2edc-4481-939a-68c69f3b4383"}
treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 2, separator='|')).value_counts()

# + {"Collapsed": "false", "persistent_id": "fdfd37f9-1ac4-4c08-bec7-1f7971cee605"}
treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 3, separator='|')).value_counts()

# + {"Collapsed": "false", "persistent_id": "2e81fa29-6255-427f-aeb8-3d9ddd615565"}
treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 4, separator='|')).value_counts()

# + {"Collapsed": "false", "persistent_id": "7a30e647-0d39-480e-82ec-897defcfac38"}
treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 5, separator='|')).value_counts()

# + [markdown] {"Collapsed": "false"}
# <!-- There are always 8 levels of the notes. As the first 6 ones are essentially always the same ("notes/Progress Notes/Social History / Family History/Social History/Social History/"), it's best to just preserve the 7th one and isolate the 8th in a new feature. This way, the split provides further insight to the model on similar notes. However, it's also worth taking note that the 8th level of `notepath` seems to be identical to the feature `notevalue`. We'll look more into it later. -->

# + {"Collapsed": "false", "persistent_id": "44721e19-f088-4cf3-be69-5201d1260d52"}
treat_df['treatmenttype'] = treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 0, separator='|'))
treat_df['treatmenttherapy'] = treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 1, separator='|'))
treat_df['treatmentdetails'] = treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 2, separator='|', till_the_end=True))
treat_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the now redundant `treatmentstring` column:

# + {"Collapsed": "false", "persistent_id": "5e8dd6a3-d086-442e-9845-ea701b36ecf8"}
treat_df = treat_df.drop('treatmentstring', axis=1)
treat_df.head()

# + {"Collapsed": "false", "persistent_id": "3e7db267-771e-4863-affb-0048ca1b7f7d"}
treat_df.treatmenttype.value_counts()

# + {"Collapsed": "false", "persistent_id": "5f75072b-b0b4-419c-a4c0-0678326257af"}
treat_df.treatmenttherapy.value_counts()

# + {"Collapsed": "false", "persistent_id": "8c77797f-a1af-4d87-a37d-3b1fa7e0132c"}
treat_df.treatmentdetails.value_counts()

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": false}
# ### Discretize categorical features
#
# Convert binary categorical features into one hot encode columns, which can later be embedded or used as is.

# + [markdown] {"Collapsed": "false"}
# #### One hot encode features

# + [markdown] {"Collapsed": "false"}
# Update list of categorical features:

# + {"Collapsed": "false", "persistent_id": "e015c37e-1537-409f-b67e-9eaabf295873"}
cat_feat = ['treatmenttype', 'treatmenttherapy', 'treatmentdetails']

# + {"Collapsed": "false", "persistent_id": "29451e32-cf24-47b0-a783-9acc9b71b1c3"}
treat_df[cat_feat].head()

# + [markdown] {"Collapsed": "false"}
# Filter just to the most common categories:

# + {"Collapsed": "false"}
for col in cat_feat:
    most_common_cat = list(treat_df[col].value_counts().nlargest(500).index)
    treat_df = treat_df[treat_df[col].isin(most_common_cat)]

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-09T16:36:38.933807Z", "iopub.execute_input": "2020-03-09T16:36:38.934065Z", "iopub.status.idle": "2020-03-09T16:36:38.937828Z", "shell.execute_reply.started": "2020-03-09T16:36:38.934035Z", "shell.execute_reply": "2020-03-09T16:36:38.936941Z"}}
old_columns = treat_df.columns

# + [markdown] {"Collapsed": "false"}
# Apply one hot encoding:

# + {"Collapsed": "false", "persistent_id": "3c0aa3dc-34e6-4a10-b01e-439fe2c6f991", "pixiedust": {"displayParams": {}}}
treat_df = du.data_processing.one_hot_encoding_dataframe(treat_df, columns=cat_feat, join_rows=False)
treat_df

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-09T16:37:27.392359Z", "iopub.status.idle": "2020-03-09T16:37:27.399976Z", "iopub.execute_input": "2020-03-09T16:37:27.392616Z", "shell.execute_reply.started": "2020-03-09T16:37:27.392582Z", "shell.execute_reply": "2020-03-09T16:37:27.399076Z"}}
new_columns = set(treat_df.columns) - set(old_columns)

# + {"Collapsed": "false", "persistent_id": "ff54e442-e896-4310-8a95-c205cd2cbf93"}
treat_df.dtypes

# + [markdown] {"Collapsed": "false"}
# Save the association between the original categorical features and the new one hot encoded columns:

# + {"Collapsed": "false", "persistent_id": "26eac7f3-9081-4a96-ae4a-40054c223fd7", "execution": {"iopub.status.busy": "2020-03-09T16:37:35.157248Z", "iopub.execute_input": "2020-03-09T16:37:35.157526Z", "iopub.status.idle": "2020-03-09T16:37:35.164656Z", "shell.execute_reply.started": "2020-03-09T16:37:35.157493Z", "shell.execute_reply": "2020-03-09T16:37:35.163771Z"}}
for orig_col in cat_feat:
    cat_feat_ohe[orig_col] = [ohe_col for ohe_col in new_columns
                              if ohe_col.startswith(orig_col)]

# + {"execution": {"iopub.status.busy": "2020-03-09T16:37:35.165864Z", "iopub.execute_input": "2020-03-09T16:37:35.166280Z", "iopub.status.idle": "2020-03-09T16:37:35.190294Z", "shell.execute_reply.started": "2020-03-09T16:37:35.166256Z", "shell.execute_reply": "2020-03-09T16:37:35.189358Z"}, "Collapsed": "false"}
cat_feat_ohe

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

# + {"Collapsed": "false", "persistent_id": "373ef393-e31d-4778-a902-78fc1ce179f3"}
stream = open(f'{data_path}/cleaned/cat_feat_ohe_treat.yaml', 'w')
yaml.dump(cat_feat_ohe, stream, default_flow_style=False)

# + [markdown] {"Collapsed": "false"}
# ### Create the timestamp feature and sort

# + [markdown] {"Collapsed": "false"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "e9364015-9dc8-45ed-a526-fbfd85a7d249"}
treat_df = treat_df.rename(columns={'treatmentoffset': 'ts'})
treat_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "a833b09c-2fb0-46fc-af33-629e67113663"}
len(treat_df)

# + {"Collapsed": "false", "persistent_id": "45d801a9-e196-4940-92c9-9e73a4cccbb1"}
treat_df = treat_df.drop_duplicates()
treat_df.head()

# + {"Collapsed": "false", "persistent_id": "6f07e6d2-788a-4c5b-a0d8-0fced42215a7"}
len(treat_df)

# + [markdown] {"Collapsed": "false"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "79121961-77a1-4657-a374-dd5d389174ed"}
treat_df = treat_df.sort_values('ts')
treat_df.head()

# + [markdown] {"Collapsed": "false"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "8672872e-aea3-480c-b5a2-383843db6e3e"}
treat_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='treatmenttype', n=5).head()

# + {"Collapsed": "false", "persistent_id": "ab644c5b-831a-4b57-a95c-fbac810e59f4"}
treat_df[treat_df.patientunitstayid == 1352520].head(10)

# + [markdown] {"Collapsed": "false"}
# We can see that there are up to 105 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# + [markdown] {"Collapsed": "false"}
# ### Join rows that have the same IDs

# + [markdown] {"Collapsed": "false"}
# Convert dataframe to Pandas, as the groupby operation in `join_repeated_rows` isn't working properly with Modin:

# + {"Collapsed": "false"}
treat_df, pd = du.utils.convert_dataframe(treat_df, to='pandas')

# + {"Collapsed": "false"}
type(treat_df)

# + {"Collapsed": "false", "persistent_id": "589931b8-fe11-439a-8b14-4857c168c023", "pixiedust": {"displayParams": {}}}
treat_df = du.embedding.join_repeated_rows(treat_df, inplace=True)
treat_df.head()

# + [markdown] {"Collapsed": "false"}
# Reconvert dataframe to Modin:

# + {"Collapsed": "false"}
treat_df, pd = du.utils.convert_dataframe(treat_df, to='modin')

# + {"Collapsed": "false"}
type(treat_df)

# + {"Collapsed": "false", "persistent_id": "e2b70eff-045e-4587-af99-329381e839b2"}
treat_df.dtypes

# + {"Collapsed": "false", "persistent_id": "08de3813-83e3-478c-bfeb-e6dbeafacdb1"}
treat_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='treatmenttype', n=5).head()

# + {"Collapsed": "false", "persistent_id": "ef1b7d12-4dd4-4c9c-9204-55ad5ac98443"}
treat_df[treat_df.patientunitstayid == 1352520].head(10)

# + [markdown] {"Collapsed": "false"}
# Comparing the output from the two previous cells with what we had before the `join_repeated_rows` method, we can see that all rows with duplicate IDs have been successfully joined.

# + [markdown] {"Collapsed": "false"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "4decdcbe-bf08-4577-b55c-dda15fbef130"}
treat_df.columns = du.data_processing.clean_naming(treat_df.columns)
treat_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Save the dataframe

# + [markdown] {"Collapsed": "false"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "7fecf050-a520-499d-ba70-96bc406e0a7e"}
treat_df.to_csv(f'{data_path}cleaned/unnormalized/ohe/treatment.csv')

# + [markdown] {"Collapsed": "false"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "8ec7af26-406d-4870-8475-0acca2b92876"}
treat_df.to_csv(f'{data_path}cleaned/normalized/ohe/treatment.csv')

# + [markdown] {"Collapsed": "false"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "23eefabe-23d7-4db4-b8e0-7c1e61ef2789"}
treat_df.describe().transpose()
# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": true}
# ## Intake output data

# + [markdown] {"Collapsed": "false"}
# ### Initialize variables

# + {"Collapsed": "false", "persistent_id": "754a96f8-d389-4968-8c13-52e5e9d0bf82", "execution": {"iopub.status.busy": "2020-02-17T05:01:35.789873Z", "iopub.execute_input": "2020-02-17T05:01:35.790092Z", "iopub.status.idle": "2020-02-17T05:01:35.794511Z", "shell.execute_reply.started": "2020-02-17T05:01:35.790060Z", "shell.execute_reply": "2020-02-17T05:01:35.793047Z"}}
# List of categorical features
cat_feat = []
# Dictionary of the one hot encoded columns originary from each categorical feature, that will be embedded
cat_feat_ohe = dict()

# + [markdown] {"Collapsed": "false"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "4a37073a-b357-4079-b6af-72125689781d", "execution": {"iopub.status.busy": "2020-02-17T05:01:37.002580Z", "iopub.execute_input": "2020-02-17T05:01:37.002849Z", "iopub.status.idle": "2020-02-17T05:01:53.516810Z", "shell.execute_reply.started": "2020-02-17T05:01:37.002809Z", "shell.execute_reply": "2020-02-17T05:01:53.515573Z"}}
in_out_df = pd.read_csv(f'{data_path}original/intakeOutput.csv')
in_out_df.head()

# + {"Collapsed": "false", "persistent_id": "6566d20f-9fcb-4d92-879b-55a0cefe54ae", "execution": {"iopub.status.busy": "2020-02-17T04:06:18.870396Z", "iopub.execute_input": "2020-02-17T04:06:18.870650Z", "iopub.status.idle": "2020-02-17T04:06:18.877115Z", "shell.execute_reply.started": "2020-02-17T04:06:18.870605Z", "shell.execute_reply": "2020-02-17T04:06:18.876169Z"}}
len(in_out_df)

# + {"Collapsed": "false", "persistent_id": "e267e007-4b72-4551-a9d2-7c916956235c", "execution": {"iopub.status.busy": "2020-02-17T04:06:18.879129Z", "iopub.execute_input": "2020-02-17T04:06:18.879384Z", "iopub.status.idle": "2020-02-17T04:06:20.662963Z", "shell.execute_reply.started": "2020-02-17T04:06:18.879345Z", "shell.execute_reply": "2020-02-17T04:06:20.661958Z"}}
in_out_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "b4f282b2-e5f4-4cfb-83c5-c79944367d70", "execution": {"iopub.status.busy": "2020-02-17T04:06:20.664981Z", "iopub.execute_input": "2020-02-17T04:06:20.665202Z", "iopub.status.idle": "2020-02-17T04:06:41.988804Z", "shell.execute_reply.started": "2020-02-17T04:06:20.665167Z", "shell.execute_reply": "2020-02-17T04:06:41.987709Z"}}
in_out_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "6a382a6d-709e-4cdf-a586-c0cc751ff853", "execution": {"iopub.status.busy": "2020-02-17T04:06:41.993271Z", "iopub.execute_input": "2020-02-17T04:06:41.993671Z", "iopub.status.idle": "2020-02-17T04:07:13.810728Z", "shell.execute_reply.started": "2020-02-17T04:06:41.993623Z", "shell.execute_reply": "2020-02-17T04:07:13.809111Z"}}
in_out_df.info()

# + [markdown] {"Collapsed": "false"}
# ### Check for missing values

# + {"Collapsed": "false", "persistent_id": "3cc624bc-6dac-4cd0-9d64-f329e29940fa", "pixiedust": {"displayParams": {}}, "execution": {"iopub.status.busy": "2020-02-17T05:01:53.519614Z", "iopub.execute_input": "2020-02-17T05:01:53.520038Z", "iopub.status.idle": "2020-02-17T05:01:56.027286Z", "shell.execute_reply.started": "2020-02-17T05:01:53.519981Z", "shell.execute_reply": "2020-02-17T05:01:56.026607Z"}}
du.search_explore.dataframe_missing_values(in_out_df)

# + [markdown] {"Collapsed": "false"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "ba513734-4c7e-455d-bfa8-9b805c59b530", "execution": {"iopub.status.busy": "2020-02-17T05:01:56.029042Z", "iopub.execute_input": "2020-02-17T05:01:56.029267Z", "iopub.status.idle": "2020-02-17T05:01:57.191687Z", "shell.execute_reply.started": "2020-02-17T05:01:56.029233Z", "shell.execute_reply": "2020-02-17T05:01:57.190729Z"}}
in_out_df.celllabel.value_counts()

# + {"Collapsed": "false", "persistent_id": "ba513734-4c7e-455d-bfa8-9b805c59b530", "execution": {"iopub.status.busy": "2020-02-17T05:01:57.193422Z", "iopub.execute_input": "2020-02-17T05:01:57.193657Z", "iopub.status.idle": "2020-02-17T05:01:58.314678Z", "shell.execute_reply.started": "2020-02-17T05:01:57.193619Z", "shell.execute_reply": "2020-02-17T05:01:58.313323Z"}}
in_out_df.celllabel.value_counts().head(50)

# + {"Collapsed": "false", "persistent_id": "4d028a9e-38e8-496e-805a-ed3f2304a07f", "execution": {"iopub.status.busy": "2020-02-17T05:01:58.315863Z", "iopub.execute_input": "2020-02-17T05:01:58.316062Z", "iopub.status.idle": "2020-02-17T05:01:58.675371Z", "shell.execute_reply.started": "2020-02-17T05:01:58.316029Z", "shell.execute_reply": "2020-02-17T05:01:58.674562Z"}}
in_out_df.cellvaluenumeric.value_counts()

# + {"Collapsed": "false", "persistent_id": "d61a9031-35b3-4c51-a141-a820f6ed70a8", "execution": {"iopub.status.busy": "2020-02-17T05:01:58.676388Z", "iopub.execute_input": "2020-02-17T05:01:58.676769Z", "iopub.status.idle": "2020-02-17T05:01:58.972846Z", "shell.execute_reply.started": "2020-02-17T05:01:58.676560Z", "shell.execute_reply": "2020-02-17T05:01:58.972035Z"}}
in_out_df.cellvaluetext.value_counts()

# + {"Collapsed": "false", "persistent_id": "502ee5b2-8eba-407e-a0c3-158c27cb9fab", "execution": {"iopub.status.busy": "2020-02-17T05:01:58.974348Z", "iopub.execute_input": "2020-02-17T05:01:58.974722Z", "iopub.status.idle": "2020-02-17T05:02:00.386080Z", "shell.execute_reply.started": "2020-02-17T05:01:58.974648Z", "shell.execute_reply": "2020-02-17T05:02:00.385337Z"}}
in_out_df.cellpath.value_counts()

# + [markdown] {"Collapsed": "false"}
# Besides the usual removal of row identifier, `intakeoutputid`, and the timestamp when data was added, `intakeoutputentryoffset`, I'm also removing the cumulative features `intaketotal`, `outputtotal`, `diaslysistotal`, and `nettotal` (cumulative data could damage the neural networks' logic with too high values and we're looking for data of the current moment), as well as `cellvaluetext`, which has redundant info with `cellvaluenumeric`.

# + {"Collapsed": "false", "persistent_id": "48f10eea-6c36-4188-91d9-4def749f2486", "execution": {"iopub.status.busy": "2020-02-17T05:02:00.388786Z", "iopub.execute_input": "2020-02-17T05:02:00.388993Z", "iopub.status.idle": "2020-02-17T05:02:01.210447Z", "shell.execute_reply.started": "2020-02-17T05:02:00.388961Z", "shell.execute_reply": "2020-02-17T05:02:01.209721Z"}}
in_out_df = in_out_df.drop(columns=['intakeoutputid', 'intakeoutputentryoffset',
                                    'intaketotal', 'outputtotal', 'dialysistotal',
                                    'nettotal', 'cellvaluetext'])
in_out_df.head()

# + [markdown] {"Collapsed": "false"}
# Additionally, we're going to focus on the most common data categories, ignoring rarer ones and those that could be redundant with data from other tables (e.g. infusion drugs).

# + {"Collapsed": "false", "persistent_id": "6c7fa8f7-3195-46a9-889a-6034ffdcaef3", "execution": {"iopub.status.busy": "2020-02-17T05:02:01.214667Z", "iopub.execute_input": "2020-02-17T05:02:01.215018Z", "iopub.status.idle": "2020-02-17T05:02:01.220180Z", "shell.execute_reply.started": "2020-02-17T05:02:01.214864Z", "shell.execute_reply": "2020-02-17T05:02:01.218743Z"}}
categories_to_keep = ['Urine', 'URINE CATHETER', 'Urinary Catheter Output: Indwelling/Continuous Ure',
                      'Bodyweight (kg)', 'P.O.', 'P.O. Intake', 'Oral Intake', 'Oral',
                      'Stool', 'Crystalloids', 'Indwelling Catheter Output', 'Nutrition Total',
                      'Enteral/Gastric Tube Intake', 'pRBCs', 'Gastric (NG)',
                      'propofol', 'LR', 'LR IVF', 'I.V.', 'Voided Amount', 'Out',
                      'Feeding Tube Flush/Water Bolus Amt (mL)', 'norepinephrine',
                      'Saline Flush (mL)', 'Volume Given (mL)',
                      'Actual Patient Fluid Removal', 'fentaNYL']

# + {"Collapsed": "false", "persistent_id": "d8b7cf79-bda7-411f-a800-e4769eb0ec00", "execution": {"iopub.status.busy": "2020-02-17T05:02:01.224973Z", "iopub.execute_input": "2020-02-17T05:02:01.225909Z", "iopub.status.idle": "2020-02-17T05:02:01.644686Z", "shell.execute_reply.started": "2020-02-17T05:02:01.225650Z", "shell.execute_reply": "2020-02-17T05:02:01.643244Z"}}
(in_out_df.celllabel.isin(categories_to_keep)).head()

# + {"Collapsed": "false", "persistent_id": "8f0a890f-ad85-457c-888e-37f9cddfe4e8", "execution": {"iopub.status.busy": "2020-02-17T05:02:01.646363Z", "iopub.execute_input": "2020-02-17T05:02:01.647007Z", "iopub.status.idle": "2020-02-17T05:02:03.005678Z", "shell.execute_reply.started": "2020-02-17T05:02:01.646903Z", "shell.execute_reply": "2020-02-17T05:02:03.004914Z"}}
in_out_df = in_out_df[in_out_df.celllabel.isin(categories_to_keep)]
in_out_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Merge redundant data
#
# Urine, oral and lactated ringer are labeled in different ways. We need to merge them into unique representations to make the machine learning models learn more efficiently.

# + [markdown] {"Collapsed": "false"}
# Unify urine data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-17T05:02:03.006637Z", "iopub.execute_input": "2020-02-17T05:02:03.006832Z", "iopub.status.idle": "2020-02-17T05:02:03.011502Z", "shell.execute_reply.started": "2020-02-17T05:02:03.006800Z", "shell.execute_reply": "2020-02-17T05:02:03.010611Z"}}
urine_labels = ['Urine', 'URINE CATHETER', 'Urinary Catheter Output: Indwelling/Continuous Ure']

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-17T05:02:03.012385Z", "iopub.execute_input": "2020-02-17T05:02:03.012589Z", "iopub.status.idle": "2020-02-17T05:02:03.563687Z", "shell.execute_reply.started": "2020-02-17T05:02:03.012542Z", "shell.execute_reply": "2020-02-17T05:02:03.562116Z"}}
in_out_df.loc[in_out_df.celllabel.isin(urine_labels), 'celllabel'] = 'urine'

# + [markdown] {"Collapsed": "false"}
# Unify oral data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-17T05:02:03.566262Z", "iopub.execute_input": "2020-02-17T05:02:03.566564Z", "iopub.status.idle": "2020-02-17T05:02:03.571518Z", "shell.execute_reply.started": "2020-02-17T05:02:03.566524Z", "shell.execute_reply": "2020-02-17T05:02:03.570822Z"}}
oral_labels = ['P.O.', 'P.O. Intake', 'Oral Intake', 'Oral']

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-17T05:02:03.572465Z", "iopub.execute_input": "2020-02-17T05:02:03.572936Z", "iopub.status.idle": "2020-02-17T05:02:04.018909Z", "shell.execute_reply.started": "2020-02-17T05:02:03.572878Z", "shell.execute_reply": "2020-02-17T05:02:04.018357Z"}}
in_out_df.loc[in_out_df.celllabel.isin(oral_labels), 'celllabel'] = 'oral'

# + [markdown] {"Collapsed": "false"}
# Unify lactated ringer data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-17T05:02:04.019855Z", "iopub.execute_input": "2020-02-17T05:02:04.020145Z", "iopub.status.idle": "2020-02-17T05:02:04.023388Z", "shell.execute_reply.started": "2020-02-17T05:02:04.020108Z", "shell.execute_reply": "2020-02-17T05:02:04.022493Z"}}
lr_labels = ['LR', 'LR IVF']

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-17T05:02:04.024277Z", "iopub.execute_input": "2020-02-17T05:02:04.024457Z", "iopub.status.idle": "2020-02-17T05:02:04.342011Z", "shell.execute_reply.started": "2020-02-17T05:02:04.024428Z", "shell.execute_reply": "2020-02-17T05:02:04.338741Z"}}
in_out_df.loc[in_out_df.celllabel.isin(lr_labels), 'celllabel'] = 'lr'

# + {"Collapsed": "false", "persistent_id": "ba513734-4c7e-455d-bfa8-9b805c59b530", "execution": {"iopub.status.busy": "2020-02-17T05:02:04.343396Z", "iopub.execute_input": "2020-02-17T05:02:04.343644Z", "iopub.status.idle": "2020-02-17T05:02:04.842099Z", "shell.execute_reply.started": "2020-02-17T05:02:04.343597Z", "shell.execute_reply": "2020-02-17T05:02:04.841477Z"}}
in_out_df.celllabel.value_counts()

# + [markdown] {"Collapsed": "false"}
# ### Distinguish intake from output data
#
# For each label, separate into an intake and an output feature.

# + [markdown] {"Collapsed": "false"}
# Get intake / output indicator:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-17T05:02:04.843179Z", "iopub.execute_input": "2020-02-17T05:02:04.843385Z", "iopub.status.idle": "2020-02-17T05:02:10.682008Z", "shell.execute_reply.started": "2020-02-17T05:02:04.843351Z", "shell.execute_reply": "2020-02-17T05:02:10.681229Z"}}
in_out_df['in_out_indctr'] = in_out_df.cellpath.apply(lambda x: du.search_explore.get_element_from_split(x, 3,
                                                                                                         separator='|'))
in_out_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-17T05:02:10.683601Z", "iopub.execute_input": "2020-02-17T05:02:10.683853Z", "iopub.status.idle": "2020-02-17T05:02:11.342945Z", "shell.execute_reply.started": "2020-02-17T05:02:10.683804Z", "shell.execute_reply": "2020-02-17T05:02:11.342209Z"}}
in_out_df['in_out_indctr'].value_counts()

# + [markdown] {"Collapsed": "false"}
# Add the appropriate intake or output indication to each label:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-17T05:02:11.343865Z", "iopub.execute_input": "2020-02-17T05:02:11.344213Z", "iopub.status.idle": "2020-02-17T05:02:12.514437Z", "shell.execute_reply.started": "2020-02-17T05:02:11.344175Z", "shell.execute_reply": "2020-02-17T05:02:12.513561Z"}}
in_out_df.loc[in_out_df.in_out_indctr == 'Intake (ml)', 'celllabel'] += '_intake'

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-17T05:02:12.516309Z", "iopub.execute_input": "2020-02-17T05:02:12.516559Z", "iopub.status.idle": "2020-02-17T05:02:13.917166Z", "shell.execute_reply.started": "2020-02-17T05:02:12.516522Z", "shell.execute_reply": "2020-02-17T05:02:13.916356Z"}}
in_out_df.loc[in_out_df.in_out_indctr == 'Output (ml)', 'celllabel'] += '_output'

# + {"Collapsed": "false", "persistent_id": "ba513734-4c7e-455d-bfa8-9b805c59b530", "execution": {"iopub.status.busy": "2020-02-17T05:02:13.929187Z", "iopub.execute_input": "2020-02-17T05:02:13.929540Z", "iopub.status.idle": "2020-02-17T05:02:14.607928Z", "shell.execute_reply.started": "2020-02-17T05:02:13.929484Z", "shell.execute_reply": "2020-02-17T05:02:14.607287Z"}}
in_out_df.celllabel.value_counts()

# + [markdown] {"Collapsed": "false"}
# Remove the now unneeded intake / output indicator and path columns:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-17T05:02:14.610469Z", "iopub.execute_input": "2020-02-17T05:02:14.610695Z", "iopub.status.idle": "2020-02-17T05:02:15.240547Z", "shell.execute_reply.started": "2020-02-17T05:02:14.610655Z", "shell.execute_reply": "2020-02-17T05:02:15.239754Z"}}
in_out_df = in_out_df.drop(columns=['in_out_indctr', 'cellpath'])
in_out_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Convert categories to features

# + [markdown] {"Collapsed": "false"}
# Transform the `celllabel` categories and `cellvaluenumeric` values into separate features:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-17T05:02:15.242249Z", "iopub.execute_input": "2020-02-17T05:02:15.242668Z", "iopub.status.idle": "2020-02-17T05:40:20.613986Z", "shell.execute_reply.started": "2020-02-17T05:02:15.242622Z", "shell.execute_reply": "2020-02-17T05:40:20.608604Z"}}
in_out_df = du.data_processing.category_to_feature(in_out_df, categories_feature='celllabel',
                                                   values_feature='cellvaluenumeric', min_len=1000, inplace=True)
in_out_df.head()

# + [markdown] {"Collapsed": "false"}
# Now we have the categories separated into their own features, as desired.

# + [markdown] {"Collapsed": "false"}
# Remove the old `celllabel` and `cellvaluenumeric` columns:

# + {"Collapsed": "false", "persistent_id": "d42cb91d-3a85-4ad6-a3c0-4e87c951781f", "execution": {"iopub.status.busy": "2020-02-17T05:40:20.626343Z", "iopub.execute_input": "2020-02-17T05:40:20.626852Z", "iopub.status.idle": "2020-02-17T05:40:24.732761Z", "shell.execute_reply.started": "2020-02-17T05:40:20.626680Z", "shell.execute_reply": "2020-02-17T05:40:24.731877Z"}}
in_out_df = in_out_df.drop(['celllabel', 'cellvaluenumeric'], axis=1)
in_out_df.head()

# + {"Collapsed": "false", "persistent_id": "2be490c2-82d9-44dc-897f-6417e41cfe96", "execution": {"iopub.status.busy": "2020-02-17T05:40:24.734469Z", "iopub.execute_input": "2020-02-17T05:40:24.734735Z", "iopub.status.idle": "2020-02-17T05:40:24.832369Z", "shell.execute_reply.started": "2020-02-17T05:40:24.734695Z", "shell.execute_reply": "2020-02-17T05:40:24.831510Z"}}
in_out_df['urine_output'].value_counts()

# + {"Collapsed": "false", "persistent_id": "cc5a84b3-994f-490a-97d5-4f4edf5b0497", "execution": {"iopub.status.busy": "2020-02-17T05:40:24.833773Z", "iopub.execute_input": "2020-02-17T05:40:24.834017Z", "iopub.status.idle": "2020-02-17T05:40:24.878771Z", "shell.execute_reply.started": "2020-02-17T05:40:24.833960Z", "shell.execute_reply": "2020-02-17T05:40:24.877891Z"}}
in_out_df['oral_intake'].value_counts()

# + {"Collapsed": "false", "persistent_id": "6488d69a-45e1-4b67-9d1c-1ce6e6c7f31a", "execution": {"iopub.status.busy": "2020-02-17T05:40:24.880664Z", "iopub.execute_input": "2020-02-17T05:40:24.881133Z", "iopub.status.idle": "2020-02-17T05:40:24.951042Z", "shell.execute_reply.started": "2020-02-17T05:40:24.880930Z", "shell.execute_reply": "2020-02-17T05:40:24.949959Z"}}
in_out_df['Bodyweight (kg)'].value_counts()

# + {"Collapsed": "false", "persistent_id": "6ae6e48f-2b69-445a-9dd7-f5875e8d1cd5", "execution": {"iopub.status.busy": "2020-02-17T05:40:24.952347Z", "iopub.execute_input": "2020-02-17T05:40:24.952874Z", "iopub.status.idle": "2020-02-17T05:40:24.995031Z", "shell.execute_reply.started": "2020-02-17T05:40:24.952735Z", "shell.execute_reply": "2020-02-17T05:40:24.994422Z"}}
in_out_df['Stool_output'].value_counts()

# + [markdown] {"Collapsed": "false"}
# ### Create the timestamp feature and sort

# + [markdown] {"Collapsed": "false"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "d0647504-f554-4d1f-8eba-d87851eb5695", "execution": {"iopub.status.busy": "2020-02-17T05:40:24.996161Z", "iopub.execute_input": "2020-02-17T05:40:24.996468Z", "iopub.status.idle": "2020-02-17T05:40:26.773137Z", "shell.execute_reply.started": "2020-02-17T05:40:24.996420Z", "shell.execute_reply": "2020-02-17T05:40:26.772572Z"}}
in_out_df = in_out_df.rename(columns={'intakeoutputoffset': 'ts'})
in_out_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "a4cc356c-5c63-41e3-b906-20a4d51ef912", "execution": {"iopub.status.busy": "2020-02-17T05:40:26.774682Z", "iopub.execute_input": "2020-02-17T05:40:26.774878Z", "iopub.status.idle": "2020-02-17T05:40:26.779641Z", "shell.execute_reply.started": "2020-02-17T05:40:26.774844Z", "shell.execute_reply": "2020-02-17T05:40:26.779041Z"}}
len(in_out_df)

# + {"Collapsed": "false", "persistent_id": "0ed1deff-4607-4456-af14-95ae472a6e05", "execution": {"iopub.status.busy": "2020-02-17T05:40:26.780926Z", "iopub.execute_input": "2020-02-17T05:40:26.781558Z", "iopub.status.idle": "2020-02-17T05:40:37.994475Z", "shell.execute_reply.started": "2020-02-17T05:40:26.781354Z", "shell.execute_reply": "2020-02-17T05:40:37.993349Z"}}
in_out_df = in_out_df.drop_duplicates()
in_out_df.head()

# + {"Collapsed": "false", "persistent_id": "e864c533-0f0d-4bde-9021-2302ea459260", "execution": {"iopub.status.busy": "2020-02-17T05:40:37.998533Z", "iopub.status.idle": "2020-02-17T05:40:38.005650Z", "iopub.execute_input": "2020-02-17T05:40:37.998986Z", "shell.execute_reply.started": "2020-02-17T05:40:37.998939Z", "shell.execute_reply": "2020-02-17T05:40:38.004926Z"}}
len(in_out_df)

# + [markdown] {"Collapsed": "false"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "9fbb6262-2cdc-4809-a2b2-ce8847793cca", "execution": {"iopub.status.busy": "2020-02-17T05:40:38.006608Z", "iopub.status.idle": "2020-02-17T05:40:41.071558Z", "iopub.execute_input": "2020-02-17T05:40:38.006883Z", "shell.execute_reply.started": "2020-02-17T05:40:38.006836Z", "shell.execute_reply": "2020-02-17T05:40:41.070973Z"}}
in_out_df = in_out_df.sort_values('ts')
in_out_df.head()

# + [markdown] {"Collapsed": "false"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "ec32dcd9-8bec-4077-9392-0f7430ddaae2", "execution": {"iopub.status.busy": "2020-02-17T13:05:28.716757Z", "iopub.status.idle": "2020-02-17T13:05:38.827012Z", "iopub.execute_input": "2020-02-17T13:05:28.718108Z", "shell.execute_reply.started": "2020-02-17T13:05:28.718013Z", "shell.execute_reply": "2020-02-17T13:05:38.824489Z"}}
in_out_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='urine_output', n=5).head()

# + {"Collapsed": "false", "persistent_id": "9baab531-96da-40e3-8bde-c4657bdc950e", "execution": {"iopub.status.busy": "2020-02-17T13:06:04.691353Z", "iopub.status.idle": "2020-02-17T13:06:04.864768Z", "iopub.execute_input": "2020-02-17T13:06:04.691719Z", "shell.execute_reply.started": "2020-02-17T13:06:04.691648Z", "shell.execute_reply": "2020-02-17T13:06:04.863484Z"}}
in_out_df[(in_out_df.patientunitstayid == 433661) & (in_out_df.ts == 661)].head(10)

# + [markdown] {"Collapsed": "false"}
# We can see that there are up to 2 rows per set of `patientunitstayid` and `ts`. As such, we must join them. However, this is a different scenario than in the other cases. Since all features are numeric, we just need to average the features.

# + [markdown] {"Collapsed": "false"}
# ### Join rows that have the same IDs

# + [markdown] {"Collapsed": "false"}
# Convert dataframe to Pandas, as the groupby operation in `join_repeated_rows` isn't working properly with Modin:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-17T13:06:53.787183Z", "iopub.status.idle": "2020-02-17T13:06:55.306457Z", "iopub.execute_input": "2020-02-17T13:06:53.787521Z", "shell.execute_reply.started": "2020-02-17T13:06:53.787460Z", "shell.execute_reply": "2020-02-17T13:06:55.305810Z"}}
in_out_df, pd = du.utils.convert_dataframe(in_out_df, to='pandas')

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-17T13:06:55.307563Z", "iopub.status.idle": "2020-02-17T13:06:55.311534Z", "iopub.execute_input": "2020-02-17T13:06:55.307774Z", "shell.execute_reply.started": "2020-02-17T13:06:55.307734Z", "shell.execute_reply": "2020-02-17T13:06:55.310874Z"}}
type(in_out_df)

# + {"Collapsed": "false", "persistent_id": "589931b8-fe11-439a-8b14-4857c168c023", "pixiedust": {"displayParams": {}}, "execution": {"iopub.status.busy": "2020-02-17T13:06:55.313155Z", "iopub.status.idle": "2020-02-17T13:07:14.296492Z", "iopub.execute_input": "2020-02-17T13:06:55.313629Z", "shell.execute_reply.started": "2020-02-17T13:06:55.313461Z", "shell.execute_reply": "2020-02-17T13:07:14.295401Z"}}
in_out_df = du.embedding.join_repeated_rows(in_out_df, cont_join_method='mean', inplace=True)
in_out_df.head()

# + [markdown] {"Collapsed": "false"}
# Reconvert dataframe to Modin:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-17T13:07:14.300378Z", "iopub.status.idle": "2020-02-17T13:10:34.431077Z", "iopub.execute_input": "2020-02-17T13:07:14.301090Z", "shell.execute_reply.started": "2020-02-17T13:07:14.300934Z", "shell.execute_reply": "2020-02-17T13:10:34.429369Z"}}
in_out_df, pd = du.utils.convert_dataframe(in_out_df, to='modin')

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-17T13:10:34.435529Z", "iopub.status.idle": "2020-02-17T13:10:34.445437Z", "iopub.execute_input": "2020-02-17T13:10:34.435944Z", "shell.execute_reply.started": "2020-02-17T13:10:34.435872Z", "shell.execute_reply": "2020-02-17T13:10:34.444418Z"}}
type(in_out_df)

# + {"Collapsed": "false", "persistent_id": "0b782718-8a92-4780-abbe-f8cab9efdfce", "execution": {"iopub.status.busy": "2020-02-17T13:10:34.446912Z", "iopub.status.idle": "2020-02-17T13:10:34.494875Z", "iopub.execute_input": "2020-02-17T13:10:34.447284Z", "shell.execute_reply.started": "2020-02-17T13:10:34.447106Z", "shell.execute_reply": "2020-02-17T13:10:34.494006Z"}}
in_out_df.dtypes

# + {"Collapsed": "false", "persistent_id": "1d5d3435-7ebf-4d13-8fdb-ac1a09f847b3", "execution": {"iopub.status.busy": "2020-02-17T13:10:34.496111Z", "iopub.status.idle": "2020-02-17T13:11:23.825486Z", "iopub.execute_input": "2020-02-17T13:10:34.496415Z", "shell.execute_reply.started": "2020-02-17T13:10:34.496374Z", "shell.execute_reply": "2020-02-17T13:11:23.823721Z"}}
in_out_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='urine_output', n=5).head()

# + {"Collapsed": "false", "persistent_id": "9baab531-96da-40e3-8bde-c4657bdc950e", "execution": {"iopub.status.busy": "2020-02-17T13:11:23.835934Z", "iopub.status.idle": "2020-02-17T13:11:26.094489Z", "iopub.execute_input": "2020-02-17T13:11:23.836294Z", "shell.execute_reply.started": "2020-02-17T13:11:23.836152Z", "shell.execute_reply": "2020-02-17T13:11:26.093554Z"}}
in_out_df[(in_out_df.patientunitstayid == 433661) & (in_out_df.ts == 661)].head(10)

# + [markdown] {"Collapsed": "false"}
# Comparing the output from the two previous cells with what we had before the `join_repeated_rows` method, we can see that all rows with duplicate IDs have been successfully joined.

# + [markdown] {"Collapsed": "false"}
# ### Rename columns

# + {"Collapsed": "false", "persistent_id": "9b090273-1c22-4122-9979-8ee2b91f0dfe", "execution": {"iopub.status.busy": "2020-02-17T13:11:26.095862Z", "iopub.status.idle": "2020-02-17T13:11:26.557065Z", "iopub.execute_input": "2020-02-17T13:11:26.096156Z", "shell.execute_reply.started": "2020-02-17T13:11:26.096037Z", "shell.execute_reply": "2020-02-17T13:11:26.556161Z"}}
in_out_df = in_out_df.rename(columns={'Out': 'dialysis_output',
                                      'Indwelling Catheter Output_output': 'indwellingcatheter_output',
                                      'Voided Amount_output' : 'voided_amount',
                                      'Feeding Tube Flush/Water Bolus Amt (mL)_intake': 'feeding_tube_flush_ml',
                                      'Volume Given (mL)_intake': 'volume_given_ml',
                                      'Actual Patient Fluid Removal_output': 'patient_fluid_removal'})
in_out_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "a12ee2e9-13d9-44ce-999c-846f320e8bfd", "execution": {"iopub.status.busy": "2020-02-17T13:11:26.558252Z", "iopub.status.idle": "2020-02-17T13:11:27.195860Z", "iopub.execute_input": "2020-02-17T13:11:26.558550Z", "shell.execute_reply.started": "2020-02-17T13:11:26.558428Z", "shell.execute_reply": "2020-02-17T13:11:27.195042Z"}}
in_out_df.columns = du.data_processing.clean_naming(in_out_df.columns)
in_out_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Normalize data

# + [markdown] {"Collapsed": "false"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "584b72a4-0de9-433b-b27f-6304a0db2b52", "execution": {"iopub.status.busy": "2020-02-17T13:11:27.198571Z", "iopub.status.idle": "2020-02-17T13:13:13.019108Z", "iopub.execute_input": "2020-02-17T13:11:27.198866Z", "shell.execute_reply.started": "2020-02-17T13:11:27.198814Z", "shell.execute_reply": "2020-02-17T13:13:13.016364Z"}}
in_out_df.to_csv(f'{data_path}cleaned/unnormalized/ohe/intakeOutput.csv')

# + {"Collapsed": "false", "persistent_id": "5d512225-ad7e-40b4-a091-b18df3f38c4c", "pixiedust": {"displayParams": {}}, "execution": {"iopub.status.busy": "2020-02-17T13:53:03.465686Z", "iopub.status.idle": "2020-02-17T13:59:29.109568Z", "iopub.execute_input": "2020-02-17T13:53:03.466969Z", "shell.execute_reply.started": "2020-02-17T13:53:03.466862Z", "shell.execute_reply": "2020-02-17T13:59:29.108001Z"}}
in_out_df_norm, mean, std = du.data_processing.normalize_data(in_out_df, get_stats=True, inplace=True)
in_out_df_norm.head()

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
stream = open(f'{data_path}/cleaned/intakeOutput_norm_stats.yaml', 'w')
yaml.dump(norm_stats, stream, default_flow_style=False)

# + [markdown] {"Collapsed": "false"}
# ### Save the dataframe

# + [markdown] {"Collapsed": "false"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "b9f11ee9-cd10-44df-961f-c7bc3642710f", "execution": {"iopub.status.busy": "2020-02-17T13:59:29.114503Z", "iopub.status.idle": "2020-02-17T14:01:01.338813Z", "iopub.execute_input": "2020-02-17T13:59:29.114845Z", "shell.execute_reply.started": "2020-02-17T13:59:29.114780Z", "shell.execute_reply": "2020-02-17T14:01:01.336011Z"}}
in_out_df.to_csv(f'{data_path}cleaned/normalized/ohe/intakeOutput.csv')

# + [markdown] {"Collapsed": "false"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "edafc000-e516-49fe-b8f6-ea4ce3969129", "execution": {"iopub.status.busy": "2020-02-17T14:01:01.343162Z", "iopub.status.idle": "2020-02-17T14:01:10.765769Z", "iopub.execute_input": "2020-02-17T14:01:01.343489Z", "shell.execute_reply.started": "2020-02-17T14:01:01.343420Z", "shell.execute_reply": "2020-02-17T14:01:10.763386Z"}}
in_out_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "3cc624bc-6dac-4cd0-9d64-f329e29940fa", "pixiedust": {"displayParams": {}}, "execution": {"iopub.status.busy": "2020-02-17T14:01:22.734862Z", "iopub.execute_input": "2020-02-17T14:01:22.735206Z", "iopub.status.idle": "2020-02-17T14:01:23.631807Z", "shell.execute_reply.started": "2020-02-17T14:01:22.735046Z", "shell.execute_reply": "2020-02-17T14:01:23.630757Z"}}
du.search_explore.dataframe_missing_values(in_out_df)

# + {"Collapsed": "false"}
