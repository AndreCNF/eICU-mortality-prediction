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
# # Nursing Data Preprocessing
# ---
#
# Reading and preprocessing nursing data of the eICU dataset from MIT with the data from over 139k patients collected in the US.
#
# This notebook addresses the preprocessing of the following eICU tables:
# * nurseAssessment
# * nurseCare
# * nurseCharting

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
import modin.pandas as pd                  # Optimized distributed version of Pandas
import data_utils as du                    # Data science and machine learning relevant methods

# + {"Collapsed": "false", "cell_type": "markdown"}
# Set the random seed for reproducibility

# + {"Collapsed": "false", "persistent_id": "39b552cd-6948-4ec8-ac04-42f850c1e05a", "last_executed_text": "du.set_random_seed(42)", "execution_event_id": "29ab85ce-b7fd-4c5a-a110-5841e741c369"}
du.set_random_seed(42)

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Nurse care data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Initialize variables

# + {"Collapsed": "false", "persistent_id": "754a96f8-d389-4968-8c13-52e5e9d0bf82"}
# List of categorical features
cat_feat = []
# Dictionary of the one hot encoded columns originary from each categorical feature, that will be embedded
cat_feat_ohe = dict()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "4a37073a-b357-4079-b6af-72125689781d"}
nurse_care_df = pd.read_csv(f'{data_path}original/nurseCare.csv')
nurse_care_df.head()

# + {"Collapsed": "false", "persistent_id": "6566d20f-9fcb-4d92-879b-55a0cefe54ae"}
len(nurse_care_df)

# + {"Collapsed": "false", "persistent_id": "e267e007-4b72-4551-a9d2-7c916956235c"}
nurse_care_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Only 13052 unit stays have nurse care data. Might not be useful to include them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "b4f282b2-e5f4-4cfb-83c5-c79944367d70"}
nurse_care_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "6a382a6d-709e-4cdf-a586-c0cc751ff853"}
nurse_care_df.columns

# + {"Collapsed": "false", "persistent_id": "f06e6c16-7c54-40c0-9055-0d8717661e4c"}
nurse_care_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "3cc624bc-6dac-4cd0-9d64-f329e29940fa"}
du.search_explore.dataframe_missing_values(nurse_care_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "ba513734-4c7e-455d-bfa8-9b805c59b530"}
nurse_care_df.celllabel.value_counts()

# + {"Collapsed": "false", "persistent_id": "4d028a9e-38e8-496e-805a-ed3f2304a07f"}
nurse_care_df.cellattribute.value_counts()

# + {"Collapsed": "false", "persistent_id": "d61a9031-35b3-4c51-a141-a820f6ed70a8"}
nurse_care_df.cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "502ee5b2-8eba-407e-a0c3-158c27cb9fab"}
nurse_care_df.cellattributepath.value_counts()

# + {"Collapsed": "false", "persistent_id": "056c6696-d254-4012-9446-cad509788c0b"}
nurse_care_df[nurse_care_df.celllabel == 'Nutrition'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "f7be37c4-ee8a-4e58-8cf9-ee17cf36f46c"}
nurse_care_df[nurse_care_df.celllabel == 'Activity'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "e39d2950-3c4c-4895-b5f4-8e865c927cb7"}
nurse_care_df[nurse_care_df.celllabel == 'Hygiene/ADLs'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "6da5000e-cfe8-4179-a0bf-159cb14240e2"}
nurse_care_df[nurse_care_df.celllabel == 'Safety'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "70c3fc3a-e7a8-4e04-b38c-a24a5f45eb91"}
nurse_care_df[nurse_care_df.celllabel == 'Treatments'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "443bb56c-2c54-4af7-901f-2d094f736859"}
nurse_care_df[nurse_care_df.celllabel == 'Isolation Precautions'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "dc2db80c-0a0c-436d-9db5-105ef2bd28b8"}
nurse_care_df[nurse_care_df.celllabel == 'Restraints'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "8c71653e-7b4f-4afc-bcf3-46b0fa1f013c"}
nurse_care_df[nurse_care_df.celllabel == 'Equipment'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Besides the usual removal of row identifier, `nursecareid`, and the timestamp when data was added, `nursecareentryoffset`, I'm also removing `cellattributepath` and `cellattribute`, which have redundant info with `celllabel`.

# + {"Collapsed": "false", "persistent_id": "48f10eea-6c36-4188-91d9-4def749f2486"}
nurse_care_df = nurse_care_df.drop(['nursecareid', 'nursecareentryoffset',
                                  'cellattributepath', 'cellattribute'], axis=1)
nurse_care_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Additionally, some information like "Equipment" and "Restraints" seem to be unnecessary. So let's remove them:

# + {"Collapsed": "false", "persistent_id": "6c7fa8f7-3195-46a9-889a-6034ffdcaef3"}
categories_to_remove = ['Safety', 'Restraints', 'Equipment', 'Airway Type',
                        'Isolation Precautions', 'Airway Size']

# + {"Collapsed": "false", "persistent_id": "d8b7cf79-bda7-411f-a800-e4769eb0ec00"}
~(nurse_care_df.celllabel.isin(categories_to_remove)).head()

# + {"Collapsed": "false", "persistent_id": "8f0a890f-ad85-457c-888e-37f9cddfe4e8"}
nurse_care_df = nurse_care_df[~(nurse_care_df.celllabel.isin(categories_to_remove))]
nurse_care_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Convert categories to features

# + {"Collapsed": "false", "cell_type": "markdown"}
# Transform the `celllabel` categories and `cellattributevalue` values into separate features:

# + {"Collapsed": "false"}
nurse_care_df = du.data_processing.category_to_feature(nurse_care_df, categories_feature='celllabel',
                                                      values_feature='cellattributevalue', min_len=1000, inplace=True)
nurse_care_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Now we have the categories separated into their own features, as desired.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the old `celllabel` and `cellattributevalue` columns:

# + {"Collapsed": "false", "persistent_id": "d42cb91d-3a85-4ad6-a3c0-4e87c951781f"}
nurse_care_df = nurse_care_df.drop(['celllabel', 'cellattributevalue'], axis=1)
nurse_care_df.head()

# + {"Collapsed": "false", "persistent_id": "2be490c2-82d9-44dc-897f-6417e41cfe96"}
nurse_care_df['Nutrition'].value_counts()

# + {"Collapsed": "false", "persistent_id": "cc5a84b3-994f-490a-97d5-4f4edf5b0497"}
nurse_care_df['Treatments'].value_counts()

# + {"Collapsed": "false", "persistent_id": "6488d69a-45e1-4b67-9d1c-1ce6e6c7f31a"}
nurse_care_df['Hygiene/ADLs'].value_counts()

# + {"Collapsed": "false", "persistent_id": "6ae6e48f-2b69-445a-9dd7-f5875e8d1cd5"}
nurse_care_df['Activity'].value_counts()

# + {"Collapsed": "false", "toc-hr-collapsed": false, "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into one hot encode columns, which can later be embedded or used as is.

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### One hot encode features

# + {"Collapsed": "false", "cell_type": "markdown"}
# Update list of categorical features:

# + {"Collapsed": "false", "persistent_id": "9836b018-de91-4b41-b107-cb8c9779b4c5"}
cat_feat = ['Nutrition', 'Treatments', 'Hygiene/ADLs', 'Activity']

# + {"Collapsed": "false", "persistent_id": "19066f2a-58d5-4edb-a33d-6d830424f40c"}
nurse_care_df[cat_feat].head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "51c18fe2-7b70-41c1-95d2-f423b3d7836b"}
nurse_care_df = du.data_processing.one_hot_encoding_dataframe(nurse_care_df, columns=cat_feat, join_rows=False,
                                                              join_by=['patientunitstayid', 'drugoffset'])
nurse_care_df

# + {"Collapsed": "false", "persistent_id": "9ceba9e2-6821-4a73-8875-5ddebef03516"}
nurse_care_df[cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "56d49f40-97fe-47d5-ad9d-ef35a4453266"}
cat_feat_ohe

# + {"Collapsed": "false", "persistent_id": "49f71013-ebc1-472e-b91b-2a96233b207b"}
nurse_care_df[cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the association between the original categorical features and the new one hot encoded columns:

# + {"Collapsed": "false", "persistent_id": "26eac7f3-9081-4a96-ae4a-40054c223fd7", "execution": {"iopub.status.busy": "2020-03-09T16:37:35.157248Z", "iopub.execute_input": "2020-03-09T16:37:35.157526Z", "iopub.status.idle": "2020-03-09T16:37:35.164656Z", "shell.execute_reply.started": "2020-03-09T16:37:35.157493Z", "shell.execute_reply": "2020-03-09T16:37:35.163771Z"}}
for orig_col in cat_feat:
    cat_feat_ohe[orig_col] = [ohe_col for ohe_col in new_columns
                              if ohe_col.startswith(orig_col)]

# + {"execution": {"iopub.status.busy": "2020-03-09T16:37:35.165864Z", "iopub.execute_input": "2020-03-09T16:37:35.166280Z", "iopub.status.idle": "2020-03-09T16:37:35.190294Z", "shell.execute_reply.started": "2020-03-09T16:37:35.166256Z", "shell.execute_reply": "2020-03-09T16:37:35.189358Z"}, "Collapsed": "false"}
cat_feat_ohe

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "398e3a5f-c8e3-4657-aa40-05967570fd66"}
stream = open(f'{data_path}/cleaned/cat_feat_ohe_nurse_care.yaml', 'w')
yaml.dump(cat_feat_ohe, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "d0647504-f554-4d1f-8eba-d87851eb5695"}
nurse_care_df = nurse_care_df.rename(columns={'nursecareoffset': 'ts'})
nurse_care_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "a4cc356c-5c63-41e3-b906-20a4d51ef912"}
len(nurse_care_df)

# + {"Collapsed": "false", "persistent_id": "0ed1deff-4607-4456-af14-95ae472a6e05"}
nurse_care_df = nurse_care_df.drop_duplicates()
nurse_care_df.head()

# + {"Collapsed": "false", "persistent_id": "e864c533-0f0d-4bde-9021-2302ea459260"}
len(nurse_care_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "9fbb6262-2cdc-4809-a2b2-ce8847793cca"}
nurse_care_df = nurse_care_df.sort_values('ts')
nurse_care_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "ec32dcd9-8bec-4077-9392-0f7430ddaae2"}
nurse_care_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='Nutrition', n=5).head()

# + {"Collapsed": "false", "persistent_id": "9baab531-96da-40e3-8bde-c4657bdc950e"}
nurse_care_df[nurse_care_df.patientunitstayid == 2798325].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 21 categories per set of `patientunitstayid` and `ts`. As such, we must join them. However, this is a different scenario than in the other cases. Since we created the features from one categorical column, it doesn't have repeated values, only different rows to indicate each of the new features' values. As such, we just need to sum the features.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"Collapsed": "false", "cell_type": "markdown"}
# Convert dataframe to Pandas, as the groupby operation in `join_repeated_rows` isn't working properly with Modin:

# + {"Collapsed": "false"}
nurse_care_df, pd = du.utils.convert_dataframe(nurse_care_df, to='pandas')

# + {"Collapsed": "false"}
type(nurse_care_df)

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "589931b8-fe11-439a-8b14-4857c168c023"}
nurse_care_df = du.embedding.join_repeated_rows(nurse_care_df, inplace=True)
nurse_care_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Reconvert dataframe to Modin:

# + {"Collapsed": "false"}
nurse_care_df, pd = du.utils.convert_dataframe(nurse_care_df, to='modin')

# + {"Collapsed": "false"}
type(nurse_care_df)

# + {"Collapsed": "false", "persistent_id": "0b782718-8a92-4780-abbe-f8cab9efdfce"}
nurse_care_df.dtypes

# + {"Collapsed": "false", "persistent_id": "1d5d3435-7ebf-4d13-8fdb-ac1a09f847b3"}
nurse_care_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='Nutrition', n=5).head()

# + {"Collapsed": "false", "persistent_id": "c4269670-2a2a-4e69-8800-ac737eaa3ebc"}
nurse_care_df[nurse_care_df.patientunitstayid == 2798325].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_repeated_rows` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Rename columns

# + {"Collapsed": "false", "persistent_id": "9b090273-1c22-4122-9979-8ee2b91f0dfe"}
nurse_care_df = nurse_care_df.rename(columns={'Treatments':'nurse_treatments'})
nurse_care_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "a12ee2e9-13d9-44ce-999c-846f320e8bfd"}
nurse_care_df.columns = du.data_processing.clean_naming(nurse_care_df.columns)
nurse_care_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "584b72a4-0de9-433b-b27f-6304a0db2b52"}
nurse_care_df.to_csv(f'{data_path}cleaned/unnormalized/ohe/nurseCare.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "b9f11ee9-cd10-44df-961f-c7bc3642710f"}
nurse_care_df.to_csv(f'{data_path}cleaned/normalized/ohe/nurseCare.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "edafc000-e516-49fe-b8f6-ea4ce3969129"}
nurse_care_df.describe().transpose()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Nurse assessment data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Initialize variables

# + {"Collapsed": "false", "persistent_id": "754a96f8-d389-4968-8c13-52e5e9d0bf82"}
# List of categorical features
cat_feat = []
# Dictionary of the one hot encoded columns originary from each categorical feature, that will be embedded
cat_feat_ohe = dict()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "51281341-db28-43a0-8f85-cd3ae11aae06"}
nurse_assess_df = pd.read_csv(f'{data_path}original/nurseAssessment.csv')
nurse_assess_df.head()

# + {"Collapsed": "false", "persistent_id": "6573ec98-4a25-4b83-b875-626d555def4e"}
len(nurse_assess_df)

# + {"Collapsed": "false", "persistent_id": "cd9958e3-dd35-4a18-b2d0-01f5dc59a708"}
nurse_assess_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Only 13001 unit stays have nurse assessment data. Might not be useful to include them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "a63bd49a-1ea0-4f3e-bb25-98040cb31c28"}
nurse_assess_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "baebbad8-8e0a-4a54-92f2-a1a3c8b63e28"}
nurse_assess_df.columns

# + {"Collapsed": "false", "persistent_id": "127f78c7-5fa1-4161-b1aa-085631f74a7b"}
nurse_assess_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "6a7adc72-3d2e-4cd5-841c-299a3a94d043"}
du.search_explore.dataframe_missing_values(nurse_assess_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "38ac8914-ae50-45ce-9585-77263c79af06"}
nurse_assess_df.celllabel.value_counts()

# + {"Collapsed": "false", "persistent_id": "5d345361-e2b7-415f-8b0e-6f706d29c4db"}
nurse_assess_df.cellattribute.value_counts()

# + {"Collapsed": "false", "persistent_id": "bae7f1bf-dd1f-4d3f-8cb0-f70365412fbc"}
nurse_assess_df.cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "50b31b51-bee1-49e6-86eb-c24d97830844"}
nurse_assess_df.cellattributepath.value_counts()

# + {"Collapsed": "false", "persistent_id": "f952c852-c8d4-4313-baa6-e0f0e33adcfc"}
nurse_assess_df[nurse_assess_df.celllabel == 'Intervention'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "1bfe013d-8f9a-496f-9bea-3091e2f01b61"}
nurse_assess_df[nurse_assess_df.celllabel == 'Neurologic'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "8984d2fa-3cac-486e-b029-ab7da972ebef"}
nurse_assess_df[nurse_assess_df.celllabel == 'Pupils'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "463936ef-481d-467e-beba-606f880dce40"}
nurse_assess_df[nurse_assess_df.celllabel == 'Edema'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "55680a6d-7fab-4fce-8eed-66dd44d011f1"}
nurse_assess_df[nurse_assess_df.celllabel == 'Secretions'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "7e511548-989e-4c78-a6bd-a56b4811dd0e"}
nurse_assess_df[nurse_assess_df.celllabel == 'Cough'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "b75398ad-65f5-4aac-a00b-4f1034c1558c"}
nurse_assess_df[nurse_assess_df.celllabel == 'Neurologic'].cellattribute.value_counts()

# + {"Collapsed": "false", "persistent_id": "d0592960-b088-4491-81d7-697cee55df06"}
nurse_assess_df[nurse_assess_df.celllabel == 'Pupils'].cellattribute.value_counts()

# + {"Collapsed": "false", "persistent_id": "b461df0f-36ef-4bdf-9eda-2a9c57bf1a44"}
nurse_assess_df[nurse_assess_df.celllabel == 'Secretions'].cellattribute.value_counts()

# + {"Collapsed": "false", "persistent_id": "c118b8b5-4c96-45d6-b2f5-a2cc2ce8ba93"}
nurse_assess_df[nurse_assess_df.celllabel == 'Cough'].cellattribute.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Besides the usual removal of row identifier, `nurseAssessID`, and the timestamp when data was added, `nurseAssessEntryOffset`, I'm also removing `cellattributepath` and `cellattribute`, which have redundant info with `celllabel`. Regarding data categories, I'm only keeping `Neurologic`, `Pupils`, `Secretions` and `Cough`, as the remaining ones either don't add much value, have too little data or are redundant with data from other tables.

# + {"Collapsed": "false", "persistent_id": "be9a8747-0209-4e96-a55a-5587f1d09882"}
nurse_assess_df = nurse_assess_df.drop(['nurseassessid', 'nurseassessentryoffset',
                                      'cellattributepath', 'cellattribute'], axis=1)
nurse_assess_df.head()

# + {"Collapsed": "false", "persistent_id": "1a520f05-7d41-4a71-a9ee-a7c359c7b84f"}
categories_to_keep = ['Neurologic', 'Pupils', 'Secretions', 'Cough']

# + {"Collapsed": "false", "persistent_id": "be99810d-b848-4966-99a8-e31c9189d173"}
nurse_assess_df.celllabel.isin(categories_to_keep).head()

# + {"Collapsed": "false", "persistent_id": "288e8cc5-c11e-4742-950c-ee2acb0d0d41"}
nurse_assess_df = nurse_assess_df[nurse_assess_df.celllabel.isin(categories_to_keep)]
nurse_assess_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Convert categories to features

# + {"Collapsed": "false", "cell_type": "markdown"}
# Transform the `celllabel` categories and `cellattributevalue` values into separate features:

# + {"Collapsed": "false"}
nurse_assess_df = du.data_processing.category_to_feature(nurse_assess_df, categories_feature='celllabel',
                                                        values_feature='cellattributevalue', min_len=1000, inplace=True)
nurse_assess_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Now we have the categories separated into their own features, as desired.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the old `celllabel` and `cellattributevalue` columns:

# + {"Collapsed": "false", "persistent_id": "d4762de3-df60-4c93-acb4-125d08c31c45"}
nurse_assess_df = nurse_assess_df.drop(['celllabel', 'cellattributevalue'], axis=1)
nurse_assess_df.head()

# + {"Collapsed": "false", "persistent_id": "b789153f-6a30-43dc-b8cc-5775b21dd7c1"}
nurse_assess_df['Neurologic'].value_counts()

# + {"Collapsed": "false", "persistent_id": "7bbe2fb5-20fa-4ff5-b7f0-f5fb2a7f26e9"}
nurse_assess_df['Pupils'].value_counts()

# + {"Collapsed": "false", "persistent_id": "465fe194-8649-49f5-9bf5-0943e2a29d9e"}
nurse_assess_df['Secretions'].value_counts()

# + {"Collapsed": "false", "persistent_id": "ef065236-1712-4dc1-aef0-f383c93a0251"}
nurse_assess_df['Cough'].value_counts()

# + {"Collapsed": "false", "toc-hr-collapsed": false, "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into one hot encode columns, which can later be embedded or used as is.

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### One hot encode features

# + {"Collapsed": "false", "cell_type": "markdown"}
# Update list of categorical features:

# + {"Collapsed": "false", "persistent_id": "93a69b04-f60a-456e-b715-6428429c34b3"}
cat_feat = ['Pupils', 'Neurologic', 'Secretions', 'Cough']

# + {"Collapsed": "false", "persistent_id": "29703baf-cba2-4f41-86e5-74912610503c"}
nurse_assess_df[cat_feat].head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "c7f73853-ecf8-409e-9351-191e07444213"}
nurse_assess_df = du.data_processing.one_hot_encoding_dataframe(nurse_assess_df, columns=cat_feat, join_rows=False,
                                                                join_by=['patientunitstayid', 'drugoffset'])
nurse_assess_df

# + {"Collapsed": "false", "persistent_id": "6101e468-8fdc-48c2-90f7-7a8db94c1b58"}
nurse_assess_df[cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "b21d68b6-f26b-469c-9139-b746f027758d"}
cat_feat_ohe

# + {"Collapsed": "false", "persistent_id": "9951459d-c61d-49cf-a9d7-630ced7dfef6"}
nurse_assess_df[cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the association between the original categorical features and the new one hot encoded columns:

# + {"Collapsed": "false", "persistent_id": "26eac7f3-9081-4a96-ae4a-40054c223fd7", "execution": {"iopub.status.busy": "2020-03-09T16:37:35.157248Z", "iopub.execute_input": "2020-03-09T16:37:35.157526Z", "iopub.status.idle": "2020-03-09T16:37:35.164656Z", "shell.execute_reply.started": "2020-03-09T16:37:35.157493Z", "shell.execute_reply": "2020-03-09T16:37:35.163771Z"}}
for orig_col in cat_feat:
    cat_feat_ohe[orig_col] = [ohe_col for ohe_col in new_columns
                              if ohe_col.startswith(orig_col)]

# + {"execution": {"iopub.status.busy": "2020-03-09T16:37:35.165864Z", "iopub.execute_input": "2020-03-09T16:37:35.166280Z", "iopub.status.idle": "2020-03-09T16:37:35.190294Z", "shell.execute_reply.started": "2020-03-09T16:37:35.166256Z", "shell.execute_reply": "2020-03-09T16:37:35.189358Z"}, "Collapsed": "false"}
cat_feat_ohe

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "0280b97a-1137-433b-a54e-6168edc4a350"}
stream = open(f'{data_path}/cleaned/cat_feat_ohe_nurse_assess.yaml', 'w')
yaml.dump(cat_feat_ohe, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "b1781127-4235-4097-9bb0-5d7dfa965735"}
nurse_assess_df = nurse_assess_df.rename(columns={'nurseassessoffset': 'ts'})
nurse_assess_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "3483fd1d-3ad6-4e93-a583-5ddeb38ad6c5"}
len(nurse_assess_df)

# + {"Collapsed": "false", "persistent_id": "ffb1df48-8666-41c0-9a74-32f39d8d37cf"}
nurse_assess_df = nurse_assess_df.drop_duplicates()
nurse_assess_df.head()

# + {"Collapsed": "false", "persistent_id": "a7dbc589-c48e-45a6-9044-67b586b11fad"}
len(nurse_assess_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "c8ba14f1-d5d7-4e2b-be19-3ec645351c8e"}
nurse_assess_df = nurse_assess_df.sort_values('ts')
nurse_assess_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "28bb9396-7d50-42a9-a807-b4408c62f815"}
nurse_assess_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='Cough', n=5).head()

# + {"Collapsed": "false", "persistent_id": "473b8ca5-32f0-45e0-ad54-c3b7ae008653"}
nurse_assess_df[nurse_assess_df.patientunitstayid == 2553254].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 80 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"Collapsed": "false", "cell_type": "markdown"}
# Convert dataframe to Pandas, as the groupby operation in `join_repeated_rows` isn't working properly with Modin:

# + {"Collapsed": "false"}
nurse_assess_df, pd = du.utils.convert_dataframe(nurse_assess_df, to='pandas')

# + {"Collapsed": "false"}
type(nurse_assess_df)

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "589931b8-fe11-439a-8b14-4857c168c023"}
nurse_assess_df = du.embedding.join_repeated_rows(nurse_assess_df, inplace=True)
nurse_assess_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Reconvert dataframe to Modin:

# + {"Collapsed": "false"}
nurse_assess_df, pd = du.utils.convert_dataframe(nurse_assess_df, to='modin')

# + {"Collapsed": "false"}
type(nurse_assess_df)

# + {"Collapsed": "false", "persistent_id": "decebaec-f14b-4521-adcc-24f485a0a781"}
nurse_assess_df.dtypes

# + {"Collapsed": "false", "persistent_id": "0ba6b50e-69db-489c-a1f1-6f324a662a68"}
nurse_assess_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='Cough', n=5).head()

# + {"Collapsed": "false", "persistent_id": "41c057d5-3d9e-454e-b71b-1f125d63842e"}
nurse_assess_df[nurse_assess_df.patientunitstayid == 2553254].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_repeated_rows` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "8b8eb574-108e-4f73-b8b8-d0314c295d7b"}
nurse_assess_df.columns = du.data_processing.clean_naming(nurse_assess_df.columns)
nurse_assess_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "70d6ca28-9b03-400e-a218-0bab68904b50"}
nurse_assess_df.to_csv(f'{data_path}cleaned/unnormalized/ohe/nurseAssessment.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "c8a9c87e-64db-4852-86ff-8f9952ba63ed"}
nurse_assess_df.to_csv(f'{data_path}cleaned/normalized/ohe/nurseAssessment.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "b009d13b-05d3-4b2b-9b35-1b9bba3dac28"}
nurse_assess_df.describe().transpose()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Nurse charting data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Initialize variables

# + {"Collapsed": "false", "persistent_id": "754a96f8-d389-4968-8c13-52e5e9d0bf82"}
# List of categorical features
cat_feat = []
# Dictionary of the one hot encoded columns originary from each categorical feature, that will be embedded
cat_feat_ohe = dict()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "783f02e4-8d50-40c7-893d-0882142aa077"}
nursechart_df = pd.read_csv(f'{data_path}original/nurseCharting.csv')
nursechart_df.head()

# + {"Collapsed": "false", "persistent_id": "798bd9b4-eba0-460d-b76a-966a0365035f"}
len(nursechart_df)

# + {"Collapsed": "false", "persistent_id": "c793ba47-3387-4135-b7a1-dabd6996127b"}
nursechart_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "71b58f5c-4b80-49ea-afdd-b917e0b8493f"}
nursechart_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "65c28a37-9b44-49cf-aa6c-b36427ada958"}
nursechart_df.columns

# + {"Collapsed": "false", "persistent_id": "09d230f6-738e-4d13-848d-38dcb5019bb1"}
nursechart_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "e6123aeb-7664-449f-b10e-a5e71f5ebb94"}
du.search_explore.dataframe_missing_values(nursechart_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "0f806c43-a1da-45d4-863b-d60298698002"}
nursechart_df.nursingchartcelltypecat.value_counts()

# + {"Collapsed": "false", "persistent_id": "2b3c6041-57b2-4593-b47d-ec1c1a3c57d7"}
nursechart_df.nursingchartcelltypevallabel.value_counts()

# + {"Collapsed": "false", "persistent_id": "13ede680-d05c-4b1e-a0a1-999bf0afc44f"}
nursechart_df.nursingchartcelltypevalname.value_counts()

# + {"Collapsed": "false", "persistent_id": "f36629b0-a2fb-4d29-bf2e-336289dcdc6d"}
nursechart_df.nursingchartvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "32391831-ef36-44a0-8c65-75e2e016feb1"}
nursechart_df[nursechart_df.nursingchartcelltypecat == 'Vital Signs'].nursingchartcelltypevallabel.value_counts()

# + {"Collapsed": "false", "persistent_id": "d086c1c5-475d-475c-8754-047ccdda305b"}
nursechart_df[nursechart_df.nursingchartcelltypecat == 'Scores'].nursingchartcelltypevallabel.value_counts()

# + {"Collapsed": "false", "persistent_id": "81d04b71-3799-4716-a889-1bbcb0e8f9d8"}
nursechart_df[nursechart_df.nursingchartcelltypecat == 'Other Vital Signs and Infusions'].nursingchartcelltypevallabel.value_counts()

# + {"Collapsed": "false", "persistent_id": "088f2234-8e1f-46e5-976d-c24f709a90d0"}
nursechart_df[nursechart_df.nursingchartcelltypecat == 'Vital Signs and Infusions'].nursingchartcelltypevallabel.value_counts()

# + {"Collapsed": "false", "persistent_id": "16f5caf8-c66a-4717-b2d1-c9c675fe3ba2"}
nursechart_df[nursechart_df.nursingchartcelltypecat == 'Invasive'].nursingchartcelltypevallabel.value_counts()

# + {"Collapsed": "false", "persistent_id": "ba37e551-e26f-4a7a-8be6-de1c20b27347"}
nursechart_df[nursechart_df.nursingchartcelltypecat == 'SVO2'].nursingchartcelltypevallabel.value_counts()

# + {"Collapsed": "false", "persistent_id": "5359fdf9-ee02-41f6-aa34-cbedfe7ff2ec"}
nursechart_df[nursechart_df.nursingchartcelltypecat == 'ECG'].nursingchartcelltypevallabel.value_counts()

# + {"Collapsed": "false", "persistent_id": "fd19b633-5509-4c70-98fa-891b137f9393"}
nursechart_df[nursechart_df.nursingchartcelltypevalname == 'Pain Score'].nursingchartcelltypevallabel.value_counts()

# + {"Collapsed": "false", "persistent_id": "aea2ed5f-ea33-4aed-b499-1fd5c5b96a34"}
nursechart_df[nursechart_df.nursingchartcelltypevalname == 'Pain Score'].nursingchartvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "5fc1a1a5-9985-4d01-b739-24a8bbec7efc"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Pain Assessment'].nursingchartcelltypevalname.value_counts()

# + {"Collapsed": "false", "persistent_id": "fb947f4d-4760-428c-860d-281794ca06f1"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Pain Assessment'].nursingchartvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "56eb63c8-e9d4-4813-9bd7-4675c79365df"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Pain Present'].nursingchartcelltypevalname.value_counts()

# + {"Collapsed": "false", "persistent_id": "68dd7c2b-676a-426c-94cc-111d71ca9cf4"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Pain Present'].nursingchartvalue.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Regarding patient's pain information, the only label that seems to be relevant is `Pain Score`. However, it's important to note that this score has different possible measurement systems (`Pain Assessment`). Due to this, we will only consider the most frequent pain scale (`WDL`). `Pain Present` has less information and, as such, is less relevant.

# + {"Collapsed": "false", "persistent_id": "d17f35ec-b431-4f29-a213-167a400293bf"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Glasgow coma score'].nursingchartcelltypevalname.value_counts()

# + {"Collapsed": "false", "persistent_id": "c437935e-e14b-451e-968b-5d648481421d"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Glasgow coma score'].nursingchartvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "20d471d8-65ed-4638-9d47-af47d8aebce3"}
nursechart_df[nursechart_df.nursingchartcelltypevalname == 'GCS Total'].nursingchartcelltypevallabel.value_counts()

# + {"Collapsed": "false", "persistent_id": "69dddaf1-35f2-43db-adf4-291f30573ed8"}
nursechart_df[nursechart_df.nursingchartcelltypevalname == 'GCS Total'].nursingchartvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "043624e8-be8c-4d26-9da7-2bed76ea87b2"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Score (Glasgow Coma Scale)'].nursingchartcelltypevalname.value_counts()

# + {"Collapsed": "false", "persistent_id": "062ae51d-9653-40a4-b0cd-41eeecab2c86"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Score (Glasgow Coma Scale)'].nursingchartvalue.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Labels `GCS Total` and `Score (Glasgow Coma Scale)` should be merged, as they represent exactly the same thing.

# + {"Collapsed": "false", "persistent_id": "873db1ef-dfc4-420f-84e4-9d58953affb0"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'SEDATION SCORE'].nursingchartcelltypevalname.value_counts()

# + {"Collapsed": "false", "persistent_id": "b4239de6-dba7-4020-af95-86424eeb5d76"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'SEDATION SCORE'].nursingchartvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "30c4d725-5c6f-458d-8394-7ad034a4453e"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Sedation Scale/Score/Goal'].nursingchartcelltypevalname.value_counts()

# + {"Collapsed": "false", "persistent_id": "c429ea30-2a2f-4bf2-8b74-8be8c94131cb"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Sedation Scale/Score/Goal'].nursingchartvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "128d50c4-4638-42c8-a1cd-38dccceaebdd"}
nursechart_df[nursechart_df.nursingchartcelltypevalname == 'Sedation Score'].nursingchartvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "408c3ffb-8df0-48df-a4cc-40a90e63b8dc"}
nursechart_df[nursechart_df.nursingchartcelltypevalname == 'Sedation Scale'].nursingchartvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "6ef7ce6c-e034-4a8f-95b2-f59bcdced712"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Delirium Scale/Score'].nursingchartcelltypevalname.value_counts()

# + {"Collapsed": "false", "persistent_id": "de2d87ef-74ba-441e-9c89-6c268f4cf7de"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Delirium Scale/Score'].nursingchartvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "e1c2af98-208a-4000-b25d-81c452193e36"}
nursechart_df[nursechart_df.nursingchartcelltypevalname == 'Delirium Score'].nursingchartvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "aaef6a44-1a98-4cee-8561-a18f6b1169d4"}
nursechart_df[nursechart_df.nursingchartcelltypevalname == 'Delirium Scale'].nursingchartvalue.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sedation and delirium scores could be interesting features, however they are presented in different scales, like in pain score, which don't seem to be directly convertable between them. Due to this, we will only consider the most frequent scale for each case (`RASS` and `CAM-ICU`, respectively).

# + {"Collapsed": "false", "persistent_id": "92d462b1-7840-4044-9da2-fe9a88b9bb28"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Best Motor Response'].nursingchartcelltypevalname.value_counts()

# + {"Collapsed": "false", "persistent_id": "df8966b5-f3c2-4237-96c5-07d2088880d2"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Best Motor Response'].nursingchartvalue.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# These "Best ___ Response" features are subparts of the total Glasgow Coma Score calculation. Because of that, and for having less data, they will be discarded.

# + {"Collapsed": "false", "persistent_id": "d1492247-9def-4ea9-b0c4-b812a20b937d"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Gastrointestinal Assessment'].nursingchartcelltypevalname.value_counts()

# + {"Collapsed": "false", "persistent_id": "39b3c5cd-389d-47aa-bc98-93e20ccf1756"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Gastrointestinal Assessment'].nursingchartvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "a06b9bdb-2140-43a6-bb0e-fbf076d31142"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Genitourinary Assessment'].nursingchartcelltypevalname.value_counts()

# + {"Collapsed": "false", "persistent_id": "c777eea2-6829-4162-abdf-83556f50f7d4"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Genitourinary Assessment'].nursingchartvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "90283bb6-8463-4006-ad4e-1d5e01e16491"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Integumentary Assessment'].nursingchartcelltypevalname.value_counts()

# + {"Collapsed": "false", "persistent_id": "f162fc4e-c297-4437-86d5-142dd1a58404"}
nursechart_df[nursechart_df.nursingchartcelltypevallabel == 'Integumentary Assessment'].nursingchartvalue.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Some other information, like these gastrointestinal, genitourinary and integumentary domains, could be relevant to add. The problem is that we only seem to have acccess to how they were measured (i.e. their scale) and not the real values.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Besides the usual removal of row identifier, `nurseAssessID`, and the timestamp when data was added, `nurseAssessEntryOffset`, I'm also removing all labels and names except those that relate to pain, coma, sedation and delirium scores. Furthermore, `nursingchartcelltypecat` doesn't add much relevant info either, so it will be removed.

# + {"Collapsed": "false", "persistent_id": "d2ed8ee0-14bc-4c87-8f77-139cbe55b97b"}
nursechart_df = nursechart_df.drop(['nursingchartid', 'nursingchartentryoffset', 'nursingchartcelltypecat'], axis=1)
nursechart_df.head()

# + {"Collapsed": "false", "persistent_id": "14134ab0-22f4-4f16-9f4f-ed0b14d59f36"}
labels_to_keep = ['Glasgow coma score', 'Score (Glasgow Coma Scale)',
                  'Sedation Scale/Score/Goal', 'Delirium Scale/Score']

# + {"Collapsed": "false", "persistent_id": "766874cf-829c-4667-af39-ca5efdd27999"}
nursechart_df = nursechart_df[nursechart_df.nursingchartcelltypevallabel.isin(labels_to_keep)]
nursechart_df.head()

# + {"Collapsed": "false", "persistent_id": "97a26d9f-b682-4b6f-8231-f1b391ecd051"}
names_to_keep = ['Pain Score', 'GCS Total', 'Value', 'Sedation Score',
                 'Sedation Scale', 'Delirium Score', 'Delirium Scale']

# + {"Collapsed": "false", "persistent_id": "f8a3403e-e661-49d7-8820-2a299b5a7a05"}
nursechart_df = nursechart_df[nursechart_df.nursingchartcelltypevalname.isin(names_to_keep)]
nursechart_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Convert categories to features

# + {"Collapsed": "false", "cell_type": "markdown"}
# Make the `nursingchartcelltypevallabel` and `nursingchartcelltypevalname` columns of type categorical:

# + {"Collapsed": "false", "persistent_id": "2aaa4ff1-472f-432d-8715-d247579dc68b"}
nursechart_df = nursechart_df.categorize(columns=['nursingchartcelltypevallabel', 'nursingchartcelltypevalname'])

# + {"Collapsed": "false", "persistent_id": "18710668-5166-46aa-be82-b1d08c3114ea"}
nursechart_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Transform the `nursingchartcelltypevallabel` categories and `nursingchartvalue` values into separate features:

# + {"Collapsed": "false", "cell_type": "markdown"}
# Now we have the categories separated into their own features, as desired.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the old `nursingchartcelltypevallabel`, `nursingchartcelltypevalname` and `nursingchartvalue` columns:

# + {"Collapsed": "false", "persistent_id": "4e229cd7-9171-4be5-967b-240c0ee17554"}
nursechart_df = nursechart_df.drop(['nursingchartcelltypevallabel', 'nursingchartcelltypevalname', 'nursingchartvalue'], axis=1)
nursechart_df.head()

# + {"Collapsed": "false", "persistent_id": "d2ca730e-23ed-4bce-b884-436277b6a91c"}
nursechart_df['Pain Score'].value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Filter the most common measurement scales
#
# Only keep data thats is in the same, most common measurement scale.

# + {"Collapsed": "false", "persistent_id": "f31f5e35-d82c-4072-8f6d-821122f963f3"}
nursechart_df = nursechart_df[((nursechart_df['Pain Assessment'] == 'WDL')
                               | (nursechart_df['Sedation Scale'] == 'RASS')
                               | (nursechart_df['Delirium Scale'] == 'CAM-ICU'))]
nursechart_df.head()


# + {"Collapsed": "false", "cell_type": "markdown"}
# Merge Glasgow coma score columns:

# + {"Collapsed": "false", "persistent_id": "87463d89-4cb2-4098-babd-542c28f4f92b"}
def set_glc(df):
    if np.isnan(df['GLC Total']):
        return df['Score (Glasgow Coma Scale)']
    else:
        return df['GLC Total']


# + {"Collapsed": "false", "persistent_id": "45e0c949-eef6-4271-bb36-f3efa39b6e1f"}
nursechart_df['glasgow_coma_score'] = nursechart_df.apply(lambda df: set_glc(df), axis=1)
nursechart_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Drop unneeded columns:

# + {"Collapsed": "false", "persistent_id": "8420f0ea-a7a6-4992-beca-6a910962cca0"}
nursechart_df = nursechart_df.drop(['Pain Assessment', 'GLC Total', 'Score (Glasgow Coma Scale)',
                                    'Value', 'Sedation Scale', 'Delirium Scale'], axis=1)
nursechart_df.head()

# + {"Collapsed": "false", "toc-hr-collapsed": false, "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into one hot encode columns, which can later be embedded or used as is.

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### One hot encode features

# + {"Collapsed": "false", "cell_type": "markdown"}
# Update list of categorical features:

# + {"Collapsed": "false", "persistent_id": "3c800d90-6e44-450b-84c7-adf0490ec664"}
cat_feat = ['nursingchartcelltypecat', 'nursingchartvalue']

# + {"Collapsed": "false", "persistent_id": "16eaf806-f079-414c-bfb8-eb56b1cf9200"}
nursechart_df[cat_feat].head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "71212e7f-2513-4b17-af44-94b893ad18bf"}
nursechart_df = du.data_processing.one_hot_encoding_dataframe(nursechart_df, columns=cat_feat, join_rows=False,
                                                              join_by=['patientunitstayid', 'drugoffset'])
nursechart_df

# + {"Collapsed": "false", "persistent_id": "d45b3fbd-7152-4b99-94c1-328a97af385f"}
nursechart_df[cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "e8519bde-dd73-4afc-9b70-7e6f0a4f85cf"}
cat_feat_ohe

# + {"Collapsed": "false", "persistent_id": "3dbfccba-d336-4338-b3a6-5f63876250ff"}
nursechart_df[cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the association between the original categorical features and the new one hot encoded columns:

# + {"Collapsed": "false", "persistent_id": "26eac7f3-9081-4a96-ae4a-40054c223fd7", "execution": {"iopub.status.busy": "2020-03-09T16:37:35.157248Z", "iopub.execute_input": "2020-03-09T16:37:35.157526Z", "iopub.status.idle": "2020-03-09T16:37:35.164656Z", "shell.execute_reply.started": "2020-03-09T16:37:35.157493Z", "shell.execute_reply": "2020-03-09T16:37:35.163771Z"}}
for orig_col in cat_feat:
    cat_feat_ohe[orig_col] = [ohe_col for ohe_col in new_columns
                              if ohe_col.startswith(orig_col)]

# + {"execution": {"iopub.status.busy": "2020-03-09T16:37:35.165864Z", "iopub.execute_input": "2020-03-09T16:37:35.166280Z", "iopub.status.idle": "2020-03-09T16:37:35.190294Z", "shell.execute_reply.started": "2020-03-09T16:37:35.166256Z", "shell.execute_reply": "2020-03-09T16:37:35.189358Z"}, "Collapsed": "false"}
cat_feat_ohe

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "439732df-2263-4fe2-81be-7535119a7170"}
stream = open(f'{data_path}/cleaned/cat_feat_ohe_nurse_chart.yaml', 'w')
yaml.dump(cat_feat_ohe, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "a7eff302-981f-40e7-8bf0-ee9ee5738b0b"}
nursechart_df = nursechart_df.rename(columns={'nursechartoffset': 'ts'})
nursechart_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "1345d11e-6a2f-455e-9846-c5762a32ad1c"}
len(nursechart_df)

# + {"Collapsed": "false", "persistent_id": "89ae3a07-24eb-43c9-97df-77ea2e0d2a46"}
nursechart_df = nursechart_df.drop_duplicates()
nursechart_df.head()

# + {"Collapsed": "false", "persistent_id": "d63a222b-e9d8-4250-934e-fb7ce40281d0"}
len(nursechart_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "bc1611c3-89ae-4223-896d-b4f42c26db74"}
nursechart_df = nursechart_df.sort_values('ts')
nursechart_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "6ae4396f-4560-400b-b37d-030a16a1889f"}
nursechart_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='nursingchartcelltypecat', n=5).head()

# + {"Collapsed": "false", "persistent_id": "e4d4ed26-00ef-4661-ad50-e8517bb60a64"}
nursechart_df[nursechart_df.patientunitstayid == 2553254].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 80 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"Collapsed": "false", "cell_type": "markdown"}
# Convert dataframe to Pandas, as the groupby operation in `join_repeated_rows` isn't working properly with Modin:

# + {"Collapsed": "false"}
nursechart_df, pd = du.utils.convert_dataframe(nursechart_df, to='pandas')

# + {"Collapsed": "false"}
type(nursechart_df)

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "589931b8-fe11-439a-8b14-4857c168c023"}
nursechart_df = du.embedding.join_repeated_rows(nursechart_df, inplace=True)
nursechart_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Reconvert dataframe to Modin:

# + {"Collapsed": "false"}
nursechart_df, pd = du.utils.convert_dataframe(nursechart_df, to='modin')

# + {"Collapsed": "false"}
type(nursechart_df)

# + {"Collapsed": "false", "persistent_id": "940ed638-e0a6-4c91-a80b-fc332ce29fc1"}
nursechart_df.dtypes

# + {"Collapsed": "false", "persistent_id": "4aee12fc-0b71-483d-8a59-dfada22a08af"}
nursechart_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='nursingchartcelltypecat', n=5).head()

# + {"Collapsed": "false", "persistent_id": "6a2b5792-c42f-403d-a79b-fb4619d6c2ba"}
nursechart_df[nursechart_df.patientunitstayid == 2553254].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_repeated_rows` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Rename columns

# + {"Collapsed": "false", "persistent_id": "69dcd73f-8e18-4672-a74f-586009f65924"}
nursechart_df = nursechart_df.rename(columns={'nursingchartcelltypecat':'nurse_assess_label',
                                                'nursingchartvalue':'nurse_assess_value'})
nursechart_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "a9a4f8f6-07ae-41e6-a2fa-4610eca487df"}
nursechart_df.columns = du.data_processing.clean_naming(nursechart_df.columns)
nursechart_df_norm.columns = du.data_processing.clean_naming(nursechart_df_norm.columns)
nursechart_df_norm.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "67a3501f-6cec-48fa-b67f-405ae0ab3628"}
nursechart_df.to_csv(f'{data_path}cleaned/unnormalized/ohe/nurseCharting.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "4b634cbe-ec47-42a2-a09c-543a53728bd6"}
nursechart_df.to_csv(f'{data_path}cleaned/normalized/ohe/nurseCharting.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "88135c50-5523-47d7-a7a8-1b7dd50e5d84"}
nursechart_df.describe().transpose()
