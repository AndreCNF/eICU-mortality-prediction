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
# # eICU Data Joining
# ---
#
# Reading and joining all parts of the eICU dataset from MIT with the data from over 139k patients collected in the US.
#
# The main goal of this notebook is to prepare a single CSV document that contains all the relevant data to be used when training a machine learning model that predicts mortality, joining tables, filtering useless columns and performing imputation.

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

# + {"Collapsed": "false", "cell_type": "markdown"}
# ## Initialize variables

# + {"Collapsed": "false", "persistent_id": "754a96f8-d389-4968-8c13-52e5e9d0bf82"}
cat_feat = []                              # List of categorical features
cat_embed_feat = []                        # List of categorical features that will be embedded
cat_embed_feat_enum = dict()               # Dictionary of the enumerations of the categorical features that will be embedded

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Nurse care data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "4a37073a-b357-4079-b6af-72125689781d"}
nursecare_df = pd.read_csv(f'{data_path}original/nurseCare.csv')
nursecare_df.head()

# + {"Collapsed": "false", "persistent_id": "6566d20f-9fcb-4d92-879b-55a0cefe54ae"}
len(nursecare_df)

# + {"Collapsed": "false", "persistent_id": "e267e007-4b72-4551-a9d2-7c916956235c"}
nursecare_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Only 13052 unit stays have nurse care data. Might not be useful to include them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "b4f282b2-e5f4-4cfb-83c5-c79944367d70"}
nursecare_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "6a382a6d-709e-4cdf-a586-c0cc751ff853"}
nursecare_df.columns

# + {"Collapsed": "false", "persistent_id": "f06e6c16-7c54-40c0-9055-0d8717661e4c"}
nursecare_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "3cc624bc-6dac-4cd0-9d64-f329e29940fa"}
du.search_explore.dataframe_missing_values(nursecare_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "ba513734-4c7e-455d-bfa8-9b805c59b530"}
nursecare_df.celllabel.value_counts()

# + {"Collapsed": "false", "persistent_id": "4d028a9e-38e8-496e-805a-ed3f2304a07f"}
nursecare_df.cellattribute.value_counts()

# + {"Collapsed": "false", "persistent_id": "d61a9031-35b3-4c51-a141-a820f6ed70a8"}
nursecare_df.cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "502ee5b2-8eba-407e-a0c3-158c27cb9fab"}
nursecare_df.cellattributepath.value_counts()

# + {"Collapsed": "false", "persistent_id": "056c6696-d254-4012-9446-cad509788c0b"}
nursecare_df[nursecare_df.celllabel == 'Nutrition'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "f7be37c4-ee8a-4e58-8cf9-ee17cf36f46c"}
nursecare_df[nursecare_df.celllabel == 'Activity'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "e39d2950-3c4c-4895-b5f4-8e865c927cb7"}
nursecare_df[nursecare_df.celllabel == 'Hygiene/ADLs'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "6da5000e-cfe8-4179-a0bf-159cb14240e2"}
nursecare_df[nursecare_df.celllabel == 'Safety'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "70c3fc3a-e7a8-4e04-b38c-a24a5f45eb91"}
nursecare_df[nursecare_df.celllabel == 'Treatments'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "443bb56c-2c54-4af7-901f-2d094f736859"}
nursecare_df[nursecare_df.celllabel == 'Isolation Precautions'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "dc2db80c-0a0c-436d-9db5-105ef2bd28b8"}
nursecare_df[nursecare_df.celllabel == 'Restraints'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "8c71653e-7b4f-4afc-bcf3-46b0fa1f013c"}
nursecare_df[nursecare_df.celllabel == 'Equipment'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Besides the usual removal of row identifier, `nursecareid`, and the timestamp when data was added, `nursecareentryoffset`, I'm also removing `cellattributepath` and `cellattribute`, which have redundant info with `celllabel`.

# + {"Collapsed": "false", "persistent_id": "48f10eea-6c36-4188-91d9-4def749f2486"}
nursecare_df = nursecare_df.drop(['nursecareid', 'nursecareentryoffset',
                                  'cellattributepath', 'cellattribute'], axis=1)
nursecare_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Additionally, some information like "Equipment" and "Restraints" seem to be unnecessary. So let's remove them:

# + {"Collapsed": "false", "persistent_id": "6c7fa8f7-3195-46a9-889a-6034ffdcaef3"}
categories_to_remove = ['Safety', 'Restraints', 'Equipment', 'Airway Type',
                        'Isolation Precautions', 'Airway Size']

# + {"Collapsed": "false", "persistent_id": "d8b7cf79-bda7-411f-a800-e4769eb0ec00"}
~(nursecare_df.celllabel.isin(categories_to_remove)).head()

# + {"Collapsed": "false", "persistent_id": "8f0a890f-ad85-457c-888e-37f9cddfe4e8"}
nursecare_df = nursecare_df[~(nursecare_df.celllabel.isin(categories_to_remove))]
nursecare_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Convert categories to features

# + {"Collapsed": "false", "cell_type": "markdown"}
# Make the `celllabel` and `cellattributevalue` columns of type categorical:

# + {"Collapsed": "false", "persistent_id": "edb243b2-7a8d-4051-a27e-e4049565490b"}
nursecare_df = nursecare_df.categorize(columns=['celllabel', 'cellattributevalue'])

# + {"Collapsed": "false", "cell_type": "markdown"}
# Transform the `celllabel` categories and `cellattributevalue` values into separate features:

# + {"Collapsed": "false", "cell_type": "markdown"}
# Now we have the categories separated into their own features, as desired.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the old `celllabel` and `cellattributevalue` columns:

# + {"Collapsed": "false", "persistent_id": "d42cb91d-3a85-4ad6-a3c0-4e87c951781f"}
nursecare_df = nursecare_df.drop(['celllabel', 'cellattributevalue'], axis=1)
nursecare_df.head()

# + {"Collapsed": "false", "persistent_id": "2be490c2-82d9-44dc-897f-6417e41cfe96"}
nursecare_df['Nutrition'].value_counts()

# + {"Collapsed": "false", "persistent_id": "cc5a84b3-994f-490a-97d5-4f4edf5b0497"}
nursecare_df['Treatments'].value_counts()

# + {"Collapsed": "false", "persistent_id": "6488d69a-45e1-4b67-9d1c-1ce6e6c7f31a"}
nursecare_df['Hygiene/ADLs'].value_counts()

# + {"Collapsed": "false", "persistent_id": "6ae6e48f-2b69-445a-9dd7-f5875e8d1cd5"}
nursecare_df['Activity'].value_counts()

# + {"toc-hr-collapsed": false, "Collapsed": "false", "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into simple numberings, one hot encode features with a low number of categories (in this case, 5) and enumerate sparse categorical features that will be embedded.

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Separate and prepare features for embedding
#
# Identify categorical features that have more than 5 unique categories, which will go through an embedding layer afterwards, and enumerate them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Update list of categorical features and add those that will need embedding (features with more than 5 unique values):

# + {"Collapsed": "false", "persistent_id": "9836b018-de91-4b41-b107-cb8c9779b4c5"}
new_cat_feat = ['Nutrition', 'Treatments', 'Hygiene/ADLs', 'Activity']
[cat_feat.append(col) for col in new_cat_feat]

# + {"Collapsed": "false", "persistent_id": "dae05725-2f44-48f6-924b-6db328111cca"}
cat_feat_nunique = [nursecare_df[feature].nunique() for feature in new_cat_feat]
cat_feat_nunique

# + {"Collapsed": "false", "persistent_id": "8493c9a1-3607-4924-87fb-21e4559f6e0f"}
new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        new_cat_embed_feat.append(new_cat_feat[i])

# + {"Collapsed": "false", "persistent_id": "19066f2a-58d5-4edb-a33d-6d830424f40c"}
nursecare_df[new_cat_feat].head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "51c18fe2-7b70-41c1-95d2-f423b3d7836b"}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    nursecare_df[feature], cat_embed_feat_enum[feature] = du.embedding.enum_categorical_feature(nursecare_df, feature)

# + {"Collapsed": "false", "persistent_id": "9ceba9e2-6821-4a73-8875-5ddebef03516"}
nursecare_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "56d49f40-97fe-47d5-ad9d-ef35a4453266"}
cat_embed_feat_enum

# + {"Collapsed": "false", "persistent_id": "49f71013-ebc1-472e-b91b-2a96233b207b"}
nursecare_df[new_cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "398e3a5f-c8e3-4657-aa40-05967570fd66"}
stream = open('cat_embed_feat_enum_nursing.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "d0647504-f554-4d1f-8eba-d87851eb5695"}
nursecare_df['ts'] = nursecare_df['nursecareoffset']
nursecare_df = nursecare_df.drop('nursecareoffset', axis=1)
nursecare_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "a4cc356c-5c63-41e3-b906-20a4d51ef912"}
len(nursecare_df)

# + {"Collapsed": "false", "persistent_id": "0ed1deff-4607-4456-af14-95ae472a6e05"}
nursecare_df = nursecare_df.drop_duplicates()
nursecare_df.head()

# + {"Collapsed": "false", "persistent_id": "e864c533-0f0d-4bde-9021-2302ea459260"}
len(nursecare_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "9fbb6262-2cdc-4809-a2b2-ce8847793cca"}
nursecare_df = nursecare_df.set_index('ts')
nursecare_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "537efbd0-2f51-4d33-90dc-a4492f94139b"}
nursecare_df.reset_index().head()

# + {"Collapsed": "false", "persistent_id": "ec32dcd9-8bec-4077-9392-0f7430ddaae2"}
nursecare_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='Nutrition').head()

# + {"Collapsed": "false", "persistent_id": "9baab531-96da-40e3-8bde-c4657bdc950e"}
nursecare_df[nursecare_df.patientunitstayid == 2798325].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 21 categories per set of `patientunitstayid` and `ts`. As such, we must join them. However, this is a different scenario than in the other cases. Since we created the features from one categorical column, it doesn't have repeated values, only different rows to indicate each of the new features' values. As such, we just need to sum the features.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "b0369d33-d5fb-45c7-aa2a-d180b987a251"}
# [TODO] Find a way to join rows while ignoring zeros
nursecare_df = du.embedding.join_categorical_enum(nursecare_df, new_cat_embed_feat)
nursecare_df.head()

# + {"Collapsed": "false", "persistent_id": "0b782718-8a92-4780-abbe-f8cab9efdfce"}
nursecare_df.dtypes

# + {"Collapsed": "false", "persistent_id": "1d5d3435-7ebf-4d13-8fdb-ac1a09f847b3"}
nursecare_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='Nutrition').head()

# + {"Collapsed": "false", "persistent_id": "c4269670-2a2a-4e69-8800-ac737eaa3ebc"}
nursecare_df[nursecare_df.patientunitstayid == 2798325].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Rename columns

# + {"Collapsed": "false", "persistent_id": "9b090273-1c22-4122-9979-8ee2b91f0dfe"}
nursecare_df = nursecare_df.rename(columns={'Treatments':'nurse_treatments'})
nursecare_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "a12ee2e9-13d9-44ce-999c-846f320e8bfd"}
nursecare_df.columns = du.data_processing.clean_naming(nursecare_df.columns)
nursecare_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "584b72a4-0de9-433b-b27f-6304a0db2b52"}
nursecare_df.to_csv(f'{data_path}cleaned/unnormalized/nurseCare.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "b9f11ee9-cd10-44df-961f-c7bc3642710f"}
nursecare_df.to_csv(f'{data_path}cleaned/normalized/nurseCare.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "edafc000-e516-49fe-b8f6-ea4ce3969129"}
nursecare_df.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

# + {"Collapsed": "false", "persistent_id": "1d849f6b-174e-4e4d-82bc-fe854cedcb19"}
nursecare_df = pd.read_csv(f'{data_path}cleaned/normalized/nurseCare.csv')
nursecare_df.head()

# + {"Collapsed": "false", "persistent_id": "3aa132e7-1a1b-47d0-9863-b3b74aafc86f"}
len(nursecare_df)

# + {"Collapsed": "false", "persistent_id": "855ec1b6-d397-4173-9994-202cf00b7537"}
nursecare_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "c8dac694-148a-403d-82b8-19fd509cc042"}
eICU_df = pd.merge_asof(eICU_df, nursecare_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Nurse assessment data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "51281341-db28-43a0-8f85-cd3ae11aae06"}
nurseassess_df = pd.read_csv(f'{data_path}original/nurseAssessment.csv')
nurseassess_df.head()

# + {"Collapsed": "false", "persistent_id": "6573ec98-4a25-4b83-b875-626d555def4e"}
len(nurseassess_df)

# + {"Collapsed": "false", "persistent_id": "cd9958e3-dd35-4a18-b2d0-01f5dc59a708"}
nurseassess_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Only 13001 unit stays have nurse care data. Might not be useful to include them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "a63bd49a-1ea0-4f3e-bb25-98040cb31c28"}
nurseassess_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "baebbad8-8e0a-4a54-92f2-a1a3c8b63e28"}
nurseassess_df.columns

# + {"Collapsed": "false", "persistent_id": "127f78c7-5fa1-4161-b1aa-085631f74a7b"}
nurseassess_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "6a7adc72-3d2e-4cd5-841c-299a3a94d043"}
du.search_explore.dataframe_missing_values(nurseassess_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "38ac8914-ae50-45ce-9585-77263c79af06"}
nurseassess_df.celllabel.value_counts()

# + {"Collapsed": "false", "persistent_id": "5d345361-e2b7-415f-8b0e-6f706d29c4db"}
nurseassess_df.cellattribute.value_counts()

# + {"Collapsed": "false", "persistent_id": "bae7f1bf-dd1f-4d3f-8cb0-f70365412fbc"}
nurseassess_df.cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "50b31b51-bee1-49e6-86eb-c24d97830844"}
nurseassess_df.cellattributepath.value_counts()

# + {"Collapsed": "false", "persistent_id": "f952c852-c8d4-4313-baa6-e0f0e33adcfc"}
nurseassess_df[nurseassess_df.celllabel == 'Intervention'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "1bfe013d-8f9a-496f-9bea-3091e2f01b61"}
nurseassess_df[nurseassess_df.celllabel == 'Neurologic'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "8984d2fa-3cac-486e-b029-ab7da972ebef"}
nurseassess_df[nurseassess_df.celllabel == 'Pupils'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "463936ef-481d-467e-beba-606f880dce40"}
nurseassess_df[nurseassess_df.celllabel == 'Edema'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "55680a6d-7fab-4fce-8eed-66dd44d011f1"}
nurseassess_df[nurseassess_df.celllabel == 'Secretions'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "7e511548-989e-4c78-a6bd-a56b4811dd0e"}
nurseassess_df[nurseassess_df.celllabel == 'Cough'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "b75398ad-65f5-4aac-a00b-4f1034c1558c"}
nurseassess_df[nurseassess_df.celllabel == 'Neurologic'].cellattribute.value_counts()

# + {"Collapsed": "false", "persistent_id": "d0592960-b088-4491-81d7-697cee55df06"}
nurseassess_df[nurseassess_df.celllabel == 'Pupils'].cellattribute.value_counts()

# + {"Collapsed": "false", "persistent_id": "b461df0f-36ef-4bdf-9eda-2a9c57bf1a44"}
nurseassess_df[nurseassess_df.celllabel == 'Secretions'].cellattribute.value_counts()

# + {"Collapsed": "false", "persistent_id": "c118b8b5-4c96-45d6-b2f5-a2cc2ce8ba93"}
nurseassess_df[nurseassess_df.celllabel == 'Cough'].cellattribute.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Besides the usual removal of row identifier, `nurseAssessID`, and the timestamp when data was added, `nurseAssessEntryOffset`, I'm also removing `cellattributepath` and `cellattribute`, which have redundant info with `celllabel`. Regarding data categories, I'm only keeping `Neurologic`, `Pupils`, `Secretions` and `Cough`, as the remaining ones either don't add much value, have too little data or are redundant with data from other tables.

# + {"Collapsed": "false", "persistent_id": "be9a8747-0209-4e96-a55a-5587f1d09882"}
nurseassess_df = nurseassess_df.drop(['nurseassessid', 'nurseassessentryoffset',
                                      'cellattributepath', 'cellattribute'], axis=1)
nurseassess_df.head()

# + {"Collapsed": "false", "persistent_id": "1a520f05-7d41-4a71-a9ee-a7c359c7b84f"}
categories_to_keep = ['Neurologic', 'Pupils', 'Secretions', 'Cough']

# + {"Collapsed": "false", "persistent_id": "be99810d-b848-4966-99a8-e31c9189d173"}
nurseassess_df.celllabel.isin(categories_to_keep).head()

# + {"Collapsed": "false", "persistent_id": "288e8cc5-c11e-4742-950c-ee2acb0d0d41"}
nurseassess_df = nurseassess_df[nurseassess_df.celllabel.isin(categories_to_keep)]
nurseassess_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Convert categories to features

# + {"Collapsed": "false", "cell_type": "markdown"}
# Make the `celllabel` and `cellattributevalue` columns of type categorical:

# + {"Collapsed": "false", "persistent_id": "35e6e508-5b51-4196-92aa-e14cb93804e0"}
nurseassess_df = nurseassess_df.categorize(columns=['celllabel', 'cellattributevalue'])

# + {"Collapsed": "false", "persistent_id": "429b6864-358f-47c2-bdc3-1cd938ffda69"}
nurseassess_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Transform the `celllabel` categories and `cellattributevalue` values into separate features:

# + {"Collapsed": "false", "cell_type": "markdown"}
# Now we have the categories separated into their own features, as desired.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the old `celllabel` and `cellattributevalue` columns:

# + {"Collapsed": "false", "persistent_id": "d4762de3-df60-4c93-acb4-125d08c31c45"}
nurseassess_df = nurseassess_df.drop(['celllabel', 'cellattributevalue'], axis=1)
nurseassess_df.head()

# + {"Collapsed": "false", "persistent_id": "b789153f-6a30-43dc-b8cc-5775b21dd7c1"}
nurseassess_df['Neurologic'].value_counts()

# + {"Collapsed": "false", "persistent_id": "7bbe2fb5-20fa-4ff5-b7f0-f5fb2a7f26e9"}
nurseassess_df['Pupils'].value_counts()

# + {"Collapsed": "false", "persistent_id": "465fe194-8649-49f5-9bf5-0943e2a29d9e"}
nurseassess_df['Secretions'].value_counts()

# + {"Collapsed": "false", "persistent_id": "ef065236-1712-4dc1-aef0-f383c93a0251"}
nurseassess_df['Cough'].value_counts()

# + {"toc-hr-collapsed": false, "Collapsed": "false", "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into simple numberings, one hot encode features with a low number of categories (in this case, 5) and enumerate sparse categorical features that will be embedded.

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Separate and prepare features for embedding
#
# Identify categorical features that have more than 5 unique categories, which will go through an embedding layer afterwards, and enumerate them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Update list of categorical features and add those that will need embedding (features with more than 5 unique values):

# + {"Collapsed": "false", "persistent_id": "93a69b04-f60a-456e-b715-6428429c34b3"}
new_cat_feat = ['Pupils', 'Neurologic', 'Secretions', 'Cough']
[cat_feat.append(col) for col in new_cat_feat]

# + {"Collapsed": "false", "persistent_id": "2fccdfbf-9fd3-4e90-abe9-8f6463dc8eb8"}
cat_feat_nunique = [nurseassess_df[feature].nunique() for feature in new_cat_feat]
cat_feat_nunique

# + {"Collapsed": "false", "persistent_id": "41b8f1fb-2bed-4a5c-9b4d-ab5b4ab07335"}
new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        new_cat_embed_feat.append(new_cat_feat[i])

# + {"Collapsed": "false", "persistent_id": "29703baf-cba2-4f41-86e5-74912610503c"}
nurseassess_df[new_cat_feat].head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "c7f73853-ecf8-409e-9351-191e07444213"}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    nurseassess_df[feature], cat_embed_feat_enum[feature] = du.embedding.enum_categorical_feature(nurseassess_df, feature)

# + {"Collapsed": "false", "persistent_id": "6101e468-8fdc-48c2-90f7-7a8db94c1b58"}
nurseassess_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "b21d68b6-f26b-469c-9139-b746f027758d"}
cat_embed_feat_enum

# + {"Collapsed": "false", "persistent_id": "9951459d-c61d-49cf-a9d7-630ced7dfef6"}
nurseassess_df[new_cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "0280b97a-1137-433b-a54e-6168edc4a350"}
stream = open('cat_embed_feat_enum_nursing.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "b1781127-4235-4097-9bb0-5d7dfa965735"}
nurseassess_df['ts'] = nurseassess_df['nurseassessoffset']
nurseassess_df = nurseassess_df.drop('nurseassessoffset', axis=1)
nurseassess_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "3483fd1d-3ad6-4e93-a583-5ddeb38ad6c5"}
len(nurseassess_df)

# + {"Collapsed": "false", "persistent_id": "ffb1df48-8666-41c0-9a74-32f39d8d37cf"}
nurseassess_df = nurseassess_df.drop_duplicates()
nurseassess_df.head()

# + {"Collapsed": "false", "persistent_id": "a7dbc589-c48e-45a6-9044-67b586b11fad"}
len(nurseassess_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "c8ba14f1-d5d7-4e2b-be19-3ec645351c8e"}
nurseassess_df = nurseassess_df.set_index('ts')
nurseassess_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "63cf5946-52ad-4abf-8bfa-0cbf70464ea7"}
nurseassess_df.reset_index().head()

# + {"Collapsed": "false", "persistent_id": "28bb9396-7d50-42a9-a807-b4408c62f815"}
nurseassess_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='Cough').head()

# + {"Collapsed": "false", "persistent_id": "473b8ca5-32f0-45e0-ad54-c3b7ae008653"}
nurseassess_df[nurseassess_df.patientunitstayid == 2553254].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 80 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "a60ffe61-f2ba-4611-9e18-d6d28d12f0a0"}
nurseassess_df = du.embedding.join_categorical_enum(nurseassess_df, new_cat_embed_feat)
nurseassess_df.head()

# + {"Collapsed": "false", "persistent_id": "decebaec-f14b-4521-adcc-24f485a0a781"}
nurseassess_df.dtypes

# + {"Collapsed": "false", "persistent_id": "0ba6b50e-69db-489c-a1f1-6f324a662a68"}
nurseassess_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='Cough').head()

# + {"Collapsed": "false", "persistent_id": "41c057d5-3d9e-454e-b71b-1f125d63842e"}
nurseassess_df[nurseassess_df.patientunitstayid == 2553254].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "8b8eb574-108e-4f73-b8b8-d0314c295d7b"}
nurseassess_df.columns = du.data_processing.clean_naming(nurseassess_df.columns)
nurseassess_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "70d6ca28-9b03-400e-a218-0bab68904b50"}
nurseassess_df.to_csv(f'{data_path}cleaned/unnormalized/nurseAssessment.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "c8a9c87e-64db-4852-86ff-8f9952ba63ed"}
nurseassess_df.to_csv(f'{data_path}cleaned/normalized/nurseAssessment.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "b009d13b-05d3-4b2b-9b35-1b9bba3dac28"}
nurseassess_df.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

# + {"Collapsed": "false", "persistent_id": "2184d99d-e508-469d-9201-4246dcb62869"}
nurseassess_df = pd.read_csv(f'{data_path}cleaned/normalized/nurseAssessment.csv')
nurseassess_df.head()

# + {"Collapsed": "false", "persistent_id": "eda02f1c-9904-4a55-9f3b-7cfd7a47afa2"}
len(nurseassess_df)

# + {"Collapsed": "false", "persistent_id": "22605092-029d-4535-96a2-3a990f8bb768"}
nurseassess_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "060d946d-2740-41f7-bcaf-68b673057a55"}
eICU_df = pd.merge_asof(eICU_df, nurseassess_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Nurse charting data

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

# + {"toc-hr-collapsed": false, "Collapsed": "false", "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into simple numberings, one hot encode features with a low number of categories (in this case, 5) and enumerate sparse categorical features that will be embedded.

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Separate and prepare features for embedding
#
# Identify categorical features that have more than 5 unique categories, which will go through an embedding layer afterwards, and enumerate them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Update list of categorical features and add those that will need embedding (features with more than 5 unique values):

# + {"Collapsed": "false", "persistent_id": "3c800d90-6e44-450b-84c7-adf0490ec664"}
new_cat_feat = ['nursingchartcelltypecat', 'nursingchartvalue']
[cat_feat.append(col) for col in new_cat_feat]

# + {"Collapsed": "false", "persistent_id": "2665e878-6555-4f96-8caa-a936ff71d322"}
cat_feat_nunique = [nursechart_df[feature].nunique() for feature in new_cat_feat]
cat_feat_nunique

# + {"Collapsed": "false", "persistent_id": "e449b27b-378b-486e-9964-79b5860e8911"}
new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        new_cat_embed_feat.append(new_cat_feat[i])

# + {"Collapsed": "false", "persistent_id": "16eaf806-f079-414c-bfb8-eb56b1cf9200"}
nursechart_df[new_cat_feat].head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "71212e7f-2513-4b17-af44-94b893ad18bf"}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    nursechart_df[feature], cat_embed_feat_enum[feature] = du.embedding.enum_categorical_feature(nursechart_df, feature)

# + {"Collapsed": "false", "persistent_id": "d45b3fbd-7152-4b99-94c1-328a97af385f"}
nursechart_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "e8519bde-dd73-4afc-9b70-7e6f0a4f85cf"}
cat_embed_feat_enum

# + {"Collapsed": "false", "persistent_id": "3dbfccba-d336-4338-b3a6-5f63876250ff"}
nursechart_df[new_cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "439732df-2263-4fe2-81be-7535119a7170"}
stream = open('cat_embed_feat_enum_nursing.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "a7eff302-981f-40e7-8bf0-ee9ee5738b0b"}
nursechart_df['ts'] = nursechart_df['nursechartoffset']
nursechart_df = nursechart_df.drop('nursechartoffset', axis=1)
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
nursechart_df = nursechart_df.set_index('ts')
nursechart_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "2737d455-c191-4fab-bae1-7b2f00110b77"}
nursechart_df.reset_index().head()

# + {"Collapsed": "false", "persistent_id": "6ae4396f-4560-400b-b37d-030a16a1889f"}
nursechart_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='nursingchartcelltypecat').head()

# + {"Collapsed": "false", "persistent_id": "e4d4ed26-00ef-4661-ad50-e8517bb60a64"}
nursechart_df[nursechart_df.patientunitstayid == 2553254].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 80 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "3db5777c-316e-4631-9dbc-165fb55d093d"}
nursechart_df = du.embedding.join_categorical_enum(nursechart_df, new_cat_embed_feat)
nursechart_df.head()

# + {"Collapsed": "false", "persistent_id": "940ed638-e0a6-4c91-a80b-fc332ce29fc1"}
nursechart_df.dtypes

# + {"Collapsed": "false", "persistent_id": "4aee12fc-0b71-483d-8a59-dfada22a08af"}
nursechart_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='nursingchartcelltypecat').head()

# + {"Collapsed": "false", "persistent_id": "6a2b5792-c42f-403d-a79b-fb4619d6c2ba"}
nursechart_df[nursechart_df.patientunitstayid == 2553254].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

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
nursechart_df.to_csv(f'{data_path}cleaned/unnormalized/nurseCharting.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "4b634cbe-ec47-42a2-a09c-543a53728bd6"}
nursechart_df.to_csv(f'{data_path}cleaned/normalized/nurseCharting.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "88135c50-5523-47d7-a7a8-1b7dd50e5d84"}
nursechart_df.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

# + {"Collapsed": "false", "persistent_id": "7e4d01f9-2b1d-4697-bac6-26623bf64e6a"}
nursechart_df = pd.read_csv(f'{data_path}cleaned/normalized/nurseCharting.csv')
nursechart_df.head()

# + {"Collapsed": "false", "persistent_id": "e86d6dd7-1e02-46d4-949d-91a84f44549b"}
len(nursechart_df)

# + {"Collapsed": "false", "persistent_id": "f05ace4a-a23c-4d65-bfa9-b726dee830a6"}
nursechart_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "b68d4d2b-7b5d-4f32-9ac5-a16b9f1dc70e"}
eICU_df = pd.merge_asof(eICU_df, nursechart_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()
