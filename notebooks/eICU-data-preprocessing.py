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
os.chdir("../../..")

# Path to the CSV dataset files
data_path = 'Documents/Datasets/Thesis/eICU/uncompressed/'

# Path to the code files
project_path = 'Documents/GitHub/eICU-mortality-prediction/'

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

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "9ee194e5-c316-4fe5-8cd8-cc4188be9447"}
patient_df = pd.read_csv(f'{data_path}original/patient.csv')
patient_df.head()

# + {"Collapsed": "false", "persistent_id": "8a040368-7c65-4d72-a4a7-622b63378c3e"}
len(patient_df)

# + {"Collapsed": "false", "persistent_id": "6144f623-8410-4651-9703-bffb19f3e9cc"}
patient_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "7022c0ab-2847-4b14-914b-69fcf3d3ca07"}
patient_df.patientunitstayid.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "d4aa2831-2d82-47d2-8538-11d1257f3891"}
patient_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "4678d828-e15a-43f1-bc52-80b13a2b7c7e"}
patient_df.columns

# + {"Collapsed": "false", "persistent_id": "e4e91c37-750a-490a-887b-3fcb133adef2"}
patient_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "9e6a1829-a25a-44a6-a1b8-074ccd5664c4"}
du.search_explore.dataframe_missing_values(patient_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features
#
# Besides removing unneeded hospital and time information, I'm also removing the admission diagnosis (`apacheadmissiondx`) as it doesn't follow the same structure as the remaining diagnosis data (which is categorized in increasingly specific categories, separated by "|").

# + {"Collapsed": "false", "persistent_id": "2b20dcee-1d77-4976-9d75-90a6359d9edc"}
patient_df = patient_df[['patientunitstayid', 'gender', 'age', 'ethnicity',  'admissionheight',
                         'hospitaldischargeoffset', 'hospitaldischargelocation', 'hospitaldischargestatus',
                         'admissionweight', 'dischargeweight', 'unitdischargeoffset']]
patient_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Make the age feature numeric
#
# In the eICU dataset, ages above 89 years old are not specified. Instead, we just receive the indication "> 89". In order to be able to work with the age feature numerically, we'll just replace the "> 89" values with "90", as if the patient is 90 years old. It might not always be the case, but it shouldn't be very different and it probably doesn't affect too much the model's logic.

# + {"Collapsed": "false", "persistent_id": "61119cf7-7f1a-4382-80d0-65ceeb6137ff"}
patient_df.age.value_counts().head()

# + {"Collapsed": "false", "persistent_id": "0ae741ed-ecdb-4b2a-acad-de7974d9301e"}
# Replace the "> 89" years old indication with 90 years
patient_df.age = patient_df.age.replace(to_replace='> 89', value=90)

# + {"Collapsed": "false", "persistent_id": "45ee35d1-b272-4e68-8305-cf6962322c2a"}
patient_df.age.value_counts().head()

# + {"Collapsed": "false", "persistent_id": "978a88e3-2ac7-41a7-abf5-c84eafd8d360"}
# Make the age feature numeric
patient_df.age = patient_df.age.astype(float)

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into simple numberings, one hot encode features with a low number of categories (in this case, 5) and enumerate sparse categorical features that will be embedded.

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Convert binary categorical features into numeric

# + {"Collapsed": "false", "persistent_id": "825d7d09-34df-43ef-a914-2a68f33723f2"}
patient_df.gender.value_counts()

# + {"Collapsed": "false", "persistent_id": "b500948a-b8fe-4fd3-a863-76d137f6ae5f"}
patient_df.gender = patient_df.gender.map(lambda x: 1 if x == 'Male' else 0 if x == 'Female' else np.nan)

# + {"Collapsed": "false", "persistent_id": "cb548210-9b6c-47dc-a094-47872216500d"}
patient_df.gender.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Separate and prepare features for embedding
#
# Identify categorical features that have more than 5 unique categories, which will go through an embedding layer afterwards, and enumerate them.
#

# + {"Collapsed": "false", "cell_type": "markdown"}
# Update list of categorical features and add those that will need embedding (features with more than 5 unique values):

# + {"Collapsed": "false", "persistent_id": "9fdc9b59-d7e5-47e6-8d32-f3ac7fbb582d"}
new_cat_feat = ['ethnicity']
[cat_feat.append(col) for col in new_cat_feat]

# + {"Collapsed": "false", "persistent_id": "78a46d88-d348-49dd-b078-d2511bdf2869"}
cat_feat_nunique = [patient_df[feature].nunique() for feature in new_cat_feat]
cat_feat_nunique

# + {"Collapsed": "false", "persistent_id": "9b6f45c2-fbdf-4a00-be73-fecca421ce93"}
new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        # Add feature to the list of the new ones (from the current table) that will be embedded
        new_cat_embed_feat.append(new_cat_feat[i])

# + {"Collapsed": "false", "persistent_id": "99d08bce-69f1-4a19-8e1d-ab2a49574506"}
patient_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "32a0ca8b-24f4-41d6-8565-038e39497c7e"}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    patient_df[feature], cat_embed_feat_enum[feature] = du.embedding.enum_categorical_feature(patient_df, feature)

# + {"Collapsed": "false", "persistent_id": "66530762-67c7-4547-953a-b5848c9e4be2"}
patient_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "a4f93b92-b778-4a56-9372-aa4f4a994c02"}
cat_embed_feat_enum

# + {"Collapsed": "false", "persistent_id": "2d79bb26-bb3f-4d3e-beac-0e809c504bdb"}
patient_df[cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "57ad30f0-8e12-482d-91c7-789b3b64b39a"}
stream = open('cat_embed_feat_enum.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create mortality label
#
# Combine info from discharge location and discharge status. Using the hospital discharge data, instead of the unit, as it has a longer perspective on the patient's status. I then save a feature called "deathOffset", which has a number if the patient is dead on hospital discharge or is NaN if the patient is still alive/unknown (presumed alive if unknown). Based on this, a label can be made later on, when all the tables are combined in a single dataframe, indicating if a patient dies in the following X time, according to how faraway we want to predict.

# + {"Collapsed": "false", "persistent_id": "dcf72825-f38a-4a63-8a36-aa1e3fd3ac86"}
patient_df.hospitaldischargestatus.value_counts()

# + {"Collapsed": "false", "persistent_id": "6b20ced2-c573-47e9-a02b-7d4aeceaddf7"}
patient_df.hospitaldischargelocation.value_counts()

# + {"Collapsed": "false", "persistent_id": "fa551a3f-d2a5-49c1-9512-cbb2dcd51478"}
patient_df['deathoffset'] = patient_df.apply(lambda df: df['hospitaldischargeoffset']
                                                        if df['hospitaldischargestatus'] == 'Expired' or
                                                        df['hospitaldischargelocation'] == 'Death' else np.nan, axis=1,
                                                        meta=('x', float))
patient_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the now unneeded hospital discharge features:

# + {"Collapsed": "false", "persistent_id": "b30bb22f-f470-4132-92e7-bed1f058f4a8"}
patient_df = patient_df.drop(['hospitaldischargeoffset', 'hospitaldischargestatus', 'hospitaldischargelocation'], axis=1)
patient_df.head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create a discharge instance and the timestamp feature

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "04909971-1649-4b66-9620-a42ca246083a"}
patient_df['ts'] = 0
patient_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create a weight feature:

# + {"Collapsed": "false", "persistent_id": "33a7d452-9d5b-4ca3-9a3d-18a8b69129c1"}
# Create feature weight and assign the initial weight that the patient has on admission
patient_df['weight'] = patient_df['admissionweight']
patient_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Duplicate every row, so as to create a discharge event:

# + {"Collapsed": "false", "persistent_id": "1dd4b55f-4756-4c7d-8b43-b96bea38460e"}
new_df = patient_df.copy()
new_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Set the `weight` and `ts` features to initially have the value on admission and, on the second timestamp, have the value on discharge:

# + {"Collapsed": "false", "persistent_id": "d9386b99-6ef8-41b3-8828-dab2ceb81437"}
new_df.ts = new_df.unitdischargeoffset
new_df.weight = new_df.dischargeweight
new_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Join the new rows to the remaining dataframe:

# + {"Collapsed": "false", "persistent_id": "a8a1529e-bcc4-4cd7-83f8-7bc556e7274e"}
patient_df = patient_df.append(new_df)
patient_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the remaining, now unneeded, weight and timestamp features:

# + {"Collapsed": "false", "persistent_id": "55f51081-1df9-4dd5-ad95-b2532a3580f2"}
patient_df = patient_df.drop(['admissionweight', 'dischargeweight', 'unitdischargeoffset'], axis=1)
patient_df.head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `patientunitstayid` so as to check the data of the each patient together:

# + {"Collapsed": "false", "persistent_id": "7c060e72-ba83-4a77-9b7b-1ce20dc7057b"}
patient_df.sort_values(by='patientunitstayid').head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "ac72ba0f-62da-44fe-837f-4535095d5210"}
patient_df = patient_df.set_index('ts')
patient_df.head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Normalize data

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "8d66425b-0aae-42d5-ac47-9a0456a080b8"}
patient_df.to_csv(f'{data_path}cleaned/unnormalized/patient.csv')

# + {"Collapsed": "false", "persistent_id": "68e8080e-5cda-4459-ac38-b0eb59e79fac"}
new_cat_feat

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "d5ad6017-ad4a-419c-badb-9454add7752d"}
patient_df_norm = du.data_processing.normalize_data(patient_df, embed_columns=new_cat_feat,
                                                 id_columns=['patientunitstayid', 'deathoffset'])
patient_df_norm.head(6)

# + {"Collapsed": "false", "persistent_id": "64492d9f-df5d-4940-b931-cbb4c3af2949"}
patient_df_norm.to_csv(f'{data_path}cleaned/normalized/patient.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "68c630e8-151b-495f-b652-c00a55d92e78"}
patient_df_norm.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the unifying dataframe

# + {"Collapsed": "false", "persistent_id": "ac29f344-d3e2-4416-899b-26241e06e141"}
eICU_df = patient_df
eICU_df.head()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Vital signs periodic data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "cd9178cd-d41f-450e-9db4-c1ff4fd4935c"}
vital_prdc_df = pd.read_csv(f'{data_path}original/vitalPeriodic.csv')
vital_prdc_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "b1274441-3c94-48b3-9bf5-e1f133f4f593"}
vital_prdc_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "20479b2d-615f-4fdd-b1e2-f257dadabd32"}
vital_prdc_df.columns

# + {"Collapsed": "false", "persistent_id": "27313084-9469-4638-979d-9003881b5fa5"}
vital_prdc_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "a97b6e41-4560-440e-885a-4d33aa44c7b8"}
du.search_explore.dataframe_missing_values(patient_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "222c3734-547f-4fb0-afa9-2e9e4de27d80"}
patient_df = patient_df[['patientunitstayid', 'gender', 'age', 'ethnicity', 'apacheadmissiondx',  'admissionheight',
                         'hospitaldischargeoffset', 'hospitaldischargelocation', 'hospitaldischargestatus',
                         'admissionweight', 'dischargeweight', 'unitdischargeoffset']]
patient_df.head()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into simple numberings, one hot encode features with a low number of categories (in this case, 5) and enumerate sparse categorical features that will be embedded.

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Convert binary categorical features into numeric

# + {"Collapsed": "false", "persistent_id": "dde8c14e-d26b-4cc3-b34b-2f1b720a7880"}
patient_df.gender.value_counts()

# + {"Collapsed": "false", "persistent_id": "be4d23c5-f4a5-496e-b1a9-576c23238c93"}
patient_df.gender = patient_df.gender.map(lambda x: 1 if x == 'Male' else 0 if x == 'Female' else np.nan)

# + {"Collapsed": "false", "persistent_id": "506a84e1-2169-4ca5-a67f-c409d393dce4"}
patient_df.gender.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Separate and prepare features for embedding
#
# Identify categorical features that have more than 5 unique categories, which will go through an embedding layer afterwards, and enumerate them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Update list of categorical features and add those that will need embedding (features with more than 5 unique values):

# + {"Collapsed": "false", "persistent_id": "90ebd7db-dbbc-4cc6-8c2a-0580962e035b"}
new_cat_feat = ['ethnicity', 'apacheadmissiondx']
[cat_feat.append(col) for col in new_cat_feat]

# + {"Collapsed": "false", "persistent_id": "d35b752d-0622-49e8-8daf-af1cf2a0a634"}
cat_feat_nunique = [patient_df[feature].nunique() for feature in new_cat_feat]
cat_feat_nunique

# + {"Collapsed": "false", "persistent_id": "7e3e5de2-2f2c-4355-8550-8c21cab261ab"}
new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        # Add feature to the list of the new ones (from the current table) that will be embedded
        new_cat_embed_feat.append(new_cat_feat[i])

# + {"Collapsed": "false", "persistent_id": "6f596545-14e4-4e66-80e0-c6ff14de4712"}
patient_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "9ea61bdf-e985-4b7a-865f-b144e1b87a2c"}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    patient_df[feature], cat_embed_feat_enum[feature] = du.embedding.enum_categorical_feature(patient_df, feature)

# + {"Collapsed": "false", "persistent_id": "b3b2937b-c97c-4f5f-a38e-0b7cdeb1d252"}
patient_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "49e8379d-d2b3-4fbb-b522-12f74a7de158"}
cat_embed_feat_enum

# + {"Collapsed": "false", "persistent_id": "03bc7f87-f58b-4861-8a43-941c5b07537e"}
patient_df[cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "6c3ecabe-097a-47be-9f76-1a1e8ab7af79"}
stream = open('cat_embed_feat_enum.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "f8e1f0ee-539b-4a32-b09b-e25cf5bb2803"}
patient_df['ts'] = 0
patient_df = patient_df.drop('observationoffset', axis=1)
patient_df.head()

# + {"Collapsed": "false", "persistent_id": "c7e3373d-3b7c-4589-89d4-c438209a3654"}
patient_df.patientunitstayid.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "7d24e681-7f07-4e72-841f-d99788c2c4ce"}
len(patient_df)

# + {"Collapsed": "false", "persistent_id": "35e31b56-8c85-40ba-b31f-de3fc946da32"}
patient_df = patient_df.drop_duplicates()
patient_df.head()

# + {"Collapsed": "false", "persistent_id": "cc3838e1-9f79-4bb1-a6ad-9cb6419b687c"}
len(patient_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "b110a331-70ea-4522-b454-bb85edf64a65"}
vital_prdc_df = vital_prdc_df.set_index('ts')
vital_prdc_df.head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "916b433e-a194-45a5-a278-6766021abfd2"}
micro_df.reset_index().head()

# + {"Collapsed": "false", "persistent_id": "080e07ca-fe7c-40fd-b5ba-60d58df367ac"}
micro_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='culturesite').head()

# + {"Collapsed": "false", "persistent_id": "da199102-daa7-4168-b64e-021c662a5567"}
micro_df[micro_df.patientunitstayid == 3069495].head(20)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "4b1e0a15-d383-4a59-8157-7b4be95bf042"}
micro_df = du.embedding.join_categorical_enum(micro_df, new_cat_embed_feat)
micro_df.head()

# + {"Collapsed": "false", "persistent_id": "f0ec040f-7283-4015-8be6-11b30e8323a6"}
micro_df.dtypes

# + {"Collapsed": "false", "persistent_id": "f9163f0d-bd73-4dc8-a23e-6f72e67cf4fb"}
micro_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='culturesite').head()

# + {"Collapsed": "false", "persistent_id": "fb8184d5-8f93-463f-b8a6-95a456f6ae11"}
micro_df[micro_df.patientunitstayid == 3069495].head(20)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Normalize data

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "18d36906-c1a2-4221-8f6c-fbf429a076b2"}
patient_df_norm = du.data_processing.normalize_data(patient_df, embed_columns=new_cat_feat,
                                                    id_columns=['patientunitstayid', 'ts', 'deathoffset'])
patient_df_norm.head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "0c7829e8-0a50-4d19-82ca-a24a8f33f62a"}
patient_df_norm.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "e3d64b8c-edf2-4890-a5a8-402fda1a3bf6"}
patient_df.columns = du.data_processing.clean_naming(patient_df.columns)
patient_df_norm.columns = du.data_processing.clean_naming(patient_df_norm.columns)
patient_df_norm.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "e8407ff8-64d1-4726-a483-a4b94e8f2099"}
patient_df.to_csv(f'{data_path}cleaned/unnormalized/patient.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "684dd75e-670e-4cf0-8cb4-33237cd47788"}
patient_df_norm.to_csv(f'{data_path}cleaned/normalized/patient.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "97e7dd86-887d-434a-86cb-34ca8be8ea74"}
patient_df_norm.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

# + {"Collapsed": "false", "persistent_id": "39228b3c-69bb-433f-8c02-fe882c35d273"}
patient_df = pd.read_csv(f'{data_path}cleaned/normalized/patient.csv')
patient_df.head()

# + {"Collapsed": "false", "persistent_id": "5ade0956-2518-4358-b5d4-59d24c908ab9"}
vital_prdc_df = pd.read_csv(f'{data_path}cleaned/normalized/vitalPeriodic.csv')
vital_prdc_df.head()

# + {"Collapsed": "false", "persistent_id": "17292c14-3a8c-420c-95e2-172f3261869a"}
eICU_df = pd.merge_asof(patient_df, vital_aprdc_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Vital signs aperiodic data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "c6d14aa0-f97e-4df0-9270-30be24a0857d"}
vital_aprdc_df = pd.read_csv(f'{data_path}original/vitalAperiodic.csv')
vital_aprdc_df.head()

# + {"Collapsed": "false", "persistent_id": "fc66c7c7-6f53-4a53-9685-2beba25630b6"}
len(vital_aprdc_df)

# + {"Collapsed": "false", "persistent_id": "40b7732d-3305-455a-96f8-8df25739c460"}
vital_aprdc_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "6fb67a70-a28f-408d-a8a3-7734c02f9916"}
vital_aprdc_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "8882ad56-cb8c-429f-8479-0616aecaa180"}
vital_aprdc_df.columns

# + {"Collapsed": "false", "persistent_id": "83d0ad98-3a84-42f2-a3a4-9a3aa589dd9a"}
vital_aprdc_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "393c57d3-b2ae-44e3-bd04-7c3f8f522ff5"}
du.search_explore.dataframe_missing_values(vital_aprdc_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "2ac7cd0d-c94a-4d80-a262-71c1c3369696"}
vital_aprdc_df = vital_aprdc_df.drop('vitalaperiodicid', axis=1)
vital_aprdc_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "b18b4887-b49f-4341-b28e-6c7866e53bb8"}
vital_aprdc_df['ts'] = vital_aprdc_df['observationoffset']
vital_aprdc_df = vital_aprdc_df.drop('observationoffset', axis=1)
vital_aprdc_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "e96d6869-1e4e-48f0-bdf6-9610b2341cc5"}
len(vital_aprdc_df)

# + {"Collapsed": "false", "persistent_id": "6973c75e-d369-4dbb-a137-8da2d62bcd2e"}
vital_aprdc_df = vital_aprdc_df.drop_duplicates()
vital_aprdc_df.head()

# + {"Collapsed": "false", "persistent_id": "484bafd6-45c4-47b1-bca6-2feb56a3f67a"}
len(vital_aprdc_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "f353a49b-c4cf-427b-8f51-987d998a8ae4"}
vital_aprdc_df = vital_aprdc_df.set_index('ts')
vital_aprdc_df.head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "32516efa-0ea4-4308-a50c-a7117fcb9d7b"}
vital_aprdc_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='noninvasivemean').head()

# + {"Collapsed": "false", "persistent_id": "112555fe-0484-4ab2-a325-71007d3db114"}
vital_aprdc_df[micro_df.patientunitstayid == 3069495].head(20)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "df09ded9-fad0-494e-a9a6-b791fdfd9030"}
micro_df = du.embedding.join_categorical_enum(micro_df, new_cat_embed_feat)
micro_df.head()

# + {"Collapsed": "false", "persistent_id": "85242cf3-1e23-4b70-902f-7c480ecc98d4"}
micro_df.dtypes

# + {"Collapsed": "false", "persistent_id": "30583cb6-8b9c-4e86-899d-98e5e6b04b5b"}
micro_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='culturesite').head()

# + {"Collapsed": "false", "persistent_id": "1b1717b8-3b9c-48da-8c4b-56ee7c4b8585"}
micro_df[micro_df.patientunitstayid == 3069495].head(20)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Normalize data

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "c1194923-4e5f-4017-a803-f0ff82777864"}
vital_aprdc_df_norm = du.data_processing.normalize_data(vital_aprdc_df,
                                                        id_columns=['patientunitstayid', 'ts'])
vital_aprdc_df_norm.head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "b6e79305-fb40-4242-84b6-35e0b0eb249d"}
vital_aprdc_df_norm.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "a55478a9-64e2-48a9-b790-2b5069e58eb7"}
vital_aprdc_df.columns = du.data_processing.clean_naming(vital_aprdc_df.columns)
vital_aprdc_df_norm.columns = du.data_processing.clean_naming(vital_aprdc_df_norm.columns)
vital_aprdc_df_norm.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "3a3abe4a-43b8-4525-9639-9362baf94878"}
vital_aprdc_df.to_csv(f'{data_path}cleaned/unnormalized/vitalAperiodic.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "ddded5d4-9dc6-49b7-8405-f44cc827de5e"}
vital_aprdc_df_norm.to_csv(f'{data_path}cleaned/normalized/vitalAperiodic.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "21223dc0-9fe2-4c30-9b23-bd7a10bc502c"}
vital_aprdc_df_norm.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

# + {"Collapsed": "false", "persistent_id": "34bfcfa0-6ea0-4eb1-9318-ea04c875fc37"}
vital_aprdc_df = pd.read_csv(f'{data_path}cleaned/normalized/vitalAperiodic.csv')
vital_aprdc_df.head()

# + {"Collapsed": "false", "persistent_id": "843051ee-39c5-47ac-a021-68b8cbed7540"}
len(vital_aprdc_df)

# + {"Collapsed": "false", "persistent_id": "0c219a72-122a-451a-821e-5dc4e9fee688"}
vital_aprdc_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "65d58a35-f921-4c45-aa61-320f096e292f"}
eICU_df = pd.merge_asof(eICU_df, vital_aprdc_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Infectious disease data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "7cda9763-a1aa-452c-94c9-a4f7b6ba5fe1"}
infect_df = pd.read_csv(f'{data_path}original/carePlanInfectiousDisease.csv')
infect_df.head()

# + {"Collapsed": "false", "persistent_id": "633afa90-fd10-4baa-b46c-654ca67a794a"}
infect_df.infectdiseasesite.value_counts().head(10)

# + {"Collapsed": "false", "persistent_id": "1083b8ef-ee83-43de-b027-8d5cd8772be2"}
infect_df.infectdiseaseassessment.value_counts().head(10)

# + {"Collapsed": "false", "persistent_id": "e9857111-c753-4fd6-b24f-037fb2e90e61"}
infect_df.responsetotherapy.value_counts().head(10)

# + {"Collapsed": "false", "persistent_id": "83c8bbdd-88d9-45e7-8b4d-11f9b117c944"}
infect_df.treatment.value_counts().head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Most features in this table either don't add much information or they have a lot of missing values. The truly relevant one seems to be `infectdiseasesite`. Even `activeupondischarge` doesn't seem very practical as we don't have complete information as to when infections end, might as well just register when they are first verified.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "6b65e5c6-f96f-4264-b15f-51faff06e0e7"}
infect_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "969eee87-ef4b-4e15-854c-f9dd8707b299"}
infect_df.columns

# + {"Collapsed": "false", "persistent_id": "4a160482-dcd8-4646-a222-816bed785008"}
infect_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "25a7cc24-e778-4007-9507-d12a88f24976"}
du.search_explore.dataframe_missing_values(infect_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "0272ddf3-bbe8-4b52-8c0c-8b091082f6ed"}
infect_df = infect_df[['patientunitstayid', 'cplinfectdiseaseoffset', 'infectdiseasesite']]
infect_df.head()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into simple numberings, one hot encode features with a low number of categories (in this case, 5) and enumerate sparse categorical features that will be embedded.

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Separate and prepare features for embedding
#
# Identify categorical features that have more than 5 unique categories, which will go through an embedding layer afterwards, and enumerate them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Update list of categorical features and add those that will need embedding (features with more than 5 unique values):

# + {"Collapsed": "false", "persistent_id": "6bae9421-c734-4cad-9693-bcdfdf6b81da"}
new_cat_feat = ['infectdiseasesite']
[cat_feat.append(col) for col in new_cat_feat]

# + {"Collapsed": "false", "persistent_id": "06eba6c6-50ac-410a-8595-35d5f3d917c3"}
cat_feat_nunique = [infect_df[feature].nunique() for feature in new_cat_feat]
cat_feat_nunique

# + {"Collapsed": "false", "persistent_id": "71f806a5-b337-4f7d-84da-2b438c3101a1"}
new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        # Add feature to the list of the new ones (from the current table) that will be embedded
        new_cat_embed_feat.append(new_cat_feat[i])

# + {"Collapsed": "false", "persistent_id": "f69d44c6-92fa-4fe5-b9c1-26fc1d859d5e"}
infect_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "7d1056c2-0522-4671-b232-ab47ad2a4bf8"}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    infect_df[feature], cat_embed_feat_enum[feature] = du.embedding.enum_categorical_feature(infect_df, feature)

# + {"Collapsed": "false", "persistent_id": "86dd313a-daeb-4ef6-9ff4-def549e84deb"}
infect_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "94bde04b-a588-4be0-9c7f-549966d572f8"}
cat_embed_feat_enum

# + {"Collapsed": "false", "persistent_id": "ada453e4-6d5d-42ac-8461-b4efc83bb769"}
infect_df[cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "81a5d0a9-20e1-48f5-a2e3-05b3f6409d9a"}
stream = open('cat_embed_feat_enum.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "6fbe3262-7b2b-401d-bdcb-8dcd19a5d1ac"}
infect_df['ts'] = infect_df['cplinfectdiseaseoffset']
infect_df = infect_df.drop('cplinfectdiseaseoffset', axis=1)
infect_df.head()

# + {"Collapsed": "false", "persistent_id": "8dfd9e62-75bf-4810-8065-dbca71b50692"}
infect_df.patientunitstayid.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Only 3620 unit stays have infection data. Might not be useful to include them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "6f0c4357-8559-4352-b005-6e918407d243"}
len(infect_df)

# + {"Collapsed": "false", "persistent_id": "2bf11f9d-5493-4107-91fa-250e702ff5ab"}
infect_df = infect_df.drop_duplicates()
infect_df.head()

# + {"Collapsed": "false", "persistent_id": "db46e90c-7286-4bd7-ad56-e4c08889350f"}
len(infect_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "85b7d8bb-34b9-4a4f-93df-3c818d1c7ac5"}
infect_df = infect_df.set_index('ts')
infect_df.head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "e97829b8-3ab6-4283-a73d-db7abc415dc8"}
infect_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='infectdiseasesite').head()

# + {"Collapsed": "false", "persistent_id": "c5a508da-88d1-4ce1-9e07-ecd1d01491e6"}
infect_df[infect_df.patientunitstayid == 3049689].head(20)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 6 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "68494ca4-58c9-473e-bfea-ac10daaee5ce"}
infect_df = du.embedding.join_categorical_enum(infect_df, new_cat_embed_feat)
infect_df.head()

# + {"Collapsed": "false", "persistent_id": "8e26bb66-0566-4ace-9614-7ee161c7898a"}
infect_df.dtypes

# + {"Collapsed": "false", "persistent_id": "5de6476e-7cb1-45c7-b686-08690bc2edcc"}
infect_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='infectdiseasesite').head()

# + {"Collapsed": "false", "persistent_id": "9678c10a-dc47-4fb7-af88-7fb807153629"}
infect_df[infect_df.patientunitstayid == 3049689].head(20)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Normalize data

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "0b478bbf-6ced-4014-ae00-aae8c96d780f"}
infect_df_norm = du.data_processing.normalize_data(infect_df, embed_columns=new_cat_feat,
                                                id_columns=['patientunitstayid'])
infect_df_norm.head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "edd7585c-3c11-4af1-8252-98bf9a530641"}
infect_df_norm.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "32d61cf3-eda8-4f53-9f4b-60b9d7061e08"}
infect_df.columns = du.data_processing.clean_naming(infect_df.columns)
infect_df_norm.columns = du.data_processing.clean_naming(infect_df_norm.columns)
infect_df_norm.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "0bbc0adb-b8e9-4cdd-94d3-87c488e2301c"}
infect_df.to_csv(f'{data_path}cleaned/unnormalized/carePlanInfectiousDisease.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "1202f15c-3b4c-4cd9-a061-6127bdbc0a1f"}
infect_df_norm.to_csv(f'{data_path}cleaned/normalized/carePlanInfectiousDisease.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "df967371-2739-49d0-86aa-7df0fcdf5277"}
infect_df_norm.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

# + {"Collapsed": "false", "persistent_id": "1cc4b871-1c7a-4c19-a814-ba79dbd8f579"}
infect_df = pd.read_csv(f'{data_path}cleaned/normalized/carePlanInfectiousDisease.csv')
infect_df.head()

# + {"Collapsed": "false", "persistent_id": "986e1552-6296-444b-bdf5-7df8743b253e"}
len(infect_df)

# + {"Collapsed": "false", "persistent_id": "52752941-71c3-4651-afa6-f69fd6987afd"}
infect_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "35f6f243-498a-4555-a84b-e018149ae2c0"}
eICU_df = pd.merge_asof(eICU_df, infect_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Microbiology data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "2ba5ce3f-58ba-4cb5-afbb-ab56cc3d9860"}
micro_df = pd.read_csv(f'{data_path}original/microLab.csv')
micro_df.head()

# + {"Collapsed": "false", "persistent_id": "ff8cd7eb-4da8-42ee-98de-01ced243bb51"}
len(micro_df)

# + {"Collapsed": "false", "persistent_id": "8c7879d4-ef60-4383-b6ad-1201569f610a"}
micro_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Only 2923 unit stays have microbiology data. Might not be useful to include them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "271f9953-c855-443c-956d-cad252be2cda"}
micro_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "3ec550d5-d260-49d0-bab6-bf8a4bd6e246"}
micro_df.columns

# + {"Collapsed": "false", "persistent_id": "da6e17b2-0261-4f0c-b63e-87c884bc0c7c"}
micro_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "a76796a4-8375-4eaa-9a10-5766269edc1b"}
du.search_explore.dataframe_missing_values(micro_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "c25394ca-57e5-4b1f-ada1-ae9d792fd181"}
micro_df.culturesite.value_counts()

# + {"Collapsed": "false", "persistent_id": "3a673167-75db-4bea-bf18-7da32ce32603"}
micro_df.organism.value_counts()

# + {"Collapsed": "false", "persistent_id": "d5c1e89e-9e4d-4ef3-aaff-582c6acefb29"}
micro_df.antibiotic.value_counts()

# + {"Collapsed": "false", "persistent_id": "85b67f10-6270-4af4-b456-ebbdbaa33fbd"}
micro_df.sensitivitylevel.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# All features appear to be relevant, except the unique identifier of the table.

# + {"Collapsed": "false", "persistent_id": "d1474fd5-f1ce-468b-a433-807ac28feb1e"}
micro_df = micro_df.drop('microlabid', axis=1)
micro_df.head()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into simple numberings, one hot encode features with a low number of categories (in this case, 5) and enumerate sparse categorical features that will be embedded.

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Separate and prepare features for embedding
#
# Identify categorical features that have more than 5 unique categories, which will go through an embedding layer afterwards, and enumerate them.
#
# In the case of microbiology data, we're also going to embed the antibiotic `sensitivitylevel`, not because it has many categories, but because there can be several rows of data per timestamp (which would be impractical on one hot encoded data).

# + {"Collapsed": "false", "cell_type": "markdown"}
# Update list of categorical features and add those that will need embedding (features with more than 5 unique values):

# + {"Collapsed": "false", "persistent_id": "176feeef-120d-487e-821d-7739594fd455"}
new_cat_feat = ['culturesite', 'organism', 'antibiotic', 'sensitivitylevel']
[cat_feat.append(col) for col in new_cat_feat]

# + {"Collapsed": "false", "persistent_id": "fa18fe2c-76cf-497e-ad7e-f62dc1251893"}
cat_feat_nunique = [micro_df[feature].nunique() for feature in new_cat_feat]
cat_feat_nunique

# + {"Collapsed": "false", "persistent_id": "4aa39c31-05ca-4761-bbdb-b1862991eba9"}
new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5 or new_cat_feat[i] == 'sensitivitylevel':
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        new_cat_embed_feat.append(new_cat_feat[i])

# + {"Collapsed": "false", "persistent_id": "41718a73-0c1a-42c5-a222-a5c605c78d84"}
micro_df[new_cat_feat].head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "53d94db8-ef6b-4f5c-8ea6-8816c0a8ce2a"}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    micro_df[feature], cat_embed_feat_enum[feature] = du.embedding.enum_categorical_feature(micro_df, feature)

# + {"Collapsed": "false", "persistent_id": "b178f2e0-4051-4a6b-b352-1f1b37cab4ff"}
micro_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "77904e0c-98b6-4e01-8e1d-c5a4f94683e6"}
cat_embed_feat_enum

# + {"Collapsed": "false", "persistent_id": "f25724cc-3d62-4ad3-a8cc-d4273d2a6dd8"}
micro_df[new_cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "20f2e12b-7e02-4663-b856-ceeeac9d2202"}
stream = open('cat_embed_feat_enum.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "e1db480d-44f4-4086-a850-4a3d2cc92286"}
micro_df['ts'] = micro_df['culturetakenoffset']
micro_df = micro_df.drop('culturetakenoffset', axis=1)
micro_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "52029cf9-eefd-479e-af95-df7ac41cc009"}
len(micro_df)

# + {"Collapsed": "false", "persistent_id": "5d35fe68-9580-40f9-ab87-b2ef4f34c433"}
micro_df = micro_df.drop_duplicates()
micro_df.head()

# + {"Collapsed": "false", "persistent_id": "d8a5d20a-e032-4407-acaa-fb5d57603449"}
len(micro_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "45d8d316-ec10-4777-804e-4a8cd3e7d2ac"}
micro_df = micro_df.set_index('ts')
micro_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "ffd19a90-d8a2-420b-a83a-04de0abab311"}
micro_df.reset_index().head()

# + {"Collapsed": "false", "persistent_id": "daa89836-1427-4874-924d-ca0b1e3128a9"}
micro_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='culturesite').head()

# + {"Collapsed": "false", "persistent_id": "08e63f76-c189-48f4-b2f6-d5da635a1761"}
micro_df[micro_df.patientunitstayid == 3069495].head(20)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 120 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "aaa129e5-7831-4cec-adcd-9fa7982544f9"}
micro_df = du.embedding.join_categorical_enum(micro_df, new_cat_embed_feat)
micro_df.head()

# + {"Collapsed": "false", "persistent_id": "ec2b646d-1155-430f-8036-e8fb829061c5"}
micro_df.dtypes

# + {"Collapsed": "false", "persistent_id": "90c18d28-2d50-4b04-8b2d-b37ad618b13f"}
micro_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='culturesite').head()

# + {"Collapsed": "false", "persistent_id": "9bf3a30b-796a-45ec-956d-c1ad3d498e00"}
micro_df[micro_df.patientunitstayid == 3069495].head(20)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Normalize data

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "571ce3b2-26cc-432f-ad27-afc497c75dc5"}
micro_df_norm = du.data_processing.normalize_data(micro_df, embed_columns=new_cat_feat,
                                               id_columns=['patientunitstayid'])
micro_df_norm.head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "e47fb477-0a20-452f-a7ee-89a6c10c9fbc"}
micro_df_norm.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "c2a0a5f7-5df5-49b6-a018-4bc059513112"}
micro_df.columns = du.data_processing.clean_naming(micro_df.columns)
micro_df_norm.columns = du.data_processing.clean_naming(micro_df_norm.columns)
micro_df_norm.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "7b515c93-2ae7-4b6a-872e-86793172393b"}
micro_df.to_csv(f'{data_path}cleaned/unnormalized/microLab.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "42a54b19-5560-4bf2-b3c7-91ca71f5666a"}
micro_df_norm.to_csv(f'{data_path}cleaned/normalized/microLab.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "8939ab80-f434-403f-8361-778c8135bc26"}
micro_df_norm.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

# + {"Collapsed": "false", "persistent_id": "23d10537-45a7-46f9-a579-c842cbd4046d"}
micro_df = pd.read_csv(f'{data_path}cleaned/normalized/microLab.csv')
micro_df.head()

# + {"Collapsed": "false", "persistent_id": "a5062d03-a549-409d-924d-69390dd6d15c"}
len(micro_df)

# + {"Collapsed": "false", "persistent_id": "54c102a1-7d98-4abd-b0fe-b1539033a153"}
micro_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "36c7e7f4-463c-4303-96c7-f75beb99ca32"}
eICU_df = pd.merge_asof(eICU_df, micro_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Respiratory care data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "31b57ee7-87a4-4461-9364-7eaf4abc43fb"}
resp_care_df = pd.read_csv(f'{data_path}original/respiratoryCare.csv', dtype={'airwayposition': 'object',
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

# + {"Collapsed": "false", "persistent_id": "7e4f5e1d-8edb-414d-99af-3c1edfef494f"}
len(resp_care_df)

# + {"Collapsed": "false", "persistent_id": "a8d54bcb-bbdc-43cf-9310-b455cd0a1b58"}
resp_care_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "d5b5a661-50c2-4321-8371-2bcc7f42fbd0"}
resp_care_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "b9d475cc-9239-4c50-94ef-fb325803d199"}
resp_care_df.columns

# + {"Collapsed": "false", "persistent_id": "b7a2385f-3b3c-485e-8941-a2632e65b5ee"}
resp_care_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "4da4745a-11cf-4122-832f-ac21821b5f6b"}
du.search_explore.dataframe_missing_values(resp_care_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "cell_type": "markdown"}
# For the respiratoryCare table, I'm not going to use any of the several features that detail what the vent in the hospital is like. Besides not appearing to be very relevant for the patient, they have a lot of missing values (>67%). Instead, I'm going to set a ventilation label (between the start and the end), and a previous ventilation label.

# + {"Collapsed": "false", "persistent_id": "3423d4d9-c1f1-47c8-8775-2d6d0c24a64c"}
resp_care_df = resp_care_df[['patientunitstayid', 'ventstartoffset',
                             'ventendoffset', 'priorventstartoffset']]
resp_care_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "1e985184-2fe7-49a9-bdad-b383420fe2c2"}
resp_care_df['ts'] = resp_care_df['ventstartoffset']
resp_care_df = resp_care_df.drop('ventstartoffset', axis=1)
resp_care_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "1bcca280-3e84-4b24-a865-5eba8ef9e462"}
len(resp_care_df)

# + {"Collapsed": "false", "persistent_id": "354abff2-b9c9-4647-bb08-380468271ca1"}
resp_care_df = resp_care_df.drop_duplicates()
resp_care_df.head()

# + {"Collapsed": "false", "persistent_id": "f9ccc3e0-b440-40b1-ad62-1a1ec5f92604"}
len(resp_care_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "5999db50-c412-4514-9924-bcf0d7b4b20f"}
resp_care_df = resp_care_df.set_index('ts')
resp_care_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "08b71ed2-1822-4f96-9fd8-fd6c88742e99"}
resp_care_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='ventendoffset').head()

# + {"Collapsed": "false", "persistent_id": "678fefff-a384-4c44-ab09-86119e6d4087"}
resp_care_df[resp_care_df.patientunitstayid == 3348331].head(20)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 5283 duplicate rows per set of `patientunitstayid` and `ts`. As such, we must join them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"Collapsed": "false", "cell_type": "markdown"}
# Even after removing duplicates rows, there are still some that have different information for the same ID and timestamp. We have to apply a groupby function, selecting the minimum value for each of the offset features, as the larger values don't make sense (in the `priorventstartoffset`).

# + {"Collapsed": "false", "persistent_id": "092a9ef1-2caa-4d53-b63a-6c641e5a6b46"}
((resp_care_df.index > resp_care_df.ventendoffset) & resp_care_df.ventendoffset != 0).value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# There are no errors of having the start vent timestamp later than the end vent timestamp.

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "6e8377a7-4e89-4371-a663-bd10b5dcf5d9"}
resp_care_df = du.embedding.join_categorical_enum(resp_care_df, cont_join_method='min')
resp_care_df.head()

# + {"Collapsed": "false", "persistent_id": "b054019f-50a4-4326-8bc6-bd31966bbeb4"}
resp_care_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='ventendoffset').head()

# + {"Collapsed": "false", "persistent_id": "741c918e-715b-467a-a775-f012e48b56ba"}
resp_care_df[resp_care_df.patientunitstayid == 1113084].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Only keep the first instance of each patient, as we're only keeping track of when they are on ventilation:

# + {"Collapsed": "false", "persistent_id": "988fcf02-dc89-4808-be0b-ed8f7c55d44a"}
resp_care_df = resp_care_df.reset_index().groupby('patientunitstayid').first().reset_index().set_index('ts')
resp_care_df.head(20)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create prior ventilation label
#
# Make a feature `priorvent` that indicates if the patient has been on ventilation before.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Convert to pandas:

# + {"Collapsed": "false", "persistent_id": "e7a14e8f-c9a5-45ae-a760-2416e3a4a740"}
resp_care_df = resp_care_df

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the prior ventilation column:

# + {"Collapsed": "false", "persistent_id": "447b27eb-02bd-42a6-8b0d-e90d350add29"}
resp_care_df['priorvent'] = (resp_care_df.priorventstartoffset < resp_care_df.index).astype(int)
resp_care_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the now unneeded `priorventstartoffset` column:

# + {"Collapsed": "false", "persistent_id": "6afed494-9efd-485a-b80d-d91b7a1f6b69"}
resp_care_df = resp_care_df.drop('priorventstartoffset', axis=1)
resp_care_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create current ventilation label
#
# Make a feature `onvent` that indicates if the patient is currently on ventilation.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create a `onvent` feature:

# + {"Collapsed": "false", "persistent_id": "ccb59a41-188f-43fe-b1ad-99dfbe0ba317"}
resp_care_df['onvent'] = 1
resp_care_df.head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Reset index to allow editing the `ts` column:

# + {"Collapsed": "false", "persistent_id": "51605a70-099e-4c45-a2c7-4bd19d164cf1"}
resp_care_df = resp_care_df.reset_index()
resp_care_df.head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Duplicate every row, so as to create a discharge event:

# + {"Collapsed": "false", "persistent_id": "601990ac-7417-4f9a-9f42-d326c284c96b"}
new_df = resp_care_df.copy()
new_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Set the new dataframe's rows to have the ventilation stop timestamp, indicating that ventilation use ended:

# + {"Collapsed": "false", "persistent_id": "c85f1bd8-15c5-41a0-a859-62316f3cc6f7"}
new_df.ts = new_df.ventendoffset
new_df.onvent = 0
new_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Join the new rows to the remaining dataframe:

# + {"Collapsed": "false", "persistent_id": "5d2b5f80-c1f4-446f-9a65-2b40a8e981ad"}
resp_care_df = resp_care_df.append(new_df)
resp_care_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "d2439d3c-3b47-4a13-a822-02540510714b"}
resp_care_df = resp_care_df.set_index('ts')
resp_care_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the now unneeded ventilation end column:

# + {"Collapsed": "false", "persistent_id": "cdb7b43c-010c-4679-89a4-d3c7f891c042"}
resp_care_df = resp_care_df.drop('ventendoffset', axis=1)
resp_care_df.head(6)

# + {"Collapsed": "false", "persistent_id": "1ddedcda-ccb2-455b-b40d-74fb31da9572"}
resp_care_df.tail(6)

# + {"Collapsed": "false", "persistent_id": "4f64da67-f9be-4600-ad56-d9e8515f5a09"}
resp_care_df[resp_care_df.patientunitstayid == 1557538]

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "c98fad52-78bf-408a-a77c-1d03a2e8afc0"}
resp_care_df.columns = du.data_processing.clean_naming(resp_care_df.columns)
resp_care_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "8da4c80c-a6a7-499f-90b2-e86416218caf"}
resp_care_df.to_csv(f'{data_path}cleaned/unnormalized/respiratoryCare.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "305ff4a7-e234-4f35-87d6-f3023063b472"}
resp_care_df.to_csv(f'{data_path}cleaned/normalized/respiratoryCare.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "749d2891-1cc2-469b-884e-2617fdfef0bf"}
resp_care_df.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

# + {"Collapsed": "false", "persistent_id": "4722cdd4-ed12-4e97-b7f5-19dff576e4d4"}
resp_care_df = pd.read_csv(f'{data_path}cleaned/normalized/respiratoryCare.csv')
resp_care_df.head()

# + {"Collapsed": "false", "persistent_id": "0646049d-7687-49cd-a54d-5d478dab509a"}
len(resp_care_df)

# + {"Collapsed": "false", "persistent_id": "fcfec6e7-d79a-4133-956c-40723e1c4582"}
resp_care_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "f65b8ef3-d72c-49de-b6af-b914d5c7c7e7"}
eICU_df = pd.merge_asof(eICU_df, resp_care_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": false, "Collapsed": "false", "cell_type": "markdown"}
# ## Respiratory charting data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "f8771315-8278-43fa-8c33-a8748c269d5a"}
resp_chart_df = pd.read_csv(f'{data_path}original/respiratoryCharting.csv')
resp_chart_df.head()

# + {"Collapsed": "false", "persistent_id": "17a209eb-a745-4c8a-9422-07d9da331dc2"}
len(resp_chart_df)

# + {"Collapsed": "false", "persistent_id": "2116419b-d1c8-4cfd-8d1f-578f2be29d95"}
resp_chart_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Only 13001 unit stays have nurse care data. Might not be useful to include them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "3ca8d218-7980-4545-9915-b081eaacbbcb"}
resp_chart_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "bf9fe2b3-b846-40d8-b804-76d1914cc57d"}
resp_chart_df.columns

# + {"Collapsed": "false", "persistent_id": "c35a5466-3be5-44f5-9d8c-ca6df564fd8f"}
resp_chart_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "5f2636df-f591-47b3-b9e1-994dd65cd912"}
du.search_explore.dataframe_missing_values(resp_chart_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "86c870c5-4c2a-4084-bb46-37d04c03cf9e"}
resp_chart_df.celllabel.value_counts()

# + {"Collapsed": "false", "persistent_id": "5a4c9cce-595f-4cf9-bb21-363726870766"}
resp_chart_df.cellattribute.value_counts()

# + {"Collapsed": "false", "persistent_id": "5af0eca1-7d4a-40c1-8c39-1597543097b4"}
resp_chart_df.cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "b5f4b97a-1b38-41e4-9bec-d1a8b882ffa7"}
resp_chart_df.cellattributepath.value_counts()

# + {"Collapsed": "false", "persistent_id": "2561d234-871d-461d-b9fc-4d11b2ff837f"}
resp_chart_df[resp_chart_df.celllabel == 'Intervention'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "d814021a-4a13-4936-9eb7-55b55c6737d4"}
resp_chart_df[resp_chart_df.celllabel == 'Neurologic'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "a9d43c15-a1b7-437f-b26a-9abbb4a88f9a"}
resp_chart_df[resp_chart_df.celllabel == 'Pupils'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "81597b29-4eb1-40af-86ef-4fc7d974d504"}
resp_chart_df[resp_chart_df.celllabel == 'Edema'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "91cdb889-0e10-43d1-94eb-cef7a99064af"}
resp_chart_df[resp_chart_df.celllabel == 'Secretions'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "ca8953ef-59d8-46cf-a552-ad84dfe40d06"}
resp_chart_df[resp_chart_df.celllabel == 'Cough'].cellattributevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "cada2e4c-8a50-4738-8721-b4e8cc3451e9"}
resp_chart_df[resp_chart_df.celllabel == 'Neurologic'].cellattribute.value_counts()

# + {"Collapsed": "false", "persistent_id": "384b7657-8020-4ad8-95f7-92ee96a3340f"}
resp_chart_df[resp_chart_df.celllabel == 'Pupils'].cellattribute.value_counts()

# + {"Collapsed": "false", "persistent_id": "ad81fe0d-124b-4501-9c68-79a40cfa6dd3"}
resp_chart_df[resp_chart_df.celllabel == 'Secretions'].cellattribute.value_counts()

# + {"Collapsed": "false", "persistent_id": "06a961f4-f377-405f-93d3-00c069e21eaa"}
resp_chart_df[resp_chart_df.celllabel == 'Cough'].cellattribute.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Besides the usual removal of row identifier, `nurseAssessID`, and the timestamp when data was added, `nurseAssessEntryOffset`, I'm also removing `cellattributepath` and `cellattribute`, which have redundant info with `celllabel`. Regarding data categories, I'm only keeping `Neurologic`, `Pupils`, `Secretions` and `Cough`, as the remaining ones either don't add much value, have too little data or are redundant with data from other tables.

# + {"Collapsed": "false", "persistent_id": "1c1baaa0-ddb2-4f75-be33-f56ca0fde887"}
resp_chart_df = resp_chart_df.drop(['nurseassessid', 'nurseassessentryoffset',
                                      'cellattributepath', 'cellattribute'], axis=1)
resp_chart_df.head()

# + {"Collapsed": "false", "persistent_id": "ba1a06ca-5881-4a2f-ab86-905cbdbcc996"}
categories_to_keep = ['Neurologic', 'Pupils', 'Secretions', 'Cough']

# + {"Collapsed": "false", "persistent_id": "26f3829d-890f-44ab-8a75-57126198240b"}
resp_chart_df.celllabel.isin(categories_to_keep).head()

# + {"Collapsed": "false", "persistent_id": "bcc38c95-1ede-4524-ba62-416a73339f9e"}
resp_chart_df = resp_chart_df[resp_chart_df.celllabel.isin(categories_to_keep)]
resp_chart_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Convert categories to features

# + {"Collapsed": "false", "cell_type": "markdown"}
# Make the `celllabel` and `cellattributevalue` columns of type categorical:

# + {"Collapsed": "false", "persistent_id": "2757b4ac-1e27-4bf3-b38c-f018904e74f3"}
resp_chart_df = resp_chart_df.categorize(columns=['celllabel', 'cellattributevalue'])

# + {"Collapsed": "false", "persistent_id": "62095f1a-f187-4849-8486-c079d2101f91"}
resp_chart_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Transform the `celllabel` categories and `cellattributevalue` values into separate features:

# + {"Collapsed": "false", "cell_type": "markdown"}
# Now we have the categories separated into their own features, as desired.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the old `celllabel` and `cellattributevalue` columns:

# + {"Collapsed": "false", "persistent_id": "46aeb95c-2887-4030-90f6-7f337fc84427"}
resp_chart_df = resp_chart_df.drop(['celllabel', 'cellattributevalue'], axis=1)
resp_chart_df.head()

# + {"Collapsed": "false", "persistent_id": "6ca2f752-535d-44c7-b0e4-e1791be24d86"}
resp_chart_df['Neurologic'].value_counts()

# + {"Collapsed": "false", "persistent_id": "4d8b4ba3-3f59-4caa-ac1f-8376dd313c2b"}
resp_chart_df['Pupils'].value_counts()

# + {"Collapsed": "false", "persistent_id": "e59e43cd-fab3-49c9-aa1a-314a02badd3c"}
resp_chart_df['Secretions'].value_counts()

# + {"Collapsed": "false", "persistent_id": "a2d59b7a-961d-4f7e-aea8-bbe1aee56b33"}
resp_chart_df['Cough'].value_counts()

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

# + {"Collapsed": "false", "persistent_id": "7baf17c0-81b2-4a1b-8a5a-7e8f383e9847"}
new_cat_feat = ['Pupils', 'Neurologic', 'Secretions', 'Cough']
[cat_feat.append(col) for col in new_cat_feat]

# + {"Collapsed": "false", "persistent_id": "8d1f787d-b95d-487d-a0cb-5877e123e666"}
cat_feat_nunique = [resp_chart_df[feature].nunique() for feature in new_cat_feat]
cat_feat_nunique

# + {"Collapsed": "false", "persistent_id": "6d2ffa56-c3fc-4bbf-ad2b-5dcbdd8e733a"}
new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        new_cat_embed_feat.append(new_cat_feat[i])

# + {"Collapsed": "false", "persistent_id": "bef728e1-1608-426e-852a-37e2cf47935c"}
resp_chart_df[new_cat_feat].head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "f691d6e7-9475-4a5f-a6b8-d1223b9eebe3"}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    resp_chart_df[feature], cat_embed_feat_enum[feature] = du.embedding.enum_categorical_feature(resp_chart_df, feature)

# + {"Collapsed": "false", "persistent_id": "151d0866-afbb-486c-b4b4-204fda79a0b8"}
resp_chart_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "8768f50d-589c-44a1-99ea-429f312df58d"}
cat_embed_feat_enum

# + {"Collapsed": "false", "persistent_id": "ab16acb6-7ba4-4ceb-b062-2bda7905acbf"}
resp_chart_df[new_cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "2135f766-d52d-4f58-bf40-ac648ce9021f"}
stream = open('cat_embed_feat_enum.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "540651fd-1fa7-425d-bce3-5027cd7761bf"}
resp_chart_df['ts'] = resp_chart_df['nurseassessoffset']
resp_chart_df = resp_chart_df.drop('nurseassessoffset', axis=1)
resp_chart_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "cb96bfe1-de84-4062-81b2-60e1de519df7"}
len(resp_chart_df)

# + {"Collapsed": "false", "persistent_id": "4d45864a-3f9a-4238-bdcd-c8f8ded00a0e"}
resp_chart_df = resp_chart_df.drop_duplicates()
resp_chart_df.head()

# + {"Collapsed": "false", "persistent_id": "4b63a057-249e-4855-b7f1-2aec53d36863"}
len(resp_chart_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "c29990be-8458-4d83-aa02-6c2733704b86"}
resp_chart_df = resp_chart_df.set_index('ts')
resp_chart_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "93dcd9c6-df9c-480c-bcc6-9a88db3bba3d"}
resp_chart_df.reset_index().head()

# + {"Collapsed": "false", "persistent_id": "acadb9e1-775f-43b3-b680-45aa2b37acd7"}
resp_chart_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='Cough').head()

# + {"Collapsed": "false", "persistent_id": "62614201-b663-4adf-9080-33e648a0a0ec"}
resp_chart_df[resp_chart_df.patientunitstayid == 2553254].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 80 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "84be170e-c8fb-47a6-a9f9-1d373cbb1eb4"}
resp_chart_df = du.embedding.join_categorical_enum(resp_chart_df, new_cat_embed_feat)
resp_chart_df.head()

# + {"Collapsed": "false", "persistent_id": "6ece20d0-6de2-4989-9394-e020fc8916ed"}
resp_chart_df.dtypes

# + {"Collapsed": "false", "persistent_id": "09692bf1-874c-49c5-a525-31f91a28c019"}
resp_chart_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='Cough').head()

# + {"Collapsed": "false", "persistent_id": "b9cbb7ec-8a59-49a2-a32c-b3baf79877ca"}
resp_chart_df[resp_chart_df.patientunitstayid == 2553254].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "a8e0f0f1-6170-473c-8611-a730f4084b64"}
resp_chart_df.columns = du.data_processing.clean_naming(resp_chart_df.columns)
resp_chart_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "6e102707-0b63-414c-97c1-92ac52203c83"}
resp_chart_df.to_csv(f'{data_path}cleaned/unnormalized/respiratoryCharting.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "5974799f-3217-4519-b448-f4ec15f9e7a9"}
resp_chart_df.to_csv(f'{data_path}cleaned/normalized/respiratoryCharting.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "0acfdac7-0ec3-4b8b-86d1-4a08563c9b4b"}
resp_chart_df.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

# + {"Collapsed": "false", "persistent_id": "9f71d94c-3761-42e1-9eea-883e3a2bc6e5"}
resp_chart_df = pd.read_csv(f'{data_path}cleaned/normalized/respiratoryCharting.csv')
resp_chart_df.head()

# + {"Collapsed": "false", "persistent_id": "e258728a-ff39-4dca-8842-5261f3a4a6a3"}
len(resp_chart_df)

# + {"Collapsed": "false", "persistent_id": "7e776a89-8833-4932-9458-0269330493d3"}
resp_chart_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "bbfbbaa4-287d-4093-973c-3e1d297c8bc1"}
eICU_df = pd.merge_asof(eICU_df, resp_chart_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Allergy data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "9383001f-c733-4cf4-8b59-abfd57cf49e5"}
alrg_df = pd.read_csv(f'{data_path}original/allergy.csv')
alrg_df.head()

# + {"Collapsed": "false", "persistent_id": "cbd138f8-8c1e-4c99-91ca-68a1244be654"}
len(alrg_df)

# + {"Collapsed": "false", "persistent_id": "5cdf0cb6-3179-459c-844d-46edb8a71619"}
alrg_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "e0483a75-628e-48bd-97e8-0350d7c4f889"}
alrg_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "c2a187d6-fb24-4a21-8f2e-26b6473c257a"}
alrg_df.columns

# + {"Collapsed": "false", "persistent_id": "df596865-c169-4738-a491-1a0694c8144a"}
alrg_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "c309a388-c464-4fa2-85b3-9f6aab6bbb1d"}
du.search_explore.dataframe_missing_values(alrg_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "e8352cc0-cec5-4fb3-a5dd-d05e05586a96"}
alrg_df[alrg_df.allergytype == 'Non Drug'].drughiclseqno.value_counts()

# + {"Collapsed": "false", "persistent_id": "32fd94b7-8ce1-4b2e-9309-4f3653be654d"}
alrg_df[alrg_df.allergytype == 'Drug'].drughiclseqno.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# As we can see, the drug features in this table only have data if the allergy derives from using the drug. As such, we don't need the `allergytype` feature. Also ignoring hospital staff related information and using just the drug codes instead of their names, as they're independent of the drug brand.

# + {"Collapsed": "false", "persistent_id": "fce01df2-fdd7-42c3-8fa8-22794989df1a"}
alrg_df.allergynotetype.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Feature `allergynotetype` also doesn't seem very relevant, discarding it.

# + {"Collapsed": "false", "persistent_id": "a090232b-049c-4386-90eb-6a68f6487a34"}
alrg_df = alrg_df[['patientunitstayid', 'allergyoffset',
                   'allergyname', 'drughiclseqno']]
alrg_df.head()

# + {"toc-hr-collapsed": false, "Collapsed": "false", "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into simple numberings, one hot encode features with a low number of categories (in this case, 5) and enumerate sparse categorical features that will be embedded.

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Separate and prepare features for embedding
#
# Identify categorical features that have more than 5 unique categories, which will go through an embedding layer afterwards, and enumerate them.
#
# In the case of microbiology data, we're also going to embed the antibiotic `sensitivitylevel`, not because it has many categories, but because there can be several rows of data per timestamp (which would be impractical on one hot encoded data).

# + {"Collapsed": "false", "cell_type": "markdown"}
# Update list of categorical features and add those that will need embedding (features with more than 5 unique values):

# + {"Collapsed": "false", "persistent_id": "621e1414-6268-4641-818b-5f8b5d54a446"}
new_cat_feat = ['allergyname', 'drughiclseqno']
[cat_feat.append(col) for col in new_cat_feat]

# + {"Collapsed": "false", "persistent_id": "664dd21a-1669-4244-abb3-18ac6c330004"}
cat_feat_nunique = [alrg_df[feature].nunique() for feature in new_cat_feat]
cat_feat_nunique

# + {"Collapsed": "false", "persistent_id": "70c0fd76-7880-4446-9574-d1779a9cce15"}
new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        new_cat_embed_feat.append(new_cat_feat[i])

# + {"Collapsed": "false", "persistent_id": "ce58accd-9f73-407c-b441-6da299604bb1"}
alrg_df[new_cat_feat].head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "9c0217cf-66d8-467b-b0df-b75441b1c0dc"}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Skip the 'drughiclseqno' from enumeration encoding
    if feature == 'drughiclseqno':
        continue
    # Prepare for embedding, i.e. enumerate categories
    alrg_df[feature], cat_embed_feat_enum[feature] = du.embedding.enum_categorical_feature(alrg_df, feature)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Fill missing values of the drug data with 0, so as to prepare for embedding:

# + {"Collapsed": "false", "persistent_id": "50a95412-7211-4780-b0da-aad5e166191e"}
alrg_df.drughiclseqno = alrg_df.drughiclseqno.fillna(0).astype(int)

# + {"Collapsed": "false", "persistent_id": "e7ebceaf-e35b-4143-8fea-b5b14016e7f3"}
alrg_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "e72804a4-82fe-4ce9-930e-6fd615c307f0"}
cat_embed_feat_enum

# + {"Collapsed": "false", "persistent_id": "1b6e81c6-87ba-44dc-9c8d-73e168e946a6"}
alrg_df[new_cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "45fdd1e4-00cd-49f8-b498-f50fb291e89a"}
stream = open('cat_embed_feat_enum.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "2f698d25-5a2b-44be-9d82-2c7790ee489f"}
alrg_df['ts'] = alrg_df['allergyoffset']
alrg_df = alrg_df.drop('allergyoffset', axis=1)
alrg_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "815f11a7-7b0d-44eb-b0f1-9061157864ca"}
len(alrg_df)

# + {"Collapsed": "false", "persistent_id": "96c4e771-2db5-461f-bf53-6b098feb26b4"}
alrg_df = alrg_df.drop_duplicates()
alrg_df.head()

# + {"Collapsed": "false", "persistent_id": "4baae0fb-9777-4abe-bcfe-f7c254e7bfc7"}
len(alrg_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "173d1236-0aad-49a4-a8fd-b25c99bc30bc"}
alrg_df = alrg_df.set_index('ts')
alrg_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "14dad389-ff55-4dc3-a435-90f9b9c8f58b"}
alrg_df.reset_index().head()

# + {"Collapsed": "false", "persistent_id": "d2660024-2f7e-4d37-b312-2a69aea35f0a"}
alrg_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='allergyname').head()

# + {"Collapsed": "false", "persistent_id": "a49094b0-6f71-4f70-ae6c-223373294b50"}
alrg_df[alrg_df.patientunitstayid == 3197554].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 47 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"Collapsed": "false", "cell_type": "markdown"}
# Even after removing duplicates rows, there are still some that have different information for the same ID and timestamp. We have to concatenate the categorical enumerations.

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "bf671749-9886-44f0-923f-24fd31d7d371"}
alrg_df = du.embedding.join_categorical_enum(alrg_df, new_cat_embed_feat)
alrg_df.head()

# + {"Collapsed": "false", "persistent_id": "96d8f554-5a5f-4b82-b5d5-47e5ce4c0f75"}
alrg_df.dtypes

# + {"Collapsed": "false", "persistent_id": "4024c555-2938-4f9a-9105-e4d77569267a"}
alrg_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='allergyname').head()

# + {"Collapsed": "false", "persistent_id": "7bb3f0b5-8f04-42f8-96aa-716762f65e5a"}
alrg_df[alrg_df.patientunitstayid == 3197554].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Rename columns

# + {"Collapsed": "false", "persistent_id": "d804e12b-79df-4a29-87ef-89c26d57b8e9"}
alrg_df = alrg_df.rename(columns={'drughiclseqno':'drugallergyhiclseqno'})
alrg_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "d6152c1e-f76f-4148-8c54-aec40a2636b1"}
alrg_df.columns = du.data_processing.clean_naming(alrg_df.columns)
alrg_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "58eeead0-9cdc-4095-9643-2009af44a9a3"}
alrg_df.to_csv(f'{data_path}cleaned/unnormalized/allergy.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "b89fd87e-1200-41ef-83dc-09e0e3309f09"}
alrg_df.to_csv(f'{data_path}cleaned/normalized/allergy.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "2297202e-d250-430b-9ecd-23efc756cb25"}
alrg_df.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

# + {"Collapsed": "false", "persistent_id": "9782a118-7e1e-4ba3-a63f-9b2654bf2fe4"}
alrg_df = pd.read_csv(f'{data_path}cleaned/normalized/allergy.csv')
alrg_df.head()

# + {"Collapsed": "false", "persistent_id": "f5e0185c-8ac8-4583-813d-c730965ff79a"}
len(alrg_df)

# + {"Collapsed": "false", "persistent_id": "3948b5c9-9ba8-491a-a436-7afd56d8b0cc"}
alrg_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "d2c24e33-72ee-4aae-b215-c5b66e132fa2"}
eICU_df = pd.merge_asof(eICU_df, alrg_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"Collapsed": "false", "persistent_id": "af67bdc2-285f-4810-8711-9e8b0d2dede8"}
# [TODO] Check if careplangeneral table could be useful. It seems to have mostly subjective data.

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## General care plan data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "613c2344-4e37-432a-b3cd-22a541edd752"}
careplangen_df = pd.read_csv(f'{data_path}original/carePlanGeneral.csv')
careplangen_df.head()

# + {"Collapsed": "false", "persistent_id": "9f1f740f-5c63-412c-bff6-ab7eae291f91"}
len(careplangen_df)

# + {"Collapsed": "false", "persistent_id": "62f66285-ee19-4aaa-a0d2-05f138ab83cc"}
careplangen_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "d23a0845-48f7-45aa-b61c-28136a01bd01"}
careplangen_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "b9135d0a-d530-43da-b8aa-18deae96b171"}
careplangen_df.columns

# + {"Collapsed": "false", "persistent_id": "dbe90cd6-a672-4e1c-8b4e-85747d90af52"}
careplangen_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "de2dbb20-4296-4a0c-9613-3d058695121d"}
du.search_explore.dataframe_missing_values(careplangen_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "d20ab161-76eb-4269-8fb7-0944fc3c3a44"}
careplangen_df.cplgroup.value_counts()

# + {"Collapsed": "false", "persistent_id": "72bb4652-7228-44c6-bc2e-4b0630c133be"}
careplangen_df.cplitemvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "9961b071-3641-4cf0-8bdc-6e6ba0294e87"}
careplangen_df[careplangen_df.cplgroup == 'Activity'].cplitemvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "09ee35a0-f9f5-4120-a1d5-76e5ad1dfb4f"}
careplangen_df[careplangen_df.cplgroup == 'Care Limitation'].cplitemvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "cd7e1ede-f930-4239-b75a-2ae239662877"}
careplangen_df[careplangen_df.cplgroup == 'Route-Status'].cplitemvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "2e383295-6667-4d7f-87e7-e476aab0e5d9"}
careplangen_df[careplangen_df.cplgroup == 'Critical Care Discharge/Transfer Planning'].cplitemvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "36b6bfb9-5200-42bb-a8e4-c333d7f2aede"}
careplangen_df[careplangen_df.cplgroup == 'Safety/Restraints'].cplitemvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "46f36240-2fd1-4560-bfe5-a2b3e38c77b1"}
careplangen_df[careplangen_df.cplgroup == 'Sedation'].cplitemvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "60c55e8b-f983-4475-af8f-208158cca575"}
careplangen_df[careplangen_df.cplgroup == 'Analgesia'].cplitemvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "004946a6-c64d-4ede-8b33-adf43110026f"}
careplangen_df[careplangen_df.cplgroup == 'Ordered Protocols'].cplitemvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "655d2ba3-45b9-4ab8-a7bc-bd32dbb5ad68"}
careplangen_df[careplangen_df.cplgroup == 'Volume Status'].cplitemvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "70c62f5e-ceea-46c6-8dfc-a9f992136d7f"}
careplangen_df[careplangen_df.cplgroup == 'Psychosocial Status'].cplitemvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "27ede8c7-fc03-4f2a-8dcd-4ed2f19a501e"}
careplangen_df[careplangen_df.cplgroup == 'Current Rate'].cplitemvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "484de861-45d1-4507-b313-5f84443088f6"}
careplangen_df[careplangen_df.cplgroup == 'Baseline Status'].cplitemvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "e2d3551f-5e70-4179-8b5c-7fe6012b823d"}
careplangen_df[careplangen_df.cplgroup == 'Protein'].cplitemvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "05df4874-3726-4480-a961-44abd3d226fe"}
careplangen_df[careplangen_df.cplgroup == 'Calories'].cplitemvalue.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# In this case, there aren't entire columns to remove. However, some specific types of care plan categories seem to be less relevant (e.g. activity, critical care discharge/transfer planning) or redundant (e.g. ventilation, infectious diseases). So, we're going to remove rows that have those categories.

# + {"Collapsed": "false", "persistent_id": "455ad80d-0a1a-442a-bf26-a9223bf8e485"}
careplangen_df = careplangen_df.drop('cplgeneralid', axis=1)
careplangen_df.head()

# + {"Collapsed": "false", "persistent_id": "cc9909b6-36ef-41a2-9181-a517f3b4db83"}
categories_to_remove = ['Ventilation', 'Airway', 'Activity', 'Care Limitation',
                        'Route-Status', 'Critical Care Discharge/Transfer Planning',
                        'Ordered Protocols', 'Acuity', 'Volume Status', 'Prognosis',
                        'Care Providers', 'Family/Health Care Proxy/Contact Info', 'Current Rate',
                        'Daily Goals/Safety Risks/Discharge Requirements', 'Goal Rate',
                        'Planned Procedures', 'Infectious Disease',
                        'Care Plan Reviewed with Patient/Family', 'Protein', 'Calories']

# + {"Collapsed": "false", "persistent_id": "d98992f6-95ce-446f-9663-31c61f7074e9"}
~(careplangen_df.cplgroup.isin(categories_to_remove)).head()

# + {"Collapsed": "false", "persistent_id": "7d24f3e0-5410-4ad7-a336-2d36e6660978"}
careplangen_df = careplangen_df[~(careplangen_df.cplgroup.isin(categories_to_remove))]
careplangen_df.head()

# + {"Collapsed": "false", "persistent_id": "4ad66aec-27fe-408e-9c4f-cea679275ea8"}
len(careplangen_df)

# + {"Collapsed": "false", "persistent_id": "c98bbaf0-4a7f-43e0-a3be-0bba71e42d78"}
careplangen_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# There's still plenty of data left, affecting around 92.48% of the unit stays, even after removing several categories.

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

# + {"Collapsed": "false", "persistent_id": "bf802b7d-4805-43cd-b356-b3b4b22d5227"}
new_cat_feat = ['cplgroup', 'cplitemvalue']
[cat_feat.append(col) for col in new_cat_feat]

# + {"Collapsed": "false", "persistent_id": "2168052c-99b8-4001-8b1a-f593e372d08c"}
cat_feat_nunique = [careplangen_df[feature].nunique() for feature in new_cat_feat]
cat_feat_nunique

# + {"Collapsed": "false", "persistent_id": "afb24ad8-137c-4b8a-ae5b-35d2128016e7"}
new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        new_cat_embed_feat.append(new_cat_feat[i])

# + {"Collapsed": "false", "persistent_id": "c9a79b95-84cb-4202-b1c3-85c142cc28e5"}
careplangen_df[new_cat_feat].head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "1f99cb0f-528a-417c-96ad-276f4bd51eea"}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    careplangen_df[feature], cat_embed_feat_enum[feature] = du.embedding.enum_categorical_feature(careplangen_df, feature)

# + {"Collapsed": "false", "persistent_id": "4db30eaa-d84a-476f-b3cc-475efb879254"}
careplangen_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "f2321342-9f71-46ad-8071-6b2a00042333"}
cat_embed_feat_enum

# + {"Collapsed": "false", "persistent_id": "8da064db-1490-4462-af76-75974765cc76"}
careplangen_df[new_cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "c4b3c022-8230-4d99-a2ad-ea40b46c88a9"}
stream = open('cat_embed_feat_enum.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "750cfef3-41f3-409f-973f-823695dfc701"}
careplangen_df['ts'] = careplangen_df['cplitemoffset']
careplangen_df = careplangen_df.drop('cplitemoffset', axis=1)
careplangen_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "a4e01751-7c79-4f18-adbd-cce61fdae2ad"}
len(careplangen_df)

# + {"Collapsed": "false", "persistent_id": "1f6d3919-01a4-4a16-9d59-305212bae2b5"}
careplangen_df = careplangen_df.drop_duplicates()
careplangen_df.head()

# + {"Collapsed": "false", "persistent_id": "d457826e-fb7b-417f-bb39-80b31f7540d2"}
len(careplangen_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "5103c257-8c0d-4e2b-824e-34aabea41d9d"}
careplangen_df = careplangen_df.set_index('ts')
careplangen_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "94c3b563-76c9-4bc7-8443-a7ce6feafb74"}
careplangen_df.reset_index().head()

# + {"Collapsed": "false", "persistent_id": "eecc6d3d-96ca-40bc-a167-d1ad1145b144"}
careplangen_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='cplgroup').head()

# + {"Collapsed": "false", "persistent_id": "29cd639a-ad9c-4dcf-a68d-23decdc85a19"}
careplangen_df[careplangen_df.patientunitstayid == 3138123].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 32 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "cb92e6b7-ef36-42e1-8755-bfae6204c7b4"}
careplangen_df = du.embedding.join_categorical_enum(careplangen_df, new_cat_embed_feat)
careplangen_df.head()

# + {"Collapsed": "false", "persistent_id": "5d5dc0a9-5dce-4370-86f5-d5098e217fdf"}
careplangen_df.dtypes

# + {"Collapsed": "false", "persistent_id": "39625b58-c493-4e09-91e7-70be59070a9a"}
careplangen_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='cplgroup').head()

# + {"Collapsed": "false", "persistent_id": "6c79f1d2-7788-410e-8ef9-f0920495f5dc"}
careplangen_df[careplangen_df.patientunitstayid == 3138123].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Rename columns
#
# Keeping the `activeupondischarge` feature so as to decide if forward fill or leave at NaN each general care plan value, when we have the full dataframe. However, we need to identify this feature's original table, general care plan, so as to not confound with other data.

# + {"Collapsed": "false", "persistent_id": "c2e06f50-2d38-4362-a5be-980164960a09"}
careplangen_df = careplangen_df.rename(columns={'activeupondischarge':'cpl_activeupondischarge'})
careplangen_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "74803039-97af-4a43-ad79-0d3c3b064c0a"}
careplangen_df.columns = du.data_processing.clean_naming(careplangen_df.columns)
careplangen_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "b7792852-30f8-4d96-8197-68afa0e1844c"}
careplangen_df.to_csv(f'{data_path}cleaned/unnormalized/carePlanGeneral.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "d5285b24-30a4-4493-ac28-447821de366d"}
careplangen_df.to_csv(f'{data_path}cleaned/normalized/carePlanGeneral.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "8a3e2317-d264-4383-b097-808886b8eb19"}
careplangen_df.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

# + {"Collapsed": "false", "persistent_id": "3e2cce77-cff2-4bce-a2d4-781f77e5a555"}
careplangen_df = pd.read_csv(f'{data_path}cleaned/normalized/carePlanGeneral.csv')
careplangen_df.head()

# + {"Collapsed": "false", "persistent_id": "0f225f11-fdd4-4049-ae28-16fb64455c4b"}
len(careplangen_df)

# + {"Collapsed": "false", "persistent_id": "513ec6cc-3eb8-4087-8acc-1c19700bc2c5"}
careplangen_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "f14ed521-2e98-4089-b46e-8eb4bc79a2d0"}
eICU_df = pd.merge_asof(eICU_df, careplangen_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Past history data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "db086782-764c-4f63-b32f-6246f7c49a9b"}
pasthist_df = pd.read_csv(f'{data_path}original/pastHistory.csv')
pasthist_df.head()

# + {"Collapsed": "false", "persistent_id": "3a3c896b-98cf-4d4c-b35b-48f2bb46645d"}
len(pasthist_df)

# + {"Collapsed": "false", "persistent_id": "c670ebb3-032d-496a-8973-f213e5b04b2d"}
pasthist_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "f65b8d4c-4832-4346-bf23-b87b2ec5c16f"}
pasthist_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "fae1abf5-445a-4d27-8f17-6379aee5fa72"}
pasthist_df.columns

# + {"Collapsed": "false", "persistent_id": "60672930-c5c9-4482-a5de-919c9dff3f75"}
pasthist_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "a417ded5-02be-4fa6-9f63-3c0a79d7f512"}
du.search_explore.dataframe_missing_values(pasthist_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "156f2c4b-029d-483f-ae2f-efc88ee80b88"}
pasthist_df.pasthistorypath.value_counts().head(20)

# + {"Collapsed": "false", "persistent_id": "a4e04c0a-863a-4b60-8028-5f6a384dc057"}
pasthist_df.pasthistorypath.value_counts().tail(20)

# + {"Collapsed": "false", "persistent_id": "be0eea31-5880-4233-b223-47401d6ac827"}
pasthist_df.pasthistoryvalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "acb0333c-2b3f-4aca-88b2-2b1cf7c479e4"}
pasthist_df.pasthistorynotetype.value_counts()

# + {"Collapsed": "false", "persistent_id": "c141a8e0-2af2-4d2c-aee8-60700ddc5301"}
pasthist_df[pasthist_df.pasthistorypath == 'notes/Progress Notes/Past History/Past History Obtain Options/Performed'].pasthistoryvalue.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# In this case, considering that it regards past diagnosis of the patients, the timestamp when that was observed probably isn't very reliable nor useful. As such, I'm going to remove the offset variables. Furthermore, `pasthistoryvaluetext` is redundant with `pasthistoryvalue`, while `pasthistorynotetype` and the past history path 'notes/Progress Notes/Past History/Past History Obtain Options/Performed' seem to be irrelevant.

# + {"Collapsed": "false", "persistent_id": "cce8b969-b435-45e5-a3aa-2f646f816491"}
pasthist_df = pasthist_df.drop(['pasthistoryid', 'pasthistoryoffset', 'pasthistoryenteredoffset',
                                'pasthistorynotetype', 'pasthistoryvaluetext'], axis=1)
pasthist_df.head()

# + {"Collapsed": "false", "persistent_id": "787315bf-b470-4538-9ad1-adcc6ef93c65"}
categories_to_remove = ['notes/Progress Notes/Past History/Past History Obtain Options/Performed']

# + {"Collapsed": "false", "persistent_id": "ce524b85-8006-4c1a-af08-aa1c6094b152"}
~(pasthist_df.pasthistorypath.isin(categories_to_remove)).head()

# + {"Collapsed": "false", "persistent_id": "23926113-127a-442a-8c33-33deb5efa772"}
pasthist_df = pasthist_df[~(pasthist_df.pasthistorypath.isin(categories_to_remove))]
pasthist_df.head()

# + {"Collapsed": "false", "persistent_id": "e9374a3c-b2be-428c-a853-0ded647a6c70"}
len(pasthist_df)

# + {"Collapsed": "false", "persistent_id": "b08768b7-7942-4cff-b4a6-8ee10322c4f1"}
pasthist_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "ac4e7984-b356-4696-9ba7-ef5763aeac89"}
pasthist_df.pasthistorypath.value_counts().head(20)

# + {"Collapsed": "false", "persistent_id": "b8a9f87a-0259-4094-8f03-eb2d64aeea8b"}
pasthist_df.pasthistorypath.value_counts().tail(20)

# + {"Collapsed": "false", "persistent_id": "4f93d732-2641-4ded-83b0-fbb8eb7f2421"}
pasthist_df.pasthistoryvalue.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# There's still plenty of data left, affecting around 81.87% of the unit stays, even after removing several categories.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Separate high level notes

# + {"Collapsed": "false", "persistent_id": "e6f07ded-5aa3-4efc-9fc8-3c7a35eadb25"}
pasthist_df.pasthistorypath.map(lambda x: x.split('/')).head().values

# + {"Collapsed": "false", "persistent_id": "d4d5ecbb-5b79-4a15-a997-7863b3facb38"}
pasthist_df.pasthistorypath.map(lambda x: len(x.split('/'))).min()

# + {"Collapsed": "false", "persistent_id": "26a98b9f-0972-482d-956f-57c3b7eac41a"}
pasthist_df.pasthistorypath.map(lambda x: len(x.split('/'))).max()

# + {"Collapsed": "false", "persistent_id": "ae522e1e-8465-4e53-8299-c0fc1f3757c1"}
pasthist_df.pasthistorypath.apply(lambda x: du.search_explore.get_element_from_split(x, 0, separator='/'),
                                  meta=('x', str)).value_counts()

# + {"Collapsed": "false", "persistent_id": "e72babaa-63b4-4804-87b5-f9ee67fd7118"}
pasthist_df.pasthistorypath.apply(lambda x: du.search_explore.get_element_from_split(x, 1, separator='/'),
                                  meta=('x', str)).value_counts()

# + {"Collapsed": "false", "persistent_id": "bf504856-1e69-40a8-9677-1878144e00f7"}
pasthist_df.pasthistorypath.apply(lambda x: du.search_explore.get_element_from_split(x, 2, separator='/'),
                                  meta=('x', str)).value_counts()

# + {"Collapsed": "false", "persistent_id": "5f79fcc3-0dfa-40fa-b2f9-4e2f1211feab"}
pasthist_df.pasthistorypath.apply(lambda x: du.search_explore.get_element_from_split(x, 3, separator='/'),
                                  meta=('x', str)).value_counts()

# + {"Collapsed": "false", "persistent_id": "271ee812-3434-47b0-9dd0-6edfea59c5fe"}
pasthist_df.pasthistorypath.apply(lambda x: du.search_explore.get_element_from_split(x, 4, separator='/'),
                                  meta=('x', str)).value_counts()

# + {"Collapsed": "false", "persistent_id": "4b03a6bd-ea02-456e-9d28-095f2b10fea0"}
pasthist_df.pasthistorypath.apply(lambda x: du.search_explore.get_element_from_split(x, 5, separator='/'),
                                  meta=('x', str)).value_counts()

# + {"Collapsed": "false", "persistent_id": "954a240a-a4b1-4e5a-b2d0-1f1864040aac"}
pasthist_df.pasthistorypath.apply(lambda x: du.search_explore.get_element_from_split(x, 6, separator='/'),
                                  meta=('x', str)).value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# There are always at least 5 levels of the notes. As the first 4 ones are essentially always the same ("notes/Progress Notes/Past History/Organ Systems/") and the 5th one tends to not be very specific (only indicates which organ system it affected, when it isn't just a case of no health problems detected), it's best to preserve the 5th and isolate the remaining string as a new feature. This way, the split provides further insight to the model on similar notes.

# + {"Collapsed": "false", "persistent_id": "abfe7998-c744-4653-96d4-752c3c7c62a8"}
pasthist_df['pasthistorytype'] = pasthist_df.pasthistorypath.apply(lambda x: du.search_explore.get_element_from_split(x, 4, separator='/'), meta=('x', str))
pasthist_df['pasthistorydetails'] = pasthist_df.pasthistorypath.apply(lambda x: du.search_explore.get_element_from_split(x, 5, separator='/', till_the_end=True), meta=('x', str))
pasthist_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# `pasthistoryvalue` seems to correspond to the last element of `pasthistorydetails`. Let's confirm it:

# + {"Collapsed": "false", "persistent_id": "d299e5c1-9355-4c3d-9af4-c54e24f289ad"}
pasthist_df['pasthistorydetails_last'] = pasthist_df.pasthistorydetails.map(lambda x: x.split('/')[-1])
pasthist_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Compare columns `pasthistoryvalue` and `pasthistorydetails`'s last element:

# + {"Collapsed": "false", "persistent_id": "f62af377-9237-4005-80b8-47aa0c83570a"}
pasthist_df[pasthist_df.pasthistoryvalue != pasthist_df.pasthistorydetails_last]

# + {"Collapsed": "false", "cell_type": "markdown"}
# The previous output confirms that the newly created `pasthistorydetails` feature's last elememt (last string in the symbol separated lists) is almost exactly equal to the already existing `pasthistoryvalue` feature, with the differences that `pasthistoryvalue` takes into account the scenarios of no health problems detected and behaves correctly in strings that contain the separator symbol in them. So, we should remove `pasthistorydetails`'s last element:

# + {"Collapsed": "false", "persistent_id": "40418862-5680-4dc5-9348-62e98599a638"}
pasthist_df = pasthist_df.drop('pasthistorydetails_last', axis=1)
pasthist_df.head()

# + {"Collapsed": "false", "persistent_id": "5022385b-8935-436e-a82b-8c402c0808f5"}
pasthist_df['pasthistorydetails'] = pasthist_df.pasthistorydetails.apply(lambda x: '/'.join(x.split('/')[:-1]), meta=('pasthistorydetails', str))
pasthist_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove irrelevant `Not Obtainable` and `Not Performed` values:

# + {"Collapsed": "false", "persistent_id": "634e9588-7d76-46d5-a152-c1b73660d558"}
pasthist_df[pasthist_df.pasthistoryvalue == 'Not Obtainable'].pasthistorydetails.value_counts()

# + {"Collapsed": "false", "persistent_id": "490b10c9-202d-4d5e-830d-cbb84272486a"}
pasthist_df[pasthist_df.pasthistoryvalue == 'Not Performed'].pasthistorydetails.value_counts()

# + {"Collapsed": "false", "persistent_id": "20746db6-d74f-49eb-bceb-d46fc0c981c0"}
pasthist_df = pasthist_df[~((pasthist_df.pasthistoryvalue == 'Not Obtainable') | (pasthist_df.pasthistoryvalue == 'Not Performed'))]
pasthist_df.head()

# + {"Collapsed": "false", "persistent_id": "aa099682-9266-4567-8c7c-11043ca3d932"}
pasthist_df.pasthistorytype.unique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Replace blank `pasthistorydetails` values:

# + {"Collapsed": "false", "persistent_id": "8a7f891f-2f84-45ac-a2d5-856531eba2bb"}
pasthist_df[pasthist_df.pasthistoryvalue == 'No Health Problems'].pasthistorydetails.value_counts()

# + {"Collapsed": "false", "persistent_id": "9fb3ddad-84bb-44c9-b76f-ff2d7475b38a"}
pasthist_df[pasthist_df.pasthistoryvalue == 'No Health Problems'].pasthistorydetails.value_counts().index

# + {"Collapsed": "false", "persistent_id": "781296cb-c8fa-49e5-826d-c9e297553c0e"}
pasthist_df[pasthist_df.pasthistorydetails == ''].head()

# + {"Collapsed": "false", "persistent_id": "23607e71-135a-4281-baaa-ccff0f9765ad"}
pasthist_df['pasthistorydetails'] = pasthist_df.apply(lambda df: 'No Health Problems' if df['pasthistorytype'] == 'No Health Problems'
                                                                 else df['pasthistorydetails'],
                                                      axis=1, meta=(None, str))
pasthist_df.head()

# + {"Collapsed": "false", "persistent_id": "6a32a636-2c60-45c4-b20f-8b82c9921cb4"}
pasthist_df[pasthist_df.pasthistorydetails == '']

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the now redundant `pasthistorypath` column:

# + {"Collapsed": "false", "persistent_id": "9268a24d-f3e7-407c-9c99-65020d7c17f0"}
pasthist_df = pasthist_df.drop('pasthistorypath', axis=1)
pasthist_df.head()

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

# + {"Collapsed": "false", "persistent_id": "e6083094-1f99-408d-9b12-dfe4d88ee39a"}
new_cat_feat = ['pasthistoryvalue', 'pasthistorytype', 'pasthistorydetails']
[cat_feat.append(col) for col in new_cat_feat]

# + {"Collapsed": "false", "persistent_id": "7d6962aa-6294-4e84-a136-1a04057781da"}
cat_feat_nunique = [pasthist_df[feature].nunique() for feature in new_cat_feat]
cat_feat_nunique

# + {"Collapsed": "false", "persistent_id": "6cfc4870-d154-40ad-b3a5-337f8697dc6c"}
new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        new_cat_embed_feat.append(new_cat_feat[i])

# + {"Collapsed": "false", "persistent_id": "aa911443-7f86-44ea-ab90-997fd38ba074"}
pasthist_df[new_cat_feat].head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "58e7c624-cfc6-4df1-83dc-a8a22fe7ffc0"}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    pasthist_df[feature], cat_embed_feat_enum[feature] = du.embedding.enum_categorical_feature(pasthist_df, feature)

# + {"Collapsed": "false", "persistent_id": "7ee06a1a-cb99-4a94-9271-6f67948fd2a6"}
pasthist_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "a77d1170-55de-492e-9c4a-2542c06da94d"}
cat_embed_feat_enum

# + {"Collapsed": "false", "persistent_id": "2b7dde92-fe4b-42ce-9705-9d505878a696"}
pasthist_df[new_cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "534d4a78-6d3e-4e3b-b318-5f353835d53a"}
stream = open('cat_embed_feat_enum.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove duplicate rows

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "1a99e9da-39a1-46ae-9d0d-68c3dd063d8f"}
len(pasthist_df)

# + {"Collapsed": "false", "persistent_id": "5296709d-b5e3-44f7-bdcc-45d17955a2b6"}
pasthist_df = pasthist_df.drop_duplicates()
pasthist_df.head()

# + {"Collapsed": "false", "persistent_id": "b3483b1e-cced-4e2b-acc0-e95870a628b2"}
len(pasthist_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "acae2295-6f7e-4290-b8a7-12d13042b65d"}
pasthist_df.groupby(['patientunitstayid']).count().nlargest(columns='pasthistoryvalue').head()

# + {"Collapsed": "false", "persistent_id": "1c71ea45-4026-43ac-8433-bd70d567bee9"}
pasthist_df[pasthist_df.patientunitstayid == 1558102].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 20 categories per `patientunitstayid`. As such, we must join them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "589931b8-fe11-439a-8b14-4857c168c023"}
pasthist_df = du.embedding.join_categorical_enum(pasthist_df, new_cat_embed_feat, id_columns=['patientunitstayid'])
pasthist_df.head()

# + {"Collapsed": "false", "persistent_id": "6edc2eea-1139-4f7f-a314-db03cd785128"}
pasthist_df.dtypes

# + {"Collapsed": "false", "persistent_id": "61f2c4df-d3b6-459b-9632-194e2736ff27"}
pasthist_df.groupby(['patientunitstayid']).count().nlargest(columns='pasthistoryvalue').head()

# + {"Collapsed": "false", "persistent_id": "aa5f247a-8e4b-4527-a265-2af71b0f8e06"}
pasthist_df[pasthist_df.patientunitstayid == 1558102].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "0cab3fc4-cb1f-4845-a0f9-7e9758df6f28"}
pasthist_df.columns = du.data_processing.clean_naming(pasthist_df.columns)
pasthist_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "01cb3dd4-7afa-454b-b8c4-e473cb305367"}
pasthist_df.to_csv(f'{data_path}cleaned/unnormalized/pastHistory.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "5924e842-7b4f-4b3a-bddf-9b89952dfe26"}
pasthist_df.to_csv(f'{data_path}cleaned/normalized/pastHistory.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "ee91beb5-1415-4960-9e84-8cbfbde07e15"}
pasthist_df.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

# + {"Collapsed": "false", "persistent_id": "beed207c-3fa8-428f-abe7-d286eb2a92ac"}
pasthist_df = pd.read_csv(f'{data_path}cleaned/normalized/pastHistory.csv')
pasthist_df.head()

# + {"Collapsed": "false", "persistent_id": "6fc058a7-6542-4bab-9b76-ae31ca82a28d"}
len(pasthist_df)

# + {"Collapsed": "false", "persistent_id": "1d562b3d-84be-4b0e-a808-d413d030647a"}
pasthist_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "af809274-88c4-407b-989e-4a59bfefc509"}
eICU_df = pd.merge_asof(eICU_df, pasthist_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Infusion drug data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "36c79435-0530-4459-8832-cb924012b62e"}
infdrug_df = pd.read_csv(f'{data_path}original/infusionDrug.csv')
infdrug_df.head()

# + {"Collapsed": "false", "persistent_id": "cf1205df-b87e-42cf-8740-f5663955860b"}
len(infdrug_df)

# + {"Collapsed": "false", "persistent_id": "fe500a2c-f9b0-41ff-a833-b61de0e87728"}
infdrug_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "08b8557e-0837-45a2-a462-3e05528756f1"}
infdrug_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "5e14aba4-9f24-4ecb-a406-5d1590c4f538"}
infdrug_df.columns

# + {"Collapsed": "false", "persistent_id": "77be4fb5-821f-4ab0-b1ee-3e4288565439"}
infdrug_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "7b3530da-0b79-4bed-935b-3a48af63d92e"}
du.search_explore.dataframe_missing_values(infdrug_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features
#
# Besides removing the row ID `infusiondrugid`, I'm also removing `infusionrate`, `volumeoffluid` and `drugamount` as they seem redundant with `drugrate` although with a lot more missing values.

# + {"Collapsed": "false", "persistent_id": "90324c6d-0d62-432c-be74-b8f5cf41de65"}
infdrug_df = infdrug_df.drop(['infusiondrugid', 'infusionrate', 'volumeoffluid', 'drugamount'], axis=1)
infdrug_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove string drug rate values

# + {"Collapsed": "false", "persistent_id": "a786da73-2e65-437b-a215-7f2ed3964df5"}
infdrug_df[infdrug_df.drugrate.map(du.utils.is_definitely_string)].head()

# + {"Collapsed": "false", "persistent_id": "4f985943-20dd-49a3-8df4-54bacc5a2863"}
infdrug_df[infdrug_df.drugrate.map(du.utils.is_definitely_string)].drugrate.value_counts()

# + {"Collapsed": "false", "persistent_id": "0182647e-311c-471f-91d8-d91fcc0ed92a"}
infdrug_df.drugrate = infdrug_df.drugrate.map(lambda x: np.nan if du.utils.is_definitely_string(x) else x)
infdrug_df.head()

# + {"Collapsed": "false", "persistent_id": "19f22442-547a-48af-8c9a-a0e89d2264b3"}
infdrug_df.patientunitstayid = infdrug_df.patientunitstayid.astype(int)
infdrug_df.infusionoffset = infdrug_df.infusionoffset.astype(int)
infdrug_df.drugname = infdrug_df.drugname.astype(str)
infdrug_df.drugrate = infdrug_df.drugrate.astype(float)
infdrug_df.patientweight = infdrug_df.patientweight.astype(float)
infdrug_df.head()

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

# + {"Collapsed": "false", "persistent_id": "d080567b-0609-4069-aecc-293e98c3277b"}
new_cat_feat = ['drugname']
[cat_feat.append(col) for col in new_cat_feat]

# + {"Collapsed": "false", "persistent_id": "7b998dd8-c470-46d3-8f15-027b3bbbfc71"}
cat_feat_nunique = [infdrug_df[feature].nunique() for feature in new_cat_feat]
cat_feat_nunique

# + {"Collapsed": "false", "persistent_id": "92fe07d2-eac6-4312-a9fa-6bb5e24bc7f4"}
new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        new_cat_embed_feat.append(new_cat_feat[i])

# + {"Collapsed": "false", "persistent_id": "82bf9aa5-c5de-433a-97d3-01d6af81e2e4"}
infdrug_df[new_cat_feat].head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "8b0f065b-fb2b-4330-b155-d86769ac1635"}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    infdrug_df[feature], cat_embed_feat_enum[feature] = du.embedding.enum_categorical_feature(infdrug_df, feature)

# + {"Collapsed": "false", "persistent_id": "c5c8d717-87c8-408d-b018-d6b6b1575549"}
infdrug_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "872eac91-7dd6-406a-b346-d23d21597022"}
cat_embed_feat_enum

# + {"Collapsed": "false", "persistent_id": "b0e7a06d-f451-470f-8cf5-d154f76e83a2"}
infdrug_df[new_cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "b40f8861-dae3-49ca-8649-d937ba3cfcf0"}
stream = open('cat_embed_feat_enum.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "b1ea5e2a-d7eb-41e6-9cad-4dcbf5997ca7"}
infdrug_df['ts'] = infdrug_df['infusionoffset']
infdrug_df = infdrug_df.drop('infusionoffset', axis=1)
infdrug_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Standardize drug names:

# + {"Collapsed": "false", "persistent_id": "54f14b33-c0e9-4760-a0e3-f904996eda99"}
infdrug_df = du.data_processing.clean_categories_naming(infdrug_df, 'drugname')
infdrug_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "edfbcec8-4ca6-430a-8caf-940a115f6cac"}
len(infdrug_df)

# + {"Collapsed": "false", "persistent_id": "e6e2fef9-878b-449c-bf08-d42dc6e4f9da"}
infdrug_df = infdrug_df.drop_duplicates()
infdrug_df.head()

# + {"Collapsed": "false", "persistent_id": "1cd6a490-f63f-458f-a274-30170c70fc66"}
len(infdrug_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "b3eb5c69-d034-45d7-ab48-0217169a48fb"}
infdrug_df = infdrug_df.set_index('ts')
infdrug_df.head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "049a3fd1-0ae4-454e-a5b5-5ce8fa94d3e1"}
infdrug_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='drugname').head()

# + {"Collapsed": "false", "persistent_id": "29b5843e-679e-4c4c-941f-e39a92965d1f"}
infdrug_df[infdrug_df.patientunitstayid == 1785711].head(20)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 17 categories per set of `patientunitstayid` and `ts`. As such, we must join them. But first, as we shouldn't mix absolute values of drug rates from different drugs, we better normalize it first.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Normalize data

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "5d512225-ad7e-40b4-a091-b18df3f38c4c"}
infdrug_df_norm = du.data_processing.normalize_data(infdrug_df,
                                                 columns_to_normalize=['patientweight'],
                                                 columns_to_normalize_cat=[('drugname', 'drugrate')])
infdrug_df_norm.head()

# + {"Collapsed": "false", "persistent_id": "48d12b1b-72f9-4884-bc0c-93a0b2a333b8"}
infdrug_df_norm.patientweight.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "cb08c783-dc90-4b03-9f21-8da8164bea89"}
infdrug_df_norm = du.embedding.join_categorical_enum(infdrug_df_norm, new_cat_embed_feat)
infdrug_df_norm.head()

# + {"Collapsed": "false", "persistent_id": "e8c3544d-cf75-45d1-8af2-e6feb8a8d587"}
infdrug_df_norm.dtypes

# + {"Collapsed": "false", "persistent_id": "bbc89593-5b2b-42b5-83f2-8bab9c1d5ef6"}
infdrug_df_norm.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='drugname').head()

# + {"Collapsed": "false", "persistent_id": "b64df6bc-254f-40b8-97cf-15737ce27db1"}
infdrug_df_norm[infdrug_df_norm.patientunitstayid == 1785711].head(20)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Rename columns

# + {"Collapsed": "false", "persistent_id": "9a27b7be-7a8a-435b-acdb-9407a325ac53"}
infdrug_df = infdrug_df.rename(columns={'patientweight': 'weight', 'drugname': 'infusion_drugname',
                                        'drugrate': 'infusion_drugrate'})
infdrug_df.head()

# + {"Collapsed": "false", "persistent_id": "8917e132-aa34-43f8-a4c9-7c38c572fc94"}
infdrug_df_norm = infdrug_df_norm.rename(columns={'patientweight': 'weight', 'drugname': 'infusion_drugname',
                                                  'drugrate': 'infusion_drugrate'})
infdrug_df_norm.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "258e5bdf-3880-4f14-8a03-7394c55c2c2e"}
infdrug_df.columns = du.data_processing.clean_naming(infdrug_df.columns)
infdrug_df_norm.columns = du.data_processing.clean_naming(infdrug_df_norm.columns)
infdrug_df_norm.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "767d2077-112b-483a-bd2c-4b578d61ba1a"}
infdrug_df.to_csv(f'{data_path}cleaned/unnormalized/infusionDrug.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "615e3df8-d467-4042-801f-a296a528b77a"}
infdrug_df_norm.to_csv(f'{data_path}cleaned/normalized/infusionDrug.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "3fe6821a-5324-4b36-94cd-7d8073c5262f"}
infdrug_df_norm.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

# + {"Collapsed": "false", "persistent_id": "3278496a-a6bc-4c4b-8e6d-93b6295375bd"}
infdrug_df = pd.read_csv(f'{data_path}cleaned/normalized/infusionDrug.csv')
infdrug_df.head()

# + {"Collapsed": "false", "persistent_id": "1aa21f85-3ed9-467c-9773-958ae79d4a9a"}
len(infdrug)

# + {"Collapsed": "false", "persistent_id": "976bb278-a2b0-4558-b925-139789c46313"}
infdrug.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "5c86375e-117f-4367-ad01-67beeef3361c"}
eICU_df = pd.merge_asof(eICU_df, infdrug_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Diagnosis data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "8f98afc2-5613-4042-94e7-98e0b72867a1"}
diagn_df = pd.read_csv(f'{data_path}original/diagnosis.csv')
diagn_df.head()

# + {"Collapsed": "false", "persistent_id": "f2bd9f00-de58-48d9-a304-5e96ba6b392d"}
len(diagn_df)

# + {"Collapsed": "false", "persistent_id": "a0b999fb-9767-43de-8ed7-59dc72f68635"}
diagn_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "3f42e4cb-0064-4b01-9f6e-496caafb08dd"}
diagn_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "aaa6012c-e776-4335-9c5b-0101f4dc153d"}
diagn_df.columns

# + {"Collapsed": "false", "persistent_id": "81262e48-301a-4230-aae0-b94bf3f584d8"}
diagn_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "a494abea-0f81-4ea8-8745-51ebfc306125"}
du.search_explore.dataframe_missing_values(diagn_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "cell_type": "markdown"}
# Besides the usual removal of row identifier, `diagnosisid`, I'm also removing apparently irrelevant (and subjective) `diagnosispriority`, redundant, with missing values and other issues `icd9code`, and `activeupondischarge`, as we don't have complete information as to when diagnosis end.

# + {"Collapsed": "false", "persistent_id": "9073b0ba-7bc9-4b9c-aac7-ac1fd4802e63"}
diagn_df = diagn_df.drop(['diagnosisid', 'diagnosispriority', 'icd9code', 'activeupondischarge'], axis=1)
diagn_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Separate high level diagnosis

# + {"Collapsed": "false", "persistent_id": "2318f39a-d909-49af-a13b-c4b1095bf161"}
diagn_df.diagnosisstring.value_counts()

# + {"Collapsed": "false", "persistent_id": "af82bc05-a5d8-4a40-a61a-d1f160c69b3d"}
diagn_df.diagnosisstring.map(lambda x: x.split('|')).head()

# + {"Collapsed": "false", "persistent_id": "08936a84-6640-4b15-bb1b-0192922d6daf"}
diagn_df.diagnosisstring.map(lambda x: len(x.split('|'))).min()

# + {"Collapsed": "false", "cell_type": "markdown"}
# There are always at least 2 higher level diagnosis. It could be beneficial to extract those first 2 levels to separate features, so as to avoid the need for the model to learn similarities that are already known.

# + {"Collapsed": "false", "persistent_id": "38475e5c-1260-4068-af50-5072366282ce"}
diagn_df['diagnosis_type_1'] = diagn_df.diagnosisstring.apply(lambda x: du.search_explore.get_element_from_split(x, 0, separator='|'), meta=('x', str))
diagn_df['diagnosis_disorder_2'] = diagn_df.diagnosisstring.apply(lambda x: du.search_explore.get_element_from_split(x, 1, separator='|'), meta=('x', str))
diagn_df['diagnosis_detailed_3'] = diagn_df.diagnosisstring.apply(lambda x: du.search_explore.get_element_from_split(x, 2, separator='|', till_the_end=True), meta=('x', str))
# Remove now redundant `diagnosisstring` feature
diagn_df = diagn_df.drop('diagnosisstring', axis=1)
diagn_df.head()

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

# + {"Collapsed": "false", "persistent_id": "67f7e344-f016-4d85-a051-df96d07b5274"}
new_cat_feat = ['diagnosis_type_1', 'diagnosis_disorder_2', 'diagnosis_detailed_3']
[cat_feat.append(col) for col in new_cat_feat]

# + {"Collapsed": "false", "persistent_id": "24e349bb-902c-4c6f-a032-647a98ac5834"}
cat_feat_nunique = [diagn_df[feature].nunique() for feature in new_cat_feat]
cat_feat_nunique

# + {"Collapsed": "false", "persistent_id": "cb4c7b2b-47ea-4976-8f6f-4e8e9956a62f"}
new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        new_cat_embed_feat.append(new_cat_feat[i])

# + {"Collapsed": "false", "persistent_id": "72f3e710-08ef-4c9d-9876-c9d8cbaaf5f0"}
diagn_df[new_cat_feat].head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "bba23ddd-c1e5-49b7-9b7f-8fe5819ee7f9"}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    diagn_df[feature], cat_embed_feat_enum[feature] = du.embedding.enum_categorical_feature(diagn_df, feature)

# + {"Collapsed": "false", "persistent_id": "64118894-5fb4-4e31-91cf-695d64a7e633"}
diagn_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "15f3d224-568b-45b1-8d6d-6bd4eaf35562"}
cat_embed_feat_enum

# + {"Collapsed": "false", "persistent_id": "cbe8c721-69d6-4af1-ba27-ef8a6c166b19"}
diagn_df[new_cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "95ac0351-6a18-41b0-9937-37d255fa34ca"}
stream = open('cat_embed_feat_enum.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "8011a320-6066-416e-bb3e-280d97088afe"}
diagn_df['ts'] = diagn_df['diagnosisoffset']
diagn_df = diagn_df.drop('diagnosisoffset', axis=1)
diagn_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "2dfd1e82-de35-40b7-a67b-f57b14389787"}
len(diagn_df)

# + {"Collapsed": "false", "persistent_id": "6c733c3e-26e8-4618-a849-cfab5e4065d8"}
diagn_df = diagn_df.drop_duplicates()
diagn_df.head()

# + {"Collapsed": "false", "persistent_id": "710c6b32-3d96-427d-9d7f-e1ecc453ba7e"}
len(diagn_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "e82d40c2-9e0c-411a-a2e2-acb1dd4f1d96"}
diagn_df = diagn_df.set_index('ts')
diagn_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "287cb2ac-7d06-4240-b896-75ab99093d32"}
diagn_df.reset_index().head()

# + {"Collapsed": "false", "persistent_id": "f28206fd-f1d0-4ca2-b006-653f60d05782"}
diagn_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='diagnosis_type_1').head()

# + {"Collapsed": "false", "persistent_id": "cac394e5-a6fa-4bc7-a496-1c97b49de381"}
diagn_df[diagn_df.patientunitstayid == 3089982].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 69 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "b6614a18-9359-48be-b659-ea4a8bb90f00"}
diagn_df = du.embedding.join_categorical_enum(diagn_df, new_cat_embed_feat)
diagn_df.head()

# + {"Collapsed": "false", "persistent_id": "e854db5e-06f3-45ee-ad27-f5214f3f6ea7"}
diagn_df.dtypes

# + {"Collapsed": "false", "persistent_id": "21a955e7-fdc3-42d8-97e0-47f4401b7c8e"}
diagn_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='diagnosis_type_1').head()

# + {"Collapsed": "false", "persistent_id": "4dddda8f-daf1-4d2b-8782-8de20201ea7f"}
diagn_df[diagn_df.patientunitstayid == 3089982].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "0f06cc9f-f14f-4c96-afdd-030ac8975f0b"}
diagn_df.columns = du.data_processing.clean_naming(diagn_df.columns)
diagn_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "0c0cc6d4-56a9-40d4-a213-288b7080fb72"}
diagn_df.to_csv(f'{data_path}cleaned/unnormalized/diagnosis.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "4c0def92-984f-4dd3-a807-be7188be38e8"}
diagn_df.to_csv(f'{data_path}cleaned/normalized/diagnosis.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "0906c011-5d47-49e4-b8d0-bfb97b575f66"}
diagn_df.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

# + {"Collapsed": "false", "persistent_id": "1107b26a-d309-4e7a-bdd8-dda075b95257"}
diagn_df = pd.read_csv(f'{data_path}cleaned/normalized/diagnosis.csv')
diagn_df.head()

# + {"Collapsed": "false", "persistent_id": "267f6d8b-c3e3-4510-8f92-39b452121037"}
len(diagn_df)

# + {"Collapsed": "false", "persistent_id": "2e0e6a07-50a4-4450-945f-915f46ee2352"}
diagn_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "f140ac92-00a6-417b-9a6d-50d2c7c93f51"}
eICU_df = pd.merge_asof(eICU_df, diagn_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Admission drug data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "d75bab34-d386-49fb-b6b7-273035226f86"}
admsdrug_df = pd.read_csv(f'{data_path}original/admissionDrug.csv')
admsdrug_df.head()

# + {"Collapsed": "false", "persistent_id": "0b16b7ea-653f-45d6-a923-ea5538cfa1a8"}
len(admsdrug_df)

# + {"Collapsed": "false", "persistent_id": "65855dd5-c78f-4596-8b10-4ad9ca706403"}
admsdrug_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# There's not much admission drug data (only around 20% of the unit stays have this data). However, it might be useful, considering also that it complements the medication table.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "cdca1bbc-5fe1-4823-8828-f2adcc14d9b5"}
admsdrug_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "7d6bb51f-8e11-44af-8427-f50a90117bd4"}
admsdrug_df.columns

# + {"Collapsed": "false", "persistent_id": "7d0197fe-9d63-4ffb-b0ce-8c15f5231b52"}
admsdrug_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "e6b1e527-f089-4418-98cc-497cc63f2454"}
du.search_explore.dataframe_missing_values(admsdrug_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "a5e3fd48-5fa3-4d9b-aa52-e1a29d7c6523"}
admsdrug_df.drugname.value_counts()

# + {"Collapsed": "false", "persistent_id": "0c6824dc-8949-4d55-96dc-932cd84a63e6"}
admsdrug_df.drughiclseqno.value_counts()

# + {"Collapsed": "false", "persistent_id": "67466a3f-3bdc-4f91-b83a-7746b15345b4"}
admsdrug_df.drugnotetype.value_counts()

# + {"Collapsed": "false", "persistent_id": "c11d4f43-468c-4175-a455-0473f5ad2807"}
admsdrug_df.drugdosage.value_counts()

# + {"Collapsed": "false", "persistent_id": "d77aa18d-8ce7-4e2b-ac94-25e5074bf371"}
admsdrug_df.drugunit.value_counts()

# + {"Collapsed": "false", "persistent_id": "12d3606e-1dec-4173-aa42-1c59370ea262"}
admsdrug_df.drugadmitfrequency.value_counts()

# + {"Collapsed": "false", "persistent_id": "09e4ec8d-8ebc-4503-a2ef-e1f20c83f3cb"}
admsdrug_df[admsdrug_df.drugdosage == 0].head(20)

# + {"Collapsed": "false", "persistent_id": "5d29249a-b8ba-4f8c-81a7-754418232261"}
admsdrug_df[admsdrug_df.drugdosage == 0].drugunit.value_counts()

# + {"Collapsed": "false", "persistent_id": "9c1a6253-713e-4a1f-be38-add4f687973d"}
admsdrug_df[admsdrug_df.drugdosage == 0].drugadmitfrequency.value_counts()

# + {"Collapsed": "false", "persistent_id": "d2eba66b-33c6-4427-a88b-c4ec28174653"}
admsdrug_df[admsdrug_df.drugunit == ' '].drugdosage.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Oddly, `drugunit` and `drugadmitfrequency` have several blank values. At the same time, when this happens, `drugdosage` tends to be 0 (which is also an unrealistic value). Considering that no NaNs are reported, these blanks and zeros probably represent missing values.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Besides removing irrelevant or hospital staff related data (e.g. `usertype`), I'm also removing the `drugname` column, which is redundant with the codes `drughiclseqno`, while also being brand dependant.

# + {"Collapsed": "false", "persistent_id": "69129157-8c81-42bc-bc88-00df3249bc86"}
admsdrug_df = admsdrug_df[['patientunitstayid', 'drugoffset', 'drugdosage',
                           'drugunit', 'drugadmitfrequency', 'drughiclseqno']]
admsdrug_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Fix missing values representation
#
# Replace blank and unrealistic zero values with NaNs.

# + {"Collapsed": "false", "persistent_id": "23d0168e-e5f3-4630-954e-3becc23307ae"}
admsdrug_df.drugdosage = admsdrug_df.drugdosage.replace(to_replace=0, value=np.nan)
admsdrug_df.drugunit = admsdrug_df.drugunit.replace(to_replace=' ', value=np.nan)
admsdrug_df.drugadmitfrequency = admsdrug_df.drugadmitfrequency.replace(to_replace=' ', value=np.nan)
admsdrug_df.head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "10eab8e7-1ef2-46cb-8d41-7953dd66ef15"}
du.search_explore.dataframe_missing_values(admsdrug_df)

# + {"toc-hr-collapsed": false, "Collapsed": "false", "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into simple numberings, one hot encode features with a low number of categories (in this case, 5) and enumerate sparse categorical features that will be embedded.

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Separate and prepare features for embedding
#
# Identify categorical features that have more than 5 unique categories, which will go through an embedding layer afterwards, and enumerate them.
#
# In the case of microbiology data, we're also going to embed the antibiotic `sensitivitylevel`, not because it has many categories, but because there can be several rows of data per timestamp (which would be impractical on one hot encoded data).

# + {"Collapsed": "false", "cell_type": "markdown"}
# Update list of categorical features and add those that will need embedding (features with more than 5 unique values):

# + {"Collapsed": "false", "persistent_id": "96f4f32e-9442-4705-acbc-37060bfba492"}
new_cat_feat = ['drugunit', 'drugadmitfrequency', 'drughiclseqno']
[cat_feat.append(col) for col in new_cat_feat]

# + {"Collapsed": "false", "persistent_id": "1dc9c231-b3c6-4211-a7ac-fb186f6e58e4"}
cat_feat_nunique = [admsdrug_df[feature].nunique() for feature in new_cat_feat]
cat_feat_nunique

# + {"Collapsed": "false", "persistent_id": "8930d1df-2c63-4c38-aa0a-ca6fe6082376"}
new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        new_cat_embed_feat.append(new_cat_feat[i])

# + {"Collapsed": "false", "persistent_id": "a33e788b-d6ba-4d79-9fee-54a32959d453"}
admsdrug_df[new_cat_feat].head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "8f69d437-6fec-4e7e-ae76-226b742b03a7"}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Skip the 'drughiclseqno' from enumeration encoding
    if feature == 'drughiclseqno':
        continue
    # Prepare for embedding, i.e. enumerate categories
    admsdrug_df[feature], cat_embed_feat_enum[feature] = du.embedding.enum_categorical_feature(admsdrug_df, feature)

# + {"Collapsed": "false", "persistent_id": "11a4cd2e-86f0-4637-a8dc-b7bf456d2bbe"}
admsdrug_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "26eac7f3-9081-4a96-ae4a-40054c223fd7"}
cat_embed_feat_enum

# + {"Collapsed": "false", "persistent_id": "b54a0213-dfda-46d3-aef5-a7a5ed8c2810"}
admsdrug_df[new_cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "fb54b948-abf1-412d-9fb3-4edd22500e97"}
stream = open('cat_embed_feat_enum.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "ceab2ec2-9b2b-4439-9c3c-7674dd5b2445"}
admsdrug_df['ts'] = admsdrug_df['drugoffset']
admsdrug_df = admsdrug_df.drop('drugoffset', axis=1)
admsdrug_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "387be9a1-51bf-43fd-9a7c-df32b5f6bfc6"}
len(admsdrug_df)

# + {"Collapsed": "false", "persistent_id": "a7fb23bc-4735-4f8a-92d0-f7e359653e5f"}
admsdrug_df = admsdrug_df.drop_duplicates()
admsdrug_df.head()

# + {"Collapsed": "false", "persistent_id": "abf09d5e-b24e-46cd-968b-4a1051ee8504"}
len(admsdrug_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "d35d4953-51aa-46ec-8107-1d4d7f3651a8"}
admsdrug_df = admsdrug_df.set_index('ts')
admsdrug_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "b656a52b-271c-42fd-b8ff-2c4c01a7d2dc"}
admsdrug_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='drughiclseqno').head()

# + {"Collapsed": "false", "persistent_id": "84c0cc0a-72eb-4fab-93e3-4c9cb83b4fc4"}
admsdrug_df[admsdrug_df.patientunitstayid == 2346930].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 48 categories per set of `patientunitstayid` and `ts`. As such, we must join them. But first, we need to normalize the dosage by the respective sets of drug code and units, so as to avoid mixing different absolute values.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Normalize data

# + {"Collapsed": "false", "persistent_id": "8645ef7e-61ca-4ae2-a7a2-4895e714086a"}
admsdrug_df_norm = admsdrug_df.reset_index()
admsdrug_df_norm.head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "417dd68c-7c54-4eaf-856c-9f6a44ee1d26"}
admsdrug_df_norm = du.data_processing.normalize_data(admsdrug_df_norm, columns_to_normalize=False,
                                                  columns_to_normalize_cat=[(['drughiclseqno', 'drugunit'], 'drugdosage')])
admsdrug_df_norm.head()

# + {"Collapsed": "false", "persistent_id": "d8f2c0a4-ba81-4958-90b7-7b5165ff1363"}
admsdrug_df_norm = admsdrug_df_norm.set_index('ts')
admsdrug_df_norm.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"Collapsed": "false", "cell_type": "markdown"}
# Even after removing duplicates rows, there are still some that have different information for the same ID and timestamp. We have to concatenate the categorical enumerations.

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "5d87f689-5f4f-4483-841a-533bc5e053c7"}
admsdrug_df_norm = du.embedding.join_categorical_enum(admsdrug_df_norm, new_cat_embed_feat)
admsdrug_df_norm.head()

# + {"Collapsed": "false", "persistent_id": "004138b3-74c4-4ca8-b8b4-3409648e00d0"}
admsdrug_df_norm.dtypes

# + {"Collapsed": "false", "persistent_id": "efd11709-c95e-4245-a662-188556d66680"}
admsdrug_df_norm.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='drughiclseqno').head()

# + {"Collapsed": "false", "persistent_id": "d6663be1-591b-4c35-b446-978dbc205444"}
admsdrug_df_norm[admsdrug_df_norm.patientunitstayid == 2346930].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "938421fc-57b7-4c40-b7a9-315b5619cbb0"}
admsdrug_df.columns = du.data_processing.clean_naming(admsdrug_df.columns)
admsdrug_df_norm.columns = du.data_processing.clean_naming(admsdrug_df_norm.columns)
admsdrug_df_norm.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "15ca1a3f-614e-49da-aa1d-7ac42bceee73"}
admsdrug_df.to_csv(f'{data_path}cleaned/unnormalized/admissionDrug.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "9a20e4a3-a8d6-4842-8470-e6bfccd03267"}
admsdrug_df_norm.to_csv(f'{data_path}cleaned/normalized/admissionDrug.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "bf406619-1133-4314-95ca-808f9fe81aee"}
admsdrug_df_norm.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

# + {"Collapsed": "false", "persistent_id": "d813ac43-2482-489f-a168-288ff5bb4814"}
admsdrug_df = pd.read_csv(f'{data_path}cleaned/normalized/admissionDrug.csv')
admsdrug_df.head()

# + {"Collapsed": "false", "persistent_id": "3c78b1bc-6152-4a90-9f29-1a48eeb3b402"}
len(admsdrug_df)

# + {"Collapsed": "false", "persistent_id": "867eeb9e-ecd6-4172-9266-5f3177c3fe93"}
admsdrug_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "2fd9114d-7062-4ad5-8a49-66e8dd1f770e"}
eICU_df = pd.merge_asof(eICU_df, admsdrug_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Medication data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "37d12a8d-d08c-41cd-b904-f005b1497fe1"}
med_df = pd.read_csv(f'{data_path}original/medication.csv', dtype={'loadingdose': 'object'})
med_df.head()

# + {"Collapsed": "false", "persistent_id": "c1bc04e9-c48c-443b-9208-8ec4618a0e2d"}
len(med_df)

# + {"Collapsed": "false", "persistent_id": "675a744b-8308-4a4d-89fb-3f8ba150343f"}
med_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# There's not much admission drug data (only around 20% of the unit stays have this data). However, it might be useful, considering also that it complements the medication table.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "2eda81ce-69df-4328-96f0-4f770bd683d3"}
med_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "c3118567-0513-47f1-a5d0-920683fc8273"}
med_df.columns

# + {"Collapsed": "false", "persistent_id": "86c4c435-d002-41ac-a635-78c7973d2aa9"}
med_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "095dbcb4-a4c4-4b9a-baca-41b954126f19"}
du.search_explore.dataframe_missing_values(med_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
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

# + {"Collapsed": "false", "cell_type": "markdown"}
# Besides removing less interesting data (e.g. `drugivadmixture`), I'm also removing the `drugname` column, which is redundant with the codes `drughiclseqno`, while also being brand dependant.

# + {"Collapsed": "false", "persistent_id": "b31b8605-5a88-4821-a65c-5337128f4264"}
med_df = med_df[['patientunitstayid', 'drugstartoffset', 'drugstopoffset',
                 'drugordercancelled', 'dosage', 'frequency', 'drughiclseqno']]
med_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove rows of which the drug has been cancelled or not specified

# + {"Collapsed": "false", "persistent_id": "4ffb498c-d6b5-46f3-9863-d74f672dd9be"}
med_df.drugordercancelled.value_counts()

# + {"Collapsed": "false", "persistent_id": "93a3e2f3-3632-4c5d-a25b-4e9195335348"}
med_df = med_df[~((med_df.drugordercancelled == 'Yes') | (np.isnan(med_df.drughiclseqno)))]
med_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the now unneeded `drugordercancelled` column:

# + {"Collapsed": "false", "persistent_id": "8c832678-d7eb-46d9-9521-a8a389cacc06"}
med_df = med_df.drop('drugordercancelled', axis=1)
med_df.head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "413613d9-1166-414d-a0c5-c51b1824d93e"}
du.search_explore.dataframe_missing_values(med_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Separating units from dosage
#
# In order to properly take into account the dosage quantities, as well as to standardize according to other tables like admission drugs, we should take the original `dosage` column and separate it to just the `drugdosage` values and the `drugunit`.

# + {"Collapsed": "false", "cell_type": "markdown"}
# No need to create a separate `pyxis` feature, which would indicate the use of the popular automated medications manager, as the frequency embedding will have that into account.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create dosage and unit features:

# + {"Collapsed": "false", "persistent_id": "68f9ce27-4230-4f52-b7f9-51c67ef4afea"}
med_df['drugdosage'] = np.nan
med_df['drugunit'] = np.nan
med_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get the dosage and unit values for each row:

# + {"Collapsed": "false", "persistent_id": "014b4c4e-31cb-487d-8bd9-b8f7048e981e"}
med_df[['drugdosage', 'drugunit']] = med_df.apply(du.data_processing.set_dosage_and_units, axis=1, result_type='expand')
med_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the now unneeded `dosage` column:

# + {"Collapsed": "false", "persistent_id": "61c02af1-fa9e-49d7-81e1-abd56d133510"}
med_df = med_df.drop('dosage', axis=1)
med_df.head()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ### Discretize categorical features
#
# Convert binary categorical features into simple numberings, one hot encode features with a low number of categories (in this case, 5) and enumerate sparse categorical features that will be embedded.

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Separate and prepare features for embedding
#
# Identify categorical features that have more than 5 unique categories, which will go through an embedding layer afterwards, and enumerate them.
#
# In the case of microbiology data, we're also going to embed the antibiotic `sensitivitylevel`, not because it has many categories, but because there can be several rows of data per timestamp (which would be impractical on one hot encoded data).

# + {"Collapsed": "false", "cell_type": "markdown"}
# Update list of categorical features and add those that will need embedding (features with more than 5 unique values):

# + {"Collapsed": "false", "persistent_id": "d714ff24-c50b-4dff-9b21-832d030d050f"}
new_cat_feat = ['drugunit', 'frequency', 'drughiclseqno']
[cat_feat.append(col) for col in new_cat_feat]

# + {"Collapsed": "false", "persistent_id": "d7e8e94d-e178-4fe6-96ff-e828eba8dc62"}
cat_feat_nunique = [med_df[feature].nunique() for feature in new_cat_feat]
cat_feat_nunique

# + {"Collapsed": "false", "persistent_id": "f3f370c1-79e0-4df7-ad6b-1c59ef8bcec6"}
new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        new_cat_embed_feat.append(new_cat_feat[i])

# + {"Collapsed": "false", "persistent_id": "ed09c3dd-50b3-48cf-aa04-76534feaf767"}
med_df[new_cat_feat].head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "51ac8fd1-cbd2-4f59-a737-f0fcc13043fd"}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Skip the 'drughiclseqno' from enumeration encoding
    if feature == 'drughiclseqno':
        continue
    # Prepare for embedding, i.e. enumerate categories
    med_df[feature], cat_embed_feat_enum[feature] = du.embedding.enum_categorical_feature(med_df, feature)

# + {"Collapsed": "false", "persistent_id": "e38470e5-73f7-4d35-b91d-ce0793b7f6f6"}
med_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "00b3ce19-756a-4de4-8b05-762de386aa29"}
cat_embed_feat_enum

# + {"Collapsed": "false", "persistent_id": "e5615265-4372-4117-a368-ec539c871763"}
med_df[new_cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "e51cc2e0-b598-484f-a3f8-8c764950777f"}
stream = open('cat_embed_feat_enum.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create drug stop event
#
# Add a timestamp corresponding to when each patient stops taking each medication.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Duplicate every row, so as to create a discharge event:

# + {"Collapsed": "false", "persistent_id": "db8e91ff-cfd7-4ddd-8873-20c20e1f9e46"}
new_df = med_df.copy()
new_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Set the new dataframe's rows to have the drug stop timestamp, with no more information on the drug that was being used:

# + {"Collapsed": "false", "persistent_id": "6d2bf14a-b75b-4b4c-ae33-403cde6781d7"}
new_df.drugstartoffset = new_df.drugstopoffset
new_df.drugunit = np.nan
new_df.drugdosage = np.nan
new_df.frequency = np.nan
new_df.drughiclseqno = np.nan
new_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Join the new rows to the remaining dataframe:

# + {"Collapsed": "false", "persistent_id": "321fe743-eae2-4b2c-a4ef-5e1c3350a231"}
med_df = med_df.append(new_df)
med_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the now unneeded medication stop column:

# + {"Collapsed": "false", "persistent_id": "6b1a48ca-fecc-467f-8071-534fbd4944bb"}
med_df = med_df.drop('drugstopoffset', axis=1)
med_df.head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "88ab8d50-b556-4c76-bb0b-33e76900018f"}
med_df['ts'] = med_df['drugstartoffset']
med_df = med_df.drop('drugstartoffset', axis=1)
med_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "74b9d214-8083-4d37-acbe-6ce26d6b1629"}
len(med_df)

# + {"Collapsed": "false", "persistent_id": "a2529353-bf59-464a-a32b-a940dd66007a"}
med_df = med_df.drop_duplicates()
med_df.head()

# + {"Collapsed": "false", "persistent_id": "be199b11-006c-4619-ac80-b3d86fd10f3b"}
len(med_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "81712e88-3b96-4f10-a536-a80268bfe805"}
med_df = med_df.set_index('ts')
med_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "ac2b0e4b-d2bd-4eb5-a629-637361a85457"}
med_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='drughiclseqno').head()

# + {"Collapsed": "false", "persistent_id": "da5e70d7-0514-4bdb-a5e2-12b6e8a1b197"}
med_df[med_df.patientunitstayid == 979183].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 41 categories per set of `patientunitstayid` and `ts`. As such, we must join them. But first, we need to normalize the dosage by the respective sets of drug code and units, so as to avoid mixing different absolute values.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Normalize data

# + {"Collapsed": "false", "persistent_id": "b5a6075e-d0f4-4766-ba4d-ee0413977e04"}
med_df_norm = med_df.reset_index()
med_df_norm.head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "a4cd949b-e561-485d-bcb6-10fccc343352"}
med_df_norm = du.data_processing.normalize_data(med_df_norm, columns_to_normalize=False,
                                             columns_to_normalize_cat=[(['drughiclseqno', 'drugunit'], 'drugdosage')])
med_df_norm.head()

# + {"Collapsed": "false", "persistent_id": "93145ae0-2204-4553-a7eb-5d235552dd82"}
med_df_norm = med_df_norm.set_index('ts')
med_df_norm.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"Collapsed": "false", "cell_type": "markdown"}
# Even after removing duplicates rows, there are still some that have different information for the same ID and timestamp. We have to concatenate the categorical enumerations.

# + {"Collapsed": "false", "persistent_id": "ed86d5a7-eeb3-44c4-9a4e-6dd67af307f2"}
list(set(med_df_norm.columns) - set(new_cat_embed_feat) - set(['patientunitstayid', 'ts']))

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "45866561-e170-4d9e-9189-e73cb5bcfc0f"}
med_df_norm = du.embedding.join_categorical_enum(med_df_norm, new_cat_embed_feat)
med_df_norm.head()

# + {"Collapsed": "false", "persistent_id": "db6b5624-e600-4d90-bc5a-ffa5a876d8dd"}
med_df_norm.dtypes

# + {"Collapsed": "false", "persistent_id": "954d2c26-4ef4-42ec-b0f4-a73febb5115d"}
med_df_norm.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='drughiclseqno').head()

# + {"Collapsed": "false", "persistent_id": "85536a51-d31a-4b25-aaee-9c9d4ec392f6"}
med_df_norm[med_df_norm.patientunitstayid == 979183].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Rename columns

# + {"Collapsed": "false", "persistent_id": "356775df-9204-4fef-b7a0-b82edf4ba85f"}
med_df_norm = med_df_norm.rename(columns={'frequency':'drugadmitfrequency'})
med_df_norm.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "0f255c7d-1d1a-4dd3-94d7-d7fd54e13da0"}
med_df.columns = du.data_processing.clean_naming(med_df.columns)
med_df_norm.columns = du.data_processing.clean_naming(med_df_norm.columns)
med_df_norm.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "7c95b423-0fd8-4e65-ac07-a0117f0c36bd"}
med_df.to_csv(f'{data_path}cleaned/unnormalized/medication.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "eae5d63f-5635-4fa0-8c42-ff6081336e18"}
med_df_norm.to_csv(f'{data_path}cleaned/normalized/medication.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "2cdabf5e-7df3-441b-b8ed-a06c404df27e"}
med_df_norm.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "9255fb38-ba28-4bc2-8f97-7596e8acbc5a"}
med_df.nlargest(columns='drugdosage')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Although the `drugdosage` looks good on mean (close to 0) and standard deviation (close to 1), it has very large magnitude minimum (-88.9) and maximum (174.1) values. Furthermore, these don't seem to be because of NaN values, whose groupby normalization could have been unideal. As such, it's hard to say if these are outliers or realistic values.

# + {"Collapsed": "false", "cell_type": "markdown"}
# [TODO] Check if these very large extreme dosage values make sense and, if not, try to fix them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

# + {"Collapsed": "false", "persistent_id": "a7c6ff09-8ff6-4347-90db-0a5193b56e5b"}
med_df = pd.read_csv(f'{data_path}cleaned/normalized/medication.csv')
med_df.head()

# + {"Collapsed": "false", "persistent_id": "4cdb2132-f4f6-4bb2-8816-2980b789cf67"}
len(med_df)

# + {"Collapsed": "false", "persistent_id": "3acf920e-fd14-4b74-8cd0-e2eda4bc14a7"}
med_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "137ee51c-ef2d-4f7c-bb1d-363306b190af"}
eICU_df = pd.merge_asof(eICU_df, med_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Notes data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "51197f67-95e2-4184-a73f-7885cb975084"}
note_df = pd.read_csv(f'{data_path}original/note.csv')
note_df.head()

# + {"Collapsed": "false", "persistent_id": "c3d45ba7-91dd-41d7-8699-c70390556018"}
len(note_df)

# + {"Collapsed": "false", "persistent_id": "17c46962-79a6-48a6-b67a-4863028ed897"}
note_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "6e745ae7-a0ea-4169-96de-d49e9f510ed9"}
note_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "ac1da815-dcd6-44df-8164-a987f9623255"}
note_df.columns

# + {"Collapsed": "false", "persistent_id": "8e30eac6-9424-4ce6-801e-5e013a964863"}
note_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "c4e967db-7ace-41df-8678-7cd11d1e002b"}
du.search_explore.dataframe_missing_values(note_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "persistent_id": "9f3b85d0-8480-4f30-8455-fde084ba7c69"}
note_df.notetype.value_counts().head(20)

# + {"Collapsed": "false", "persistent_id": "e41f6a9d-c5ea-45e3-b38b-89ba70de1b87"}
note_df.notepath.value_counts().head(40)

# + {"Collapsed": "false", "persistent_id": "b12dcddf-9703-4aff-8a20-df1d147ef554"}
note_df.notevalue.value_counts().head(20)

# + {"Collapsed": "false", "persistent_id": "faeab8ba-1cd0-4bac-801d-64b7c91a0637"}
note_df[note_df.notepath.str.contains('notes/Progress Notes/Social History')].head(20)

# + {"Collapsed": "false", "persistent_id": "0a5ddffd-b815-4b55-93ee-d5aeba54be0a"}
note_df[note_df.notepath.str.contains('notes/Progress Notes/Social History')].notepath.value_counts().head(20)

# + {"Collapsed": "false", "persistent_id": "3261378f-0eee-477f-bdfd-01cb1af45334"}
note_df[note_df.notepath.str.contains('notes/Progress Notes/Social History')].notevalue.value_counts().head(20)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Out of all the possible notes, only those addressing the patient's social history seem to be interesting and containing information not found in other tables. As scuh, we'll only keep the note paths that mention social history:

# + {"Collapsed": "false", "persistent_id": "d3a0e8a8-68d6-4c90-aded-0f5940c3936b"}
note_df = note_df[note_df.notepath.str.contains('notes/Progress Notes/Social History')]
note_df.head()

# + {"Collapsed": "false", "persistent_id": "8409525f-e11f-4cf5-acd7-56fcdcf6c130"}
len(note_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# There are still rows that seem to contain irrelevant data. Let's remove them by finding rows that contain specific words, like "obtain" and "print", that only appear in said irrelevant rows:

# + {"Collapsed": "false", "persistent_id": "2eefd7ee-b1af-4bcc-aece-7813b4ed2b29"}
category_types_to_remove = ['obtain', 'print', 'copies', 'options']

# + {"Collapsed": "false", "persistent_id": "094b7adf-176e-4989-8559-a114fef6e6e0"}
du.search_explore.find_row_contains_word(note_df, feature='notepath', words=category_types_to_remove).value_counts()

# + {"Collapsed": "false", "persistent_id": "d7ac4099-e9ed-489d-97ae-c4af5879c9ba"}
note_df = note_df[~du.search_explore.find_row_contains_word(note_df, feature='notepath', words=category_types_to_remove)]
note_df.head()

# + {"Collapsed": "false", "persistent_id": "0399795d-cff1-4df9-a322-a5fcd7ba11d7"}
len(note_df)

# + {"Collapsed": "false", "persistent_id": "d3b2d0c7-c8eb-404d-a0ac-e10e761d10fb"}
note_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "47ec86ae-999d-4678-889a-1653ca1a8bfb"}
note_df.notetype.value_counts().head(20)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Filtering just for interesting social history data greatly reduced the data volume of the notes table, now only present in around 20.5% of the unit stays. Still, it might be useful to include.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Besides the usual removal of row identifier, `noteid`, I'm also removing apparently irrelevant (`noteenteredoffset`, `notetype`) and redundant (`notetext`) columns:

# + {"Collapsed": "false", "persistent_id": "5beb1b97-7a5b-446c-934d-74b99556151f"}
note_df = note_df.drop(['noteid', 'noteenteredoffset', 'notetype', 'notetext'], axis=1)
note_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Separate high level notes

# + {"Collapsed": "false", "persistent_id": "4bc9189b-e06f-49a6-8613-d9d46f4ac4f7"}
note_df.notepath.value_counts().head(20)

# + {"Collapsed": "false", "persistent_id": "1cf61f46-6540-4445-b9ff-9be941f513bc"}
note_df.notepath.map(lambda x: x.split('/')).head().values

# + {"Collapsed": "false", "persistent_id": "93e3bc60-2c47-49f6-896b-f966f7fd6e42"}
note_df.notepath.map(lambda x: len(x.split('/'))).min()

# + {"Collapsed": "false", "persistent_id": "7a73ec52-0d6e-4627-a31f-e34ecfc79648"}
note_df.notepath.map(lambda x: len(x.split('/'))).max()

# + {"Collapsed": "false", "persistent_id": "99a73002-27a5-4796-9681-0dab429335bf"}
note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 1, separator='/'),
                       meta=('x', str)).value_counts()

# + {"Collapsed": "false", "persistent_id": "e907d8c4-ee87-4432-9cd4-9cd4f6999f2b"}
note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 2, separator='/'),
                       meta=('x', str)).value_counts()

# + {"Collapsed": "false", "persistent_id": "4a2a2b6b-9466-4592-a45e-698c9a7c944d"}
note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 3, separator='/'),
                       meta=('x', str)).value_counts()

# + {"Collapsed": "false", "persistent_id": "caea8162-79c0-4e2f-8b7b-cffbd26120ed"}
note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 4, separator='/'),
                       meta=('x', str)).value_counts()

# + {"Collapsed": "false", "persistent_id": "b222d3b9-caec-4b7f-ab63-c86e6dd3697c"}
note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 5, separator='/'),
                       meta=('x', str)).value_counts()

# + {"Collapsed": "false", "persistent_id": "32a752c7-5605-406d-acbd-2e0f420b7514"}
note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 6, separator='/'),
                       meta=('x', str)).value_counts()

# + {"Collapsed": "false", "persistent_id": "c65ed003-e440-457e-8e90-4940d2392a30"}
note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 7, separator='/'),
                       meta=('x', str)).value_counts()

# + {"Collapsed": "false", "persistent_id": "d85d05c9-21b3-4ecf-9e02-be0904f549dc"}
note_df.notevalue.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# There are always 8 levels of the notes. As the first 6 ones are essentially always the same ("notes/Progress Notes/Social History / Family History/Social History/Social History/"), it's best to just preserve the 7th one and isolate the 8th in a new feature. This way, the split provides further insight to the model on similar notes. However, it's also worth taking note that the 8th level of `notepath` seems to be identical to the feature `notevalue`. We'll look more into it later.

# + {"Collapsed": "false", "persistent_id": "a84d611a-871e-44ab-83ac-bbda639710cf"}
note_df['notetopic'] = note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 6, separator='/'), meta=('x', str))
note_df['notedetails'] = note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 7, separator='/'), meta=('x', str))
note_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the now redundant `notepath` column:

# + {"Collapsed": "false", "persistent_id": "3b6b3f09-7a5f-4f3a-bf17-baffb1ac975b"}
note_df = note_df.drop('notepath', axis=1)
note_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Compare columns `notevalue` and `notedetails`:

# + {"Collapsed": "false", "persistent_id": "5e4b5f52-a9f6-411d-87e5-c4a524942fe1"}
note_df[note_df.notevalue != note_df.notedetails]

# + {"Collapsed": "false", "cell_type": "markdown"}
# The previous blank output confirms that the newly created `notedetails` feature is exactly equal to the already existing `notevalue` feature. So, we should remove one of them:

# + {"Collapsed": "false", "persistent_id": "c73dcd18-e561-44ac-b3c5-3aabac45217a"}
note_df = note_df.drop('notedetails', axis=1)
note_df.head()

# + {"Collapsed": "false", "persistent_id": "3c4ccb54-901b-496a-be2d-e722cbf2ccc2"}
note_df[note_df.notetopic == 'Smoking Status'].notevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "feec565a-91e4-4555-ae56-6d4b658f09be"}
note_df[note_df.notetopic == 'Ethanol Use'].notevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "e2e20fd6-27e0-4e9b-bbf1-a801b240753f"}
note_df[note_df.notetopic == 'CAD'].notevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "193e722b-65a4-4057-9242-0c66a34f882c"}
note_df[note_df.notetopic == 'Cancer'].notevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "7e5e40b4-15be-443a-b2f4-995334bd1628"}
note_df[note_df.notetopic == 'Recent Travel'].notevalue.value_counts()

# + {"Collapsed": "false", "persistent_id": "c70bbd0c-b46c-4fd6-9ba3-c68f94c9e71f"}
note_df[note_df.notetopic == 'Bleeding Disorders'].notevalue.value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Considering how only the categories of "Smoking Status" and "Ethanol Use" in `notetopic` have more than one possible `notevalue` category, with the remaining being only 2 useful ones (categories "Recent Travel" and "Bleeding Disorders" have too little samples), it's probably best to just turn them into features, instead of packing in the same embedded feature.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Convert categories to features

# + {"Collapsed": "false", "cell_type": "markdown"}
# Make the `notetopic` and `notevalue` columns of type categorical:

# + {"Collapsed": "false", "persistent_id": "5b887ae0-4d27-4ef0-aaa5-b84e65d27fd5"}
note_df = note_df.categorize(columns=['notetopic', 'notevalue'])

# + {"Collapsed": "false", "cell_type": "markdown"}
# Transform the `notetopic` categories and `notevalue` values into separate features:

# + {"Collapsed": "false", "cell_type": "markdown"}
# Now we have the categories separated into their own features, as desired. Notice also how categories `Bleeding Disorders` and `Recent Travel` weren't added, as they appeared in less than the specified minimum of 1000 rows.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the old `notevalue` and `notetopic` columns:

# + {"Collapsed": "false", "persistent_id": "01dbc119-9a63-4561-acff-0835a88048a3"}
note_df = note_df.drop(['notevalue', 'notetopic'], axis=1)
note_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# While `Ethanol Use` and `Smoking Status` have several unique values, `CAD` and `Cancer` only have 1, indicating when that characteristic is present. As such,we should turn `CAD` and `Cancer` into binary features:

# + {"Collapsed": "false", "persistent_id": "c9bb3585-dfab-40cd-b71b-e5fb95d3218d"}
note_df['CAD'] = note_df['CAD'].apply(lambda x: 1 if x == 'CAD' else 0, meta=('CAD', int))
note_df['Cancer'] = note_df['Cancer'].apply(lambda x: 1 if x == 'Cancer' else 0, meta=('Cancer', int))
note_df.head()

# + {"Collapsed": "false", "persistent_id": "ad342024-94c5-4f9e-a9b2-82bfd6353db2"}
note_df['CAD'].value_counts()

# + {"Collapsed": "false", "persistent_id": "67b46f3d-d5bc-4cda-9c2d-4db10304f268"}
note_df['Cancer'].value_counts()

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

# + {"Collapsed": "false", "persistent_id": "0ea70c94-33a9-46b0-b987-dac01e78ec21"}
new_cat_feat = ['Smoking Status', 'Ethanol Use', 'CAD', 'Cancer']
[cat_feat.append(col) for col in new_cat_feat]

# + {"Collapsed": "false", "persistent_id": "09bb5a42-4cdc-43cd-9acb-9029e34dc279"}
cat_feat_nunique = [note_df[feature].nunique() for feature in new_cat_feat]
cat_feat_nunique

# + {"Collapsed": "false", "persistent_id": "22c11ea5-130d-4f5a-806f-5a8f63b27b10"}
new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        new_cat_embed_feat.append(new_cat_feat[i])

# + {"Collapsed": "false", "persistent_id": "a4a2b9a5-0f9b-442c-9042-ed940501b71e"}
note_df[new_cat_feat].head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "318dc10d-8369-45f1-9deb-5acc83616c04"}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    note_df[feature], cat_embed_feat_enum[feature] = du.embedding.enum_categorical_feature(note_df, feature)

# + {"Collapsed": "false", "persistent_id": "781ff06e-15f5-495c-967d-97c3dd790be7"}
note_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "d8fbd1fe-a0fb-4542-99e3-3696b5629e74"}
cat_embed_feat_enum

# + {"Collapsed": "false", "persistent_id": "c7a413ec-d61e-49ba-a7ae-13949fc6f092"}
note_df[new_cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "4342f002-c60a-4724-a542-9b7f906d3f2b"}
stream = open('cat_embed_feat_enum.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "dfab2799-af6c-4475-8341-ec3f40546ed1"}
note_df['ts'] = note_df['noteoffset']
note_df = note_df.drop('noteoffset', axis=1)
note_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "8af4dd26-9eb8-4edf-8bcf-361b10c94979"}
len(note_df)

# + {"Collapsed": "false", "persistent_id": "de85aa8f-0d02-4c35-868c-16116a83cf7f"}
note_df = note_df.drop_duplicates()
note_df.head()

# + {"Collapsed": "false", "persistent_id": "bb6efd0a-aa95-40d6-84b2-8916705a4cf4"}
len(note_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "a03573d9-f345-4ff4-84b9-2b2a3f73ce27"}
note_df = note_df.set_index('ts')
note_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "57b4548e-684d-4a17-bcef-ba9c968af375"}
note_df.reset_index().head()

# + {"Collapsed": "false", "persistent_id": "c94e7b7b-dc34-478b-842b-c34c926c934d"}
note_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='CAD').head()

# + {"Collapsed": "false", "persistent_id": "b8bcf17b-3d52-4cc9-bfbb-7f7d8fe83b3b"}
note_df[note_df.patientunitstayid == 3091883].head(10)

# + {"Collapsed": "false", "persistent_id": "a63112fe-a224-4b36-810f-f0d087be43b0"}
note_df[note_df.patientunitstayid == 3052175].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 5 categories per set of `patientunitstayid` and `ts`. As such, we must join them. However, this is a different scenario than in the other cases. Since we created the features from one categorical column, it doesn't have repeated values, only different rows to indicate each of the new features' values. As such, we just need to sum the features.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "591b2ccd-fa5c-4eb2-bec1-8ac21de1c890"}
note_df = du.embedding.join_categorical_enum(note_df, cont_join_method='max')
note_df.head()

# + {"Collapsed": "false", "persistent_id": "d3040cd3-4500-4129-ae90-23f3753045f8"}
note_df.dtypes

# + {"Collapsed": "false", "persistent_id": "b4d9884b-d8bb-49f4-8a38-4146f751708e"}
note_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='CAD').head()

# + {"Collapsed": "false", "persistent_id": "93c6457a-a187-421a-a9f7-6cbf844c1365"}
note_df[note_df.patientunitstayid == 3091883].head(10)

# + {"Collapsed": "false", "persistent_id": "d25a8707-20d0-40d7-ad0b-6efd2306686d"}
note_df[note_df.patientunitstayid == 3052175].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "32450572-639e-4539-b35a-181078ed3335"}
note_df.columns = du.data_processing.clean_naming(note_df.columns)
note_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "e42f577a-db00-4ecf-9e3c-433007a3bdaf"}
note_df.to_csv(f'{data_path}cleaned/unnormalized/note.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "812e7eb1-ff92-4a26-a970-2f40fc5bbdb1"}
note_df.to_csv(f'{data_path}cleaned/normalized/note.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "eebc356f-507e-4872-be9d-a1d774f2fd7a"}
note_df.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

# + {"Collapsed": "false", "persistent_id": "d9a7db2d-03ed-4bdd-8417-c36603617a60"}
note_df = pd.read_csv(f'{data_path}cleaned/normalized/note.csv')
note_df.head()

# + {"Collapsed": "false", "persistent_id": "56e6ff4f-d3e0-4e7f-9ce9-0182b6677de4"}
len(note_df)

# + {"Collapsed": "false", "persistent_id": "8e504335-d454-4068-b44e-e684526e9389"}
note_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "6fa9e548-fde8-432b-92d4-63daafcd3827"}
eICU_df = pd.merge_asof(eICU_df, note_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Treatment data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "d0a4e172-9a38-4bef-a052-6310b7fb5214"}
treat_df = pd.read_csv(f'{data_path}original/treatment.csv')
treat_df.head()

# + {"Collapsed": "false", "persistent_id": "bb5ccda2-5a79-46ed-9be2-8bb21b995771"}
len(treat_df)

# + {"Collapsed": "false", "persistent_id": "ad5c7291-d2ba-4e05-baea-5a887de8f535"}
treat_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "594cd2ff-12e5-4c90-83e7-722818eaa39e"}
treat_df.describe().transpose()

# + {"Collapsed": "false", "persistent_id": "3cca7683-fa6c-414f-854f-cd1f36cc6e42"}
treat_df.columns

# + {"Collapsed": "false", "persistent_id": "ce410a02-9154-40e5-bf5f-ddbdf1724bcb"}
treat_df.dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Check for missing values

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "ba586d2f-b03e-4a8d-942f-a2440be92101"}
du.search_explore.dataframe_missing_values(treat_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Remove unneeded features

# + {"Collapsed": "false", "cell_type": "markdown"}
# Besides the usual removal of row identifier, `treatmentid`, I'm also removing `activeupondischarge`, as we don't have complete information as to when diagnosis end.

# + {"Collapsed": "false", "persistent_id": "64f644db-fb9b-45c2-9c93-e4878850ed08"}
treat_df = treat_df.drop(['treatmentid', 'activeupondischarge'], axis=1)
treat_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Separate high level diagnosis

# + {"Collapsed": "false", "persistent_id": "99292fb1-2fff-46c0-9f57-e686a52bce75"}
treat_df.treatmentstring.value_counts()

# + {"Collapsed": "false", "persistent_id": "56252187-e07d-4042-a7c2-9bad34e8da22"}
treat_df.treatmentstring.map(lambda x: x.split('|')).head()

# + {"Collapsed": "false", "persistent_id": "d3c5a8e5-b60d-4b04-ad9f-9df30e1b63c3"}
treat_df.treatmentstring.map(lambda x: len(x.split('|'))).min()

# + {"Collapsed": "false", "persistent_id": "0fb9adae-17a9-433d-aa4a-9367511846c1"}
treat_df.treatmentstring.map(lambda x: len(x.split('|'))).max()

# + {"Collapsed": "false", "cell_type": "markdown"}
# There are always at least 3 higher level diagnosis. It could be beneficial to extract those first 3 levels to separate features, with the last one getting values until the end of the string, so as to avoid the need for the model to learn similarities that are already known.

# + {"Collapsed": "false", "persistent_id": "aee50693-322a-4a01-9ef2-fac27c51e45f"}
treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 0, separator='|'),
                               meta=('x', str)).value_counts()

# + {"Collapsed": "false", "persistent_id": "6ffd154f-ab1e-4614-b990-3414c2e8abf5"}
treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 1, separator='|'),
                               meta=('x', str)).value_counts()

# + {"Collapsed": "false", "persistent_id": "add7fc79-2edc-4481-939a-68c69f3b4383"}
treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 2, separator='|'),
                               meta=('x', str)).value_counts()

# + {"Collapsed": "false", "persistent_id": "fdfd37f9-1ac4-4c08-bec7-1f7971cee605"}
treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 3, separator='|'),
                               meta=('x', str)).value_counts()

# + {"Collapsed": "false", "persistent_id": "2e81fa29-6255-427f-aeb8-3d9ddd615565"}
treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 4, separator='|'),
                               meta=('x', str)).value_counts()

# + {"Collapsed": "false", "persistent_id": "7a30e647-0d39-480e-82ec-897defcfac38"}
treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 5, separator='|'),
                               meta=('x', str)).value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# <!-- There are always 8 levels of the notes. As the first 6 ones are essentially always the same ("notes/Progress Notes/Social History / Family History/Social History/Social History/"), it's best to just preserve the 7th one and isolate the 8th in a new feature. This way, the split provides further insight to the model on similar notes. However, it's also worth taking note that the 8th level of `notepath` seems to be identical to the feature `notevalue`. We'll look more into it later. -->

# + {"Collapsed": "false", "persistent_id": "44721e19-f088-4cf3-be69-5201d1260d52"}
treat_df['treatmenttype'] = treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 0, separator='|'), meta=('x', str))
treat_df['treatmenttherapy'] = treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 1, separator='|'), meta=('x', str))
treat_df['treatmentdetails'] = treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 2, separator='|', till_the_end=True), meta=('x', str))
treat_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
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

# + {"Collapsed": "false", "persistent_id": "e015c37e-1537-409f-b67e-9eaabf295873"}
new_cat_feat = ['treatmenttype', 'treatmenttherapy', 'treatmentdetails']
[cat_feat.append(col) for col in new_cat_feat]

# + {"Collapsed": "false", "persistent_id": "00289bd2-57fb-48ed-a61d-125d0bdefedb"}
cat_feat_nunique = [treat_df[feature].nunique() for feature in new_cat_feat]
cat_feat_nunique

# + {"Collapsed": "false", "persistent_id": "fa8c4e78-e6fc-4af7-94e7-14dfbbf3e756"}
new_cat_embed_feat = []
for i in range(len(new_cat_feat)):
    if cat_feat_nunique[i] > 5:
        # Add feature to the list of those that will be embedded
        cat_embed_feat.append(new_cat_feat[i])
        new_cat_embed_feat.append(new_cat_feat[i])

# + {"Collapsed": "false", "persistent_id": "29451e32-cf24-47b0-a783-9acc9b71b1c3"}
treat_df[new_cat_feat].head()

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "3c0aa3dc-34e6-4a10-b01e-439fe2c6f991"}
for i in range(len(new_cat_embed_feat)):
    feature = new_cat_embed_feat[i]
    # Prepare for embedding, i.e. enumerate categories
    treat_df[feature], cat_embed_feat_enum[feature] = du.embedding.enum_categorical_feature(treat_df, feature)

# + {"Collapsed": "false", "persistent_id": "a69a1a9d-9330-4f68-9c17-e8af92847439"}
treat_df[new_cat_feat].head()

# + {"Collapsed": "false", "persistent_id": "519ddcf4-c7d0-43a5-9896-39aaad9ded94"}
cat_embed_feat_enum

# + {"Collapsed": "false", "persistent_id": "ff54e442-e896-4310-8a95-c205cd2cbf93"}
treat_df[new_cat_feat].dtypes

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "373ef393-e31d-4778-a902-78fc1ce179f3"}
stream = open('cat_embed_feat_enum.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "e9364015-9dc8-45ed-a526-fbfd85a7d249"}
treat_df['ts'] = treat_df['treatmentoffset']
treat_df = treat_df.drop('treatmentoffset', axis=1)
treat_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "a833b09c-2fb0-46fc-af33-629e67113663"}
len(treat_df)

# + {"Collapsed": "false", "persistent_id": "45d801a9-e196-4940-92c9-9e73a4cccbb1"}
treat_df = treat_df.drop_duplicates()
treat_df.head()

# + {"Collapsed": "false", "persistent_id": "6f07e6d2-788a-4c5b-a0d8-0fced42215a7"}
len(treat_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "79121961-77a1-4657-a374-dd5d389174ed"}
treat_df = treat_df.set_index('ts')
treat_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "c79e4b14-61d9-48c3-bfff-da6e794d6df9"}
treat_df.reset_index().head()

# + {"Collapsed": "false", "persistent_id": "8672872e-aea3-480c-b5a2-383843db6e3e"}
treat_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='treatmenttype').head()

# + {"Collapsed": "false", "persistent_id": "ab644c5b-831a-4b57-a95c-fbac810e59f4"}
treat_df[treat_df.patientunitstayid == 1352520].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 105 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "e08bac51-50b2-47d1-aea3-8f72a80223aa"}
treat_df = du.embedding.join_categorical_enum(treat_df, new_cat_embed_feat)
treat_df.head()

# + {"Collapsed": "false", "persistent_id": "e2b70eff-045e-4587-af99-329381e839b2"}
treat_df.dtypes

# + {"Collapsed": "false", "persistent_id": "08de3813-83e3-478c-bfeb-e6dbeafacdb1"}
treat_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='treatmenttype').head()

# + {"Collapsed": "false", "persistent_id": "ef1b7d12-4dd4-4c9c-9204-55ad5ac98443"}
treat_df[treat_df.patientunitstayid == 1352520].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "4decdcbe-bf08-4577-b55c-dda15fbef130"}
treat_df.columns = du.data_processing.clean_naming(treat_df.columns)
treat_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Save the dataframe

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "7fecf050-a520-499d-ba70-96bc406e0a7e"}
treat_df.to_csv(f'{data_path}cleaned/unnormalized/diagnosis.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "8ec7af26-406d-4870-8475-0acca2b92876"}
treat_df.to_csv(f'{data_path}cleaned/normalized/diagnosis.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "23eefabe-23d7-4db4-b8e0-7c1e61ef2789"}
treat_df.describe().transpose()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join dataframes
#
# Merge dataframes by the unit stay, `patientunitstayid`, and the timestamp, `ts`, with a tolerence for a difference of up to 30 minutes.

# + {"Collapsed": "false", "persistent_id": "9f483dd2-1ac4-4476-91b8-1e0433af905f"}
treat_df = pd.read_csv(f'{data_path}cleaned/normalized/diagnosis.csv')
treat_df.head()

# + {"Collapsed": "false", "persistent_id": "93ba8928-d23a-4200-8001-252cd46481fd"}
len(treat_df)

# + {"Collapsed": "false", "persistent_id": "39d602af-5e6c-444c-af30-79b07c1d5ef2"}
treat_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "persistent_id": "ccc79c6c-497d-4682-a860-6aea8dfe2944"}
eICU_df = pd.merge_asof(eICU_df, treat_df, on='ts', by='patientunitstayid', direction='nearest', tolerance=30)
eICU_df.head()

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
stream = open('cat_embed_feat_enum.yaml', 'w')
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

# + {"toc-hr-collapsed": false, "Collapsed": "false", "cell_type": "markdown"}
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
stream = open('cat_embed_feat_enum.yaml', 'w')
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
stream = open('cat_embed_feat_enum.yaml', 'w')
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
