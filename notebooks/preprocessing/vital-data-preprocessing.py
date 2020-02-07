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
stream = open('cat_embed_feat_enum_vital.yaml', 'w')
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
