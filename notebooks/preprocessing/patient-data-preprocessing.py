# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: eICU-mortality-prediction
#     language: python
#     name: eicu-mortality-prediction
# ---

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": false}
# # Patient Data Preprocessing
# ---
#
# Reading and preprocessing patient data of the eICU dataset from MIT with the data from over 139k patients collected in the US.
#
# This notebook addresses the preprocessing of the following eICU tables:
# * patient
# * note

# + [markdown] {"Collapsed": "false", "colab_type": "text", "id": "KOdmFzXqF7nq"}
# ## Importing the necessary packages

# + {"Collapsed": "false", "colab": {}, "colab_type": "code", "execution_event_id": "897396d2-3f1a-416c-bd55-24da2aad1e55", "id": "G5RrWE9R_Nkl", "last_executed_text": "import os                                  # os handles directory/workspace changes\nimport numpy as np                         # NumPy to handle numeric and NaN operations\nimport yaml                                # Save and load YAML files", "persistent_id": "522745b5-b5bf-479f-b697-5c7e9e12fc33"}
import os                                  # os handles directory/workspace changes
import numpy as np                         # NumPy to handle numeric and NaN operations
import yaml                                # Save and load YAML files

# + {"Collapsed": "false", "execution_event_id": "b09d25cb-f680-46ae-ab23-989f74324844", "last_executed_text": "# Debugging packages\nimport pixiedust                           # Debugging in Jupyter Notebook cells", "persistent_id": "02accdbf-be7e-415c-ba11-165906e66c50"}
# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# + {"Collapsed": "false", "execution_event_id": "baeb346a-1c34-42d1-a501-7ae37369255e", "last_executed_text": "# Change to parent directory (presumably \"Documents\")\nos.chdir(\"../../..\")\n\n# Path to the CSV dataset files\ndata_path = 'Documents/Datasets/Thesis/eICU/uncompressed/'\n\n# Path to the code files\nproject_path = 'Documents/GitHub/eICU-mortality-prediction/'", "persistent_id": "a1f6ee7f-36d4-489d-b2dd-ec2a38f15d11"}
# Change to parent directory (presumably "Documents")
os.chdir("../../../..")
# Path to the parquet dataset files
data_path = 'data/eICU/cleaned/'
# Path to the code files
project_path = 'code/eICU-mortality-prediction/'

# + {"Collapsed": "false"}
# # Make sure that every large operation can be handled, by using the disk as an overflow for the memory
# # !export MODIN_OUT_OF_CORE=true
# # Another trick to do with Pandas so as to be able to allocate bigger objects to memory
# # !sudo bash -c 'echo 1 > /proc/sys/vm/overcommit_memory'

# + {"Collapsed": "false", "execution_event_id": "82ef68be-443a-4bb8-8abd-7457a7005b4d", "last_executed_text": "import modin.pandas as pd                  # Optimized distributed version of Pandas\nimport data_utils as du                    # Data science and machine learning relevant methods", "persistent_id": "c0c2e356-d4f4-4a9d-bec2-88bdf9eb6a38"}
# import modin.pandas as pd                  # Optimized distributed version of Pandas
import pandas as pd
import data_utils as du                    # Data science and machine learning relevant methods

# + [markdown] {"Collapsed": "false"}
# Allow pandas to show more columns:

# + {"Collapsed": "false"}
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

# + [markdown] {"Collapsed": "false"}
# Set the random seed for reproducibility

# + {"Collapsed": "false", "execution_event_id": "9f0e32aa-08d7-4d35-bde8-7855e1788d5e", "last_executed_text": "du.set_random_seed(42)", "persistent_id": "39b552cd-6948-4ec8-ac04-42f850c1e05a"}
du.set_random_seed(42)

# + [markdown] {"Collapsed": "false"}
# ## Patient data

# + [markdown] {"Collapsed": "false"}
# ### Initialize variables

# + {"Collapsed": "false", "persistent_id": "754a96f8-d389-4968-8c13-52e5e9d0bf82"}
# List of categorical features
cat_feat = []
# Dictionary of the one hot encoded columns originary from each categorical feature, that will be embedded
cat_feat_ohe = dict()

# + [markdown] {"Collapsed": "false"}
# ### Read the data

# + {"Collapsed": "false", "execution_event_id": "28c41b31-f70e-4de0-a16a-ad04b572ecf1", "last_executed_text": "patient_df = pd.read_csv(f'{data_path}original/patient.csv')\npatient_df.head()", "persistent_id": "9ee194e5-c316-4fe5-8cd8-cc4188be9447"}
patient_df = pd.read_csv(f'{data_path}original/patient.csv')
patient_df.head()

# + {"Collapsed": "false"}
patient_df.dtypes

# + {"Collapsed": "false", "execution_event_id": "3e769f41-3675-4ef5-bd30-491cec9fd3bb", "last_executed_text": "len(patient_df)", "persistent_id": "8a040368-7c65-4d72-a4a7-622b63378c3e"}
len(patient_df)

# + {"Collapsed": "false", "execution_event_id": "72759c29-4ee9-45c6-a6fb-aca6b08b1496", "last_executed_text": "patient_df.patientunitstayid.nunique()", "persistent_id": "6144f623-8410-4651-9703-bffb19f3e9cc"}
patient_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "execution_event_id": "a72e876f-562d-482c-b256-5f5bdf7d1463", "last_executed_text": "patient_df.patientunitstayid.value_counts()", "persistent_id": "7022c0ab-2847-4b14-914b-69fcf3d3ca07"}
patient_df.patientunitstayid.value_counts()

# + [markdown] {"Collapsed": "false"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "execution_event_id": "f42f4420-5088-4096-b8e4-dd77dd1cb2d5", "last_executed_text": "patient_df.describe().transpose()", "persistent_id": "d4aa2831-2d82-47d2-8538-11d1257f3891"}
patient_df.describe().transpose()

# + {"Collapsed": "false", "execution_event_id": "a40cfc9a-9936-4e65-b89d-a8cc11c43913", "last_executed_text": "patient_df.columns", "persistent_id": "4678d828-e15a-43f1-bc52-80b13a2b7c7e"}
patient_df.columns

# + {"Collapsed": "false"}
patient_df.dtypes

# + [markdown] {"Collapsed": "false"}
# ### Check for missing values

# + {"Collapsed": "false", "execution_event_id": "ea095970-5dbe-4425-8c18-ea7ae79b2e04", "last_executed_text": "du.search_explore.dataframe_missing_values(patient_df)", "persistent_id": "9e6a1829-a25a-44a6-a1b8-074ccd5664c4", "pixiedust": {"displayParams": {}}}
du.search_explore.dataframe_missing_values(patient_df)

# + [markdown] {"Collapsed": "false"}
# ### Remove unneeded features
#
# Besides removing unneeded hospital and time information, I'm also removing the admission diagnosis (`apacheadmissiondx`) as it doesn't follow the same structure as the remaining diagnosis data (which is categorized in increasingly specific categories, separated by "|").

# + {"Collapsed": "false", "execution_event_id": "8d352b03-4c33-4e8b-98d1-215f0ed2da33", "last_executed_text": "patient_df = patient_df[['patientunitstayid', 'gender', 'age', 'ethnicity',  'admissionheight',\n                         'hospitaldischargeoffset', 'hospitaldischargelocation', 'hospitaldischargestatus',\n                         'admissionweight']]\npatient_df.head()", "persistent_id": "2b20dcee-1d77-4976-9d75-90a6359d9edc"}
patient_df = patient_df[['patientunitstayid', 'gender', 'age', 'ethnicity',  'admissionheight',
                         'hospitaldischargeoffset', 'hospitaldischargelocation', 'hospitaldischargestatus',
                         'unitdischargeoffset', 'unitdischargelocation', 'unitdischargestatus',
                         'admissionweight']]
patient_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Make the age feature numeric
#
# In the eICU dataset, ages above 89 years old are not specified. Instead, we just receive the indication "> 89". In order to be able to work with the age feature numerically, we'll just replace the "> 89" values with "90", as if the patient is 90 years old. It might not always be the case, but it shouldn't be very different and it probably doesn't affect too much the model's logic.

# + {"Collapsed": "false", "execution_event_id": "8e79a3a5-1cc3-44ea-a911-20532298ff06", "last_executed_text": "patient_df.age.value_counts().head()", "persistent_id": "61119cf7-7f1a-4382-80d0-65ceeb6137ff"}
patient_df.age.value_counts().head()

# + {"Collapsed": "false", "execution_event_id": "b5e1da10-fe5f-45fc-9da8-0761581940c9", "last_executed_text": "# Replace the \"> 89\" years old indication with 90 years\npatient_df.age = patient_df.age.replace(to_replace='> 89', value=90)", "persistent_id": "0ae741ed-ecdb-4b2a-acad-de7974d9301e"}
# Replace the "> 89" years old indication with 90 years
patient_df.age = patient_df.age.replace(to_replace='> 89', value=90)

# + {"Collapsed": "false", "execution_event_id": "8400cdcc-df79-4e9c-ab0d-88419a94d589", "last_executed_text": "patient_df.age.value_counts().head()", "persistent_id": "45ee35d1-b272-4e68-8305-cf6962322c2a"}
patient_df.age.value_counts().head()

# + {"Collapsed": "false", "execution_event_id": "c8ea1329-9c73-4240-bfcf-7ccc382fc0ad", "last_executed_text": "# Make the age feature numeric\npatient_df.age = patient_df.age.astype(float)", "persistent_id": "978a88e3-2ac7-41a7-abf5-c84eafd8d360"}
# Make the age feature numeric
patient_df.age = patient_df.age.astype(float)

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": false}
# ### Discretize categorical features
#
# Convert binary categorical features into one hot encode columns, which can later be embedded or used as is.

# + [markdown] {"Collapsed": "false"}
# #### One hot encode features

# + [markdown] {"Collapsed": "false"}
# Update list of categorical features:

# + {"Collapsed": "false", "execution_event_id": "a5b9f20c-a2c1-4a75-ace1-6656ab4cdc5f", "last_executed_text": "patient_df.gender.value_counts()", "persistent_id": "825d7d09-34df-43ef-a914-2a68f33723f2"}
patient_df.gender.value_counts()

# + {"Collapsed": "false", "execution_event_id": "8326b7b6-04a5-468c-8a05-1d9994a051b2", "last_executed_text": "patient_df.gender = patient_df.gender.map(lambda x: 1 if x == 'Male' else 0 if x == 'Female' else np.nan)", "persistent_id": "b500948a-b8fe-4fd3-a863-76d137f6ae5f"}
patient_df.gender = patient_df.gender.map(lambda x: 1 if x == 'Male' else 0 if x == 'Female' else np.nan)

# + {"Collapsed": "false", "execution_event_id": "063a69b8-5cea-4add-b11c-3225c8eedb78", "last_executed_text": "patient_df.gender.value_counts()", "persistent_id": "cb548210-9b6c-47dc-a094-47872216500d"}
patient_df.gender.value_counts()

# + [markdown] {"Collapsed": "false"}
# #### Separate and prepare features for embedding
#
# Identify categorical features that have more than 5 unique categories, which will go through an embedding layer afterwards, and enumerate them.
#

# + [markdown] {"Collapsed": "false"}
# Update list of categorical features and add those that will need embedding (features with more than 5 unique values):

# + {"Collapsed": "false", "execution_event_id": "6cb657da-f34c-4a83-a358-65adf97f4a5d", "last_executed_text": "cat_feat = ['ethnicity']\n[cat_feat.append(col) for col in cat_feat]", "persistent_id": "9fdc9b59-d7e5-47e6-8d32-f3ac7fbb582d"}
cat_feat = ['ethnicity']

# + {"Collapsed": "false", "execution_event_id": "1a89c528-9de1-4250-a6e6-2a29c6045770", "last_executed_text": "patient_df[cat_feat].head()", "persistent_id": "99d08bce-69f1-4a19-8e1d-ab2a49574506"}
patient_df[cat_feat].head()

# + [markdown] {"Collapsed": "false"}
# Apply one hot encoding:

# + {"Collapsed": "false", "execution_event_id": "9152d797-82f8-46cb-9822-36e46b66b3ac", "last_executed_text": "for i in range(len(new_cat_embed_feat)):\n    feature = new_cat_embed_feat[i]\n    # Prepare for embedding, i.e. enumerate categories\n    patient_df[feature], cat_feat_ohe[feature] = du.embedding.enum_categorical_feature(patient_df, feature)", "persistent_id": "32a0ca8b-24f4-41d6-8565-038e39497c7e"}
patient_df, new_columns = du.data_processing.one_hot_encoding_dataframe(patient_df, columns=cat_feat,
                                                                        join_rows=False,
                                                                        get_new_column_names=True,
                                                                        inplace=True)
patient_df

# + {"Collapsed": "false", "execution_event_id": "56578e03-6482-46bb-91fa-271c875f77f2", "last_executed_text": "patient_df[cat_feat].dtypes", "persistent_id": "2d79bb26-bb3f-4d3e-beac-0e809c504bdb"}
patient_df.dtypes

# + [markdown] {"Collapsed": "false"}
# Save the association between the original categorical features and the new one hot encoded columns:

# + {"Collapsed": "false", "persistent_id": "26eac7f3-9081-4a96-ae4a-40054c223fd7"}
for orig_col in cat_feat:
    cat_feat_ohe[orig_col] = [ohe_col for ohe_col in new_columns
                              if ohe_col.startswith(orig_col)]

# + {"Collapsed": "false"}
cat_feat_ohe

# + [markdown] {"Collapsed": "false"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "execution_event_id": "fbfc56d3-0060-43b5-937a-c2721d8112d6", "last_executed_text": "stream = open('cat_feat_ohe_patient.yml', 'w')\nyaml.dump(cat_feat_ohe, stream, default_flow_style=False)", "persistent_id": "57ad30f0-8e12-482d-91c7-789b3b64b39a"}
stream = open(f'{data_path}/cleaned/cat_feat_ohe_patient.yml', 'w')
yaml.dump(cat_feat_ohe, stream, default_flow_style=False)

# + [markdown] {"Collapsed": "false"}
# ### Create mortality label
#
# Combine info from discharge location and discharge status. Using the hospital discharge data, instead of the unit, as it has a longer perspective on the patient's status. I then save a feature called "deathOffset", which has a number if the patient is dead on hospital discharge or is NaN if the patient is still alive/unknown (presumed alive if unknown). Based on this, a label can be made later on, when all the tables are combined in a single dataframe, indicating if a patient dies in the following X time, according to how faraway we want to predict.

# + {"Collapsed": "false", "execution_event_id": "7f386899-12e5-4fe2-8292-91c86cc3f1fd", "last_executed_text": "patient_df.hospitaldischargestatus.value_counts()", "persistent_id": "dcf72825-f38a-4a63-8a36-aa1e3fd3ac86"}
patient_df.hospitaldischargestatus.value_counts()

# + {"Collapsed": "false", "execution_event_id": "7f386899-12e5-4fe2-8292-91c86cc3f1fd", "last_executed_text": "patient_df.hospitaldischargestatus.value_counts()", "persistent_id": "dcf72825-f38a-4a63-8a36-aa1e3fd3ac86"}
patient_df.unitdischargestatus.value_counts()

# + {"Collapsed": "false", "execution_event_id": "3a0e6a8a-0784-442b-8925-c6487437eb24", "last_executed_text": "patient_df.hospitaldischargelocation.value_counts()", "persistent_id": "6b20ced2-c573-47e9-a02b-7d4aeceaddf7"}
patient_df.hospitaldischargelocation.value_counts()

# + {"Collapsed": "false", "execution_event_id": "3a0e6a8a-0784-442b-8925-c6487437eb24", "last_executed_text": "patient_df.hospitaldischargelocation.value_counts()", "persistent_id": "6b20ced2-c573-47e9-a02b-7d4aeceaddf7"}
patient_df.unitdischargelocation.value_counts()


# + {"Collapsed": "false", "execution_event_id": "6821f025-536a-45dd-b299-4f4a97d14196", "last_executed_text": "def get_death_ts(df):\n    if df['hospitaldischargestatus'] == 'Expired':\n        df['death_ts'] = df['hospitaldischargeoffset']\n    else:\n        df['death_ts'] = np.nan\n    return df", "persistent_id": "8527fbdc-1731-45bb-9a68-e8b04a3ed4d2"}
def get_death_ts(df):
    if df['unitdischargestatus'] == 'Expired':
        df['death_ts'] = df['unitdischargeoffset']
    elif df['hospitaldischargestatus'] == 'Expired':
        df['death_ts'] = df['hospitaldischargeoffset']
    else:
        df['death_ts'] = np.nan
    return df


# + {"Collapsed": "false", "execution_event_id": "a1782cdd-c154-4b47-a532-45e5f2e32f8d", "last_executed_text": "tmp_df = patient_df.copy()\ntmp_df['death_ts'] = tmp_df['hospitaldischargeoffset']", "persistent_id": "8a1ad954-7559-4c0b-83b5-fbb847479dcd"}
tmp_df = patient_df.copy()
tmp_df['death_ts'] = tmp_df['hospitaldischargeoffset']

# + {"Collapsed": "false", "execution_event_id": "0e0ee4cd-528f-4fbc-a596-d93df8205528", "last_executed_text": "tmp_df = tmp_df.apply(get_death_ts, axis=1, result_type='broadcast')\ntmp_df.head()", "persistent_id": "766f6f87-bf5b-4032-a3b4-bedda6f5efc7"}
tmp_df = tmp_df.apply(get_death_ts, axis=1, result_type='broadcast')
tmp_df.head()

# + {"Collapsed": "false", "execution_event_id": "a21551b3-564a-4258-9ecb-5a7036e49606", "last_executed_text": "tmp_df['death_ts'].index.value_counts()", "persistent_id": "77e65a7d-36b7-410e-80df-7bf1806bfcff"}
tmp_df['death_ts'].index.value_counts()

# + {"Collapsed": "false", "execution_event_id": "899503f2-683b-4963-a0e8-865c4a3601bf", "last_executed_text": "tmp_df['death_ts'][100783]", "persistent_id": "7f64c658-2a41-4ac9-b4f9-9eafd33900a7"}
tmp_df['death_ts'][100783]

# + {"Collapsed": "false", "execution_event_id": "1537f386-c4a8-4594-8472-497e591f219e", "last_executed_text": "(~tmp_df['death_ts'].isnull()).sum()", "persistent_id": "f87e7659-f89f-4353-a695-4dc670caca8d"}
(~tmp_df['death_ts'].isnull()).sum()

# + {"Collapsed": "false", "execution_event_id": "c9cd6a9b-c1d5-4062-b80b-2ffc60f43255", "last_executed_text": "tmp_col = tmp_df.groupby('patientunitstayid').death_ts.max()\ntmp_col", "persistent_id": "55cc99b9-b702-4303-8e2e-a9f410ee60d1"}
tmp_col = tmp_df.groupby('patientunitstayid').death_ts.max()
tmp_col

# + {"Collapsed": "false", "execution_event_id": "194e8afe-389f-4eb7-869e-ea6035378305", "last_executed_text": "tmp_col.index.value_counts()", "persistent_id": "77e65a7d-36b7-410e-80df-7bf1806bfcff"}
tmp_col.index.value_counts()

# + {"Collapsed": "false", "execution_event_id": "f6638f12-10f9-4d8a-9a3f-270b4b0c828f", "last_executed_text": "(~tmp_col.isnull()).sum()", "persistent_id": "f87e7659-f89f-4353-a695-4dc670caca8d"}
(~tmp_col.isnull()).sum()

# + {"Collapsed": "false", "execution_event_id": "0783867e-c163-4b21-8ff5-e83079eaccb6", "last_executed_text": "patient_df['death_ts'] = tmp_col\npatient_df.head()", "persistent_id": "fa551a3f-d2a5-49c1-9512-cbb2dcd51478"}
patient_df.set_index('patientunitstayid', inplace=True)
patient_df['death_ts'] = tmp_col
patient_df.reset_index(inplace=True)
patient_df.head()
# -

(~patient_df['death_ts'].isnull()).sum()

# + [markdown] {"Collapsed": "false"}
# Remove the now unneeded hospital discharge features:

# + {"Collapsed": "false", "execution_event_id": "e81a2ecc-aa82-4434-b79d-f9416658dc52", "last_executed_text": "patient_df = patient_df.drop(['hospitaldischargeoffset', 'hospitaldischargestatus', 'hospitaldischargelocation'], axis=1)\npatient_df.head(6)", "persistent_id": "b30bb22f-f470-4132-92e7-bed1f058f4a8"}
patient_df = patient_df.drop(['hospitaldischargeoffset', 'hospitaldischargestatus', 'hospitaldischargelocation',
                              'unitdischargeoffset', 'unitdischargestatus', 'unitdischargelocation'], axis=1)
patient_df.head(6)

# + [markdown] {"Collapsed": "false"}
# ### Create the timestamp feature and sort

# + [markdown] {"Collapsed": "false"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "execution_event_id": "0fe48344-f367-4402-a310-93d06ec954bb", "last_executed_text": "patient_df['ts'] = 0\npatient_df.head()", "persistent_id": "dfab2799-af6c-4475-8341-ec3f40546ed1"}
patient_df['ts'] = 0
patient_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Normalize data

# + [markdown] {"Collapsed": "false"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "execution_event_id": "e1e9c146-1b0e-4659-969e-22838a05b5a5", "last_executed_text": "patient_df.to_csv(f'{data_path}cleaned/unnormalized/ohe/patient.csv')", "persistent_id": "8d66425b-0aae-42d5-ac47-9a0456a080b8"}
# patient_df.to_csv(f'{data_path}cleaned/unnormalized/ohe/patient.csv')

# + {"Collapsed": "false", "execution_event_id": "3d6d0a5c-9160-4ffc-87d4-85632a968a1d", "last_executed_text": "patient_df_norm = du.data_processing.normalize_data(patient_df, categ_columns=cat_feat,\n                                                    id_columns=['patientunitstayid', 'ts', 'death_ts'])\npatient_df_norm.head(6)", "persistent_id": "d5ad6017-ad4a-419c-badb-9454add7752d", "pixiedust": {"displayParams": {}}}
patient_df, mean, std = du.data_processing.normalize_data(patient_df,
                                                          id_columns=['patientunitstayid', 'ts', 'death_ts'],
                                                          get_stats=True, inplace=True)
patient_df.head(6)

# + [markdown] {"Collapsed": "false"}
# Save a dictionary with the mean and standard deviation values of each column that was normalized:

# + {"Collapsed": "false"}
norm_stats = dict()
for key, _ in mean.items():
    norm_stats[key] = dict()
    norm_stats[key]['mean'] = mean[key]
    norm_stats[key]['std'] = std[key]
norm_stats

# + {"Collapsed": "false"}
stream = open(f'{data_path}/cleaned/patient_norm_stats.yml', 'w')
yaml.dump(norm_stats, stream, default_flow_style=False)

# + {"Collapsed": "false", "execution_event_id": "3eed71a9-b6b3-4f0f-99b3-0b80313faf98", "last_executed_text": "patient_df_norm.to_csv(f'{data_path}cleaned/normalized/ohe/patient.csv')", "persistent_id": "64492d9f-df5d-4940-b931-cbb4c3af2949"}
patient_df.to_csv(f'{data_path}cleaned/normalized/ohe/patient.csv')

# + [markdown] {"Collapsed": "false"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "execution_event_id": "08f10c87-1a59-4890-9af4-32c93f0cba7a", "last_executed_text": "patient_df_norm.describe().transpose()", "persistent_id": "68c630e8-151b-495f-b652-c00a55d92e78"}
patient_df.describe().transpose()

# + {"Collapsed": "false", "execution_event_id": "3146d2bf-b3f7-410b-95a0-5e1b52b40f8a", "last_executed_text": "# [TODO] Remove the rows with ts = 0 if there are no matching rows in other tables", "persistent_id": "826b9069-c468-47a7-aa00-a92edc829e13"}
# [TODO] Remove the rows with ts = 0 if there are no matching rows in other tables

# + {"Collapsed": "false"}
patient_df = pd.read_csv(f'{data_path}cleaned/normalized/ohe/patient.csv', usecols=['patientunitstayid', 'death_ts'])
patient_df

# + {"Collapsed": "false"}
note_df = pd.read_csv(f'{data_path}cleaned/normalized/ohe/note.csv')
note_df
# -

len(note_df)

note_df.set_index('patientunitstayid', inplace=True)
patient_df.set_index('patientunitstayid', inplace=True)

len(note_df)

note_df = note_df.join(patient_df, how='left')
note_df

len(note_df)

patient_df

patient_df.loc[141168]

patient_df.loc[263814]

patient_df.loc[263815]

note_df.loc[141168]

(~patient_df['death_ts'].isnull()).sum()

patient_df[patient_df['death_ts'] == 0]

# + {"Collapsed": "false"}
patient_dies = ~(patient_df.groupby('patientunitstayid').death_ts.max().isna())
patient_dies

# + {"Collapsed": "false"}
patient_dies.value_counts()

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": false}
# ## Notes data

# + [markdown] {"Collapsed": "false"}
# ### Initialize variables

# + {"Collapsed": "false", "persistent_id": "754a96f8-d389-4968-8c13-52e5e9d0bf82"}
# List of categorical features
cat_feat = []
# Dictionary of the one hot encoded columns originary from each categorical feature, that will be embedded
cat_feat_ohe = dict()

# + [markdown] {"Collapsed": "false"}
# ### Read the data

# + {"Collapsed": "false", "execution_event_id": "bf6e34cd-7ae9-4636-a9b9-27d404b49610", "last_executed_text": "note_df = pd.read_csv(f'{data_path}original/note.csv')\nnote_df.head()", "persistent_id": "51197f67-95e2-4184-a73f-7885cb975084"}
note_df = pd.read_csv(f'{data_path}original/note.csv')
note_df.head()

# + {"Collapsed": "false", "execution_event_id": "d2a30b69-ae10-4a5e-be98-56280de75b37", "last_executed_text": "len(note_df)", "persistent_id": "c3d45ba7-91dd-41d7-8699-c70390556018"}
len(note_df)

# + {"Collapsed": "false", "execution_event_id": "e7d22357-65a7-4573-bd2e-42fc3794e931", "last_executed_text": "note_df.patientunitstayid.nunique()", "persistent_id": "17c46962-79a6-48a6-b67a-4863028ed897"}
note_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Get an overview of the dataframe through the `describe` method:

# + {"Collapsed": "false", "execution_event_id": "546023f0-9a90-47da-a108-5f59f57fa5af", "last_executed_text": "note_df.describe().transpose()", "persistent_id": "6e745ae7-a0ea-4169-96de-d49e9f510ed9"}
note_df.describe().transpose()

# + {"Collapsed": "false", "execution_event_id": "dac11dab-f33a-4ab5-bf3e-c78447f9cdc4", "last_executed_text": "note_df.columns", "persistent_id": "ac1da815-dcd6-44df-8164-a987f9623255"}
note_df.columns

# + {"Collapsed": "false", "execution_event_id": "ab10d0e5-46be-4824-92ac-b1939b57792d", "last_executed_text": "note_df.dtypes", "persistent_id": "8e30eac6-9424-4ce6-801e-5e013a964863"}
note_df.dtypes

# + [markdown] {"Collapsed": "false"}
# ### Check for missing values

# + {"Collapsed": "false", "execution_event_id": "a04c5f38-289a-4569-8c57-89eddcda2b0d", "last_executed_text": "du.search_explore.dataframe_missing_values(note_df)", "persistent_id": "c4e967db-7ace-41df-8678-7cd11d1e002b", "pixiedust": {"displayParams": {}}}
du.search_explore.dataframe_missing_values(note_df)

# + [markdown] {"Collapsed": "false"}
# ### Remove unneeded features

# + {"Collapsed": "false", "execution_event_id": "39bf557e-d53a-4ecd-8118-a9719aa49188", "last_executed_text": "note_df.notetype.value_counts().head(20)", "persistent_id": "9f3b85d0-8480-4f30-8455-fde084ba7c69"}
note_df.notetype.value_counts().head(20)

# + {"Collapsed": "false", "execution_event_id": "0df9919f-4d46-4c34-a326-df8e8cf7f18c", "last_executed_text": "note_df.notepath.value_counts().head(40)", "persistent_id": "e41f6a9d-c5ea-45e3-b38b-89ba70de1b87"}
note_df.notepath.value_counts().head(40)

# + {"Collapsed": "false", "execution_event_id": "04986ee5-3cc1-453c-8127-05f513370b5d", "last_executed_text": "note_df.notevalue.value_counts().head(20)", "persistent_id": "b12dcddf-9703-4aff-8a20-df1d147ef554"}
note_df.notevalue.value_counts().head(20)

# + {"Collapsed": "false", "execution_event_id": "3528d324-be32-4076-9a34-d99c3e768b62", "last_executed_text": "note_df[note_df.notepath.str.contains('notes/Progress Notes/Social History')].head(20)", "persistent_id": "faeab8ba-1cd0-4bac-801d-64b7c91a0637"}
note_df[note_df.notepath.str.contains('notes/Progress Notes/Social History')].head(20)

# + {"Collapsed": "false", "execution_event_id": "97fce90a-cb98-480d-a76b-fa02cdc744b9", "last_executed_text": "note_df[note_df.notepath.str.contains('notes/Progress Notes/Social History')].notepath.value_counts().head(20)", "persistent_id": "0a5ddffd-b815-4b55-93ee-d5aeba54be0a"}
note_df[note_df.notepath.str.contains('notes/Progress Notes/Social History')].notepath.value_counts().head(20)

# + {"Collapsed": "false", "execution_event_id": "45af80ba-5e5a-424d-a5d1-cc83a54a8ca3", "last_executed_text": "note_df[note_df.notepath.str.contains('notes/Progress Notes/Social History')].notevalue.value_counts().head(20)", "persistent_id": "3261378f-0eee-477f-bdfd-01cb1af45334"}
note_df[note_df.notepath.str.contains('notes/Progress Notes/Social History')].notevalue.value_counts().head(20)

# + [markdown] {"Collapsed": "false"}
# Out of all the possible notes, only those addressing the patient's social history seem to be interesting and containing information not found in other tables. As such, we'll only keep the note paths that mention social history:

# + {"Collapsed": "false", "execution_event_id": "58f39665-bca9-4c38-b6ce-1b8177332e40", "last_executed_text": "note_df = note_df[note_df.notepath.str.contains('notes/Progress Notes/Social History')]\nnote_df.head()", "persistent_id": "d3a0e8a8-68d6-4c90-aded-0f5940c3936b"}
note_df = note_df[note_df.notepath.str.contains('notes/Progress Notes/Social History')]
note_df.head()

# + {"Collapsed": "false", "execution_event_id": "ae4aa363-8c1c-46ec-a3ed-2a89c95acd21", "last_executed_text": "len(note_df)", "persistent_id": "8409525f-e11f-4cf5-acd7-56fcdcf6c130"}
len(note_df)

# + [markdown] {"Collapsed": "false"}
# There are still rows that seem to contain irrelevant data. Let's remove them by finding rows that contain specific words, like "obtain" and "print", that only appear in said irrelevant rows:

# + {"Collapsed": "false", "execution_event_id": "47ec5a85-7a34-42d5-ab38-80d338ae3bbb", "last_executed_text": "category_types_to_remove = ['obtain', 'print', 'copies', 'options']", "persistent_id": "2eefd7ee-b1af-4bcc-aece-7813b4ed2b29"}
category_types_to_remove = ['obtain', 'print', 'copies', 'options']

# + {"Collapsed": "false", "execution_event_id": "ae917580-7c61-4845-9023-d046443fcde7", "last_executed_text": "du.search_explore.find_row_contains_word(note_df, feature='notepath', words=category_types_to_remove).value_counts()", "persistent_id": "094b7adf-176e-4989-8559-a114fef6e6e0"}
du.search_explore.find_row_contains_word(note_df, feature='notepath', words=category_types_to_remove).value_counts()

# + {"Collapsed": "false", "execution_event_id": "7c2e47e8-a71a-4c1b-9d1b-4597af8bfdd9", "last_executed_text": "note_df = note_df[~du.search_explore.find_row_contains_word(note_df, feature='notepath', words=category_types_to_remove)]\nnote_df.head()", "persistent_id": "d7ac4099-e9ed-489d-97ae-c4af5879c9ba"}
note_df = note_df[~du.search_explore.find_row_contains_word(note_df, feature='notepath', words=category_types_to_remove)]
note_df.head()

# + {"Collapsed": "false", "execution_event_id": "6d25dea6-3bfd-4311-897a-577b19db33eb", "last_executed_text": "len(note_df)", "persistent_id": "0399795d-cff1-4df9-a322-a5fcd7ba11d7"}
len(note_df)

# + {"Collapsed": "false", "execution_event_id": "ee5e4d43-535e-43ee-a1d7-706e41fbd9bb", "last_executed_text": "note_df.patientunitstayid.nunique()", "persistent_id": "d3b2d0c7-c8eb-404d-a0ac-e10e761d10fb"}
note_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "execution_event_id": "16cfc6a4-d5a3-4a00-9215-3f210db8a340", "last_executed_text": "note_df.notetype.value_counts().head(20)", "persistent_id": "47ec86ae-999d-4678-889a-1653ca1a8bfb"}
note_df.notetype.value_counts().head(20)

# + [markdown] {"Collapsed": "false"}
# Filtering just for interesting social history data greatly reduced the data volume of the notes table, now only present in around 20.5% of the unit stays. Still, it might be useful to include.

# + [markdown] {"Collapsed": "false"}
# Besides the usual removal of row identifier, `noteid`, I'm also removing apparently irrelevant (`noteenteredoffset`, `notetype`) and redundant (`notetext`) columns:

# + {"Collapsed": "false", "execution_event_id": "2e499a1a-e9a0-42cf-b351-98273b224c15", "last_executed_text": "note_df = note_df.drop(['noteid', 'noteenteredoffset', 'notetype', 'notetext'], axis=1)\nnote_df.head()", "persistent_id": "5beb1b97-7a5b-446c-934d-74b99556151f"}
note_df = note_df.drop(['noteid', 'noteenteredoffset', 'notetype', 'notetext'], axis=1)
note_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Separate high level notes

# + {"Collapsed": "false", "execution_event_id": "86ed6fec-5a95-403d-b0fb-741624d37089", "last_executed_text": "note_df.notepath.value_counts().head(20)", "persistent_id": "4bc9189b-e06f-49a6-8613-d9d46f4ac4f7"}
note_df.notepath.value_counts().head(20)

# + {"Collapsed": "false", "execution_event_id": "45d1a28c-caf8-4911-acd2-7f93e21f15ef", "last_executed_text": "note_df.notepath.map(lambda x: x.split('/')).head().values", "persistent_id": "1cf61f46-6540-4445-b9ff-9be941f513bc"}
note_df.notepath.map(lambda x: x.split('/')).head().values

# + {"Collapsed": "false", "execution_event_id": "aeed908a-bf58-4665-bcb2-d986bb9d4677", "last_executed_text": "note_df.notepath.map(lambda x: len(x.split('/'))).min()", "persistent_id": "93e3bc60-2c47-49f6-896b-f966f7fd6e42"}
note_df.notepath.map(lambda x: len(x.split('/'))).min()

# + {"Collapsed": "false", "execution_event_id": "2fdb7585-2fa4-412a-ac6b-57f43f27dcf8", "last_executed_text": "note_df.notepath.map(lambda x: len(x.split('/'))).max()", "persistent_id": "7a73ec52-0d6e-4627-a31f-e34ecfc79648"}
note_df.notepath.map(lambda x: len(x.split('/'))).max()

# + {"Collapsed": "false", "execution_event_id": "3b05f725-2b8f-415a-9df0-898485ae78d7", "last_executed_text": "note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 1, separator='/')).value_counts()", "persistent_id": "99a73002-27a5-4796-9681-0dab429335bf"}
note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 1, separator='/')).value_counts()

# + {"Collapsed": "false", "execution_event_id": "06e1f61d-c40c-4c3b-8a09-927e90c37c9a", "last_executed_text": "note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 2, separator='/')).value_counts()", "persistent_id": "e907d8c4-ee87-4432-9cd4-9cd4f6999f2b"}
note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 2, separator='/')).value_counts()

# + {"Collapsed": "false", "execution_event_id": "3fa1a2e4-4623-4dcf-b3e6-09788fddfb53", "last_executed_text": "note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 3, separator='/')).value_counts()", "persistent_id": "4a2a2b6b-9466-4592-a45e-698c9a7c944d"}
note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 3, separator='/')).value_counts()

# + {"Collapsed": "false", "execution_event_id": "d8c5c05e-d595-4365-9ab2-bbe17f990450", "last_executed_text": "note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 4, separator='/')).value_counts()", "persistent_id": "caea8162-79c0-4e2f-8b7b-cffbd26120ed"}
note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 4, separator='/')).value_counts()

# + {"Collapsed": "false", "execution_event_id": "e1090a0d-12da-4be3-b8cb-ef6a10edb3c3", "last_executed_text": "note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 5, separator='/')).value_counts()", "persistent_id": "b222d3b9-caec-4b7f-ab63-c86e6dd3697c"}
note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 5, separator='/')).value_counts()

# + {"Collapsed": "false", "execution_event_id": "2e464c6c-ab0d-42a5-9dfd-baabc2d416e6", "last_executed_text": "note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 6, separator='/')).value_counts()", "persistent_id": "32a752c7-5605-406d-acbd-2e0f420b7514"}
note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 6, separator='/')).value_counts()

# + {"Collapsed": "false", "execution_event_id": "1fbf1fa2-4f18-4635-b102-99c96ab4a946", "last_executed_text": "note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 7, separator='/')).value_counts()", "persistent_id": "c65ed003-e440-457e-8e90-4940d2392a30"}
note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 7, separator='/')).value_counts()

# + {"Collapsed": "false", "execution_event_id": "035d026b-094f-41db-a4da-54bbdef838fe", "last_executed_text": "note_df.notevalue.value_counts()", "persistent_id": "d85d05c9-21b3-4ecf-9e02-be0904f549dc"}
note_df.notevalue.value_counts()

# + [markdown] {"Collapsed": "false"}
# There are always 8 levels of the notes. As the first 6 ones are essentially always the same ("notes/Progress Notes/Social History / Family History/Social History/Social History/"), it's best to just preserve the 7th one and isolate the 8th in a new feature. This way, the split provides further insight to the model on similar notes. However, it's also worth taking note that the 8th level of `notepath` seems to be identical to the feature `notevalue`. We'll look more into it later.

# + {"Collapsed": "false", "execution_event_id": "086df1e1-073d-4907-a8e3-e390fa773047", "last_executed_text": "note_df['notetopic'] = note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 6, separator='/'))\nnote_df['notedetails'] = note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 7, separator='/'))\nnote_df.head()", "persistent_id": "a84d611a-871e-44ab-83ac-bbda639710cf"}
note_df['notetopic'] = note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 6, separator='/'))
note_df['notedetails'] = note_df.notepath.apply(lambda x: du.search_explore.get_element_from_split(x, 7, separator='/'))
note_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the now redundant `notepath` column:

# + {"Collapsed": "false", "execution_event_id": "9587e482-17b8-44c1-a90a-d28e5e6a9fcc", "last_executed_text": "note_df = note_df.drop('notepath', axis=1)\nnote_df.head()", "persistent_id": "3b6b3f09-7a5f-4f3a-bf17-baffb1ac975b"}
note_df = note_df.drop('notepath', axis=1)
note_df.head()

# + [markdown] {"Collapsed": "false"}
# Compare columns `notevalue` and `notedetails`:

# + {"Collapsed": "false", "execution_event_id": "64c5f440-21d1-48c9-acbf-88d2bbd0007f", "last_executed_text": "note_df[note_df.notevalue != note_df.notedetails]", "persistent_id": "5e4b5f52-a9f6-411d-87e5-c4a524942fe1"}
note_df[note_df.notevalue != note_df.notedetails]

# + [markdown] {"Collapsed": "false"}
# The previous blank output confirms that the newly created `notedetails` feature is exactly equal to the already existing `notevalue` feature. So, we should remove one of them:

# + {"Collapsed": "false", "execution_event_id": "f292b84b-167a-41fc-aa27-9edb54153eda", "last_executed_text": "note_df = note_df.drop('notedetails', axis=1)\nnote_df.head()", "persistent_id": "c73dcd18-e561-44ac-b3c5-3aabac45217a"}
note_df = note_df.drop('notedetails', axis=1)
note_df.head()

# + {"Collapsed": "false", "execution_event_id": "cb7580b6-b2a8-4ef9-b784-bf1dc3258025", "last_executed_text": "note_df[note_df.notetopic == 'Smoking Status'].notevalue.value_counts()", "persistent_id": "3c4ccb54-901b-496a-be2d-e722cbf2ccc2"}
note_df[note_df.notetopic == 'Smoking Status'].notevalue.value_counts()

# + {"Collapsed": "false", "execution_event_id": "f4c1b6a1-a80b-4b33-be5f-74aeef2ced7f", "last_executed_text": "note_df[note_df.notetopic == 'Ethanol Use'].notevalue.value_counts()", "persistent_id": "feec565a-91e4-4555-ae56-6d4b658f09be"}
note_df[note_df.notetopic == 'Ethanol Use'].notevalue.value_counts()

# + {"Collapsed": "false", "execution_event_id": "8b686166-7ef1-4c88-922a-0e304e15b71f", "last_executed_text": "note_df[note_df.notetopic == 'CAD'].notevalue.value_counts()", "persistent_id": "e2e20fd6-27e0-4e9b-bbf1-a801b240753f"}
note_df[note_df.notetopic == 'CAD'].notevalue.value_counts()

# + {"Collapsed": "false", "execution_event_id": "2ed1ee51-93e9-42a0-9cc8-840549273a16", "last_executed_text": "note_df[note_df.notetopic == 'Cancer'].notevalue.value_counts()", "persistent_id": "193e722b-65a4-4057-9242-0c66a34f882c"}
note_df[note_df.notetopic == 'Cancer'].notevalue.value_counts()

# + {"Collapsed": "false", "execution_event_id": "38c48b5f-62c0-48d3-88db-f314abb1313d", "last_executed_text": "note_df[note_df.notetopic == 'Recent Travel'].notevalue.value_counts()", "persistent_id": "7e5e40b4-15be-443a-b2f4-995334bd1628"}
note_df[note_df.notetopic == 'Recent Travel'].notevalue.value_counts()

# + {"Collapsed": "false", "execution_event_id": "4245a20a-23bc-4366-86ce-e0ed37ffc4a6", "last_executed_text": "note_df[note_df.notetopic == 'Bleeding Disorders'].notevalue.value_counts()", "persistent_id": "c70bbd0c-b46c-4fd6-9ba3-c68f94c9e71f"}
note_df[note_df.notetopic == 'Bleeding Disorders'].notevalue.value_counts()

# + [markdown] {"Collapsed": "false"}
# Considering how only the categories of "Smoking Status" and "Ethanol Use" in `notetopic` have more than one possible `notevalue` category, with the remaining being only 2 useful ones (categories "Recent Travel" and "Bleeding Disorders" have too little samples), it's probably best to just turn them into features, instead of packing in the same embedded feature.

# + [markdown] {"Collapsed": "false"}
# ### Convert categories to features

# + [markdown] {"Collapsed": "false"}
# Make the `notetopic` and `notevalue` columns of type categorical:

# + {"Collapsed": "false", "execution_event_id": "728243f3-1e45-4a9b-aeee-3b2945bd920c", "last_executed_text": "# Only needed while using Dask, not with Modin or Pandas\n# note_df = note_df.categorize(columns=['notetopic', 'notevalue'])", "persistent_id": "5b887ae0-4d27-4ef0-aaa5-b84e65d27fd5"}
# Only needed while using Dask, not with Modin or Pandas
# note_df = note_df.categorize(columns=['notetopic', 'notevalue'])

# + [markdown] {"Collapsed": "false"}
# Transform the `notetopic` categories and `notevalue` values into separate features:

# + {"Collapsed": "false", "execution_event_id": "531b8a3f-58ec-4acc-b25e-e0b44ab6cedf", "last_executed_text": "note_df = du.data_processing.category_to_feature(note_df, categories_feature='notetopic', \n                                                 values_feature='notevalue', min_len=1000, inplace=True)\nnote_df.head()", "persistent_id": "80f2910a-7d8d-4bac-8afe-fa4fd54c7b84"}
note_df = du.data_processing.category_to_feature(note_df, categories_feature='notetopic',
                                                 values_feature='notevalue', min_len=1000, inplace=True)
note_df.head()

# + [markdown] {"Collapsed": "false"}
# Now we have the categories separated into their own features, as desired. Notice also how categories `Bleeding Disorders` and `Recent Travel` weren't added, as they appeared in less than the specified minimum of 1000 rows.

# + [markdown] {"Collapsed": "false"}
# Remove the old `notevalue` and `notetopic` columns:

# + {"Collapsed": "false", "execution_event_id": "25cd8dc4-9494-49a5-8245-0217b397a9bc", "last_executed_text": "note_df = note_df.drop(['notevalue', 'notetopic'], axis=1)\nnote_df.head()", "persistent_id": "01dbc119-9a63-4561-acff-0835a88048a3"}
note_df = note_df.drop(['notevalue', 'notetopic'], axis=1)
note_df.head()

# + [markdown] {"Collapsed": "false"}
# While `Ethanol Use` and `Smoking Status` have several unique values, `CAD` and `Cancer` only have 1, indicating when that characteristic is present. As such,we should turn `CAD` and `Cancer` into binary features:

# + {"Collapsed": "false", "execution_event_id": "a81c6b64-03d4-47e6-9b52-3f00963c2577", "last_executed_text": "note_df['CAD'] = note_df['CAD'].apply(lambda x: 1 if x == 'CAD' else 0)\nnote_df['Cancer'] = note_df['Cancer'].apply(lambda x: 1 if x == 'Cancer' else 0)\nnote_df.head()", "persistent_id": "c9bb3585-dfab-40cd-b71b-e5fb95d3218d"}
note_df['CAD'] = note_df['CAD'].apply(lambda x: 1 if x == 'CAD' else 0)
note_df['Cancer'] = note_df['Cancer'].apply(lambda x: 1 if x == 'Cancer' else 0)
note_df.head()

# + {"Collapsed": "false", "persistent_id": "ad342024-94c5-4f9e-a9b2-82bfd6353db2"}
note_df['CAD'].value_counts()

# + {"Collapsed": "false", "persistent_id": "67b46f3d-d5bc-4cda-9c2d-4db10304f268"}
note_df['Cancer'].value_counts()

# + {"Collapsed": "false", "persistent_id": "ad342024-94c5-4f9e-a9b2-82bfd6353db2"}
note_df['Smoking Status'].value_counts()

# + {"Collapsed": "false", "persistent_id": "67b46f3d-d5bc-4cda-9c2d-4db10304f268"}
note_df['Ethanol Use'].value_counts()

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": false}
# ### Discretize categorical features
#
# Convert binary categorical features into one hot encode columns, which can later be embedded or used as is.

# + [markdown] {"Collapsed": "false"}
# #### One hot encode features

# + [markdown] {"Collapsed": "false"}
# Update list of categorical features:

# + {"Collapsed": "false", "persistent_id": "0ea70c94-33a9-46b0-b987-dac01e78ec21"}
cat_feat = ['Smoking Status', 'Ethanol Use']

# + {"Collapsed": "false", "persistent_id": "a4a2b9a5-0f9b-442c-9042-ed940501b71e"}
note_df[cat_feat].head()

# + [markdown] {"Collapsed": "false"}
# Convert dataframe to Pandas, as the `one_hot_encoding_dataframe` isn't working properly with Modin:

# + {"Collapsed": "false"}
note_df, pd = du.utils.convert_dataframe(note_df, to='pandas', dtypes=dict(note_df.dtypes))

# + [markdown] {"Collapsed": "false"}
# Apply one hot encoding:

# + {"Collapsed": "false", "persistent_id": "318dc10d-8369-45f1-9deb-5acc83616c04", "pixiedust": {"displayParams": {}}}
note_df, new_columns = du.data_processing.one_hot_encoding_dataframe(note_df, columns=cat_feat,
                                                                     join_rows=False,
                                                                     get_new_column_names=True,
                                                                     inplace=True)
note_df

# + {"Collapsed": "false", "persistent_id": "c7a413ec-d61e-49ba-a7ae-13949fc6f092"}
note_df.dtypes

# + [markdown] {"Collapsed": "false"}
# Save the association between the original categorical features and the new one hot encoded columns:

# + {"Collapsed": "false", "persistent_id": "26eac7f3-9081-4a96-ae4a-40054c223fd7"}
for orig_col in cat_feat:
    cat_feat_ohe[orig_col] = [ohe_col for ohe_col in new_columns
                              if ohe_col.startswith(orig_col)]

# + {"Collapsed": "false"}
cat_feat_ohe

# + [markdown] {"Collapsed": "false"}
# #### Save enumeration encoding mapping
#
# Save the dictionary that maps from the original categories/strings to the new numerical encondings.

# + {"Collapsed": "false", "persistent_id": "4342f002-c60a-4724-a542-9b7f906d3f2b"}
stream = open(f'{data_path}/cleaned/cat_feat_ohe_note.yml', 'w')
yaml.dump(cat_feat_ohe, stream, default_flow_style=False)

# + [markdown] {"Collapsed": "false"}
# ### Create the timestamp feature and sort

# + [markdown] {"Collapsed": "false"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "dfab2799-af6c-4475-8341-ec3f40546ed1"}
note_df = note_df.rename(columns={'noteoffset': 'ts'})
note_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove duplicate rows:

# + {"Collapsed": "false", "persistent_id": "8af4dd26-9eb8-4edf-8bcf-361b10c94979"}
len(note_df)

# + {"Collapsed": "false", "persistent_id": "de85aa8f-0d02-4c35-868c-16116a83cf7f"}
note_df = note_df.drop_duplicates()
note_df.head()

# + {"Collapsed": "false", "execution_event_id": "0f6fb1fb-5d50-4f2c-acd1-804100222250", "last_executed_text": "len(note_df)", "persistent_id": "bb6efd0a-aa95-40d6-84b2-8916705a4cf4"}
len(note_df)

# + [markdown] {"Collapsed": "false"}
# Sort by `ts` so as to be easier to merge with other dataframes later:

# + {"Collapsed": "false", "persistent_id": "a03573d9-f345-4ff4-84b9-2b2a3f73ce27"}
note_df = note_df.sort_values('ts')
note_df.head()

# + [markdown] {"Collapsed": "false"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "c94e7b7b-dc34-478b-842b-c34c926c934d"}
note_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='CAD', n=5).head()

# + {"Collapsed": "false", "persistent_id": "b8bcf17b-3d52-4cc9-bfbb-7f7d8fe83b3b"}
note_df[note_df.patientunitstayid == 3091883].head(10)

# + {"Collapsed": "false", "persistent_id": "a63112fe-a224-4b36-810f-f0d087be43b0"}
note_df[note_df.patientunitstayid == 3052175].head(10)

# + [markdown] {"Collapsed": "false"}
# We can see that there are up to 5 categories per set of `patientunitstayid` and `ts`. As such, we must join them. However, this is a different scenario than in the other cases. Since we created the features from one categorical column, it doesn't have repeated values, only different rows to indicate each of the new features' values. As such, we just need to sum the features.

# + [markdown] {"Collapsed": "false"}
# ### Join rows that have the same IDs

# + {"Collapsed": "false", "execution_event_id": "c6c89c91-ec15-4636-99d0-6ed07bcc921c", "last_executed_text": "note_df = du.embedding.join_repeated_rows(note_df, cont_join_method='max')\nnote_df.head()", "persistent_id": "591b2ccd-fa5c-4eb2-bec1-8ac21de1c890", "pixiedust": {"displayParams": {}}}
note_df = du.embedding.join_repeated_rows(note_df, inplace=True)
note_df.head()

# + {"Collapsed": "false", "execution_event_id": "22163577-ad6a-4eed-8b09-c87d5c740199", "last_executed_text": "note_df.dtypes", "persistent_id": "d3040cd3-4500-4129-ae90-23f3753045f8"}
note_df.dtypes

# + {"Collapsed": "false", "execution_event_id": "612ae128-11bd-46db-a875-83a1701d51f6", "last_executed_text": "note_df.reset_index().groupby(['patientunitstayid', 'ts']).count().nlargest(columns='CAD').head()", "persistent_id": "b4d9884b-d8bb-49f4-8a38-4146f751708e"}
note_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='CAD', n=5).head()

# + {"Collapsed": "false", "persistent_id": "93c6457a-a187-421a-a9f7-6cbf844c1365"}
note_df[note_df.patientunitstayid == 3091883].head(10)

# + {"Collapsed": "false", "persistent_id": "d25a8707-20d0-40d7-ad0b-6efd2306686d"}
note_df[note_df.patientunitstayid == 3052175].head(10)

# + [markdown] {"Collapsed": "false"}
# Comparing the output from the two previous cells with what we had before the `join_repeated_rows` method, we can see that all rows with duplicate IDs have been successfully joined.

# + [markdown] {"Collapsed": "false"}
# ### Clean column names
#
# Standardize all column names to be on lower case, have spaces replaced by underscores and remove comas.

# + {"Collapsed": "false", "persistent_id": "32450572-639e-4539-b35a-181078ed3335"}
note_df.columns = du.data_processing.clean_naming(note_df.columns, lower_case=False)
note_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Save the dataframe

# + [markdown] {"Collapsed": "false"}
# Save the dataframe before normalizing:

# + {"Collapsed": "false", "persistent_id": "e42f577a-db00-4ecf-9e3c-433007a3bdaf"}
# note_df.to_csv(f'{data_path}cleaned/unnormalized/ohe/note.csv')

# + [markdown] {"Collapsed": "false"}
# Save the dataframe after normalizing:

# + {"Collapsed": "false", "persistent_id": "812e7eb1-ff92-4a26-a970-2f40fc5bbdb1"}
note_df.to_csv(f'{data_path}cleaned/normalized/ohe/note.csv')

# + [markdown] {"Collapsed": "false"}
# Confirm that everything is ok through the `describe` method:

# + {"Collapsed": "false", "persistent_id": "eebc356f-507e-4872-be9d-a1d774f2fd7a"}
note_df.describe().transpose()

# + {"Collapsed": "false"}
