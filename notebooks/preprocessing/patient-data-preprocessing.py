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
# ## Patient data

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
stream = open('cat_embed_feat_enum_patient.yaml', 'w')
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
stream = open('cat_embed_feat_enum_patient.yaml', 'w')
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
