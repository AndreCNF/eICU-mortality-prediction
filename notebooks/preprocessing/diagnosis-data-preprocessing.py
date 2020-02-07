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
