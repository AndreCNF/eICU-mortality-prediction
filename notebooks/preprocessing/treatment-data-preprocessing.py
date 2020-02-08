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
stream = open('cat_embed_feat_enum_treat.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "b1ea5e2a-d7eb-41e6-9cad-4dcbf5997ca7"}
infdrug_df = infdrug_df.rename(columns={'infusionoffset': 'ts'})
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
infdrug_df = infdrug_df.sort_values('ts')
infdrug_df.head(6)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "049a3fd1-0ae4-454e-a5b5-5ce8fa94d3e1"}
infdrug_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='drugname', n=5).head()

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
infdrug_df_norm = du.embedding.join_categorical_enum(infdrug_df_norm, new_cat_embed_feat, inplace=True)
infdrug_df_norm.head()

# + {"Collapsed": "false", "persistent_id": "e8c3544d-cf75-45d1-8af2-e6feb8a8d587"}
infdrug_df_norm.dtypes

# + {"Collapsed": "false", "persistent_id": "bbc89593-5b2b-42b5-83f2-8bab9c1d5ef6"}
infdrug_df_norm.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='drugname', n=5).head()

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
stream = open('cat_embed_feat_enum_treat.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "ceab2ec2-9b2b-4439-9c3c-7674dd5b2445"}
admsdrug_df = admsdrug_df.rename(columns={'drugoffset': 'ts'})
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
admsdrug_df = admsdrug_df.sort_values('ts')
admsdrug_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "b656a52b-271c-42fd-b8ff-2c4c01a7d2dc"}
admsdrug_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='drughiclseqno', n=5).head()

# + {"Collapsed": "false", "persistent_id": "84c0cc0a-72eb-4fab-93e3-4c9cb83b4fc4"}
admsdrug_df[admsdrug_df.patientunitstayid == 2346930].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 48 categories per set of `patientunitstayid` and `ts`. As such, we must join them. But first, we need to normalize the dosage by the respective sets of drug code and units, so as to avoid mixing different absolute values.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Normalize data

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "417dd68c-7c54-4eaf-856c-9f6a44ee1d26"}
admsdrug_df_norm = du.data_processing.normalize_data(admsdrug_df_norm, columns_to_normalize=False,
                                                  columns_to_normalize_cat=[(['drughiclseqno', 'drugunit'], 'drugdosage')])
admsdrug_df_norm.head()

# + {"Collapsed": "false", "persistent_id": "d8f2c0a4-ba81-4958-90b7-7b5165ff1363"}
admsdrug_df_norm = admsdrug_df_norm.sort_values('ts')
admsdrug_df_norm.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"Collapsed": "false", "cell_type": "markdown"}
# Even after removing duplicates rows, there are still some that have different information for the same ID and timestamp. We have to concatenate the categorical enumerations.

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "5d87f689-5f4f-4483-841a-533bc5e053c7"}
admsdrug_df_norm = du.embedding.join_categorical_enum(admsdrug_df_norm, new_cat_embed_feat, inplace=True)
admsdrug_df_norm.head()

# + {"Collapsed": "false", "persistent_id": "004138b3-74c4-4ca8-b8b4-3409648e00d0"}
admsdrug_df_norm.dtypes

# + {"Collapsed": "false", "persistent_id": "efd11709-c95e-4245-a662-188556d66680"}
admsdrug_df_norm.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='drughiclseqno', n=5).head()

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
stream = open('cat_embed_feat_enum_treat.yaml', 'w')
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
med_df = med_df.rename(columns={'drugstopoffset': 'ts'})
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
med_df = med_df.sort_values('ts')
med_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "ac2b0e4b-d2bd-4eb5-a629-637361a85457"}
med_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='drughiclseqno', n=5).head()

# + {"Collapsed": "false", "persistent_id": "da5e70d7-0514-4bdb-a5e2-12b6e8a1b197"}
med_df[med_df.patientunitstayid == 979183].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 41 categories per set of `patientunitstayid` and `ts`. As such, we must join them. But first, we need to normalize the dosage by the respective sets of drug code and units, so as to avoid mixing different absolute values.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Normalize data

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "a4cd949b-e561-485d-bcb6-10fccc343352"}
med_df_norm = du.data_processing.normalize_data(med_df_norm, columns_to_normalize=False,
                                             columns_to_normalize_cat=[(['drughiclseqno', 'drugunit'], 'drugdosage')])
med_df_norm.head()

# + {"Collapsed": "false", "persistent_id": "93145ae0-2204-4553-a7eb-5d235552dd82"}
med_df_norm = med_df_norm.sort_values('ts')
med_df_norm.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"Collapsed": "false", "cell_type": "markdown"}
# Even after removing duplicates rows, there are still some that have different information for the same ID and timestamp. We have to concatenate the categorical enumerations.

# + {"Collapsed": "false", "persistent_id": "ed86d5a7-eeb3-44c4-9a4e-6dd67af307f2"}
list(set(med_df_norm.columns) - set(new_cat_embed_feat) - set(['patientunitstayid', 'ts']))

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "45866561-e170-4d9e-9189-e73cb5bcfc0f"}
med_df_norm = du.embedding.join_categorical_enum(med_df_norm, new_cat_embed_feat, inplace=True)
med_df_norm.head()

# + {"Collapsed": "false", "persistent_id": "db6b5624-e600-4d90-bc5a-ffa5a876d8dd"}
med_df_norm.dtypes

# + {"Collapsed": "false", "persistent_id": "954d2c26-4ef4-42ec-b0f4-a73febb5115d"}
med_df_norm.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='drughiclseqno', n=5).head()

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
med_df.nlargest(columns='drugdosage', n=5)

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

# + {"Collapsed": "false", "cell_type": "markdown"}
# <!-- There are always 8 levels of the notes. As the first 6 ones are essentially always the same ("notes/Progress Notes/Social History / Family History/Social History/Social History/"), it's best to just preserve the 7th one and isolate the 8th in a new feature. This way, the split provides further insight to the model on similar notes. However, it's also worth taking note that the 8th level of `notepath` seems to be identical to the feature `notevalue`. We'll look more into it later. -->

# + {"Collapsed": "false", "persistent_id": "44721e19-f088-4cf3-be69-5201d1260d52"}
treat_df['treatmenttype'] = treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 0, separator='|'))
treat_df['treatmenttherapy'] = treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 1, separator='|'))
treat_df['treatmentdetails'] = treat_df.treatmentstring.apply(lambda x: du.search_explore.get_element_from_split(x, 2, separator='|', till_the_end=True))
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
stream = open('cat_embed_feat_enum_treat.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "e9364015-9dc8-45ed-a526-fbfd85a7d249"}
treat_df = treat_df.rename(columns={'treatmentoffset': 'ts'})
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
treat_df = treat_df.sort_values('ts')
treat_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "8672872e-aea3-480c-b5a2-383843db6e3e"}
treat_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='treatmenttype', n=5).head()

# + {"Collapsed": "false", "persistent_id": "ab644c5b-831a-4b57-a95c-fbac810e59f4"}
treat_df[treat_df.patientunitstayid == 1352520].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 105 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "e08bac51-50b2-47d1-aea3-8f72a80223aa"}
treat_df = du.embedding.join_categorical_enum(treat_df, new_cat_embed_feat, inplace=True)
treat_df.head()

# + {"Collapsed": "false", "persistent_id": "e2b70eff-045e-4587-af99-329381e839b2"}
treat_df.dtypes

# + {"Collapsed": "false", "persistent_id": "08de3813-83e3-478c-bfeb-e6dbeafacdb1"}
treat_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='treatmenttype', n=5).head()

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
