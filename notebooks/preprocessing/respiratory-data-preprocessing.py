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
# # Respiratory Data Preprocessing
# ---
#
# Reading and preprocessing respiratory data of the eICU dataset from MIT with the data from over 139k patients collected in the US.
#
# This notebook addresses the preprocessing of the following eICU tables:
# * respiratoryCare
# * respiratoryCharting

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
# ## Respiratory care data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Initialize variables

# + {"Collapsed": "false", "persistent_id": "754a96f8-d389-4968-8c13-52e5e9d0bf82"}
cat_feat = []                              # List of categorical features
cat_embed_feat = []                        # List of categorical features that will be embedded
cat_embed_feat_enum = dict()               # Dictionary of the enumerations of the categorical features that will be embedded

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Read the data

# + {"Collapsed": "false", "persistent_id": "31b57ee7-87a4-4461-9364-7eaf4abc43fb"}
resp_care_df = pd.read_csv(f'{data_path}original/respiratoryCare.csv', dtype={'airwayposition': 'object',
                                                                              'airwaytype': 'object',
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
resp_care_df = resp_care_df.rename(columns={'ventstartoffset': 'ts'})
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
resp_care_df = resp_care_df.sort_values('ts')
resp_care_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "08b71ed2-1822-4f96-9fd8-fd6c88742e99"}
resp_care_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='ventendoffset', n=5).head()

# + {"Collapsed": "false", "persistent_id": "678fefff-a384-4c44-ab09-86119e6d4087"}
resp_care_df[resp_care_df.patientunitstayid == 1113084].head(20)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 5283 duplicate rows per set of `patientunitstayid` and `ts`. As such, we must join them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"Collapsed": "false", "cell_type": "markdown"}
# Even after removing duplicates rows, there are still some that have different information for the same ID and timestamp. We have to apply a groupby function, selecting the minimum value for each of the offset features, as the larger values don't make sense (in the `priorventstartoffset`).

# + {"Collapsed": "false", "persistent_id": "092a9ef1-2caa-4d53-b63a-6c641e5a6b46"}
((resp_care_df.ts > resp_care_df.ventendoffset) & resp_care_df.ventendoffset != 0).value_counts()

# + {"Collapsed": "false", "cell_type": "markdown"}
# There are no errors of having the start vent timestamp later than the end vent timestamp.

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "6e8377a7-4e89-4371-a663-bd10b5dcf5d9"}
resp_care_df = du.embedding.join_categorical_enum(resp_care_df, cont_join_method='min', inplace=True)
resp_care_df.head()

# + {"Collapsed": "false", "persistent_id": "b054019f-50a4-4326-8bc6-bd31966bbeb4"}
resp_care_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='ventendoffset', n=5).head()

# + {"Collapsed": "false", "persistent_id": "741c918e-715b-467a-a775-f012e48b56ba"}
resp_care_df[resp_care_df.patientunitstayid == 1113084].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Comparing the output from the two previous cells with what we had before the `join_categorical_enum` method, we can see that all rows with duplicate IDs have been successfully joined.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Convert dataframe to Pandas, as the next cells aren't working properly with Modin:

# + {"Collapsed": "false"}
resp_care_df, pd = du.utils.convert_dataframe(resp_care_df, to='pandas')

# + {"Collapsed": "false"}
type(resp_care_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Only keep the first instance of each patient, as we're only keeping track of when they are on ventilation:

# + {"Collapsed": "false", "persistent_id": "988fcf02-dc89-4808-be0b-ed8f7c55d44a"}
resp_care_df = resp_care_df.groupby('patientunitstayid').first().sort_values('ts').reset_index()
resp_care_df.head(20)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create prior ventilation label
#
# Make a feature `priorvent` that indicates if the patient has been on ventilation before.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the prior ventilation column:

# + {"Collapsed": "false", "persistent_id": "447b27eb-02bd-42a6-8b0d-e90d350add29"}
resp_care_df['priorvent'] = (resp_care_df.priorventstartoffset < resp_care_df.ts).astype(int)
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
resp_care_df = resp_care_df.sort_values('ts')
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
# Reconvert dataframe to Modin:

# + {"Collapsed": "false"}
resp_care_df, pd = du.utils.convert_dataframe(resp_care_df, to='modin')

# + {"Collapsed": "false"}
type(resp_care_df)

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

# + {"toc-hr-collapsed": true, "Collapsed": "false", "cell_type": "markdown"}
# ## Respiratory charting data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Initialize variables

# + {"Collapsed": "false", "persistent_id": "754a96f8-d389-4968-8c13-52e5e9d0bf82"}
cat_feat = []                              # List of categorical features
cat_embed_feat = []                        # List of categorical features that will be embedded
cat_embed_feat_enum = dict()               # Dictionary of the enumerations of the categorical features that will be embedded

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
    resp_chart_df[feature], cat_embed_feat_enum[feature] = du.embedding.enum_categorical_feature(resp_chart_df, feature, nan_value=0,
                                                                                                 forbidden_digit=0)

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
stream = open(f'{data_path}/cleaned/cat_embed_feat_enum_resp.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Create the timestamp feature and sort

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create the timestamp (`ts`) feature:

# + {"Collapsed": "false", "persistent_id": "540651fd-1fa7-425d-bce3-5027cd7761bf"}
resp_chart_df = resp_chart_df.rename(columns={'nurseassessoffset': 'ts'})
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
resp_chart_df = resp_chart_df.sort_values('ts')
resp_chart_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check for possible multiple rows with the same unit stay ID and timestamp:

# + {"Collapsed": "false", "persistent_id": "acadb9e1-775f-43b3-b680-45aa2b37acd7"}
resp_chart_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='Cough', n=5).head()

# + {"Collapsed": "false", "persistent_id": "62614201-b663-4adf-9080-33e648a0a0ec"}
resp_chart_df[resp_chart_df.patientunitstayid == 2553254].head(10)

# + {"Collapsed": "false", "cell_type": "markdown"}
# We can see that there are up to 80 categories per set of `patientunitstayid` and `ts`. As such, we must join them.

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Join rows that have the same IDs

# + {"Collapsed": "false", "cell_type": "markdown"}
# Convert dataframe to Pandas, as the groupby operation in `join_categorical_enum` isn't working properly with Modin:

# + {"Collapsed": "false"}
resp_chart_df, pd = du.utils.convert_dataframe(resp_chart_df, to='pandas')

# + {"Collapsed": "false"}
type(resp_chart_df)

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false", "persistent_id": "589931b8-fe11-439a-8b14-4857c168c023"}
resp_chart_df = du.embedding.join_categorical_enum(resp_chart_df, new_cat_embed_feat, inplace=True)
resp_chart_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Reconvert dataframe to Modin:

# + {"Collapsed": "false"}
resp_chart_df, pd = du.utils.convert_dataframe(resp_chart_df, to='modin')

# + {"Collapsed": "false"}
type(resp_chart_df)

# + {"Collapsed": "false", "persistent_id": "6ece20d0-6de2-4989-9394-e020fc8916ed"}
resp_chart_df.dtypes

# + {"Collapsed": "false", "persistent_id": "09692bf1-874c-49c5-a525-31f91a28c019"}
resp_chart_df.groupby(['patientunitstayid', 'ts']).count().nlargest(columns='Cough', n=5).head()

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
