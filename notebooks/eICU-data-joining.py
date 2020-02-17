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
# Reading and joining all preprocessed parts of the eICU dataset from MIT with the data from over 139k patients collected in the US.
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
data_path = 'Datasets/Thesis/eICU/uncompressed/cleaned/'
# Path to the code files
project_path = 'GitHub/eICU-mortality-prediction/'

# + {"Collapsed": "false", "persistent_id": "c0c2e356-d4f4-4a9d-bec2-88bdf9eb6a38", "last_executed_text": "import modin.pandas as pd                  # Optimized distributed version of Pandas\nimport data_utils as du                    # Data science and machine learning relevant methods", "execution_event_id": "82ef68be-443a-4bb8-8abd-7457a7005b4d"}
# import ray                                 # Parallelization backend used by Modin
# import modin.pandas as pd                  # Optimized distributed version of Pandas
import pandas as pd
import data_utils as du                    # Data science and machine learning relevant methods

# + {"Collapsed": "false"}
# Setting ray's object size limit to 5GB
# ray.shutdown()
# ray.init(memory=8000 * 1024 * 1024,
#          object_store_memory=5000 * 1024 * 1024)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Set the random seed for reproducibility

# + {"Collapsed": "false", "persistent_id": "39b552cd-6948-4ec8-ac04-42f850c1e05a", "last_executed_text": "du.set_random_seed(42)", "execution_event_id": "29ab85ce-b7fd-4c5a-a110-5841e741c369"}
du.set_random_seed(42)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ## Loading the data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Patient information

# + {"Collapsed": "false"}
patient_df = pd.read_csv(f'{data_path}normalized/patient.csv')
patient_df.head()

# + {"Collapsed": "false"}
du.search_explore.dataframe_missing_values(patient_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove rows that don't identify the patient:

# + {"Collapsed": "false"}
patient_df = patient_df[~patient_df.patientunitstayid.isnull()]

# + {"Collapsed": "false"}
du.search_explore.dataframe_missing_values(patient_df)

# + {"Collapsed": "false"}
patient_df.patientunitstayid = patient_df.patientunitstayid.astype(int)
patient_df.ts = patient_df.ts.astype(int)
patient_df.dtypes

# + {"Collapsed": "false"}
note_df = pd.read_csv(f'{data_path}normalized/note.csv')
note_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false"}
patient_df = patient_df.drop(columns='Unnamed: 0')
# note_df = note_df.drop(columns='Unnamed: 0')

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Diagnosis

# + {"Collapsed": "false"}
diagns_df = pd.read_csv(f'{data_path}normalized/diagnosis.csv')
diagns_df.head()

# + {"Collapsed": "false"}
alrg_df = pd.read_csv(f'{data_path}normalized/allergy.csv')
alrg_df.head()

# + {"Collapsed": "false"}
past_hist_df = pd.read_csv(f'{data_path}normalized/pastHistory.csv')
past_hist_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false"}
diagns_df = diagns_df.drop(columns='Unnamed: 0')
alrg_df = alrg_df.drop(columns='Unnamed: 0')
past_hist_df = past_hist_df.drop(columns='Unnamed: 0')

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Treatments

# + {"Collapsed": "false"}
treat_df = pd.read_csv(f'{data_path}normalized/treatment.csv')
treat_df.head()

# + {"Collapsed": "false"}
adms_drug_df = pd.read_csv(f'{data_path}normalized/admissionDrug.csv')
adms_drug_df.head()

# + {"Collapsed": "false"}
inf_drug_df = pd.read_csv(f'{data_path}normalized/infusionDrug.csv')
inf_drug_df.head()

# + {"Collapsed": "false"}
med_df = pd.read_csv(f'{data_path}normalized/medication.csv')
med_df.head()

# + {"Collapsed": "false"}
# in_out_df = pd.read_csv(f'{data_path}normalized/intakeOutake.csv')
# in_out_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false"}
treat_df = treat_df.drop(columns='Unnamed: 0')
adms_drug_df = adms_drug_df.drop(columns='Unnamed: 0')
inf_drug_df = inf_drug_df.drop(columns='Unnamed: 0')
med_df = med_df.drop(columns='Unnamed: 0')
# in_out_df = in_out_df.drop(columns='Unnamed: 0')

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Nursing data

# + {"Collapsed": "false"}
nurse_care_df = pd.read_csv(f'{data_path}normalized/nurseCare.csv')
nurse_care_df.head()

# + {"Collapsed": "false"}
nurse_assess_df = pd.read_csv(f'{data_path}normalized/nurseAssessment.csv')
nurse_assess_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false"}
nurse_care_df = nurse_care_df.drop(columns='Unnamed: 0')
nurse_assess_df = nurse_assess_df.drop(columns='Unnamed: 0')

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Respiratory data

# + {"Collapsed": "false"}
resp_care_df = pd.read_csv(f'{data_path}normalized/respiratoryCare.csv')
resp_care_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false"}
resp_care_df = resp_care_df.drop(columns='Unnamed: 0')

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Vital signals

# + {"Collapsed": "false"}
# vital_aprdc_df = pd.read_csv(f'{data_path}normalized/vitalAperiodic.csv')
# vital_aprdc_df.head()

# + {"Collapsed": "false"}
# vital_prdc_df = pd.read_csv(f'{data_path}normalized/vitalPeriodic.csv')
# vital_prdc_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false"}
# patient_df = patient_df.drop(columns='Unnamed: 0')
# note_df = note_df.drop(columns='Unnamed: 0')

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Exams data

# + {"Collapsed": "false"}
# lab_df = pd.read_csv(f'{data_path}normalized/lab.csv')
# lab_df.head()

# + {"Collapsed": "false"}
# phys_exam_df = pd.read_csv(f'{data_path}normalized/physicalExam.csv')
# phys_exam_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false"}
# patient_df = patient_df.drop(columns='Unnamed: 0')
# note_df = note_df.drop(columns='Unnamed: 0')

# + {"Collapsed": "false", "cell_type": "markdown"}
# ## Joining dataframes

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Checking the matching of unit stays IDs

# + {"Collapsed": "false"}
full_stays_list = set(patient_df.patientunitstayid.unique())

# + {"Collapsed": "false", "cell_type": "markdown"}
# Total number of unit stays:

# + {"Collapsed": "false"}
len(full_stays_list)

# + {"Collapsed": "false"}
note_stays_list = set(note_df.patientunitstayid.unique())

# + {"Collapsed": "false"}
len(note_stays_list)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Number of unit stays that have note data:

# + {"Collapsed": "false"}
len(set.intersection(full_stays_list, note_stays_list))

# + {"Collapsed": "false"}
diagns_stays_list = set(diagns_df.patientunitstayid.unique())

# + {"Collapsed": "false"}
len(diagns_stays_list)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Number of unit stays that have diagnosis data:

# + {"Collapsed": "false"}
len(set.intersection(full_stays_list, diagns_stays_list))

# + {"Collapsed": "false"}
alrg_stays_list = set(alrg_df.patientunitstayid.unique())

# + {"Collapsed": "false"}
len(alrg_stays_list)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Number of unit stays that have allergy data:

# + {"Collapsed": "false"}
len(set.intersection(full_stays_list, alrg_stays_list))

# + {"Collapsed": "false"}
past_hist_stays_list = set(past_hist_df.patientunitstayid.unique())

# + {"Collapsed": "false"}
len(past_hist_stays_list)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Number of unit stays that have past history data:

# + {"Collapsed": "false"}
len(set.intersection(full_stays_list, past_hist_stays_list))

# + {"Collapsed": "false"}
treat_stays_list = set(treat_df.patientunitstayid.unique())

# + {"Collapsed": "false"}
len(treat_stays_list)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Number of unit stays that have treatment data:

# + {"Collapsed": "false"}
len(set.intersection(full_stays_list, treat_stays_list))

# + {"Collapsed": "false"}
adms_drug_stays_list = set(adms_drug_df.patientunitstayid.unique())

# + {"Collapsed": "false"}
len(adms_drug_stays_list)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Number of unit stays that have admission drug data:

# + {"Collapsed": "false"}
len(set.intersection(full_stays_list, adms_drug_stays_list))

# + {"Collapsed": "false"}
inf_drug_stays_list = set(inf_drug_df.patientunitstayid.unique())

# + {"Collapsed": "false"}
len(inf_drug_stays_list)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Number of unit stays that have infusion drug data:

# + {"Collapsed": "false"}
len(set.intersection(full_stays_list, inf_drug_stays_list))

# + {"Collapsed": "false"}
med_stays_list = set(med_df.patientunitstayid.unique())

# + {"Collapsed": "false"}
len(med_stays_list)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Number of unit stays that have medication data:

# + {"Collapsed": "false"}
len(set.intersection(full_stays_list, med_stays_list))

# + {"Collapsed": "false"}
nurse_care_stays_list = set(nurse_care_df.patientunitstayid.unique())

# + {"Collapsed": "false"}
len(nurse_care_stays_list)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Number of unit stays that have nurse care data:

# + {"Collapsed": "false"}
len(set.intersection(full_stays_list, nurse_care_stays_list))

# + {"Collapsed": "false"}
nurse_assess_stays_list = set(nurse_assess_df.patientunitstayid.unique())

# + {"Collapsed": "false"}
len(nurse_assess_stays_list)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Number of unit stays that have nurse assessment data:

# + {"Collapsed": "false"}
len(set.intersection(full_stays_list, nurse_assess_stays_list))

# + {"Collapsed": "false"}
resp_care_stays_list = set(resp_care_df.patientunitstayid.unique())

# + {"Collapsed": "false"}
len(resp_care_stays_list)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Number of unit stays that have nurse care data:

# + {"Collapsed": "false"}
len(set.intersection(full_stays_list, resp_care_stays_list))

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Joining patient with note data

# + {"Collapsed": "false"}
eICU_df = pd.merge(patient_df, note_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Joining with diagnosis data

# + {"Collapsed": "false"}
eICU_df = pd.merge(eICU_df, diagns_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Joining with allergy data

# + {"Collapsed": "false"}
eICU_df = pd.merge(eICU_df, alrg_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Joining with past history data

# + {"Collapsed": "false"}
len(eICU_df)

# + {"Collapsed": "false"}
eICU_df = pd.merge(eICU_df, past_hist_df, how='outer', on='patientunitstayid')
eICU_df.head()

# + {"Collapsed": "false"}
len(eICU_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Joining with treatment data

# + {"Collapsed": "false"}
eICU_df = pd.merge(eICU_df, treat_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Joining with admission drug data

# + {"Collapsed": "false"}
eICU_df = pd.merge(eICU_df, adms_drug_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Joining with medication data

# + {"Collapsed": "false"}
eICU_df = pd.merge(eICU_df, med_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Joining with intake outake data

# + {"Collapsed": "false"}
# eICU_df = pd.merge(eICU_df, in_out_df, how='outer', on=['patientunitstayid', 'ts'])
# eICU_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Joining with nurse care data

# + {"Collapsed": "false"}
eICU_df = pd.merge(eICU_df, nurse_care_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Joining with nurse assessment data

# + {"Collapsed": "false"}
eICU_df = pd.merge(eICU_df, nurse_assess_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Joining with nurse charting data

# + {"Collapsed": "false"}
# eICU_df = pd.merge(eICU_df, nurse_chart_df, how='outer', on=['patientunitstayid', 'ts'])
# eICU_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Joining with respiratory care data

# + {"Collapsed": "false"}
eICU_df = pd.merge(eICU_df, resp_care_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Joining with aperiodic vital signals data

# + {"Collapsed": "false"}
# eICU_df = pd.merge(eICU_df, vital_aprdc_df, how='outer', on=['patientunitstayid', 'ts'])
# eICU_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Joining with periodic vital signals data

# + {"Collapsed": "false"}
# eICU_df = pd.merge(eICU_df, vital_prdc_df, how='outer', on=['patientunitstayid', 'ts'])
# eICU_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Joining with lab data

# + {"Collapsed": "false"}
# eICU_df = pd.merge(eICU_df, lab_df, how='outer', on=['patientunitstayid', 'ts'])
# eICU_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Joining with physical exam data

# + {"Collapsed": "false"}
# eICU_df = pd.merge(eICU_df, phys_exam_df, how='outer', on=['patientunitstayid', 'ts'])
# eICU_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ## Cleaning the joined data

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Removing unit stays that are too short

# + {"Collapsed": "false"}
eICU_df.info()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Make sure that the dataframe is ordered by time `ts`:

# + {"Collapsed": "false"}
eICU_df = eICU_df.sort_values('ts')
eICU_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove unit stays that have less than 10 records:

# + {"Collapsed": "false"}
unit_stay_len = eICU_df.groupby('patientunitstayid').patientunitstayid.count()
unit_stay_len

# + {"Collapsed": "false"}
unit_stay_short = set(unit_stay_len[unit_stay_len < 10].index)
unit_stay_short

# + {"Collapsed": "false"}
len(unit_stay_short)

# + {"Collapsed": "false"}
eICU_df.patientunitstayid.nunique()

# + {"Collapsed": "false"}
eICU_df = eICU_df[~eICU_df.patientunitstayid.isin(unit_stay_short)]

# + {"Collapsed": "false"}
eICU_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove unit stays that have data that represent less than 48h:

# + {"Collapsed": "false"}
unit_stay_duration = eICU_df.groupby('patientunitstayid').ts.apply(lambda x: x.max() - x.min())
unit_stay_duration

# + {"Collapsed": "false"}
unit_stay_short = set(unit_stay_duration[unit_stay_duration < 48*60].index)
unit_stay_short

# + {"Collapsed": "false"}
len(unit_stay_short)

# + {"Collapsed": "false"}
eICU_df.patientunitstayid.nunique()

# + {"Collapsed": "false"}
eICU_df = eICU_df[~eICU_df.patientunitstayid.isin(unit_stay_short)]

# + {"Collapsed": "false"}
eICU_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Joining duplicate columns

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Continuous features

# + {"Collapsed": "false"}
set([col.split('_x')[0].split('_y')[0] for col in eICU_df.columns if col.endswith('_x') or col.endswith('_y')])

# + {"Collapsed": "false"}
eICU_df[['drugdosage_x', 'drughiclseqno_x',
         'drugdosage_y', 'drughiclseqno_y']].head(20)

# + {"Collapsed": "false"}
eICU_df[eICU_df.index == 2564878][['drugdosage_x', 'drughiclseqno_x',
                                   'drugdosage_y', 'drughiclseqno_y']]

# + {"Collapsed": "false", "pixiedust": {"displayParams": {}}}
eICU_df = du.data_processing.merge_columns(eICU_df, cols_to_merge=['drugdosage', 'drughiclseqno'])
eICU_df.sample(20)

# + {"Collapsed": "false"}
eICU_df[['drugdosage', 'drughiclseqno']].head(20)

# + {"Collapsed": "false"}
eICU_df[eICU_df.index == 2564878][['drugdosage', 'drughiclseqno']]

# + {"Collapsed": "false", "cell_type": "markdown"}
# #### Categorical features
#
# Join encodings of the same features, from different tables.

# + {"Collapsed": "false", "cell_type": "markdown"}
# Load encoding dictionaries:

# + {"Collapsed": "false"}
stream_adms_drug = open(f'{data_path}cat_embed_feat_enum_adms_drug.yaml', 'r')
stream_med = open(f'{data_path}cat_embed_feat_enum_med.yaml', 'r')
cat_embed_feat_enum_adms_drug = yaml.load(stream_adms_drug, Loader=yaml.FullLoader)
cat_embed_feat_enum_med = yaml.load(stream_med, Loader=yaml.FullLoader)

# + {"Collapsed": "false"}
eICU_df[['drugadmitfrequency_x', 'drugunit_x',
         'drugadmitfrequency_y', 'drugunit_y']].head(20)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Standardize the encoding of similar columns:

# + {"Collapsed": "false"}
list(cat_embed_feat_enum_adms_drug.keys())

# + {"Collapsed": "false"}
list(cat_embed_feat_enum_med.keys())

# + {"Collapsed": "false"}
eICU_df.to_csv(f'{data_path}normalized/eICU_post_merge_continuous_cols.csv')

# + {"Collapsed": "false"}
eICU_df = pd.read_csv(f'{data_path}normalized/eICU_post_merge_continuous_cols.csv')
eICU_df.head()

# + {"Collapsed": "false", "pixiedust": {"displayParams": {}}}
eICU_df, cat_embed_feat_enum['drugadmitfrequency'] = du.embedding.converge_enum(eICU_df, cat_feat_name=['drugadmitfrequency_x', 
                                                                                                        'drugadmitfrequency_y'], 
                                                                                dict1=cat_embed_feat_enum_adms_drug['drugadmitfrequency'],
                                                                                dict2=cat_embed_feat_enum_med['frequency'],
                                                                                nan_value=0, sort=True, inplace=True)

# + {"Collapsed": "false"}
eICU_df, cat_embed_feat_enum['drugunit'] = du.embedding.converge_enum(eICU_df, cat_feat_name=['drugunit_x', 
                                                                                              'drugunit_y'],
                                                                      dict1=cat_embed_feat_enum_adms_drug['drugunit'],
                                                                      dict2=cat_embed_feat_enum_med['drugunit'],
                                                                      nan_value=0, sort=True, inplace=True)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Merge the features:

# + {"Collapsed": "false"}
eICU_df = du.data_processing.merge_columns(eICU_df, cols_to_merge=['drugadmitfrequency', 'drugunit'])
eICU_df.sample(20)

# + {"Collapsed": "false"}
eICU_df[['drugadmitfrequency', 'drugunit']].head(20)

# + {"Collapsed": "false"}
eICU_df.to_csv(f'{data_path}normalized/eICU_post_merge_categorical_cols.csv')

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Creating a single encoding dictionary for the complete dataframe
#
# Combine the encoding dictionaries of all tables, having in account the converged ones, into a single dictionary representative of all the categorical features in the resulting dataframe.

# + {"Collapsed": "false"}
# [TODO] Add dictionaries if their keys aren't already in the overall dictionary `cat_embed_feat_enum`

# + {"Collapsed": "false", "cell_type": "markdown"}
# Save the final encoding dictionary:

# + {"Collapsed": "false"}
stream = open(f'{data_path}/cleaned/cat_embed_feat_enum_eICU.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Removing columns with too many missing values
#
# We should remove features that have too many missing values (in this case, those that have more than 40% of missing values). Without enough data, it's even risky to do imputation, as it's unlikely for the imputation to correctly model the missing feature.

# + {"Collapsed": "false"}
du.search_explore.dataframe_missing_values(eICU_df)

# + {"Collapsed": "false"}
prev_features = eICU_df.columns
len(prev_features)

# + {"Collapsed": "false"}
eICU_df = du.data_processing.remove_cols_with_many_nans(eICU_df, nan_percent_thrsh=70, inplace=True)

# + {"Collapsed": "false"}
features = eICU_df.columns
len(features)

# + {"Collapsed": "false", "cell_type": "markdown"}
# Removed features:

# + {"Collapsed": "false"}
set(prev_features) - set(features)

# + {"Collapsed": "false"}
eICU_df.head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Removing rows with too many missing values
#
# This actually might not make sense to do, as some tables, such as `patient`, are set in a timestamp that is unlikely to have matches in other tables, although it's still useful to add.

# + {"Collapsed": "false"}
# len(eICU_df)

# + {"Collapsed": "false"}
# n_features = len(eICU_df.columns)
# n_features

# + {"Collapsed": "false"}
# eICU_df = eICU_df[eICU_df.isnull().sum(axis=1) < 0.5 * n_features]

# + {"Collapsed": "false"}
# len(eICU_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Removing unit stays with too many missing values
#
# Consider removing all unit stays that have, combining rows and columns, a very high percentage of missing values.

# + {"Collapsed": "false"}
n_features = len(eICU_df.columns)
n_features

# + {"Collapsed": "false", "cell_type": "markdown"}
# Create a temporary column that counts each row's number of missing values:

# + {"Collapsed": "false"}
eICU_df['row_msng_val'] = eICU_df.isnull().sum(axis=1)
eICU_df[['patientunitstay', 'ts', 'row_msng_val']].head()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Check each unit stay's percentage of missing data points:

# + {"Collapsed": "false"}
# Number of possible data points in each unit stay
n_data_points = eICU_df.groupby('patientunitstayid').ts.count() * n_features
n_data_points

# + {"Collapsed": "false"}
# Number of missing values in each unit stay
n_msng_val = eICU_df.groupby('patientunitstayid').row_msng_val.sum()
n_msng_val

# + {"Collapsed": "false"}
# Percentage of missing values in each unit stay
msng_val_prct = (n_msng_val / n_data_points) * 100
msng_val_prct

# + {"Collapsed": "false"}
msng_val_prct.describe()

# + {"Collapsed": "false", "cell_type": "markdown"}
# Remove unit stays that have too many missing values (>70% of their respective data points):

# + {"Collapsed": "false"}
unit_stay_high_msgn = set(msng_val_prct[msng_val_prct > 70].index)
unit_stay_high_msgn

# + {"Collapsed": "false"}
eICU_df.patientunitstayid.nunique()

# + {"Collapsed": "false"}
eICU_df = eICU_df[~eICU_df.patientunitstayid.isin(unit_stay_high_msgn)]

# + {"Collapsed": "false"}
eICU_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "cell_type": "markdown"}
# ### Performing imputation

# + {"Collapsed": "false"}
du.search_explore.dataframe_missing_values(eICU_df)

# + {"Collapsed": "false"}
# [TODO] Be careful to avoid interpolating categorical features (e.g. `drugunit`); these must only 
# be imputated through zero filling
eICU_df = du.data_processing.missing_values_imputation(eICU_df, method='interpolation', 
                                                       id_column='patientunitstay', inplace=True)
eICU_df.head()

# + {"Collapsed": "false"}
du.search_explore.dataframe_missing_values(eICU_df)

# + {"Collapsed": "false", "cell_type": "markdown"}
# ## Setting the label
#
# Define the label column considering the desired time window on which we want to predict mortality (0, 24h, 48h, 72h, etc).

# + {"Collapsed": "false"}
time_window_h = 24

# + {"Collapsed": "false"}
eICU_df['label'] = eICU_df[eICU_df.death_ts - eICU_df.ts <= time_window_h * 60]
eICU_df.head()
