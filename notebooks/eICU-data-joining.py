# # eICU Data Joining
# ---
#
# Reading and joining all preprocessed parts of the eICU dataset from MIT with the data from over 139k patients collected in the US.
#
# The main goal of this notebook is to prepare a single parquet document that contains all the relevant data to be used when training a machine learning model that predicts mortality, joining tables, filtering useless columns and performing imputation.

# ## Importing the necessary packages

import os                                  # os handles directory/workspace changes
import numpy as np                         # NumPy to handle numeric and NaN operations
import yaml                                # Save and load YAML files

# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# Path to the parquet dataset files
data_path = 'data/eICU/cleaned/'
# Path to the code files
project_path = 'code/eICU-mortality-prediction/'

# Make sure that every large operation can be handled, by using the disk as an overflow for the memory
# !export MODIN_OUT_OF_CORE=true
# Another trick to do with Pandas so as to be able to allocate bigger objects to memory
# !sudo bash -c 'echo 1 > /proc/sys/vm/overcommit_memory'

import modin.pandas as mpd                  # Optimized distributed version of Pandas
import pandas as pd
import data_utils as du                    # Data science and machine learning relevant methods

du.set_pandas_library('pandas')

# Allow pandas to show more columns:

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

# Set the random seed for reproducibility

du.set_random_seed(42)

# ## Initializing variables

dtype_dict = {'patientunitstayid': 'uint32',
              'gender': 'UInt8',
              'age': 'float32',
              'admissionheight': 'float32',
              'admissionweight': 'float32',
              'death_ts': 'Int32',
              'ts': 'int32',
              'cad': 'UInt8',
              'cancer': 'UInt8',
              # 'diagnosis_type_1': 'UInt64',
              # 'diagnosis_disorder_2': 'UInt64',
              # 'diagnosis_detailed_3': 'UInt64',
              # 'allergyname': 'UInt64',
              # 'drugallergyhiclseqno': 'UInt64',
              # 'pasthistoryvalue': 'UInt64',
              # 'pasthistorytype': 'UInt64',
              # 'pasthistorydetails': 'UInt64',
              # 'treatmenttype': 'UInt64',
              # 'treatmenttherapy': 'UInt64',
              # 'treatmentdetails': 'UInt64',
              # 'drugunit_x': 'UInt64',
              # 'drugadmitfrequency_x': 'UInt64',
              # 'drughiclseqno_x': 'UInt64',
              'drugdosage_x': 'float32',
              # 'drugadmitfrequency_y': 'UInt64',
              # 'drughiclseqno_y': 'UInt64',
              'drugdosage_y': 'float32',
              # 'drugunit_y': 'UInt64',
              'bodyweight_(kg)': 'float32',
              'oral_intake': 'float32',
              'urine_output': 'float32',
              'i.v._intake	': 'float32',
              'saline_flush_(ml)_intake': 'float32',
              'volume_given_ml': 'float32',
              'stool_output': 'float32',
              'prbcs_intake': 'float32',
              'gastric_(ng)_output': 'float32',
              'dialysis_output': 'float32',
              'propofol_intake': 'float32',
              'lr_intake': 'float32',
              'indwellingcatheter_output': 'float32',
              'feeding_tube_flush_ml': 'float32',
              'patient_fluid_removal': 'float32',
              'fentanyl_intake': 'float32',
              'norepinephrine_intake': 'float32',
              'crystalloids_intake': 'float32',
              'voided_amount': 'float32',
              'nutrition_total_intake': 'float32',
              # 'nutrition': 'UInt64',
              # 'nurse_treatments': 'UInt64',
              # 'hygiene/adls': 'UInt64',
              # 'activity': 'UInt64',
              # 'pupils': 'UInt64',
              # 'neurologic': 'UInt64',
              # 'secretions': 'UInt64',
              # 'cough': 'UInt64',
              'priorvent': 'UInt8',
              'onvent': 'UInt8',
              'noninvasivesystolic': 'float32',
              'noninvasivediastolic': 'float32',
              'noninvasivemean': 'float32',
              'paop': 'float32',
              'cardiacoutput': 'float32',
              'cardiacinput': 'float32',
              'svr': 'float32',
              'svri': 'float32',
              'pvr': 'float32',
              'pvri': 'float32',
              'temperature': 'float32',
              'sao2': 'float32',
              'heartrate': 'float32',
              'respiration': 'float32',
              'cvp': 'float32',
              'etco2': 'float32',
              'systemicsystolic': 'float32',
              'systemicdiastolic': 'float32',
              'systemicmean': 'float32',
              'pasystolic': 'float32',
              'padiastolic': 'float32',
              'pamean': 'float32',
              'st1': 'float32',
              'st2': 'float32',
              'st3': 'float32',
              'icp': 'float32',
              # 'labtypeid': 'UInt64',
              # 'labname': 'UInt64',
              # 'lab_units': 'UInt64',
              'lab_result': 'float32'}

# Load the lists of one hot encoded columns:

# stream_adms_drug = open(f'{data_path}cat_feat_ohe_adms_drug.yaml', 'r')
# stream_inf_drug = open(f'{data_path}cat_feat_ohe_inf_drug.yaml', 'r')
stream_med = open(f'{data_path}cat_feat_ohe_med.yml', 'r')
stream_treat = open(f'{data_path}cat_feat_ohe_treat.yml', 'r')
stream_diag = open(f'{data_path}cat_feat_ohe_diag.yml', 'r')
# stream_alrg = open(f'{data_path}cat_feat_ohe_alrg.yml', 'r')
stream_past_hist = open(f'{data_path}cat_feat_ohe_past_hist.yml', 'r')
stream_lab = open(f'{data_path}cat_feat_ohe_lab.yml', 'r')
stream_patient = open(f'{data_path}cat_feat_ohe_patient.yml', 'r')
stream_notes = open(f'{data_path}cat_feat_ohe_note.yml', 'r')

# cat_feat_ohe_adms_drug = yaml.load(stream_adms_drug, Loader=yaml.FullLoader)
# cat_feat_ohe_inf_drug = yaml.load(stream_inf_drug, Loader=yaml.FullLoader)
cat_feat_ohe_med = yaml.load(stream_med, Loader=yaml.FullLoader)
cat_feat_ohe_treat = yaml.load(stream_treat, Loader=yaml.FullLoader)
cat_feat_ohe_diag = yaml.load(stream_diag, Loader=yaml.FullLoader)
# cat_feat_ohe_alrg = yaml.load(stream_alrg, Loader=yaml.FullLoader)
cat_feat_ohe_past_hist = yaml.load(stream_past_hist, Loader=yaml.FullLoader)
cat_feat_ohe_lab = yaml.load(stream_lab, Loader=yaml.FullLoader)
cat_feat_ohe_patient = yaml.load(stream_patient, Loader=yaml.FullLoader)
cat_feat_ohe_notes = yaml.load(stream_notes, Loader=yaml.FullLoader)

cat_feat_ohe = du.utils.merge_dicts([cat_feat_ohe_med, cat_feat_ohe_treat,
                                     cat_feat_ohe_diag,
                                     cat_feat_ohe_past_hist, cat_feat_ohe_lab,
                                     cat_feat_ohe_patient, cat_feat_ohe_notes])

ohe_columns = du.utils.merge_lists(list(cat_feat_ohe.values()))
ohe_columns = du.data_processing.clean_naming(ohe_columns, lower_case=False)

# Add the one hot encoded columns to the dtypes dictionary, specifying them with type `UInt8`

for col in ohe_columns:
    dtype_dict[col] = 'UInt8'

dtype_dict

# ## Loading the data

# ### Patient information

patient_df = mpd.read_csv(f'{data_path}normalized/ohe/patient.csv', dtype=dtype_dict)
patient_df = du.utils.convert_dataframe(patient_df, to='pandas', return_library=False, dtypes=dtype_dict)
patient_df = patient_df.drop(columns='Unnamed: 0')
patient_df.head()

note_df = mpd.read_csv(f'{data_path}normalized/ohe/note.csv', dtype=dtype_dict)
note_df = du.utils.convert_dataframe(note_df, to='pandas', return_library=False, dtypes=dtype_dict)
note_df = note_df.drop(columns='Unnamed: 0')
note_df.head()

# ### Diagnosis

diagns_df = mpd.read_csv(f'{data_path}normalized/ohe/diagnosis.csv', dtype=dtype_dict)
diagns_df = du.utils.convert_dataframe(diagns_df, to='pandas', return_library=False, dtypes=dtype_dict)
diagns_df = diagns_df.drop(columns='Unnamed: 0')
diagns_df.head()

# +
# alrg_df = mpd.read_csv(f'{data_path}normalized/ohe/allergy.csv', dtype=dtype_dict)
# alrg_df = du.utils.convert_dataframe(alrg_df, to='pandas', return_library=False, dtypes=dtype_dict)
# alrg_df = alrg_df.drop(columns='Unnamed: 0')
# alrg_df.head()
# -

past_hist_df = mpd.read_csv(f'{data_path}normalized/ohe/pastHistory.csv', dtype=dtype_dict)
past_hist_df = du.utils.convert_dataframe(past_hist_df, to='pandas', return_library=False, dtypes=dtype_dict)
past_hist_df = past_hist_df.drop(columns='Unnamed: 0')
past_hist_df.head()

# ### Treatments

treat_df = mpd.read_csv(f'{data_path}normalized/ohe/treatment.csv', dtype=dtype_dict)
treat_df = du.utils.convert_dataframe(treat_df, to='pandas', return_library=False, dtypes=dtype_dict)
treat_df = treat_df.drop(columns='Unnamed: 0')
treat_df.head()

adms_drug_df = mpd.read_csv(f'{data_path}normalized/ohe/admissionDrug.csv', dtype=dtype_dict)
adms_drug_df = du.utils.convert_dataframe(adms_drug_df, to='pandas', return_library=False, dtypes=dtype_dict)
adms_drug_df = adms_drug_df.drop(columns='Unnamed: 0')
adms_drug_df.head()

# +
# inf_drug_df = mpd.read_csv(f'{data_path}normalized/ohe/infusionDrug.csv', dtype=dtype_dict)
# inf_drug_df = du.utils.convert_dataframe(inf_drug_df, to='pandas', return_library=False, dtypes=dtype_dict)
# inf_drug_df = inf_drug_df.drop(columns='Unnamed: 0')
# inf_drug_df.head()
# -

med_df = mpd.read_csv(f'{data_path}normalized/ohe/medication.csv', dtype=dtype_dict)
med_df = du.utils.convert_dataframe(med_df, to='pandas', return_library=False, dtypes=dtype_dict)
med_df = med_df.drop(columns='Unnamed: 0')
med_df.head()

# +
# in_out_df = mpd.read_csv(f'{data_path}normalized/intakeOutput.csv', dtype=dtype_dict)
# in_out_df = du.utils.convert_dataframe(in_out_df, to='pandas', return_library=False, dtypes=dtype_dict)
# in_out_df = in_out_df.drop(columns='Unnamed: 0')
# in_out_df.head()
# -

# ### Nursing data

# +
# nurse_care_df = mpd.read_csv(f'{data_path}normalized/ohe/nurseCare.csv')
# nurse_care_df.head()

# +
# nurse_assess_df = mpd.read_csv(f'{data_path}normalized/ohe/nurseAssessment.csv')
# nurse_assess_df.head()
# -

# Remove the uneeded 'Unnamed: 0' column:

# +
# nurse_care_df = nurse_care_df.drop(columns='Unnamed: 0')
# nurse_assess_df = nurse_assess_df.drop(columns='Unnamed: 0')
# -

# ### Respiratory data

resp_care_df = mpd.read_csv(f'{data_path}normalized/ohe/respiratoryCare.csv', dtype=dtype_dict)
resp_care_df = du.utils.convert_dataframe(resp_care_df, to='pandas', return_library=False, dtypes=dtype_dict)
resp_care_df = resp_care_df.drop(columns='Unnamed: 0')
resp_care_df.head()

# ### Vital signals

# vital_aprdc_df = mpd.read_csv(f'{data_path}normalized/vitalAperiodic.csv', dtype=dtype_dict)
# vital_aprdc_df = du.utils.convert_dataframe(vital_aprdc_df, to='pandas', return_library=False, dtypes=dtype_dict)
vital_aprdc_df = pd.read_csv(f'{data_path}normalized/vitalAperiodic.csv', dtype=dtype_dict)
vital_aprdc_df = vital_aprdc_df.drop(columns='Unnamed: 0')
vital_aprdc_df.head()

# vital_prdc_df = mpd.read_csv(f'{data_path}normalized/ohe/vitalPeriodic.csv', dtype=dtype_dict)
# vital_prdc_df = du.utils.convert_dataframe(vital_prdc_df, to='pandas', return_library=False, dtypes=dtype_dict)
vital_prdc_df = pd.read_csv(f'{data_path}normalized/ohe/vitalPeriodic.csv', dtype=dtype_dict)
vital_prdc_df = vital_prdc_df.drop(columns='Unnamed: 0')
vital_prdc_df.head()

# ### Exams data

lab_df = mpd.read_csv(f'{data_path}normalized/ohe/lab.csv', dtype=dtype_dict)
lab_df = du.utils.convert_dataframe(lab_df, to='pandas', return_library=False, dtypes=dtype_dict)
lab_df = lab_df.drop(columns='Unnamed: 0')
lab_df.head()

# ## Joining dataframes

# ### Checking the matching of unit stays IDs

full_stays_list = set(patient_df.patientunitstayid.unique())

# Total number of unit stays:

len(full_stays_list)

note_stays_list = set(note_df.patientunitstayid.unique())

len(note_stays_list)

# Number of unit stays that have note data:

len(set.intersection(full_stays_list, note_stays_list))

diagns_stays_list = set(diagns_df.patientunitstayid.unique())

len(diagns_stays_list)

# Number of unit stays that have diagnosis data:

len(set.intersection(full_stays_list, diagns_stays_list))

# +
# alrg_stays_list = set(alrg_df.patientunitstayid.unique())

# +
# len(alrg_stays_list)
# -

# Number of unit stays that have allergy data:

# +
# len(set.intersection(full_stays_list, alrg_stays_list))
# -

past_hist_stays_list = set(past_hist_df.patientunitstayid.unique())

len(past_hist_stays_list)

# Number of unit stays that have past history data:

len(set.intersection(full_stays_list, past_hist_stays_list))

treat_stays_list = set(treat_df.patientunitstayid.unique())

len(treat_stays_list)

# Number of unit stays that have treatment data:

len(set.intersection(full_stays_list, treat_stays_list))

adms_drug_stays_list = set(adms_drug_df.patientunitstayid.unique())

len(adms_drug_stays_list)

# Number of unit stays that have admission drug data:

len(set.intersection(full_stays_list, adms_drug_stays_list))

# +
# inf_drug_stays_list = set(inf_drug_df.patientunitstayid.unique())

# +
# len(inf_drug_stays_list)
# -

# Number of unit stays that have infusion drug data:

# +
# len(set.intersection(full_stays_list, inf_drug_stays_list))
# -

med_stays_list = set(med_df.patientunitstayid.unique())

len(med_stays_list)

# Number of unit stays that have medication data:

len(set.intersection(full_stays_list, med_stays_list))

# +
# in_out_stays_list = set(in_out_df.patientunitstayid.unique())

# +
# len(in_out_stays_list)
# -

# Number of unit stays that have intake and output data:

# +
# len(set.intersection(full_stays_list, in_out_stays_list))

# +
# nurse_care_stays_list = set(nurse_care_df.patientunitstayid.unique())

# +
# len(nurse_care_stays_list)
# -

# Number of unit stays that have nurse care data:

# +
# len(set.intersection(full_stays_list, nurse_care_stays_list))

# +
# nurse_assess_stays_list = set(nurse_assess_df.patientunitstayid.unique())

# +
# len(nurse_assess_stays_list)
# -

# Number of unit stays that have nurse assessment data:

# +
# len(set.intersection(full_stays_list, nurse_assess_stays_list))
# -

resp_care_stays_list = set(resp_care_df.patientunitstayid.unique())

len(resp_care_stays_list)

# Number of unit stays that have respiratory care data:

len(set.intersection(full_stays_list, resp_care_stays_list))

vital_aprdc_stays_list = set(vital_aprdc_df.patientunitstayid.unique())

len(vital_aprdc_stays_list)

# Number of unit stays that have vital aperiodic data:

len(set.intersection(full_stays_list, vital_aprdc_stays_list))

vital_prdc_stays_list = set(vital_prdc_df.patientunitstayid.unique())

len(vital_prdc_stays_list)

# Number of unit stays that have vital periodic data:

len(set.intersection(full_stays_list, vital_prdc_stays_list))

lab_stays_list = set(lab_df.patientunitstayid.unique())

len(lab_stays_list)

# Number of unit stays that have lab data:

len(set.intersection(full_stays_list, lab_stays_list))

# ### Joining patient with note data

# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

eICU_df = patient_df
eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
note_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# Merge the dataframes:

eICU_df = eICU_df.join(note_df, how='left')
eICU_df.head()

# Save the current dataframe:

eICU_df.reset_index(inplace=True)
eICU_df.head()

eICU_df.groupby(['patientunitstayid', 'ts']).size().sort_values()

eICU_df.dtypes

eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_diagns.ftr')

# ### Joining with diagnosis data

eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_diagns.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()

# Filter to the unit stays that also have data in the other tables:

diagns_df = diagns_df[diagns_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

patient_dies = ~(eICU_df.groupby('patientunitstayid').death_ts.max().isna())
patient_dies

unit_stay_patient_dies = set(patient_dies[patient_dies == True].index)
unit_stay_patient_dies

len(unit_stay_patient_dies)

# Also filter only to the unit stays that have data in this new table, considering its importance:

eICU_df.patientunitstayid.nunique()

eICU_df = eICU_df[eICU_df.patientunitstayid.isin(set(diagns_df.patientunitstayid.unique()) | unit_stay_patient_dies)]

eICU_df.patientunitstayid.nunique()

# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
diagns_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# Merge the dataframes:

eICU_df = eICU_df.join(diagns_df, how='outer')
eICU_df.head()

# Save the current dataframe:

eICU_df.reset_index(inplace=True)
eICU_df.head()

eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_past_hist.ftr')

# ### Joining with past history data

eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_past_hist.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()

# Filter to the unit stays that also have data in the other tables:

past_hist_df = past_hist_df[past_hist_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

patient_dies = ~(eICU_df.groupby('patientunitstayid').death_ts.max().isna())
patient_dies

unit_stay_patient_dies = set(patient_dies[patient_dies == True].index)
unit_stay_patient_dies

len(unit_stay_patient_dies)

# Also filter only to the unit stays that have data in this new table, considering its importance:

eICU_df.patientunitstayid.nunique()

eICU_df = eICU_df[eICU_df.patientunitstayid.isin(set(past_hist_df.patientunitstayid.unique()) | unit_stay_patient_dies)]

eICU_df.patientunitstayid.nunique()

# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

eICU_df.set_index('patientunitstayid', inplace=True)
past_hist_df.set_index('patientunitstayid', inplace=True)

# Merge the dataframes:

len(eICU_df)

eICU_df = eICU_df.join(past_hist_df, how='outer')
eICU_df.head()

len(eICU_df)

# Save the current dataframe:

eICU_df.reset_index(inplace=True)
eICU_df.head()

eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_treat.ftr')

# ### Joining with treatment data

eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_treat.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()

# Filter to the unit stays that also have data in the other tables:

treat_df = treat_df[treat_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

patient_dies = ~(eICU_df.groupby('patientunitstayid').death_ts.max().isna())
patient_dies

unit_stay_patient_dies = set(patient_dies[patient_dies == True].index)
unit_stay_patient_dies

len(unit_stay_patient_dies)

# Also filter only to the unit stays that have data in this new table, considering its importance:

eICU_df.patientunitstayid.nunique()

eICU_df = eICU_df[eICU_df.patientunitstayid.isin(set(treat_df.patientunitstayid.unique()) | unit_stay_patient_dies)]

eICU_df.patientunitstayid.nunique()

# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
treat_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# Merge the dataframes:

eICU_df = eICU_df.join(treat_df, how='outer')
eICU_df.head()

# Save the current dataframe:

eICU_df.reset_index(inplace=True)
eICU_df.head()

# +
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)
# -

eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_med.ftr')

# ### Joining with medication data

# #### Joining medication and admission drug data

# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

med_df.set_index(['patientunitstayid', 'ts'], inplace=True)
adms_drug_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# Merge the dataframes:

med_df = med_df.join(adms_drug_df, how='outer', lsuffix='_x', rsuffix='_y')
med_df.head()

# #### Merging duplicate columns

set([col.split('_x')[0].split('_y')[0] for col in med_df.columns if col.endswith('_x') or col.endswith('_y')])

med_df[['drugadmitfrequency_twice_a_day_x', 'drugadmitfrequency_twice_a_day_y',
        'drughiclseqno_10321_x', 'drughiclseqno_10321_y']].head(20)

med_df = du.data_processing.merge_columns(med_df, inplace=True)
med_df.sample(20)

set([col.split('_x')[0].split('_y')[0] for col in med_df.columns if col.endswith('_x') or col.endswith('_y')])

med_df[['drugadmitfrequency_twice_a_day', 'drughiclseqno_10321']].head(20)

# Save the current dataframe:

med_df.reset_index(inplace=True)
med_df.head()

med_df.to_feather(f'{data_path}normalized/ohe/med_and_adms_drug.ftr')

# #### Joining with the rest of the eICU data

med_drug_df = pd.read_feather(f'{data_path}normalized/ohe/med_and_adms_drug.ftr')
med_drug_df = du.utils.convert_dtypes(med_drug_df, dtypes=dtype_dict, inplace=True)
med_drug_df.head()

eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_med.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()

# Filter to the unit stays that also have data in the other tables:

med_drug_df = med_drug_df[med_drug_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

patient_dies = ~(eICU_df.groupby('patientunitstayid').death_ts.max().isna())
patient_dies

unit_stay_patient_dies = set(patient_dies[patient_dies == True].index)
unit_stay_patient_dies

len(unit_stay_patient_dies)

# Also filter only to the unit stays that have data in this new table, considering its importance:

eICU_df.patientunitstayid.nunique()

eICU_df = eICU_df[eICU_df.patientunitstayid.isin(set(med_drug_df.patientunitstayid.unique()) | unit_stay_patient_dies)]

eICU_df.patientunitstayid.nunique()

# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
med_drug_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# Merge the dataframes:

len(eICU_df)

eICU_df = eICU_df.join(med_drug_df, how='outer')
eICU_df.head()

len(eICU_df)

# Save the current dataframe:

eICU_df.reset_index(inplace=True)
eICU_df.head()

eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_resp_care.ftr')

# ### Joining with respiratory care data

eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_resp_care.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()

# Filter to the unit stays that also have data in the other tables:

resp_care_df = resp_care_df[resp_care_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
resp_care_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# Merge the dataframes:

eICU_df = eICU_df.join(resp_care_df, how='outer')
eICU_df.head()

# Save the current dataframe:

eICU_df.reset_index(inplace=True)
eICU_df.head()

eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_vital_aprdc.ftr')

# ### Joining with aperiodic vital signals data

eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_vital_aprdc.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()

# Filter to the unit stays that also have data in the other tables:

vital_aprdc_df = vital_aprdc_df[vital_aprdc_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

patient_dies = ~(eICU_df.groupby('patientunitstayid').death_ts.max().isna())
patient_dies

unit_stay_patient_dies = set(patient_dies[patient_dies == True].index)
unit_stay_patient_dies

len(unit_stay_patient_dies)

# Also filter only to the unit stays that have data in this new table, considering its importance:

eICU_df.patientunitstayid.nunique()

eICU_df = eICU_df[eICU_df.patientunitstayid.isin(set(vital_aprdc_df.patientunitstayid.unique()) | unit_stay_patient_dies)]

eICU_df.patientunitstayid.nunique()

# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
vital_aprdc_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# Merge the dataframes:

eICU_df = eICU_df.join(vital_aprdc_df, how='outer')
eICU_df.head()

# #### Filtering for the lengthiest unit stays
#
# Filter to the 10k lengthiest unit stays, also including those where the patient dies (even if it's not one of the lengthiest, so as to avoid making the dataset even more unbalanced)

eICU_df.reset_index(inplace=True)
eICU_df.head()

# Get the 10k lengthiest unit stays:

unit_stay_len = eICU_df.groupby('patientunitstayid').patientunitstayid.count().sort_values(ascending=False)
unit_stay_len

unit_stay_len.value_counts()

unit_stay_long = set(unit_stay_len[:10000].index)
unit_stay_long

len(unit_stay_long)

patient_dies = ~(eICU_df.groupby('patientunitstayid').death_ts.max().isna())
patient_dies

unit_stay_patient_dies = set(patient_dies[patient_dies == True].index)
unit_stay_patient_dies

len(unit_stay_patient_dies)

eICU_df.patientunitstayid.nunique()

eICU_df = eICU_df[eICU_df.patientunitstayid.isin(unit_stay_long | unit_stay_patient_dies)]

eICU_df.patientunitstayid.nunique()

# Save the current dataframe:

du.data_processing.save_chunked_data(eICU_df, file_name='eICU_before_joining_vital_prdc', n_chunks=8, 
                                     data_path=f'{data_path}normalized/ohe/')

# ### Joining with periodic vital signals data

eICU_df = du.data_processing.load_chunked_data(file_name='eICU_before_joining_vital_prdc', n_chunks=8, 
                                               data_path=f'{data_path}normalized/ohe/', dtypes=dtype_dict)
eICU_df.head()

# Filter to the unit stays that also have data in the other tables:

vital_prdc_df = vital_prdc_df[vital_prdc_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

patient_dies = ~(eICU_df.groupby('patientunitstayid').death_ts.max().isna())
patient_dies

unit_stay_patient_dies = set(patient_dies[patient_dies == True].index)
unit_stay_patient_dies

len(unit_stay_patient_dies)

# Also filter only to the unit stays that have data in this new table, considering its importance:

eICU_df.patientunitstayid.nunique()

eICU_df = eICU_df[eICU_df.patientunitstayid.isin(set(vital_prdc_df.patientunitstayid.unique()) | unit_stay_patient_dies)]

eICU_df.patientunitstayid.nunique()

# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
vital_prdc_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# Merge the dataframes:

eICU_df = eICU_df.join(vital_prdc_df, how='outer')
eICU_df.head()

# Save the current dataframe:

eICU_df.reset_index(inplace=True)
eICU_df.head()

du.data_processing.save_chunked_data(eICU_df, file_name='eICU_before_joining_lab', n_chunks=8, 
                                     data_path=f'{data_path}normalized/ohe/')

# ### Joining with lab data

eICU_df = du.data_processing.load_chunked_data(file_name='eICU_before_joining_lab', n_chunks=8, 
                                               data_path=f'{data_path}normalized/ohe/', dtypes=dtype_dict)
eICU_df.head()

# Filter to the unit stays that also have data in the other tables:

lab_df = lab_df[lab_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

patient_dies = ~(eICU_df.groupby('patientunitstayid').death_ts.max().isna())
patient_dies

unit_stay_patient_dies = set(patient_dies[patient_dies == True].index)
unit_stay_patient_dies

len(unit_stay_patient_dies)

# Also filter only to the unit stays that have data in this new table, considering its importance:

eICU_df.patientunitstayid.nunique()

eICU_df = eICU_df[eICU_df.patientunitstayid.isin(set(lab_df.patientunitstayid.unique()) | unit_stay_patient_dies)]

eICU_df.patientunitstayid.nunique()

# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
lab_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# Merge the dataframes:

eICU_df = eICU_df.join(lab_df, how='outer')
eICU_df.head()

# Save the current dataframe:

eICU_df.reset_index(inplace=True)
eICU_df.head()

du.data_processing.save_chunked_data(eICU_df, file_name='eICU_post_joining', n_chunks=8, 
                                     data_path=f'{data_path}normalized/ohe/')

# ## Cleaning the joined data

eICU_df = du.data_processing.load_chunked_data(file_name='eICU_post_joining', n_chunks=8, 
                                               data_path=f'{data_path}normalized/ohe/', dtypes=dtype_dict)
eICU_df.head()

# +
# eICU_df.info(memory_usage='deep')
# -

# ### Removing unit stays that are too short

# Make sure that the dataframe is ordered by time `ts`:

eICU_df = eICU_df.sort_values('ts')
eICU_df.head()

# Remove unit stays that have data that represent less than 48h:

unit_stay_duration = eICU_df.groupby('patientunitstayid').ts.apply(lambda x: x.max() - x.min())
unit_stay_duration

unit_stay_long = set(unit_stay_duration[unit_stay_duration >= 48*60].index)
unit_stay_long

len(unit_stay_long)

patient_dies = ~(eICU_df.groupby('patientunitstayid').death_ts.max().isna())
patient_dies

unit_stay_patient_dies = set(patient_dies[patient_dies == True].index)
unit_stay_patient_dies

len(unit_stay_patient_dies)

eICU_df.patientunitstayid.nunique()

eICU_df = eICU_df[eICU_df.patientunitstayid.isin(unit_stay_long | unit_stay_patient_dies)]

eICU_df.patientunitstayid.nunique()

du.data_processing.save_chunked_data(eICU_df, file_name='eICU_post_short_stay_removal', n_chunks=8, 
                                     data_path=f'{data_path}normalized/ohe/')

# ### Removing unit stays with too many missing values
#
# Consider removing all unit stays that have, combining rows and columns, a very high percentage of missing values.

eICU_df = du.data_processing.load_chunked_data(file_name='eICU_post_short_stay_removal', n_chunks=8, 
                                               data_path=f'{data_path}normalized/ohe/', dtypes=dtype_dict)
eICU_df.head()

# Reconvert dataframe to Modin:

eICU_df = du.utils.convert_dataframe(vital_prdc_df, to='modin', return_library=False, dtypes=dtype_dict)

n_features = len(eICU_df.columns)
n_features

# Create a temporary column that counts each row's number of missing values:

eICU_df['row_msng_val'] = eICU_df.isnull().sum(axis=1)
eICU_df[['patientunitstayid', 'ts', 'row_msng_val']].head()

# Check each unit stay's percentage of missing data points:

# Number of possible data points in each unit stay
n_data_points = eICU_df.groupby('patientunitstayid').ts.count() * n_features
n_data_points

# Number of missing values in each unit stay
n_msng_val = eICU_df.groupby('patientunitstayid').row_msng_val.sum()
n_msng_val

# Percentage of missing values in each unit stay
msng_val_prct = (n_msng_val / n_data_points) * 100
msng_val_prct

msng_val_prct.describe()

# Remove unit stays that have too many missing values (>70% of their respective data points):

unit_stay_high_msgn = set(msng_val_prct[msng_val_prct > 70].index)
unit_stay_high_msgn

eICU_df.patientunitstayid.nunique()

eICU_df = eICU_df[~eICU_df.patientunitstayid.isin(unit_stay_high_msgn)]

eICU_df.patientunitstayid.nunique()

du.data_processing.save_chunked_data(eICU_df, file_name='eICU_post_high_missing_stay_removal', n_chunks=8, 
                                     data_path=f'{data_path}normalized/ohe/')

# ### Removing columns with too many missing values
#
# We should remove features that have too many missing values (in this case, those that have more than 40% of missing values). Without enough data, it's even risky to do imputation, as it's unlikely for the imputation to correctly model the missing feature.

eICU_df = du.data_processing.load_chunked_data(file_name='eICU_post_high_missing_stay_removal', n_chunks=8, 
                                               data_path=f'{data_path}normalized/ohe/', dtypes=dtype_dict)
eICU_df.head()

du.search_explore.dataframe_missing_values(eICU_df)

prev_features = eICU_df.columns
len(prev_features)

eICU_df = du.data_processing.remove_cols_with_many_nans(eICU_df, nan_percent_thrsh=70, inplace=True)

features = eICU_df.columns
len(features)

# Removed features:

set(prev_features) - set(features)

eICU_df.head()

du.data_processing.save_chunked_data(eICU_df, file_name='eICU_post_high_missing_cols_removal', n_chunks=8, 
                                     data_path=f'{data_path}normalized/ohe/')

# ### Performing imputation

eICU_df = du.data_processing.load_chunked_data(file_name='eICU_post_high_missing_cols_removal', n_chunks=8, 
                                               data_path=f'{data_path}normalized/ohe/', dtypes=dtype_dict)
eICU_df.head()

du.search_explore.dataframe_missing_values(eICU_df)

# Imputate patient and past history features separately, as they should remain the same regardless of time:

eICU_columns = list(eICU_df.columns)
patient_columns = list(set((list(patient_df.columns) + list(past_history_df.columns))))
const_columns = list()
for col in patient_columns:
    if col in eICU_columns:
        const_columns.append(col)

# Forward fill
eICU_df.loc[:, const_columns] = (eICU_df.set_index('patientunitstayid', append=True)
                                 .groupby('patientunitstayid')[const_columns].fillna(method='ffill'))
# Backward fill
eICU_df.loc[:, const_columns] = (eICU_df.set_index('patientunitstayid', append=True)
                                 .groupby('patientunitstayid')[const_columns].fillna(method='bfill')
# Replace remaining missing values with zero
eICU_df.loc[:, const_columns] = eICU_df[const_columns].fillna(value=0)

# Imputate the remaining features:

eICU_df = du.data_processing.missing_values_imputation(eICU_df, method='interpolation',
                                                       id_column='patientunitstayid', inplace=True)
eICU_df.head()

du.search_explore.dataframe_missing_values(eICU_df)

du.data_processing.save_chunked_data(eICU_df, file_name='eICU_post_imputation', n_chunks=8, 
                                     data_path=f'{data_path}normalized/ohe/')

# ### Rearranging columns
#
# For ease of use and for better intuition, we should make sure that the ID columns (`patientunitstayid` and `ts`) are the first ones in the dataframe.

eICU_df = du.data_processing.load_chunked_data(file_name='eICU_post_imputation', n_chunks=8, 
                                               data_path=f'{data_path}normalized/ohe/', dtypes=dtype_dict)
eICU_df.head()

columns = list(eICU_df.columns)
columns

columns.remove('patientunitstayid')
columns.remove('ts')

columns = ['patientunitstayid', 'ts'] + columns
columns

eICU_df = eICU_df[columns]
eICU_df.head()

du.data_processing.save_chunked_data(eICU_df, file_name='eICU', n_chunks=8, 
                                     data_path=f'{data_path}normalized/ohe/')

# ## Setting the label
#
# Define the label column considering the desired time window on which we want to predict mortality (0, 24h, 48h, 72h, etc).

time_window_h = 24

eICU_df['label'] = eICU_df[eICU_df.death_ts - eICU_df.ts <= time_window_h * 60]
eICU_df.head()

# ## Creating a single encoding dictionary for the complete dataframe
#
# Combine the one hot encoding dictionaries of all tables, having in account the converged ones, into a single dictionary representative of all the categorical features in the resulting dataframe.

stream_adms_drug = open(f'{data_path}cat_feat_ohe_adms_drug.yaml', 'r')
stream_inf_drug = open(f'{data_path}cat_feat_ohe_inf_drug.yaml', 'r')
stream_med = open(f'{data_path}cat_feat_ohe_med.yaml', 'r')
stream_treat = open(f'{data_path}cat_feat_ohe_treat.yaml', 'r')
stream_diag = open(f'{data_path}cat_feat_ohe_diag.yaml', 'r')
stream_alrg = open(f'{data_path}cat_feat_ohe_alrg.yaml', 'r')
stream_past_hist = open(f'{data_path}cat_feat_ohe_past_hist.yaml', 'r')
stream_lab = open(f'{data_path}cat_feat_ohe_lab.yaml', 'r')
stream_patient = open(f'{data_path}cat_feat_ohe_patient.yaml', 'r')
stream_notes = open(f'{data_path}cat_feat_ohe_note.yaml', 'r')

cat_feat_ohe_adms_drug = yaml.load(stream_adms_drug, Loader=yaml.FullLoader)
cat_feat_ohe_inf_drug = yaml.load(stream_inf_drug, Loader=yaml.FullLoader)
cat_feat_ohe_med = yaml.load(stream_med, Loader=yaml.FullLoader)
cat_feat_ohe_treat = yaml.load(stream_treat, Loader=yaml.FullLoader)
cat_feat_ohe_diag = yaml.load(stream_diag, Loader=yaml.FullLoader)
cat_feat_ohe_alrg = yaml.load(stream_alrg, Loader=yaml.FullLoader)
cat_feat_ohe_past_hist = yaml.load(stream_past_hist, Loader=yaml.FullLoader)
cat_feat_ohe_lab = yaml.load(stream_lab, Loader=yaml.FullLoader)
cat_feat_ohe_patient = yaml.load(stream_patient, Loader=yaml.FullLoader)
cat_feat_ohe_notes = yaml.load(stream_notes, Loader=yaml.FullLoader)

cat_feat_ohe = du.utils.merge_dicts([cat_feat_ohe_adms_drug, cat_feat_ohe_inf_drug,
                                     cat_feat_ohe_med, cat_feat_ohe_treat,
                                     cat_feat_ohe_diag, cat_feat_ohe_alrg,
                                     cat_feat_ohe_past_hist, cat_feat_ohe_lab,
                                     cat_feat_ohe_patient, cat_feat_ohe_notes])

list(cat_feat_ohe.keys())

# Clean the one hot encoded column names, as they are in the final dataframe:

for key, val in cat_feat_ohe.items():
    cat_feat_ohe[key] = du.data_processing.clean_naming(cat_feat_ohe[key], lower_case=False)

# Save the final encoding dictionary:

stream = open(f'{data_path}/cleaned/cat_feat_ohe_eICU.yaml', 'w')
yaml.dump(cat_feat_ohe, stream, default_flow_style=False)

# ## Creating a single normalization dictionary for the complete dataframe
#
# Combine the normalization stats dictionaries of all tables into a single dictionary representative of all the continuous features in the resulting dataframe.

stream_adms_drug = open(f'{data_path}admissionDrug_norm_stats.yaml', 'r')
stream_inf_drug = open(f'{data_path}infusionDrug_norm_stats.yaml', 'r')
stream_med = open(f'{data_path}medication_norm_stats.yaml', 'r')
# stream_in_out = open(f'{data_path}cat_embed_feat_enum_in_out.yaml', 'r')
stream_lab = open(f'{data_path}lab_norm_stats.yaml', 'r')
stream_patient = open(f'{data_path}patient_norm_stats.yaml', 'r')
# stream_vital_aprdc = open(f'{data_path}vitalAperiodic_norm_stats.yaml', 'r')
# stream_vital_prdc = open(f'{data_path}vitalPeriodic_norm_stats.yaml', 'r')

norm_stats_adms_drug = yaml.load(stream_adms_drug, Loader=yaml.FullLoader)
norm_stats_inf_drug = yaml.load(stream_inf_drug, Loader=yaml.FullLoader)
norm_stats_med = yaml.load(stream_med, Loader=yaml.FullLoader)
# norm_stats__in_out = yaml.load(stream_in_out, Loader=yaml.FullLoader)
norm_stats_lab = yaml.load(stream_lab, Loader=yaml.FullLoader)
norm_stats_patient = yaml.load(stream_patient, Loader=yaml.FullLoader)
# norm_stats_vital_aprdc = yaml.load(stream_vital_aprdc, Loader=yaml.FullLoader)
# norm_stats_vital_prdc = yaml.load(stream_vital_prdc, Loader=yaml.FullLoader)

norm_stats = du.utils.merge_dicts([norm_stats_adms_drug, norm_stats_inf_drug,
                                   norm_stats_med,
#                                    norm_stats_in_out,
                                   norm_stats_lab, norm_stats_patient,
                                   norm_stats_vital_aprdc, norm_stats_vital_prdc])

list(norm_stats.keys())

# Save the final encoding dictionary:

stream = open(f'{data_path}/cleaned/eICU_norm_stats.yaml', 'w')
yaml.dump(norm_stats, stream, default_flow_style=False)
