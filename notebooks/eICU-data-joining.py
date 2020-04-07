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

# import modin.pandas as mpd                  # Optimized distributed version of Pandas
import pandas as pd
import data_utils as du                     # Data science and machine learning relevant methods

du.set_pandas_library('pandas')

# Allow pandas to show more columns:

pd.set_option('display.max_columns', 3000)
pd.set_option('display.max_rows', 3000)

# Set the random seed for reproducibility

du.set_random_seed(42)

# ## Initializing variables

stream_dtypes = open(f'{data_path}eICU_dtype_dict.yml', 'r')
stream_const_columns = open(f'{data_path}eICU_const_columns.yml', 'r')

dtype_dict = yaml.load(stream_dtypes, Loader=yaml.FullLoader)
dtype_dict

const_columns = yaml.load(stream_const_columns, Loader=yaml.FullLoader)
const_columns

# ## Loading the data

# ### Patient information

patient_df = pd.read_csv(f'{data_path}normalized/ohe/patient.csv', dtype=dtype_dict)
# patient_df = du.utils.convert_dataframe(patient_df, to='pandas', return_library=False, dtypes=dtype_dict)
patient_df = patient_df.drop(columns='Unnamed: 0')
patient_df.head()

patient_df.dtypes

patient_dies = ~(patient_df.groupby('patientunitstayid').death_ts.max().isna())
patient_dies

unit_stay_patient_dies = set(patient_dies[patient_dies == True].index)
unit_stay_patient_dies

len(unit_stay_patient_dies)

(~patient_df['death_ts'].isnull()).sum()

patient_df.to_numpy()

type(patient_df.to_numpy())

import sys

sys.getsizeof(patient_df)

sys.getsizeof(patient_df.values)

patient_df.values.nbytes

note_df = mpd.read_csv(f'{data_path}normalized/ohe/note.csv', dtype=dtype_dict)
note_df = du.utils.convert_dataframe(note_df, to='pandas', return_library=False, dtypes=dtype_dict)
note_df = note_df.drop(columns='Unnamed: 0')
note_df.head()

# ### Diagnosis

diagns_df = mpd.read_csv(f'{data_path}normalized/ohe/diagnosis.csv', dtype=dtype_dict)
diagns_df = du.utils.convert_dataframe(diagns_df, to='pandas', return_library=False, dtypes=dtype_dict)
diagns_df = diagns_df.drop(columns='Unnamed: 0')
diagns_df.head()

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

adms_drug_df.dtypes

med_df = mpd.read_csv(f'{data_path}normalized/ohe/medication.csv', dtype=dtype_dict)
med_df = du.utils.convert_dataframe(med_df, to='pandas', return_library=False, dtypes=dtype_dict)
med_df = med_df.drop(columns='Unnamed: 0')
med_df.head()

med_df.dtypes

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

lab_df.dtypes.value_counts()

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

eICU_df.dtypes.value_counts()

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

eICU_df.dtypes.value_counts()

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

eICU_df.dtypes

eICU_df.dtypes.value_counts()

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

eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_med.ftr')

eICU_df.dtypes.value_counts()

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

med_df.dtypes.value_counts()

med_df = du.data_processing.merge_columns(med_df, inplace=True)
med_df.sample(20)

set([col.split('_x')[0].split('_y')[0] for col in med_df.columns if col.endswith('_x') or col.endswith('_y')])

med_df[['drugadmitfrequency_twice_a_day', 'drughiclseqno_10321']].head(20)

# Save the current dataframe:

med_df.reset_index(inplace=True)
med_df.head()

med_df.to_feather(f'{data_path}normalized/ohe/med_and_adms_drug.ftr')

med_df.dtypes.value_counts()

# #### Joining with the rest of the eICU data

med_drug_df = pd.read_feather(f'{data_path}normalized/ohe/med_and_adms_drug.ftr')
med_drug_df = du.utils.convert_dtypes(med_drug_df, dtypes=dtype_dict, inplace=True)
med_drug_df.head()

med_drug_df.dtypes.value_counts()

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

eICU_df.dtypes.value_counts()

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

eICU_df.dtypes.value_counts()

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

# #### Filtering for the unit stays with less missing data
#
# Filter to the 10k unit stays with less missing data, also including those where the patient dies (even if it's not one of those with less missing data, so as to avoid making the dataset even more unbalanced)

eICU_df.reset_index(inplace=True)
eICU_df.head()

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

msng_val_prct = msng_val_prct.sort_values(ascending=False)
msng_val_prct

# Get the 10k unit stays with less missing data points:

unit_stay_low_msgn = set(msng_val_prct[:10000].index)
unit_stay_low_msgn

len(unit_stay_low_msgn)

patient_dies = ~(eICU_df.groupby('patientunitstayid').death_ts.max().isna())
patient_dies

unit_stay_patient_dies = set(patient_dies[patient_dies == True].index)
unit_stay_patient_dies

len(unit_stay_patient_dies)

eICU_df.patientunitstayid.nunique()

# eICU_df = eICU_df[eICU_df.patientunitstayid.isin(unit_stay_low_msgn | unit_stay_patient_dies)]
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(unit_stay_low_msgn)]

eICU_df.patientunitstayid.nunique()

patient_dies = ~(eICU_df.groupby('patientunitstayid').death_ts.max().isna())
patient_dies

unit_stay_patient_dies = set(patient_dies[patient_dies == True].index)
unit_stay_patient_dies

len(unit_stay_patient_dies)

eICU_df.drop(columns='row_msng_val', inplace=True)

# Save the current dataframe:

du.data_processing.save_chunked_data(eICU_df, file_name='eICU_before_joining_vital_prdc_10k', n_chunks=8, 
                                     data_path=f'{data_path}normalized/ohe/')

eICU_df.dtypes.value_counts()

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

eICU_df.dtypes.value_counts()

# ### Joining with lab data

eICU_df = du.data_processing.load_chunked_data(file_name='eICU_before_joining_lab', n_chunks=8, 
                                               data_path=f'{data_path}normalized/ohe/', dtypes=dtype_dict)
eICU_df.head()

# Filter to the unit stays that also have data in the other tables:

lab_df = lab_df[lab_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

# +
# patient_dies = ~(eICU_df.groupby('patientunitstayid').death_ts.max().isna())
# patient_dies

# +
# unit_stay_patient_dies = set(patient_dies[patient_dies == True].index)
# unit_stay_patient_dies

# +
# len(unit_stay_patient_dies)
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# +
# eICU_df.patientunitstayid.nunique()

# +
# eICU_df = eICU_df[eICU_df.patientunitstayid.isin(set(lab_df.patientunitstayid.unique()) | unit_stay_patient_dies)]

# +
# eICU_df.patientunitstayid.nunique()
# -

# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
lab_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# Merge the dataframes:

eICU_df = eICU_df.join(lab_df, how='outer')
eICU_df.head()

# Save the current dataframe:

eICU_df.reset_index(inplace=True)
eICU_df.head()

len(eICU_df)

du.data_processing.save_chunked_data(eICU_df, file_name='eICU_post_joining', n_chunks=8, 
                                     data_path=f'{data_path}normalized/ohe/')

eICU_df.dtypes.value_counts()

patient_dies = ~(eICU_df.groupby('patientunitstayid').death_ts.max().isna())
unit_stay_patient_dies = set(patient_dies[patient_dies == True].index)
len(unit_stay_patient_dies)

eICU_df.dtypes[eICU_df.dtypes=='float64']

# ## Cleaning the joined data

eICU_df = du.data_processing.load_chunked_data(file_name='eICU_post_joining', n_chunks=8, 
                                               data_path=f'{data_path}normalized/ohe/', dtypes=dtype_dict)
eICU_df.head()

eICU_df.dtypes.value_counts()

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

len(unit_stay_long | unit_stay_patient_dies)

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

msng_val_prct = msng_val_prct.sort_values(ascending=False)
msng_val_prct

# Only keep the 7k unit stays that have less missing data points:

unit_stay_low_msgn = set(msng_val_prct[:8000].index)
unit_stay_low_msgn

# ~Remove unit stays that have too many missing values (>99% of their respective data points):~

# +
# unit_stay_low_msgn = set(msng_val_prct[msng_val_prct < 99].index)
# unit_stay_low_msgn
# -

len(unit_stay_low_msgn)

patient_dies = ~(eICU_df.groupby('patientunitstayid').death_ts.max().isna())
patient_dies

unit_stay_patient_dies = set(patient_dies[patient_dies == True].index)
unit_stay_patient_dies

len(unit_stay_patient_dies)

len(unit_stay_low_msgn | unit_stay_patient_dies)

len(unit_stay_low_msgn & unit_stay_patient_dies)

eICU_df.patientunitstayid.nunique()

# eICU_df = eICU_df[eICU_df.patientunitstayid.isin(unit_stay_low_msgn | unit_stay_patient_dies)]
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(unit_stay_low_msgn)]

eICU_df.patientunitstayid.nunique()

patient_dies = ~(eICU_df.groupby('patientunitstayid').death_ts.max().isna())
patient_dies

unit_stay_patient_dies = set(patient_dies[patient_dies == True].index)
unit_stay_patient_dies

len(unit_stay_patient_dies)

eICU_df.drop(columns='row_msng_val', inplace=True)

du.data_processing.save_chunked_data(eICU_df, file_name='eICU_post_high_missing_stay_removal', n_chunks=8, 
                                     data_path=f'{data_path}normalized/ohe/')

tmp_arr = eICU_df.to_records(index=False)
tmp_arr

# ### Removing columns with too many missing values
#
# We should remove features that have too many missing values (in this case, those that have more than 40% of missing values). Without enough data, it's even risky to do imputation, as it's unlikely for the imputation to correctly model the missing feature.
#
# Actually, it doesn't seem to be worth it, as most columns have around 98% missing data points.

# +
# eICU_df = du.data_processing.load_chunked_data(file_name='eICU_post_high_missing_stay_removal', n_chunks=8, 
#                                                data_path=f'{data_path}normalized/ohe/', dtypes=dtype_dict)
# eICU_df.head()

# +
# missing_values = du.search_explore.dataframe_missing_values(eICU_df)
# missing_values

# +
# len(eICU_df)

# +
# missing_values.percent_missing.describe()

# +
# prev_features = eICU_df.columns
# len(prev_features)

# +
# eICU_df = du.data_processing.remove_cols_with_many_nans(eICU_df, nan_percent_thrsh=99, inplace=True)

# +
# features = eICU_df.columns
# len(features)
# -

# Removed features:

# +
# set(prev_features) - set(features)

# +
# eICU_df.head()

# +
# du.data_processing.save_chunked_data(eICU_df, file_name='eICU_post_high_missing_cols_removal', n_chunks=8, 
#                                      data_path=f'{data_path}normalized/ohe/')
# -

# ### Performing imputation

eICU_df = du.data_processing.load_chunked_data(file_name='eICU_post_high_missing_stay_removal', n_chunks=8, 
                                               data_path=f'{data_path}normalized/ohe/', dtypes=dtype_dict)
eICU_df.head()

eICU_df.shape

eICU_df.patientunitstayid.nunique()

# #### Constant columns
#
# Imputate patient and past history features separately, as they should remain the same regardless of time.

eICU_df.reset_index(drop=True, inplace=True)

# +
# eICU_df.loc[0, const_columns]

# +
# # Trying to imputate data of just one unit stay
# id_const_columns = ['patientunitstayid'] + const_columns
# # Forward fill and backward fill
# eICU_df[eICU_df.patientunitstayid == 2385766][id_const_columns].groupby('patientunitstayid').apply(lambda group: group.ffill().bfill())
# -

du.search_explore.dataframe_missing_values(eICU_df)

id_const_columns = ['patientunitstayid'] + const_columns
# Forward fill and backward fill
eICU_df.loc[:, id_const_columns] = eICU_df[id_const_columns].groupby('patientunitstayid').apply(lambda group: group.ffill().bfill())


# Replace remaining missing values with zero
eICU_df.loc[:, const_columns] = eICU_df[const_columns].fillna(value=0)


du.data_processing.save_chunked_data(eICU_df, file_name='eICU_post_const_imputation', n_chunks=8, 
                                     data_path=f'{data_path}normalized/ohe/')

du.search_explore.dataframe_missing_values(eICU_df)

# +
# eICU_df[eICU_df.patientunitstayid == 2385766][id_const_columns]
# -

# #### Boolean columns
#
# Boolean columns should have their missing values filled with zeros, so as to signal the absense of each feature.

eICU_df = du.data_processing.load_chunked_data(file_name='eICU_post_const_imputation', n_chunks=8, 
                                               data_path=f'{data_path}normalized/ohe/', dtypes=dtype_dict)
eICU_df.head()

du.search_explore.dataframe_missing_values(eICU_df)

bool_columns = du.search_explore.list_boolean_columns(eICU_df, search_by_dtypes=True)
bool_columns

eICU_df.dtypes.value_counts()

eICU_df.dtypes[eICU_df.dtypes != 'UInt8']

# Replace missing values with zero
eICU_df.loc[:, bool_columns] = eICU_df[bool_columns].fillna(value=0)

du.data_processing.save_chunked_data(eICU_df, file_name='eICU_post_bool_imputation', n_chunks=8, 
                                     data_path=f'{data_path}normalized/ohe/')

du.search_explore.dataframe_missing_values(eICU_df)

# #### Remaining features

eICU_df = du.data_processing.load_chunked_data(file_name='eICU_post_bool_imputation', n_chunks=8, 
                                               data_path=f'{data_path}normalized/ohe/', dtypes=dtype_dict)
eICU_df.head()

du.search_explore.dataframe_missing_values(eICU_df)

bool_columns = du.search_explore.list_boolean_columns(eICU_df, search_by_dtypes=True)
bool_columns

columns_to_imputate = list(eICU_df.columns)
columns_to_imputate.remove('death_ts')
columns_to_imputate.remove('ts')
columns_to_imputate = list(set(columns_to_imputate) - set(const_columns) - set(bool_columns))
columns_to_imputate

# +
# # %%pixie_debugger
# # Trying to imputate data of just one unit stay
# du.data_processing.missing_values_imputation(eICU_df[eICU_df.patientunitstayid == 2385766], 
#                                              columns_to_imputate=columns_to_imputate,
#                                              method='interpolation', id_column='patientunitstayid',
#                                              inplace=True)
# -

eICU_df = du.data_processing.missing_values_imputation(eICU_df, method='interpolation',
                                                       columns_to_imputate=columns_to_imputate,
                                                       id_column='patientunitstayid', 
                                                       zero_bool=False, inplace=True)
eICU_df.head()

# +
# eICU_df = eICU_df.fillna(value=0)
# eICU_df.head()
# -

du.data_processing.save_chunked_data(eICU_df, file_name='eICU', batch_size=32,
                                     id_column='patientunitstayid', data_path=f'{data_path}normalized/ohe/final/')

du.search_explore.dataframe_missing_values(eICU_df)

eICU_df = du.data_processing.load_chunked_data(file_name='eICU', n_chunks=250, 
                                               data_path=f'{data_path}normalized/ohe/final/', dtypes=dtype_dict)
eICU_df.head()

# Set the label columns
eICU_df['label_0h'] = (eICU_df.death_ts - eICU_df.ts <= 0).astype('float16')
eICU_df['label_24h'] = (eICU_df.death_ts - eICU_df.ts <= 24 * 60).astype('float16')
eICU_df['label_48h'] = (eICU_df.death_ts - eICU_df.ts <= 48 * 60).astype('float16')
eICU_df['label_72h'] = (eICU_df.death_ts - eICU_df.ts <= 72 * 60).astype('float16')
eICU_df['label_96h'] = (eICU_df.death_ts - eICU_df.ts <= 96 * 60).astype('float16')

(eICU_df['label_0h'].sum() / len(eICU_df)) * 100

(eICU_df['label_24h'].sum() / len(eICU_df)) * 100

(eICU_df['label_48h'].sum() / len(eICU_df)) * 100

(eICU_df['label_72h'].sum() / len(eICU_df)) * 100

(eICU_df['label_96h'].sum() / len(eICU_df)) * 100

n_unit_stays = eICU_df.patientunitstayid.nunique()
n_unit_stays

((eICU_df.groupby('patientunitstayid').label_0h.max() == 1).sum() / n_unit_stays) * 100

((eICU_df.groupby('patientunitstayid').label_24h.max() == 1).sum() / n_unit_stays) * 100

((eICU_df.groupby('patientunitstayid').label_48h.max() == 1).sum() / n_unit_stays) * 100

((eICU_df.groupby('patientunitstayid').label_72h.max() == 1).sum() / n_unit_stays) * 100

((eICU_df.groupby('patientunitstayid').label_96h.max() == 1).sum() / n_unit_stays) * 100

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

du.data_processing.save_chunked_data(eICU_df, file_name='eICU', batch_size=32,
                                     id_column='patientunitstayid', data_path=f'{data_path}normalized/ohe/')
