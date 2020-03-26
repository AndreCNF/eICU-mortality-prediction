# # eICU Data Dictionaries
# ---
#
# The main goal of this notebook is to prepare dictionaries containing important information on the eICU data, namely the column data types, the categorization of each set of one hot encoded columns and the normalization stats.

# ## Importing the necessary packages

import os                                  # os handles directory/workspace changes
import yaml                                # Save and load YAML files

import pixiedust                           # Debugging in Jupyter Notebook cells

# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# Path to the parquet dataset files
data_path = 'data/eICU/cleaned/'
# Path to the code files
project_path = 'code/eICU-mortality-prediction/'

import data_utils as du                    # Data science and machine learning relevant methods

# Set the random seed for reproducibility

du.set_random_seed(42)

# ## One hot encoding columns categorization
#
# Combine the one hot encoding dictionaries of all tables, having in account the converged ones, into a single dictionary representative of all the categorical features in the resulting dataframe.

stream_adms_drug = open(f'{data_path}cat_feat_ohe_adms_drug.yml', 'r')
stream_med = open(f'{data_path}cat_feat_ohe_med.yml', 'r')
stream_treat = open(f'{data_path}cat_feat_ohe_treat.yml', 'r')
stream_diag = open(f'{data_path}cat_feat_ohe_diag.yml', 'r')
stream_past_hist = open(f'{data_path}cat_feat_ohe_past_hist.yml', 'r')
stream_lab = open(f'{data_path}cat_feat_ohe_lab.yml', 'r')
stream_patient = open(f'{data_path}cat_feat_ohe_patient.yml', 'r')
stream_notes = open(f'{data_path}cat_feat_ohe_note.yml', 'r')

cat_feat_ohe_adms_drug = yaml.load(stream_adms_drug, Loader=yaml.FullLoader)
cat_feat_ohe_med = yaml.load(stream_med, Loader=yaml.FullLoader)
cat_feat_ohe_treat = yaml.load(stream_treat, Loader=yaml.FullLoader)
cat_feat_ohe_diag = yaml.load(stream_diag, Loader=yaml.FullLoader)
cat_feat_ohe_past_hist = yaml.load(stream_past_hist, Loader=yaml.FullLoader)
cat_feat_ohe_lab = yaml.load(stream_lab, Loader=yaml.FullLoader)
cat_feat_ohe_patient = yaml.load(stream_patient, Loader=yaml.FullLoader)
cat_feat_ohe_notes = yaml.load(stream_notes, Loader=yaml.FullLoader)

set(cat_feat_ohe_adms_drug.keys())

set(cat_feat_ohe_med.keys())

set(cat_feat_ohe_adms_drug.keys()).intersection(set(cat_feat_ohe_med.keys()))

len(set(cat_feat_ohe_adms_drug.keys()).intersection(set(cat_feat_ohe_diag.keys())))

len(cat_feat_ohe_adms_drug['drughiclseqno'])

len(cat_feat_ohe_med['drughiclseqno'])

len(set(cat_feat_ohe_adms_drug['drughiclseqno']) | set(cat_feat_ohe_med['drughiclseqno']))

cat_feat_ohe = du.utils.merge_dicts([cat_feat_ohe_adms_drug, cat_feat_ohe_med, 
                                     cat_feat_ohe_treat, cat_feat_ohe_diag,
                                     cat_feat_ohe_past_hist, cat_feat_ohe_lab,
                                     cat_feat_ohe_patient, cat_feat_ohe_notes])

len(cat_feat_ohe['drughiclseqno'])

list(cat_feat_ohe.keys())

# Clean the one hot encoded column names, as they are in the final dataframe:

for key, val in cat_feat_ohe.items():
    cat_feat_ohe[key] = du.data_processing.clean_naming(cat_feat_ohe[key], lower_case=False)

cat_feat_ohe

# Save the final encoding dictionary:

stream = open(f'{data_path}eICU_cat_feat_ohe.yml', 'w')
yaml.dump(cat_feat_ohe, stream, default_flow_style=False)

# ## Data types

dtype_dict = {'patientunitstayid': 'uint32',
              'gender': 'UInt8',
              'age': 'float32',
              'admissionheight': 'float32',
              'admissionweight': 'float32',
              'death_ts': 'Int32',
              'ts': 'int32',
              'CAD': 'UInt8',
              'Cancer': 'UInt8',
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

ohe_columns = du.utils.merge_lists(list(cat_feat_ohe.values()))
ohe_columns = du.data_processing.clean_naming(ohe_columns, lower_case=False)

# Add the one hot encoded columns to the dtypes dictionary, specifying them with type `UInt8`

for col in ohe_columns:
    dtype_dict[col] = 'UInt8'

dtype_dict

# Save the data types dictionary:

stream = open(f'{data_path}eICU_dtype_dict.yml', 'w')
yaml.dump(dtype_dict, stream, default_flow_style=False)

# ## Normalization stats
#
# Combine the normalization stats dictionaries of all tables into a single dictionary representative of all the continuous features in the resulting dataframe.

stream_adms_drug = open(f'{data_path}admissionDrug_norm_stats.yml', 'r')
stream_med = open(f'{data_path}medication_norm_stats.yml', 'r')
# stream_lab = open(f'{data_path}lab_norm_stats.yml', 'r')
# stream_patient = open(f'{data_path}patient_norm_stats.yml', 'r')
# stream_vital_aprdc = open(f'{data_path}vitalAperiodic_norm_stats.yml', 'r')
stream_vital_prdc = open(f'{data_path}vitalPeriodic_norm_stats.yml', 'r')

norm_stats_adms_drug = yaml.load(stream_adms_drug, Loader=yaml.FullLoader)
norm_stats_med = yaml.load(stream_med, Loader=yaml.FullLoader)
# norm_stats_lab = yaml.load(stream_lab, Loader=yaml.FullLoader)
# norm_stats_patient = yaml.load(stream_patient, Loader=yaml.FullLoader)
# norm_stats_vital_aprdc = yaml.load(stream_vital_aprdc, Loader=yaml.FullLoader)
norm_stats_vital_prdc = yaml.load(stream_vital_prdc, Loader=yaml.FullLoader)

norm_stats = du.utils.merge_dicts([norm_stats_adms_drug, norm_stats_med,
#                                    norm_stats_lab, norm_stats_patient,
#                                    norm_stats_vital_aprdc, 
                                   norm_stats_vital_prdc])

list(norm_stats.keys())

# Save the normalization stats dictionary:

stream = open(f'{data_path}eICU_norm_stats.yml', 'w')
yaml.dump(norm_stats, stream, default_flow_style=False)


