# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] {"toc-hr-collapsed": false, "Collapsed": "false"}
# # eICU Data Joining
# ---
#
# Reading and joining all preprocessed parts of the eICU dataset from MIT with the data from over 139k patients collected in the US.
#
# The main goal of this notebook is to prepare a single parquet document that contains all the relevant data to be used when training a machine learning model that predicts mortality, joining tables, filtering useless columns and performing imputation.

# + [markdown] {"colab_type": "text", "id": "KOdmFzXqF7nq", "toc-hr-collapsed": true, "Collapsed": "false"}
# ## Importing the necessary packages

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl", "Collapsed": "false", "persistent_id": "522745b5-b5bf-479f-b697-5c7e9e12fc33", "last_executed_text": "import os                                  # os handles directory/workspace changes\nimport numpy as np                         # NumPy to handle numeric and NaN operations\nimport yaml                                # Save and load YAML files", "execution_event_id": "deb57b39-6a79-4b3a-95ed-02f8089ff593", "execution": {"iopub.status.busy": "2020-03-17T05:04:12.632590Z", "iopub.execute_input": "2020-03-17T05:04:12.632895Z", "iopub.status.idle": "2020-03-17T05:04:12.658261Z", "shell.execute_reply.started": "2020-03-17T05:04:12.632850Z", "shell.execute_reply": "2020-03-17T05:04:12.657196Z"}}
import os                                  # os handles directory/workspace changes
import numpy as np                         # NumPy to handle numeric and NaN operations
import yaml                                # Save and load YAML files

# + {"Collapsed": "false", "persistent_id": "02accdbf-be7e-415c-ba11-165906e66c50", "last_executed_text": "# Debugging packages\nimport pixiedust                           # Debugging in Jupyter Notebook cells", "execution_event_id": "fa33a2f7-7127-49c6-bbe9-f89555b1f2be", "execution": {"iopub.status.busy": "2020-03-17T05:04:12.928919Z", "iopub.execute_input": "2020-03-17T05:04:12.929205Z", "iopub.status.idle": "2020-03-17T05:04:13.963384Z", "shell.execute_reply.started": "2020-03-17T05:04:12.929161Z", "shell.execute_reply": "2020-03-17T05:04:13.962156Z"}}
# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# + {"Collapsed": "false", "persistent_id": "a1f6ee7f-36d4-489d-b2dd-ec2a38f15d11", "last_executed_text": "# Change to parent directory (presumably \"Documents\")\nos.chdir(\"../../..\")\n\n# Path to the parquet dataset files\ndata_path = 'Documents/Datasets/Thesis/eICU/'\n\n# Path to the code files\nproject_path = 'Documents/GitHub/eICU-mortality-prediction/'", "execution_event_id": "baeb346a-1c34-42d1-a501-7ae37369255e", "execution": {"iopub.status.busy": "2020-03-17T05:04:13.964818Z", "iopub.execute_input": "2020-03-17T05:04:13.965104Z", "iopub.status.idle": "2020-03-17T05:04:13.969606Z", "shell.execute_reply.started": "2020-03-17T05:04:13.965052Z", "shell.execute_reply": "2020-03-17T05:04:13.968670Z"}}
# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# Path to the parquet dataset files
data_path = 'data/eICU/cleaned/'
# Path to the code files
project_path = 'code/eICU-mortality-prediction/'

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-17T05:04:13.971271Z", "iopub.execute_input": "2020-03-17T05:04:13.971543Z", "iopub.status.idle": "2020-03-17T05:04:14.231019Z", "shell.execute_reply.started": "2020-03-17T05:04:13.971494Z", "shell.execute_reply": "2020-03-17T05:04:14.229698Z"}}
# Make sure that every large operation can be handled, by using the disk as an overflow for the memory
# !export MODIN_OUT_OF_CORE=true
# Another trick to do with Pandas so as to be able to allocate bigger objects to memory
# !sudo bash -c 'echo 1 > /proc/sys/vm/overcommit_memory'

# + {"Collapsed": "false", "persistent_id": "c0c2e356-d4f4-4a9d-bec2-88bdf9eb6a38", "last_executed_text": "import modin.pandas as pd                  # Optimized distributed version of Pandas\nimport data_utils as du                    # Data science and machine learning relevant methods", "execution_event_id": "82ef68be-443a-4bb8-8abd-7457a7005b4d", "execution": {"iopub.status.busy": "2020-03-17T05:04:14.232763Z", "iopub.execute_input": "2020-03-17T05:04:14.233040Z", "iopub.status.idle": "2020-03-17T05:04:16.754402Z", "shell.execute_reply.started": "2020-03-17T05:04:14.232988Z", "shell.execute_reply": "2020-03-17T05:04:16.753405Z"}}
import modin.pandas as mpd                  # Optimized distributed version of Pandas
import pandas as pd
import data_utils as du                    # Data science and machine learning relevant methods
# -

# Allow pandas to show more columns:

# + {"execution": {"iopub.status.busy": "2020-03-17T05:04:16.755997Z", "iopub.execute_input": "2020-03-17T05:04:16.756363Z", "iopub.status.idle": "2020-03-17T05:04:16.765786Z", "shell.execute_reply.started": "2020-03-17T05:04:16.756295Z", "shell.execute_reply": "2020-03-17T05:04:16.764781Z"}}
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

# + [markdown] {"Collapsed": "false"}
# Set the random seed for reproducibility

# + {"Collapsed": "false", "persistent_id": "39b552cd-6948-4ec8-ac04-42f850c1e05a", "last_executed_text": "du.set_random_seed(42)", "execution_event_id": "29ab85ce-b7fd-4c5a-a110-5841e741c369", "execution": {"iopub.status.busy": "2020-03-17T05:04:16.767511Z", "iopub.execute_input": "2020-03-17T05:04:16.767795Z", "iopub.status.idle": "2020-03-17T05:04:16.774038Z", "shell.execute_reply.started": "2020-03-17T05:04:16.767750Z", "shell.execute_reply": "2020-03-17T05:04:16.773051Z"}}
du.set_random_seed(42)
# -

# ## Initializing variables

# + {"execution": {"iopub.status.busy": "2020-03-17T05:04:16.776054Z", "iopub.execute_input": "2020-03-17T05:04:16.776356Z", "iopub.status.idle": "2020-03-17T05:04:16.788073Z", "shell.execute_reply.started": "2020-03-17T05:04:16.776289Z", "shell.execute_reply": "2020-03-17T05:04:16.787076Z"}}
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

# + [markdown] {"Collapsed": "false"}
# Load the lists of one hot encoded columns:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-17T05:04:16.789642Z", "iopub.execute_input": "2020-03-17T05:04:16.789976Z", "iopub.status.idle": "2020-03-17T05:04:16.798590Z", "shell.execute_reply.started": "2020-03-17T05:04:16.789923Z", "shell.execute_reply": "2020-03-17T05:04:16.797721Z"}}
# stream_adms_drug = open(f'{data_path}cat_feat_ohe_adms_drug.yaml', 'r')
# stream_inf_drug = open(f'{data_path}cat_feat_ohe_inf_drug.yaml', 'r')
stream_med = open(f'{data_path}cat_feat_ohe_med.yaml', 'r')
stream_treat = open(f'{data_path}cat_feat_ohe_treat.yaml', 'r')
stream_diag = open(f'{data_path}cat_feat_ohe_diag.yaml', 'r')
# stream_alrg = open(f'{data_path}cat_feat_ohe_alrg.yaml', 'r')
stream_past_hist = open(f'{data_path}cat_feat_ohe_past_hist.yaml', 'r')
stream_lab = open(f'{data_path}cat_feat_ohe_lab.yaml', 'r')
stream_patient = open(f'{data_path}cat_feat_ohe_patient.yaml', 'r')
stream_notes = open(f'{data_path}cat_feat_ohe_note.yaml', 'r')

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-17T05:04:16.799653Z", "iopub.status.idle": "2020-03-17T05:04:17.051304Z", "iopub.execute_input": "2020-03-17T05:04:16.799881Z", "shell.execute_reply.started": "2020-03-17T05:04:16.799836Z", "shell.execute_reply": "2020-03-17T05:04:17.050427Z"}}
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

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-17T05:04:17.052444Z", "iopub.status.idle": "2020-03-17T05:04:17.057414Z", "iopub.execute_input": "2020-03-17T05:04:17.052704Z", "shell.execute_reply.started": "2020-03-17T05:04:17.052657Z", "shell.execute_reply": "2020-03-17T05:04:17.056552Z"}}
cat_feat_ohe = du.utils.merge_dicts([cat_feat_ohe_adms_drug, cat_feat_ohe_inf_drug,
                                     cat_feat_ohe_med, cat_feat_ohe_treat,
                                     cat_feat_ohe_diag, cat_feat_ohe_alrg,
                                     cat_feat_ohe_past_hist, cat_feat_ohe_lab,
                                     cat_feat_ohe_patient, cat_feat_ohe_notes])

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-17T05:04:17.059324Z", "iopub.status.idle": "2020-03-17T05:04:17.066908Z", "iopub.execute_input": "2020-03-17T05:04:17.059753Z", "shell.execute_reply.started": "2020-03-17T05:04:17.059683Z", "shell.execute_reply": "2020-03-17T05:04:17.065392Z"}}
ohe_columns = du.utils.merge_lists(list(cat_feat_ohe.values()))
ohe_columns = du.data_processing.clean_naming(ohe_columns, lower_case=False)

# + [markdown] {"Collapsed": "false"}
# Add the one hot encoded columns to the dtypes dictionary, specifying them with type `UInt8`

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-17T05:04:18.098450Z", "iopub.status.idle": "2020-03-17T05:04:18.104125Z", "iopub.execute_input": "2020-03-17T05:04:18.098761Z", "shell.execute_reply.started": "2020-03-17T05:04:18.098707Z", "shell.execute_reply": "2020-03-17T05:04:18.102797Z"}}
for col in ohe_columns:
    dtype_dict[col] = 'UInt8'

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-17T05:04:51.160325Z", "iopub.execute_input": "2020-03-17T05:04:51.160650Z", "iopub.status.idle": "2020-03-17T05:04:51.180638Z", "shell.execute_reply.started": "2020-03-17T05:04:51.160591Z", "shell.execute_reply": "2020-03-17T05:04:51.179498Z"}}
dtype_dict

# + [markdown] {"Collapsed": "false"}
# ## Loading the data

# + [markdown] {"Collapsed": "false"}
# ### Patient information

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-17T05:05:20.998312Z", "iopub.execute_input": "2020-03-17T05:05:20.998608Z", "iopub.status.idle": "2020-03-17T05:05:22.129647Z", "shell.execute_reply.started": "2020-03-17T05:05:20.998563Z", "shell.execute_reply": "2020-03-17T05:05:22.128691Z"}}
patient_df = mpd.read_csv(f'{data_path}normalized/ohe/patient.csv', dtype=dtype_dict)
patient_df = du.utils.convert_dataframe(patient_df, to='pandas', return_library=False, dtypes=dtype_dict)
patient_df = patient_df.drop(columns='Unnamed: 0')
patient_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-17T05:04:22.707095Z", "iopub.status.idle": "2020-03-17T05:04:22.707700Z", "iopub.execute_input": "2020-03-17T02:12:19.851527Z", "shell.execute_reply.started": "2020-03-17T02:12:19.851463Z", "shell.execute_reply": "2020-03-17T02:12:20.044451Z"}}
note_df = mpd.read_csv(f'{data_path}normalized/ohe/note.csv', dtype=dtype_dict)
note_df = du.utils.convert_dataframe(note_df, to='pandas', return_library=False, dtypes=dtype_dict)
note_df = note_df.drop(columns='Unnamed: 0')
note_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Diagnosis

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-17T02:18:33.721453Z", "iopub.execute_input": "2020-03-17T02:18:33.721778Z", "iopub.status.idle": "2020-03-17T02:21:02.390931Z", "shell.execute_reply.started": "2020-03-17T02:18:33.721720Z", "shell.execute_reply": "2020-03-17T02:21:02.390002Z"}}
diagns_df = mpd.read_csv(f'{data_path}normalized/ohe/diagnosis.csv', dtype=dtype_dict)
diagns_df = du.utils.convert_dataframe(diagns_df, to='pandas', return_library=False, dtypes=dtype_dict)
diagns_df = diagns_df.drop(columns='Unnamed: 0')
diagns_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T19:00:14.051303Z", "iopub.execute_input": "2020-03-16T19:00:14.051698Z", "iopub.status.idle": "2020-03-16T19:00:21.933263Z", "shell.execute_reply.started": "2020-03-16T19:00:14.051642Z", "shell.execute_reply": "2020-03-16T19:00:21.932045Z"}}
# alrg_df = mpd.read_csv(f'{data_path}normalized/ohe/allergy.csv', dtype=dtype_dict)
# alrg_df = du.utils.convert_dataframe(alrg_df, to='pandas', return_library=False, dtypes=dtype_dict)
# alrg_df = alrg_df.drop(columns='Unnamed: 0')
# alrg_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T19:00:21.934520Z", "iopub.execute_input": "2020-03-16T19:00:21.934850Z", "iopub.status.idle": "2020-03-16T19:00:33.637145Z", "shell.execute_reply.started": "2020-03-16T19:00:21.934806Z", "shell.execute_reply": "2020-03-16T19:00:33.636199Z"}}
past_hist_df = mpd.read_csv(f'{data_path}normalized/ohe/pastHistory.csv', dtype=dtype_dict)
past_hist_df = du.utils.convert_dataframe(past_hist_df, to='pandas', return_library=False, dtypes=dtype_dict)
past_hist_df = past_hist_df.drop(columns='Unnamed: 0')
past_hist_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Treatments

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T19:00:33.657779Z", "iopub.execute_input": "2020-03-16T19:00:33.658038Z", "iopub.status.idle": "2020-03-16T19:01:32.449135Z", "shell.execute_reply.started": "2020-03-16T19:00:33.657989Z", "shell.execute_reply": "2020-03-16T19:01:32.447991Z"}}
treat_df = mpd.read_csv(f'{data_path}normalized/ohe/treatment.csv', dtype=dtype_dict)
treat_df = du.utils.convert_dataframe(treat_df, to='pandas', return_library=False, dtypes=dtype_dict)
treat_df = treat_df.drop(columns='Unnamed: 0')
treat_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T19:01:32.450530Z", "iopub.execute_input": "2020-03-16T19:01:32.450844Z", "iopub.status.idle": "2020-03-16T19:01:36.426296Z", "shell.execute_reply.started": "2020-03-16T19:01:32.450786Z", "shell.execute_reply": "2020-03-16T19:01:36.425424Z"}}
adms_drug_df = mpd.read_csv(f'{data_path}normalized/ohe/admissionDrug.csv', dtype=dtype_dict)
adms_drug_df = du.utils.convert_dataframe(adms_drug_df, to='pandas', return_library=False, dtypes=dtype_dict)
adms_drug_df = adms_drug_df.drop(columns='Unnamed: 0')
adms_drug_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T19:19:22.198988Z", "iopub.execute_input": "2020-03-16T19:19:22.199248Z", "iopub.status.idle": "2020-03-16T19:20:27.980609Z", "shell.execute_reply.started": "2020-03-16T19:19:22.199195Z", "shell.execute_reply": "2020-03-16T19:20:27.978951Z"}}
# inf_drug_df = mpd.read_csv(f'{data_path}normalized/ohe/infusionDrug.csv', dtype=dtype_dict)
# inf_drug_df = du.utils.convert_dataframe(inf_drug_df, to='pandas', return_library=False, dtypes=dtype_dict)
# inf_drug_df = inf_drug_df.drop(columns='Unnamed: 0')
# inf_drug_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T05:24:19.815390Z", "iopub.execute_input": "2020-03-15T05:24:19.815588Z", "iopub.status.idle": "2020-03-15T05:25:38.111300Z", "shell.execute_reply.started": "2020-03-15T05:24:19.815551Z", "shell.execute_reply": "2020-03-15T05:25:38.110165Z"}}
med_df = mpd.read_csv(f'{data_path}normalized/ohe/medication.csv', dtype=dtype_dict)
med_df = du.utils.convert_dataframe(med_df, to='pandas', return_library=False, dtypes=dtype_dict)
med_df = med_df.drop(columns='Unnamed: 0')
med_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T14:14:52.586985Z", "iopub.execute_input": "2020-03-15T14:14:52.587453Z", "iopub.status.idle": "2020-03-15T14:17:00.683297Z", "shell.execute_reply.started": "2020-03-15T14:14:52.587405Z", "shell.execute_reply": "2020-03-15T14:17:00.682629Z"}}
# in_out_df = mpd.read_csv(f'{data_path}normalized/intakeOutput.csv', dtype=dtype_dict)
# in_out_df = du.utils.convert_dataframe(in_out_df, to='pandas', return_library=False, dtypes=dtype_dict)
# in_out_df = in_out_df.drop(columns='Unnamed: 0')
# in_out_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Nursing data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:14.978567Z", "iopub.status.idle": "2020-03-15T04:51:14.978880Z", "iopub.execute_input": "2020-03-15T02:14:05.712354Z", "shell.execute_reply.started": "2020-03-15T02:14:05.712316Z", "shell.execute_reply": "2020-03-15T02:14:05.714353Z"}}
# nurse_care_df = mpd.read_csv(f'{data_path}normalized/ohe/nurseCare.csv')
# nurse_care_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:14.979722Z", "iopub.status.idle": "2020-03-15T04:51:14.980057Z", "iopub.execute_input": "2020-03-15T02:14:05.716226Z", "shell.execute_reply.started": "2020-03-15T02:14:05.716191Z", "shell.execute_reply": "2020-03-15T02:14:05.718753Z"}}
# nurse_assess_df = mpd.read_csv(f'{data_path}normalized/ohe/nurseAssessment.csv')
# nurse_assess_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:14.980824Z", "iopub.status.idle": "2020-03-15T04:51:14.981136Z", "iopub.execute_input": "2020-03-15T02:14:05.720500Z", "shell.execute_reply.started": "2020-03-15T02:14:05.720462Z", "shell.execute_reply": "2020-03-15T02:14:05.723574Z"}}
# nurse_care_df = nurse_care_df.drop(columns='Unnamed: 0')
# nurse_assess_df = nurse_assess_df.drop(columns='Unnamed: 0')

# + [markdown] {"Collapsed": "false"}
# ### Respiratory data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T19:47:09.343476Z", "iopub.status.idle": "2020-03-15T19:47:10.917576Z", "iopub.execute_input": "2020-03-15T19:47:09.343759Z", "shell.execute_reply.started": "2020-03-15T19:47:09.343714Z", "shell.execute_reply": "2020-03-15T19:47:10.916896Z"}}
resp_care_df = mpd.read_csv(f'{data_path}normalized/ohe/respiratoryCare.csv', dtype=dtype_dict)
resp_care_df = du.utils.convert_dataframe(resp_care_df, to='pandas', return_library=False, dtypes=dtype_dict)
resp_care_df = resp_care_df.drop(columns='Unnamed: 0')
resp_care_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Vital signals

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T02:53:29.992331Z", "iopub.status.idle": "2020-03-16T02:53:32.936100Z", "iopub.execute_input": "2020-03-16T02:53:29.992664Z", "shell.execute_reply.started": "2020-03-16T02:53:29.992604Z", "shell.execute_reply": "2020-03-16T02:53:32.935087Z"}}
vital_aprdc_df = mpd.read_csv(f'{data_path}normalized/vitalAperiodic.csv', dtype=dtype_dict)
vital_aprdc_df = du.utils.convert_dataframe(vital_aprdc_df, to='pandas', return_library=False, dtypes=dtype_dict)
vital_aprdc_df = vital_aprdc_df.drop(columns='Unnamed: 0')
vital_aprdc_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T02:43:09.710064Z", "iopub.status.idle": "2020-03-15T04:52:12.341001Z", "iopub.execute_input": "2020-03-16T02:43:09.710278Z", "shell.execute_reply.started": "2020-03-15T04:51:36.795517Z", "shell.execute_reply": "2020-03-15T04:52:12.340358Z"}}
vital_prdc_df = mpd.read_csv(f'{data_path}normalized/vitalPeriodic.csv', dtype=dtype_dict)
vital_prdc_df = du.utils.convert_dataframe(vital_prdc_df, to='pandas', return_library=False, dtypes=dtype_dict)
vital_prdc_df = vital_prdc_df.drop(columns='Unnamed: 0')
vital_prdc_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Exams data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:19:23.890361Z", "iopub.status.idle": "2020-03-15T02:21:58.259579Z", "iopub.execute_input": "2020-03-15T02:19:23.890554Z", "shell.execute_reply.started": "2020-03-15T02:19:23.890518Z", "shell.execute_reply": "2020-03-15T02:21:58.258747Z"}}
lab_df = mpd.read_csv(f'{data_path}normalized/ohe/lab.csv', dtype=dtype_dict)
lab_df = du.utils.convert_dataframe(lab_df, to='pandas', return_library=False, dtypes=dtype_dict)
lab_df = lab_df.drop(columns='Unnamed: 0')
lab_df.head()

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": false}
# ## Joining dataframes

# + [markdown] {"Collapsed": "false"}
# ### Checking the matching of unit stays IDs

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:39.678332Z", "iopub.status.idle": "2020-02-26T17:06:40.093070Z", "iopub.execute_input": "2020-02-26T17:06:39.678558Z", "shell.execute_reply.started": "2020-02-26T17:06:39.678520Z", "shell.execute_reply": "2020-02-26T17:06:40.092337Z"}}
full_stays_list = set(patient_df.patientunitstayid.unique())

# + [markdown] {"Collapsed": "false"}
# Total number of unit stays:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:40.094402Z", "iopub.status.idle": "2020-02-26T17:06:40.099961Z", "iopub.execute_input": "2020-02-26T17:06:40.094749Z", "shell.execute_reply.started": "2020-02-26T17:06:40.094687Z", "shell.execute_reply": "2020-02-26T17:06:40.099190Z"}}
len(full_stays_list)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:40.101255Z", "iopub.status.idle": "2020-02-26T17:06:40.226963Z", "iopub.execute_input": "2020-02-26T17:06:40.101929Z", "shell.execute_reply.started": "2020-02-26T17:06:40.101857Z", "shell.execute_reply": "2020-02-26T17:06:40.225874Z"}}
note_stays_list = set(note_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:40.228664Z", "iopub.status.idle": "2020-02-26T17:06:40.234264Z", "iopub.execute_input": "2020-02-26T17:06:40.229023Z", "shell.execute_reply.started": "2020-02-26T17:06:40.228961Z", "shell.execute_reply": "2020-02-26T17:06:40.233560Z"}}
len(note_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have note data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:40.235590Z", "iopub.status.idle": "2020-02-26T17:06:40.253443Z", "iopub.execute_input": "2020-02-26T17:06:40.235905Z", "shell.execute_reply.started": "2020-02-26T17:06:40.235847Z", "shell.execute_reply": "2020-02-26T17:06:40.252738Z"}}
len(set.intersection(full_stays_list, note_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:40.254692Z", "iopub.status.idle": "2020-02-26T17:06:40.648925Z", "iopub.execute_input": "2020-02-26T17:06:40.254923Z", "shell.execute_reply.started": "2020-02-26T17:06:40.254868Z", "shell.execute_reply": "2020-02-26T17:06:40.648116Z"}}
diagns_stays_list = set(diagns_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:40.649995Z", "iopub.status.idle": "2020-02-26T17:06:40.654306Z", "iopub.execute_input": "2020-02-26T17:06:40.650198Z", "shell.execute_reply.started": "2020-02-26T17:06:40.650161Z", "shell.execute_reply": "2020-02-26T17:06:40.653650Z"}}
len(diagns_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have diagnosis data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:40.655215Z", "iopub.status.idle": "2020-02-26T17:06:40.704945Z", "iopub.execute_input": "2020-02-26T17:06:40.655413Z", "shell.execute_reply.started": "2020-02-26T17:06:40.655377Z", "shell.execute_reply": "2020-02-26T17:06:40.704010Z"}}
len(set.intersection(full_stays_list, diagns_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:40.705977Z", "iopub.status.idle": "2020-02-26T17:06:40.880141Z", "iopub.execute_input": "2020-02-26T17:06:40.706193Z", "shell.execute_reply.started": "2020-02-26T17:06:40.706156Z", "shell.execute_reply": "2020-02-26T17:06:40.879409Z"}}
# alrg_stays_list = set(alrg_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:40.881198Z", "iopub.status.idle": "2020-02-26T17:06:40.885978Z", "iopub.execute_input": "2020-02-26T17:06:40.881405Z", "shell.execute_reply.started": "2020-02-26T17:06:40.881369Z", "shell.execute_reply": "2020-02-26T17:06:40.885401Z"}}
# len(alrg_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have allergy data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:40.886881Z", "iopub.status.idle": "2020-02-26T17:06:40.910256Z", "iopub.execute_input": "2020-02-26T17:06:40.887083Z", "shell.execute_reply.started": "2020-02-26T17:06:40.887046Z", "shell.execute_reply": "2020-02-26T17:06:40.909552Z"}}
# len(set.intersection(full_stays_list, alrg_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:40.911370Z", "iopub.status.idle": "2020-02-26T17:06:41.116675Z", "iopub.execute_input": "2020-02-26T17:06:40.911618Z", "shell.execute_reply.started": "2020-02-26T17:06:40.911579Z", "shell.execute_reply": "2020-02-26T17:06:41.115974Z"}}
past_hist_stays_list = set(past_hist_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:41.117749Z", "iopub.status.idle": "2020-02-26T17:06:41.122087Z", "iopub.execute_input": "2020-02-26T17:06:41.117964Z", "shell.execute_reply.started": "2020-02-26T17:06:41.117925Z", "shell.execute_reply": "2020-02-26T17:06:41.121497Z"}}
len(past_hist_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have past history data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:41.123031Z", "iopub.status.idle": "2020-02-26T17:06:41.155381Z", "iopub.execute_input": "2020-02-26T17:06:41.123220Z", "shell.execute_reply.started": "2020-02-26T17:06:41.123186Z", "shell.execute_reply": "2020-02-26T17:06:41.154649Z"}}
len(set.intersection(full_stays_list, past_hist_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:41.156558Z", "iopub.status.idle": "2020-02-26T17:06:41.531138Z", "iopub.execute_input": "2020-02-26T17:06:41.156784Z", "shell.execute_reply.started": "2020-02-26T17:06:41.156745Z", "shell.execute_reply": "2020-02-26T17:06:41.530382Z"}}
treat_stays_list = set(treat_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:41.532226Z", "iopub.status.idle": "2020-02-26T17:06:41.536740Z", "iopub.execute_input": "2020-02-26T17:06:41.532436Z", "shell.execute_reply.started": "2020-02-26T17:06:41.532398Z", "shell.execute_reply": "2020-02-26T17:06:41.536123Z"}}
len(treat_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have treatment data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:41.537742Z", "iopub.status.idle": "2020-02-26T17:06:41.577114Z", "iopub.execute_input": "2020-02-26T17:06:41.538002Z", "shell.execute_reply.started": "2020-02-26T17:06:41.537957Z", "shell.execute_reply": "2020-02-26T17:06:41.576445Z"}}
len(set.intersection(full_stays_list, treat_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:41.578084Z", "iopub.status.idle": "2020-02-26T17:06:41.743253Z", "iopub.execute_input": "2020-02-26T17:06:41.578290Z", "shell.execute_reply.started": "2020-02-26T17:06:41.578253Z", "shell.execute_reply": "2020-02-26T17:06:41.742093Z"}}
adms_drug_stays_list = set(adms_drug_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:41.744209Z", "iopub.status.idle": "2020-02-26T17:06:41.748862Z", "iopub.execute_input": "2020-02-26T17:06:41.744412Z", "shell.execute_reply.started": "2020-02-26T17:06:41.744375Z", "shell.execute_reply": "2020-02-26T17:06:41.748142Z"}}
len(adms_drug_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have admission drug data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:41.749745Z", "iopub.status.idle": "2020-02-26T17:06:41.764151Z", "iopub.execute_input": "2020-02-26T17:06:41.749931Z", "shell.execute_reply.started": "2020-02-26T17:06:41.749897Z", "shell.execute_reply": "2020-02-26T17:06:41.763601Z"}}
len(set.intersection(full_stays_list, adms_drug_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:41.764961Z", "iopub.status.idle": "2020-02-26T17:06:42.309915Z", "iopub.execute_input": "2020-02-26T17:06:41.765156Z", "shell.execute_reply.started": "2020-02-26T17:06:41.765122Z", "shell.execute_reply": "2020-02-26T17:06:42.309105Z"}}
# inf_drug_stays_list = set(inf_drug_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:42.311041Z", "iopub.status.idle": "2020-02-26T17:06:42.315850Z", "iopub.execute_input": "2020-02-26T17:06:42.311311Z", "shell.execute_reply.started": "2020-02-26T17:06:42.311265Z", "shell.execute_reply": "2020-02-26T17:06:42.315150Z"}}
# len(inf_drug_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have infusion drug data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:42.316935Z", "iopub.status.idle": "2020-02-26T17:06:42.341538Z", "iopub.execute_input": "2020-02-26T17:06:42.317160Z", "shell.execute_reply.started": "2020-02-26T17:06:42.317119Z", "shell.execute_reply": "2020-02-26T17:06:42.340907Z"}}
# len(set.intersection(full_stays_list, inf_drug_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:42.342546Z", "iopub.status.idle": "2020-02-26T17:06:43.217103Z", "iopub.execute_input": "2020-02-26T17:06:42.342753Z", "shell.execute_reply.started": "2020-02-26T17:06:42.342716Z", "shell.execute_reply": "2020-02-26T17:06:43.216344Z"}}
med_stays_list = set(med_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:43.218112Z", "iopub.status.idle": "2020-02-26T17:06:43.222301Z", "iopub.execute_input": "2020-02-26T17:06:43.218322Z", "shell.execute_reply.started": "2020-02-26T17:06:43.218285Z", "shell.execute_reply": "2020-02-26T17:06:43.221665Z"}}
len(med_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have medication data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:43.223261Z", "iopub.status.idle": "2020-02-26T17:06:43.268485Z", "iopub.execute_input": "2020-02-26T17:06:43.223466Z", "shell.execute_reply.started": "2020-02-26T17:06:43.223429Z", "shell.execute_reply": "2020-02-26T17:06:43.267653Z"}}
len(set.intersection(full_stays_list, med_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:43.269468Z", "iopub.status.idle": "2020-02-26T17:06:44.458225Z", "iopub.execute_input": "2020-02-26T17:06:43.269688Z", "shell.execute_reply.started": "2020-02-26T17:06:43.269651Z", "shell.execute_reply": "2020-02-26T17:06:44.457486Z"}}
# in_out_stays_list = set(in_out_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:44.459345Z", "iopub.status.idle": "2020-02-26T17:06:44.464285Z", "iopub.execute_input": "2020-02-26T17:06:44.459571Z", "shell.execute_reply.started": "2020-02-26T17:06:44.459530Z", "shell.execute_reply": "2020-02-26T17:06:44.463605Z"}}
# len(in_out_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have intake and output data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:44.465226Z", "iopub.status.idle": "2020-02-26T17:06:44.842575Z", "iopub.execute_input": "2020-02-26T17:06:44.465431Z", "shell.execute_reply.started": "2020-02-26T17:06:44.465394Z", "shell.execute_reply": "2020-02-26T17:06:44.841735Z"}}
# len(set.intersection(full_stays_list, in_out_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:44.843620Z", "iopub.status.idle": "2020-02-26T17:06:44.846905Z", "iopub.execute_input": "2020-02-26T17:06:44.844142Z", "shell.execute_reply.started": "2020-02-26T17:06:44.844096Z", "shell.execute_reply": "2020-02-26T17:06:44.846256Z"}}
# nurse_care_stays_list = set(nurse_care_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:44.847765Z", "iopub.status.idle": "2020-02-26T17:06:44.852734Z", "iopub.execute_input": "2020-02-26T17:06:44.847965Z", "shell.execute_reply.started": "2020-02-26T17:06:44.847929Z", "shell.execute_reply": "2020-02-26T17:06:44.852149Z"}}
# len(nurse_care_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have nurse care data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:44.853502Z", "iopub.status.idle": "2020-02-26T17:06:44.857298Z", "iopub.execute_input": "2020-02-26T17:06:44.853706Z", "shell.execute_reply.started": "2020-02-26T17:06:44.853668Z", "shell.execute_reply": "2020-02-26T17:06:44.856683Z"}}
# len(set.intersection(full_stays_list, nurse_care_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:44.858075Z", "iopub.status.idle": "2020-02-26T17:06:44.861888Z", "iopub.execute_input": "2020-02-26T17:06:44.858259Z", "shell.execute_reply.started": "2020-02-26T17:06:44.858226Z", "shell.execute_reply": "2020-02-26T17:06:44.861335Z"}}
# nurse_assess_stays_list = set(nurse_assess_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:44.881703Z", "iopub.status.idle": "2020-02-26T17:06:44.884822Z", "iopub.execute_input": "2020-02-26T17:06:44.881939Z", "shell.execute_reply.started": "2020-02-26T17:06:44.881899Z", "shell.execute_reply": "2020-02-26T17:06:44.884167Z"}}
# len(nurse_assess_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have nurse assessment data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:44.887122Z", "iopub.status.idle": "2020-02-26T17:06:44.890836Z", "iopub.execute_input": "2020-02-26T17:06:44.887349Z", "shell.execute_reply.started": "2020-02-26T17:06:44.887299Z", "shell.execute_reply": "2020-02-26T17:06:44.889727Z"}}
# len(set.intersection(full_stays_list, nurse_assess_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:44.891668Z", "iopub.status.idle": "2020-02-26T17:06:45.008553Z", "iopub.execute_input": "2020-02-26T17:06:44.891859Z", "shell.execute_reply.started": "2020-02-26T17:06:44.891825Z", "shell.execute_reply": "2020-02-26T17:06:45.007454Z"}}
resp_care_stays_list = set(resp_care_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:45.009556Z", "iopub.status.idle": "2020-02-26T17:06:45.013955Z", "iopub.execute_input": "2020-02-26T17:06:45.009763Z", "shell.execute_reply.started": "2020-02-26T17:06:45.009726Z", "shell.execute_reply": "2020-02-26T17:06:45.013334Z"}}
len(resp_care_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have respiratory care data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:45.014823Z", "iopub.status.idle": "2020-02-26T17:06:45.032842Z", "iopub.execute_input": "2020-02-26T17:06:45.015043Z", "shell.execute_reply.started": "2020-02-26T17:06:45.015006Z", "shell.execute_reply": "2020-02-26T17:06:45.032224Z"}}
len(set.intersection(full_stays_list, resp_care_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:45.033691Z", "iopub.status.idle": "2020-02-26T17:06:48.311737Z", "iopub.execute_input": "2020-02-26T17:06:45.033892Z", "shell.execute_reply.started": "2020-02-26T17:06:45.033849Z", "shell.execute_reply": "2020-02-26T17:06:48.310937Z"}}
vital_aprdc_stays_list = set(vital_aprdc_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:48.312988Z", "iopub.status.idle": "2020-02-26T17:06:48.317433Z", "iopub.execute_input": "2020-02-26T17:06:48.313221Z", "shell.execute_reply.started": "2020-02-26T17:06:48.313180Z", "shell.execute_reply": "2020-02-26T17:06:48.316834Z"}}
len(vital_aprdc_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have vital aperiodic data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:48.318331Z", "iopub.status.idle": "2020-02-26T17:06:48.373587Z", "iopub.execute_input": "2020-02-26T17:06:48.318552Z", "shell.execute_reply.started": "2020-02-26T17:06:48.318516Z", "shell.execute_reply": "2020-02-26T17:06:48.372779Z"}}
len(set.intersection(full_stays_list, vital_aprdc_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:48.374671Z", "iopub.status.idle": "2020-02-26T17:07:13.514154Z", "iopub.execute_input": "2020-02-26T17:06:48.374897Z", "shell.execute_reply.started": "2020-02-26T17:06:48.374855Z", "shell.execute_reply": "2020-02-26T17:07:13.513440Z"}}
vital_prdc_stays_list = set(vital_prdc_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:07:13.515189Z", "iopub.status.idle": "2020-02-26T17:07:13.519472Z", "iopub.execute_input": "2020-02-26T17:07:13.515395Z", "shell.execute_reply.started": "2020-02-26T17:07:13.515357Z", "shell.execute_reply": "2020-02-26T17:07:13.518805Z"}}
len(vital_prdc_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have vital periodic data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:07:13.520320Z", "iopub.status.idle": "2020-02-26T17:07:22.498386Z", "iopub.execute_input": "2020-02-26T17:07:13.520519Z", "shell.execute_reply.started": "2020-02-26T17:07:13.520484Z", "shell.execute_reply": "2020-02-26T17:07:22.497724Z"}}
len(set.intersection(full_stays_list, vital_prdc_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:07:22.499818Z", "iopub.status.idle": "2020-02-26T17:07:26.687838Z", "iopub.execute_input": "2020-02-26T17:07:22.500158Z", "shell.execute_reply.started": "2020-02-26T17:07:22.500117Z", "shell.execute_reply": "2020-02-26T17:07:26.687020Z"}}
lab_stays_list = set(lab_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:07:26.688879Z", "iopub.status.idle": "2020-02-26T17:07:26.693687Z", "iopub.execute_input": "2020-02-26T17:07:26.689281Z", "shell.execute_reply.started": "2020-02-26T17:07:26.689058Z", "shell.execute_reply": "2020-02-26T17:07:26.693035Z"}}
len(lab_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have lab data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:07:26.694609Z", "iopub.status.idle": "2020-02-26T17:07:27.104262Z", "iopub.execute_input": "2020-02-26T17:07:26.694807Z", "shell.execute_reply.started": "2020-02-26T17:07:26.694770Z", "shell.execute_reply": "2020-02-26T17:07:27.103549Z"}}
len(set.intersection(full_stays_list, lab_stays_list))

# + [markdown] {"Collapsed": "false"}
# ### Joining patient with note data

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-17T02:12:32.587992Z", "iopub.execute_input": "2020-03-17T02:12:32.588262Z", "iopub.status.idle": "2020-03-17T02:12:32.643325Z", "shell.execute_reply.started": "2020-03-17T02:12:32.588222Z", "shell.execute_reply": "2020-03-17T02:12:32.642479Z"}}
eICU_df = patient_df
eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
note_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# + [markdown] {"Collapsed": "false"}
# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-17T02:12:34.917364Z", "iopub.status.idle": "2020-03-17T02:12:35.706182Z", "iopub.execute_input": "2020-03-17T02:12:34.917662Z", "shell.execute_reply.started": "2020-03-17T02:12:34.917611Z", "shell.execute_reply": "2020-03-17T02:12:35.705426Z"}}
eICU_df = eICU_df.join(note_df, how='left')
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-17T02:00:24.902573Z", "iopub.execute_input": "2020-03-17T02:00:24.902960Z", "iopub.status.idle": "2020-03-17T02:00:26.259042Z", "shell.execute_reply.started": "2020-03-17T02:00:24.902909Z", "shell.execute_reply": "2020-03-17T02:00:26.257437Z"}}
# eICU_df, pd = du.utils.convert_dataframe(eICU_df, to='pandas')

# + {"execution": {"iopub.status.busy": "2020-03-17T02:08:20.974931Z", "iopub.execute_input": "2020-03-17T02:08:20.975295Z", "iopub.status.idle": "2020-03-17T02:08:21.192568Z", "shell.execute_reply.started": "2020-03-17T02:08:20.975233Z", "shell.execute_reply": "2020-03-17T02:08:21.190944Z"}}
# eICU_df[(eICU_df.patientunitstayid == 838771) & (eICU_df.ts == 0)]

# + {"execution": {"iopub.status.busy": "2020-03-17T02:08:52.916163Z", "iopub.execute_input": "2020-03-17T02:08:52.916529Z", "iopub.status.idle": "2020-03-17T02:08:53.073988Z", "shell.execute_reply.started": "2020-03-17T02:08:52.916463Z", "shell.execute_reply": "2020-03-17T02:08:53.072787Z"}}
# eICU_df[(eICU_df.patientunitstayid == 2696741) & (eICU_df.ts == 0)]

# +
# eICU_df.groupby(['patientunitstayid', 'ts']).size().value_counts()

# + {"execution": {"iopub.status.busy": "2020-03-17T02:10:21.326289Z", "iopub.execute_input": "2020-03-17T02:10:21.326698Z", "iopub.status.idle": "2020-03-17T02:10:21.506281Z", "shell.execute_reply.started": "2020-03-17T02:10:21.326589Z", "shell.execute_reply": "2020-03-17T02:10:21.504593Z"}}
# eICU_df = eICU_df.groupby(['patientunitstayid', 'ts']).first()
# eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-17T02:10:31.909109Z", "iopub.execute_input": "2020-03-17T02:10:31.909489Z", "iopub.status.idle": "2020-03-17T02:11:06.016198Z", "shell.execute_reply.started": "2020-03-17T02:10:31.909437Z", "shell.execute_reply": "2020-03-17T02:11:06.015350Z"}}
# eICU_df.groupby(['patientunitstayid', 'ts']).size().value_counts()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
eICU_df.reset_index(inplace=True)
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"execution": {"iopub.status.busy": "2020-03-17T02:12:54.058760Z", "iopub.execute_input": "2020-03-17T02:12:54.059203Z", "iopub.status.idle": "2020-03-17T02:12:54.153541Z", "shell.execute_reply.started": "2020-03-17T02:12:54.059139Z", "shell.execute_reply": "2020-03-17T02:12:54.152508Z"}}
eICU_df.groupby(['patientunitstayid', 'ts']).size().sort_values()

# + {"execution": {"iopub.status.busy": "2020-03-17T02:13:07.955827Z", "iopub.execute_input": "2020-03-17T02:13:07.956140Z", "iopub.status.idle": "2020-03-17T02:13:08.250772Z", "shell.execute_reply.started": "2020-03-17T02:13:07.956089Z", "shell.execute_reply": "2020-03-17T02:13:08.249833Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_diagns.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with diagnosis data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-17T02:21:02.530014Z", "iopub.status.idle": "2020-03-17T02:21:02.570799Z", "iopub.execute_input": "2020-03-17T02:21:02.530354Z", "shell.execute_reply.started": "2020-03-17T02:21:02.530289Z", "shell.execute_reply": "2020-03-17T02:21:02.569848Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_diagns.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-17T02:21:02.572058Z", "iopub.status.idle": "2020-03-17T02:21:03.733294Z", "iopub.execute_input": "2020-03-17T02:21:02.572327Z", "shell.execute_reply.started": "2020-03-17T02:21:02.572267Z", "shell.execute_reply": "2020-03-17T02:21:03.732468Z"}}
diagns_df = diagns_df[diagns_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-03-17T02:21:03.734428Z", "iopub.status.idle": "2020-03-17T02:21:03.779961Z", "iopub.execute_input": "2020-03-17T02:21:03.734674Z", "shell.execute_reply.started": "2020-03-17T02:21:03.734627Z", "shell.execute_reply": "2020-03-17T02:21:03.779224Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(diagns_df.patientunitstayid.unique())]

# + {"execution": {"iopub.status.busy": "2020-03-17T02:21:03.781208Z", "iopub.execute_input": "2020-03-17T02:21:03.781508Z", "iopub.status.idle": "2020-03-17T02:21:03.792554Z", "shell.execute_reply.started": "2020-03-17T02:21:03.781449Z", "shell.execute_reply": "2020-03-17T02:21:03.791579Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-17T02:21:03.793552Z", "iopub.execute_input": "2020-03-17T02:21:03.793838Z", "iopub.status.idle": "2020-03-17T02:21:03.908162Z", "shell.execute_reply.started": "2020-03-17T02:21:03.793792Z", "shell.execute_reply": "2020-03-17T02:21:03.907312Z"}}
eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
diagns_df.set_index(['patientunitstayid', 'ts'], inplace=True)
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-17T02:21:03.910110Z", "iopub.status.idle": "2020-03-17T02:21:18.726951Z", "iopub.execute_input": "2020-03-17T02:21:03.910430Z", "shell.execute_reply.started": "2020-03-17T02:21:03.910350Z", "shell.execute_reply": "2020-03-17T02:21:18.726037Z"}}
eICU_df = eICU_df.join(diagns_df, how='outer')
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
eICU_df.reset_index(inplace=True)
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"execution": {"iopub.status.busy": "2020-03-17T02:21:19.272691Z", "iopub.execute_input": "2020-03-17T02:21:19.272995Z", "iopub.status.idle": "2020-03-17T02:21:20.789968Z", "shell.execute_reply.started": "2020-03-17T02:21:19.272938Z", "shell.execute_reply": "2020-03-17T02:21:20.789231Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_past_hist.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with allergy data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-17T02:21:35.032410Z", "iopub.status.idle": "2020-03-17T02:21:37.106768Z", "iopub.execute_input": "2020-03-17T02:21:35.032721Z", "shell.execute_reply.started": "2020-03-17T02:21:35.032645Z", "shell.execute_reply": "2020-03-17T02:21:37.105903Z"}}
# eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_alrg.ftr')
# eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
# eICU_df.head()
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:01:52.209225Z", "iopub.status.idle": "2020-03-16T19:01:52.311109Z", "iopub.execute_input": "2020-03-16T19:01:52.209502Z", "shell.execute_reply.started": "2020-03-16T19:01:52.209446Z", "shell.execute_reply": "2020-03-16T19:01:52.310204Z"}}
# alrg_df = alrg_df[alrg_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

# + {"execution": {"iopub.status.busy": "2020-03-16T19:01:52.312436Z", "iopub.execute_input": "2020-03-16T19:01:52.312681Z", "iopub.status.idle": "2020-03-16T19:01:52.363719Z", "shell.execute_reply.started": "2020-03-16T19:01:52.312641Z", "shell.execute_reply": "2020-03-16T19:01:52.362786Z"}}
# eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T19:01:52.365093Z", "iopub.execute_input": "2020-03-16T19:01:52.365354Z", "iopub.status.idle": "2020-03-16T19:01:52.714861Z", "shell.execute_reply.started": "2020-03-16T19:01:52.365304Z", "shell.execute_reply": "2020-03-16T19:01:52.713691Z"}}
# eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
# alrg_df.set_index(['patientunitstayid', 'ts'], inplace=True)
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T19:01:52.716196Z", "iopub.execute_input": "2020-03-16T19:01:52.716575Z", "iopub.status.idle": "2020-03-16T19:01:56.577042Z", "shell.execute_reply.started": "2020-03-16T19:01:52.716521Z", "shell.execute_reply": "2020-03-16T19:01:56.575856Z"}}
# eICU_df = eICU_df.join(alrg_df, how='outer')
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
# eICU_df.reset_index(inplace=True)
# eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"execution": {"iopub.status.busy": "2020-03-15T06:24:26.414433Z", "iopub.execute_input": "2020-03-15T06:24:26.414647Z", "iopub.status.idle": "2020-03-15T01:58:25.351098Z", "shell.execute_reply.started": "2020-02-26T17:33:36.720857Z", "shell.execute_reply": "2020-02-26T18:02:20.126190Z"}}
# eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_past_hist.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with past history data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T14:12:15.122958Z", "iopub.status.idle": "2020-03-15T14:14:52.585252Z", "iopub.execute_input": "2020-03-15T14:12:15.123155Z", "shell.execute_reply.started": "2020-03-15T14:12:15.123117Z", "shell.execute_reply": "2020-03-15T14:14:52.584305Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_past_hist.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:01:56.705317Z", "iopub.status.idle": "2020-03-16T19:01:56.860152Z", "iopub.execute_input": "2020-03-16T19:01:56.705542Z", "shell.execute_reply.started": "2020-03-16T19:01:56.705504Z", "shell.execute_reply": "2020-03-16T19:01:56.859095Z"}}
past_hist_df = past_hist_df[past_hist_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:01:56.861285Z", "iopub.status.idle": "2020-03-16T19:01:57.049920Z", "iopub.execute_input": "2020-03-16T19:01:56.861539Z", "shell.execute_reply.started": "2020-03-16T19:01:56.861485Z", "shell.execute_reply": "2020-03-16T19:01:57.048938Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(past_hist_df.patientunitstayid.unique())]

# + {"execution": {"iopub.status.busy": "2020-03-16T19:01:57.051671Z", "iopub.execute_input": "2020-03-16T19:01:57.051925Z", "iopub.status.idle": "2020-03-16T19:01:57.114886Z", "shell.execute_reply.started": "2020-03-16T19:01:57.051877Z", "shell.execute_reply": "2020-03-16T19:01:57.113386Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T19:01:57.116774Z", "iopub.execute_input": "2020-03-16T19:01:57.117189Z", "iopub.status.idle": "2020-03-16T19:01:57.316047Z", "shell.execute_reply.started": "2020-03-16T19:01:57.117066Z", "shell.execute_reply": "2020-03-16T19:01:57.315119Z"}}
eICU_df.set_index('patientunitstayid', inplace=True)
past_hist_df.set_index('patientunitstayid', inplace=True)
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T19:01:57.317417Z", "iopub.execute_input": "2020-03-16T19:01:57.318295Z", "iopub.status.idle": "2020-03-16T19:01:57.323594Z", "shell.execute_reply.started": "2020-03-16T19:01:57.318083Z", "shell.execute_reply": "2020-03-16T19:01:57.322766Z"}}
len(eICU_df)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T19:01:57.327544Z", "iopub.status.idle": "2020-03-16T19:02:01.158458Z", "iopub.execute_input": "2020-03-16T19:01:57.328065Z", "shell.execute_reply.started": "2020-03-16T19:01:57.327999Z", "shell.execute_reply": "2020-03-16T19:02:01.157591Z"}}
eICU_df = eICU_df.join(past_hist_df, how='outer')
eICU_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T19:02:01.160188Z", "iopub.execute_input": "2020-03-16T19:02:01.160449Z", "iopub.status.idle": "2020-03-16T19:02:01.166005Z", "shell.execute_reply.started": "2020-03-16T19:02:01.160404Z", "shell.execute_reply": "2020-03-16T19:02:01.164803Z"}}
len(eICU_df)

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
eICU_df.reset_index(inplace=True)
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"execution": {"iopub.status.busy": "2020-03-15T06:24:26.414433Z", "iopub.execute_input": "2020-03-15T06:24:26.414647Z", "iopub.status.idle": "2020-03-15T01:58:25.351098Z", "shell.execute_reply.started": "2020-02-26T17:33:36.720857Z", "shell.execute_reply": "2020-02-26T18:02:20.126190Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_treat.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with treatment data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T14:12:15.122958Z", "iopub.status.idle": "2020-03-15T14:14:52.585252Z", "iopub.execute_input": "2020-03-15T14:12:15.123155Z", "shell.execute_reply.started": "2020-03-15T14:12:15.123117Z", "shell.execute_reply": "2020-03-15T14:14:52.584305Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_treat.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:02:01.281986Z", "iopub.status.idle": "2020-03-16T19:02:01.467518Z", "iopub.execute_input": "2020-03-16T19:02:01.282244Z", "shell.execute_reply.started": "2020-03-16T19:02:01.282194Z", "shell.execute_reply": "2020-03-16T19:02:01.465827Z"}}
treat_df = treat_df[treat_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:02:01.468921Z", "iopub.status.idle": "2020-03-16T19:02:01.924822Z", "iopub.execute_input": "2020-03-16T19:02:01.469269Z", "shell.execute_reply.started": "2020-03-16T19:02:01.469198Z", "shell.execute_reply": "2020-03-16T19:02:01.923876Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(treat_df.patientunitstayid.unique())]

# + {"execution": {"iopub.status.busy": "2020-03-16T19:02:01.925955Z", "iopub.execute_input": "2020-03-16T19:02:01.926212Z", "iopub.status.idle": "2020-03-16T19:02:01.992041Z", "shell.execute_reply.started": "2020-03-16T19:02:01.926158Z", "shell.execute_reply": "2020-03-16T19:02:01.990767Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T19:02:01.993262Z", "iopub.status.idle": "2020-03-16T19:02:02.915521Z", "iopub.execute_input": "2020-03-16T19:02:01.993516Z", "shell.execute_reply.started": "2020-03-16T19:02:01.993457Z", "shell.execute_reply": "2020-03-16T19:02:02.914574Z"}}
eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
treat_df.set_index(['patientunitstayid', 'ts'], inplace=True)
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T19:02:02.917017Z", "iopub.status.idle": "2020-03-16T19:02:22.808392Z", "iopub.execute_input": "2020-03-16T19:02:02.917356Z", "shell.execute_reply.started": "2020-03-16T19:02:02.917291Z", "shell.execute_reply": "2020-03-16T19:02:22.807300Z"}}
eICU_df = eICU_df.join(treat_df, how='outer')
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
eICU_df.reset_index(inplace=True)
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"execution": {"iopub.status.busy": "2020-03-15T06:24:26.414433Z", "iopub.execute_input": "2020-03-15T06:24:26.414647Z", "iopub.status.idle": "2020-03-15T01:58:25.351098Z", "shell.execute_reply.started": "2020-02-26T17:33:36.720857Z", "shell.execute_reply": "2020-02-26T18:02:20.126190Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_med.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with infusion drug data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T19:20:27.994634Z", "iopub.status.idle": "2020-03-16T19:20:34.030648Z", "iopub.execute_input": "2020-03-16T19:20:27.994911Z", "shell.execute_reply.started": "2020-03-16T19:20:27.994860Z", "shell.execute_reply": "2020-03-16T19:20:34.029788Z"}}
# eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_inf_drug.ftr')
# eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
# eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-16T19:12:55.552875Z", "iopub.execute_input": "2020-03-16T19:12:55.553777Z", "iopub.status.idle": "2020-03-16T19:12:55.560627Z", "shell.execute_reply.started": "2020-03-16T19:12:55.553708Z", "shell.execute_reply": "2020-03-16T19:12:55.559734Z"}}
# eICU_df.dtypes

# + {"execution": {"iopub.status.busy": "2020-03-16T19:12:56.645538Z", "iopub.execute_input": "2020-03-16T19:12:56.646154Z", "iopub.status.idle": "2020-03-16T19:12:56.710660Z", "shell.execute_reply.started": "2020-03-16T19:12:56.646095Z", "shell.execute_reply": "2020-03-16T19:12:56.709564Z"}}
# eICU_df.astype({'patientunitstayid': 'uint16'}).dtypes

# + {"execution": {"iopub.status.busy": "2020-03-16T19:13:11.222179Z", "iopub.execute_input": "2020-03-16T19:13:11.222575Z", "iopub.status.idle": "2020-03-16T19:13:11.227481Z", "shell.execute_reply.started": "2020-03-16T19:13:11.222516Z", "shell.execute_reply": "2020-03-16T19:13:11.226485Z"}}
# df_columns = list(eICU_df.columns)

# + {"execution": {"iopub.status.busy": "2020-03-16T19:13:12.467473Z", "iopub.execute_input": "2020-03-16T19:13:12.467784Z", "iopub.status.idle": "2020-03-16T19:13:12.473863Z", "shell.execute_reply.started": "2020-03-16T19:13:12.467723Z", "shell.execute_reply": "2020-03-16T19:13:12.472659Z"}}
# tmp_dict = dict()
# for key, val in dtype_dict.items():
#     if key in df_columns:
#         tmp_dict[key] = dtype_dict[key]

# + {"execution": {"iopub.status.busy": "2020-03-16T19:13:14.164630Z", "iopub.execute_input": "2020-03-16T19:13:14.164914Z", "iopub.status.idle": "2020-03-16T19:13:14.171708Z", "shell.execute_reply.started": "2020-03-16T19:13:14.164873Z", "shell.execute_reply": "2020-03-16T19:13:14.170807Z"}}
# tmp_dict

# + {"execution": {"iopub.status.busy": "2020-03-16T19:13:16.899070Z", "iopub.execute_input": "2020-03-16T19:13:16.899528Z", "iopub.status.idle": "2020-03-16T19:13:16.956361Z", "shell.execute_reply.started": "2020-03-16T19:13:16.899468Z", "shell.execute_reply": "2020-03-16T19:13:16.954960Z"}}
# # eICU_df = eICU_df.astype(tmp_dict, copy=False)
# eICU_df = eICU_df.astype(tmp_dict)

# + {"execution": {"iopub.status.busy": "2020-03-16T19:13:17.943038Z", "iopub.execute_input": "2020-03-16T19:13:17.943337Z", "iopub.status.idle": "2020-03-16T19:13:17.953315Z", "shell.execute_reply.started": "2020-03-16T19:13:17.943294Z", "shell.execute_reply": "2020-03-16T19:13:17.952113Z"}}
# eICU_df.dtypes
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:20:34.032065Z", "iopub.status.idle": "2020-03-16T19:20:34.488917Z", "iopub.execute_input": "2020-03-16T19:20:34.032347Z", "shell.execute_reply.started": "2020-03-16T19:20:34.032296Z", "shell.execute_reply": "2020-03-16T19:20:34.487923Z"}}
# inf_drug_df = inf_drug_df[inf_drug_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

# + {"execution": {"iopub.status.busy": "2020-03-16T19:20:34.490228Z", "iopub.execute_input": "2020-03-16T19:20:34.490565Z", "iopub.status.idle": "2020-03-16T19:20:34.542088Z", "shell.execute_reply.started": "2020-03-16T19:20:34.490504Z", "shell.execute_reply": "2020-03-16T19:20:34.540951Z"}}
# eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T19:20:34.543767Z", "iopub.execute_input": "2020-03-16T19:20:34.544114Z", "iopub.status.idle": "2020-03-16T19:20:36.948907Z", "shell.execute_reply.started": "2020-03-16T19:20:34.544042Z", "shell.execute_reply": "2020-03-16T19:20:36.948045Z"}}
# eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
# inf_drug_df.set_index(['patientunitstayid', 'ts'], inplace=True)
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T19:20:36.950096Z", "iopub.status.idle": "2020-03-16T19:21:53.629422Z", "iopub.execute_input": "2020-03-16T19:20:36.950356Z", "shell.execute_reply.started": "2020-03-16T19:20:36.950305Z", "shell.execute_reply": "2020-03-16T19:21:53.628433Z"}}
# eICU_df = eICU_df.join(inf_drug_df, how='outer')
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
# eICU_df.reset_index(inplace=True)
# eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"execution": {"iopub.status.busy": "2020-03-16T19:22:05.277569Z", "iopub.execute_input": "2020-03-16T19:22:05.277916Z", "iopub.status.idle": "2020-03-16T19:22:05.578952Z", "shell.execute_reply.started": "2020-03-16T19:22:05.277778Z", "shell.execute_reply": "2020-03-16T19:22:05.578030Z"}}
# eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_med.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with medication data

# + [markdown] {"Collapsed": "false"}
# #### Joining medication and admission drug data

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false"}
med_df.set_index(['patientunitstayid', 'ts'], inplace=True)
adms_drug_df.set_index(['patientunitstayid', 'ts'], inplace=True)
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T05:30:57.714810Z", "iopub.status.idle": "2020-03-15T06:24:26.413378Z", "iopub.execute_input": "2020-03-15T05:30:57.715006Z", "shell.execute_reply.started": "2020-03-15T05:30:57.714971Z", "shell.execute_reply": "2020-03-15T06:24:26.412632Z"}}
med_df = med_df.join(adms_drug_df, how='outer')
med_df.head()

# + [markdown] {"Collapsed": "false"}
# #### Merging duplicate columns

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:19:40.055657Z", "iopub.status.idle": "2020-03-02T02:19:40.061775Z", "iopub.execute_input": "2020-03-02T02:19:40.055928Z", "shell.execute_reply.started": "2020-03-02T02:19:40.055877Z", "shell.execute_reply": "2020-03-02T02:19:40.061167Z"}}
set([col.split('_x')[0].split('_y')[0] for col in eICU_df.columns if col.endswith('_x') or col.endswith('_y')])

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:19:40.062791Z", "iopub.status.idle": "2020-03-02T02:19:40.801829Z", "iopub.execute_input": "2020-03-02T02:19:40.063021Z", "shell.execute_reply.started": "2020-03-02T02:19:40.062965Z", "shell.execute_reply": "2020-03-02T02:19:40.800838Z"}}
eICU_df[['drugdosage_x', 'drugdosage_y']].head(20)
# -

# + {"Collapsed": "false", "pixiedust": {"displayParams": {}}, "execution": {"iopub.status.busy": "2020-03-02T02:36:29.271829Z", "iopub.status.idle": "2020-03-02T02:40:49.633520Z", "iopub.execute_input": "2020-03-02T02:36:29.272090Z", "shell.execute_reply.started": "2020-03-02T02:36:29.272048Z", "shell.execute_reply": "2020-03-02T02:40:49.632359Z"}}
eICU_df = du.data_processing.merge_columns(eICU_df, inplace=True)
eICU_df.sample(20)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:19:40.055657Z", "iopub.status.idle": "2020-03-02T02:19:40.061775Z", "iopub.execute_input": "2020-03-02T02:19:40.055928Z", "shell.execute_reply.started": "2020-03-02T02:19:40.055877Z", "shell.execute_reply": "2020-03-02T02:19:40.061167Z"}}
set([col.split('_x')[0].split('_y')[0] for col in eICU_df.columns if col.endswith('_x') or col.endswith('_y')])

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:42:58.060632Z", "iopub.status.idle": "2020-03-02T02:42:58.071196Z", "iopub.execute_input": "2020-03-02T02:42:58.062830Z", "shell.execute_reply.started": "2020-03-02T02:42:58.062000Z", "shell.execute_reply": "2020-03-02T02:42:58.070236Z"}}
eICU_df['drugdosage'].head(20)

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
med_df.reset_index(inplace=True)
med_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
# med_df = du.utils.convert_pyarrow_dtypes(med_df, inplace=True)

# + {"execution": {"iopub.status.busy": "2020-03-15T06:24:26.414433Z", "iopub.execute_input": "2020-03-15T06:24:26.414647Z", "iopub.status.idle": "2020-03-15T01:58:25.351098Z", "shell.execute_reply.started": "2020-02-26T17:33:36.720857Z", "shell.execute_reply": "2020-02-26T18:02:20.126190Z"}}
med_df.to_feather(f'{data_path}normalized/ohe/med_and_adms_drug.ftr')

# + [markdown] {"Collapsed": "false"}
# #### Joining with the rest of the eICU data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T02:53:41.307799Z", "iopub.status.idle": "2020-03-16T02:57:40.916608Z", "iopub.execute_input": "2020-03-16T02:53:41.308067Z", "shell.execute_reply.started": "2020-03-16T02:53:41.308027Z", "shell.execute_reply": "2020-03-16T02:57:40.915726Z"}}
med_drug_df = pd.read_feather(f'{data_path}normalized/ohe/med_and_adms_drug.ftr')
med_drug_df = du.utils.convert_dtypes(med_drug_df, dtypes=dtype_dict, inplace=True)
med_drug_df.head()
# -

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T02:53:41.307799Z", "iopub.status.idle": "2020-03-16T02:57:40.916608Z", "iopub.execute_input": "2020-03-16T02:53:41.308067Z", "shell.execute_reply.started": "2020-03-16T02:53:41.308027Z", "shell.execute_reply": "2020-03-16T02:57:40.915726Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_med.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-16T02:57:40.917873Z", "iopub.status.idle": "2020-03-16T02:57:55.127824Z", "iopub.execute_input": "2020-03-16T02:57:40.918142Z", "shell.execute_reply.started": "2020-03-16T02:57:40.918103Z", "shell.execute_reply": "2020-03-16T02:57:55.127171Z"}}
med_drug_df = med_drug_df[med_drug_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-03-16T02:57:55.129155Z", "iopub.status.idle": "2020-03-16T02:58:17.481997Z", "iopub.execute_input": "2020-03-16T02:57:55.129394Z", "shell.execute_reply.started": "2020-03-16T02:57:55.129356Z", "shell.execute_reply": "2020-03-16T02:58:17.481293Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(med_drug_df.patientunitstayid.unique())]

# + {"execution": {"iopub.status.busy": "2020-02-27T05:20:54.131063Z", "iopub.execute_input": "2020-02-27T05:20:54.131275Z", "iopub.status.idle": "2020-02-27T05:21:15.499582Z", "shell.execute_reply.started": "2020-02-27T05:20:54.131238Z", "shell.execute_reply": "2020-02-27T05:21:15.498801Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false"}
eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
med_drug_df.set_index(['patientunitstayid', 'ts'], inplace=True)
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T02:58:17.483050Z", "iopub.status.idle": "2020-03-16T03:12:21.035320Z", "iopub.execute_input": "2020-03-16T02:58:17.483254Z", "shell.execute_reply.started": "2020-03-16T02:58:17.483218Z", "shell.execute_reply": "2020-03-16T03:12:21.034360Z"}}
eICU_df = eICU_df.join(med_drug_df, how='outer')
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
eICU_df.reset_index(inplace=True)
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"execution": {"iopub.status.busy": "2020-03-16T03:12:21.036182Z", "iopub.execute_input": "2020-02-26T17:33:36.720916Z", "iopub.status.idle": "2020-03-16T03:12:21.036482Z", "shell.execute_reply.started": "2020-02-26T17:33:36.720857Z", "shell.execute_reply": "2020-02-26T18:02:20.126190Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_resp_care.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with intake outake data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T14:12:15.122958Z", "iopub.status.idle": "2020-03-15T14:14:52.585252Z", "iopub.execute_input": "2020-03-15T14:12:15.123155Z", "shell.execute_reply.started": "2020-03-15T14:12:15.123117Z", "shell.execute_reply": "2020-03-15T14:14:52.584305Z"}}
# eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_in_out.ftr')
# eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
# eICU_df.head()
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-15T14:17:00.692190Z", "iopub.status.idle": "2020-03-15T14:17:02.377755Z", "iopub.execute_input": "2020-03-15T14:17:00.692386Z", "shell.execute_reply.started": "2020-03-15T14:17:00.692351Z", "shell.execute_reply": "2020-03-15T14:17:02.376997Z"}}
# in_out_df = in_out_df[in_out_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-03-15T14:17:02.378883Z", "iopub.status.idle": "2020-03-15T14:17:12.519271Z", "iopub.execute_input": "2020-03-15T14:17:02.379120Z", "shell.execute_reply.started": "2020-03-15T14:17:02.379080Z", "shell.execute_reply": "2020-03-15T14:17:12.518572Z"}}
# eICU_df = eICU_df[eICU_df.patientunitstayid.isin(in_out_df.patientunitstayid.unique())]

# + {"execution": {"iopub.status.busy": "2020-02-27T05:20:54.131063Z", "iopub.execute_input": "2020-02-27T05:20:54.131275Z", "iopub.status.idle": "2020-02-27T05:21:15.499582Z", "shell.execute_reply.started": "2020-02-27T05:20:54.131238Z", "shell.execute_reply": "2020-02-27T05:21:15.498801Z"}}
# eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false"}
# eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
# in_out_df.set_index(['patientunitstayid', 'ts'], inplace=True)
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T14:17:12.520211Z", "iopub.status.idle": "2020-03-15T15:14:28.597564Z", "iopub.execute_input": "2020-03-15T14:17:12.520423Z", "shell.execute_reply.started": "2020-03-15T14:17:12.520386Z", "shell.execute_reply": "2020-03-15T15:14:28.596911Z"}}
# eICU_df = eICU_df.join(in_out_df, how='outer')
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
# eICU_df.reset_index(inplace=True)
# eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"execution": {"iopub.status.busy": "2020-03-15T15:14:28.822010Z", "iopub.execute_input": "2020-03-15T15:14:28.822215Z", "iopub.status.idle": "2020-03-15T01:58:25.351098Z", "shell.execute_reply.started": "2020-02-26T17:33:36.720857Z", "shell.execute_reply": "2020-02-26T18:02:20.126190Z"}}
# eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_resp_care.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with nurse care data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:14.996900Z", "iopub.status.idle": "2020-03-15T04:51:14.997207Z", "iopub.execute_input": "2020-02-26T17:13:35.213020Z", "shell.execute_reply.started": "2020-02-26T17:13:35.212983Z", "shell.execute_reply": "2020-02-26T17:13:35.215332Z"}}
# eICU_df = pd.merge(eICU_df, nurse_care_df, how='outer', on=['patientunitstayid', 'ts'], copy=False)
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Joining with nurse assessment data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:14.997878Z", "iopub.status.idle": "2020-03-15T04:51:14.998163Z", "iopub.execute_input": "2020-02-26T17:13:35.217016Z", "shell.execute_reply.started": "2020-02-26T17:13:35.216967Z", "shell.execute_reply": "2020-02-26T17:13:37.232810Z"}}
# eICU_df = pd.merge(eICU_df, nurse_assess_df, how='outer', on=['patientunitstayid', 'ts'], copy=False)
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Joining with nurse charting data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:14.998878Z", "iopub.status.idle": "2020-03-15T04:51:14.999189Z", "iopub.execute_input": "2020-02-26T17:13:37.234780Z", "shell.execute_reply.started": "2020-02-26T17:13:37.234737Z", "shell.execute_reply": "2020-02-26T17:13:37.238206Z"}}
# eICU_df = pd.merge(eICU_df, nurse_chart_df, how='outer', on=['patientunitstayid', 'ts'], copy=False)
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Joining with respiratory care data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T19:47:15.661355Z", "iopub.status.idle": "2020-03-15T19:51:19.649887Z", "iopub.execute_input": "2020-03-15T19:47:15.661977Z", "shell.execute_reply.started": "2020-03-15T19:47:15.661904Z", "shell.execute_reply": "2020-03-15T19:51:19.649194Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_resp_care.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-15T19:51:19.650973Z", "iopub.status.idle": "2020-03-15T19:51:21.246918Z", "iopub.execute_input": "2020-03-15T19:51:19.651360Z", "shell.execute_reply.started": "2020-03-15T19:51:19.651316Z", "shell.execute_reply": "2020-03-15T19:51:21.246220Z"}}
resp_care_df = resp_care_df[resp_care_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

# + {"execution": {"iopub.status.busy": "2020-02-27T05:20:54.131063Z", "iopub.execute_input": "2020-02-27T05:20:54.131275Z", "iopub.status.idle": "2020-02-27T05:21:15.499582Z", "shell.execute_reply.started": "2020-02-27T05:20:54.131238Z", "shell.execute_reply": "2020-02-27T05:21:15.498801Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false"}
eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
resp_care_df.set_index(['patientunitstayid', 'ts'], inplace=True)
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T19:51:21.248439Z", "iopub.status.idle": "2020-03-15T21:08:52.365932Z", "iopub.execute_input": "2020-03-15T19:51:21.249062Z", "shell.execute_reply.started": "2020-03-15T19:51:21.249000Z", "shell.execute_reply": "2020-03-15T21:08:52.365287Z"}}
eICU_df = eICU_df.join(resp_care_df, how='outer')
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
eICU_df.reset_index(inplace=True)
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"execution": {"iopub.status.busy": "2020-03-15T21:08:52.366982Z", "iopub.execute_input": "2020-03-15T21:08:52.367210Z", "iopub.status.idle": "2020-03-15T19:21:13.549139Z", "shell.execute_reply.started": "2020-02-26T17:33:36.720857Z", "shell.execute_reply": "2020-02-26T18:02:20.126190Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_vital_aprdc.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with aperiodic vital signals data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T02:53:41.307799Z", "iopub.status.idle": "2020-03-16T02:57:40.916608Z", "iopub.execute_input": "2020-03-16T02:53:41.308067Z", "shell.execute_reply.started": "2020-03-16T02:53:41.308027Z", "shell.execute_reply": "2020-03-16T02:57:40.915726Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_vital_aprdc.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-16T02:57:40.917873Z", "iopub.status.idle": "2020-03-16T02:57:55.127824Z", "iopub.execute_input": "2020-03-16T02:57:40.918142Z", "shell.execute_reply.started": "2020-03-16T02:57:40.918103Z", "shell.execute_reply": "2020-03-16T02:57:55.127171Z"}}
vital_aprdc_df = vital_aprdc_df[vital_aprdc_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-03-16T02:57:55.129155Z", "iopub.status.idle": "2020-03-16T02:58:17.481997Z", "iopub.execute_input": "2020-03-16T02:57:55.129394Z", "shell.execute_reply.started": "2020-03-16T02:57:55.129356Z", "shell.execute_reply": "2020-03-16T02:58:17.481293Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(vital_aprdc_df.patientunitstayid.unique())]

# + {"execution": {"iopub.status.busy": "2020-02-27T05:20:54.131063Z", "iopub.execute_input": "2020-02-27T05:20:54.131275Z", "iopub.status.idle": "2020-02-27T05:21:15.499582Z", "shell.execute_reply.started": "2020-02-27T05:20:54.131238Z", "shell.execute_reply": "2020-02-27T05:21:15.498801Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false"}
eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
vital_aprdc_df.set_index(['patientunitstayid', 'ts'], inplace=True)
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-16T02:58:17.483050Z", "iopub.status.idle": "2020-03-16T03:12:21.035320Z", "iopub.execute_input": "2020-03-16T02:58:17.483254Z", "shell.execute_reply.started": "2020-03-16T02:58:17.483218Z", "shell.execute_reply": "2020-03-16T03:12:21.034360Z"}}
eICU_df = eICU_df.join(vital_aprdc_df, how='outer')
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
eICU_df.reset_index(inplace=True)
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"execution": {"iopub.status.busy": "2020-03-16T03:12:21.036182Z", "iopub.execute_input": "2020-02-26T17:33:36.720916Z", "iopub.status.idle": "2020-03-16T03:12:21.036482Z", "shell.execute_reply.started": "2020-02-26T17:33:36.720857Z", "shell.execute_reply": "2020-02-26T18:02:20.126190Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_vital_prdc.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with periodic vital signals data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T20:27:00.928018Z", "iopub.status.idle": "2020-02-26T20:27:30.650260Z", "iopub.execute_input": "2020-02-26T20:27:00.928261Z", "shell.execute_reply.started": "2020-02-26T20:27:00.928221Z", "shell.execute_reply": "2020-02-26T20:27:30.649669Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_vital_prdc.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-15T01:58:25.348497Z", "iopub.status.idle": "2020-03-15T01:58:25.348812Z", "iopub.execute_input": "2020-02-26T20:27:30.651538Z", "shell.execute_reply.started": "2020-02-26T20:27:30.651486Z", "shell.execute_reply": "2020-02-26T20:29:03.629871Z"}}
vital_prdc_df = vital_prdc_df[vital_prdc_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-03-15T01:58:25.349824Z", "iopub.status.idle": "2020-03-15T01:58:25.350118Z", "iopub.execute_input": "2020-02-26T20:29:03.632286Z", "shell.execute_reply.started": "2020-02-26T20:29:03.632247Z", "shell.execute_reply": "2020-02-26T20:30:05.977157Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(vital_prdc_df.patientunitstayid.unique())]

# + {"execution": {"iopub.status.busy": "2020-02-27T05:20:54.131063Z", "iopub.execute_input": "2020-02-27T05:20:54.131275Z", "iopub.status.idle": "2020-02-27T05:21:15.499582Z", "shell.execute_reply.started": "2020-02-27T05:20:54.131238Z", "shell.execute_reply": "2020-02-27T05:21:15.498801Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false"}
eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
vital_prdc_df.set_index(['patientunitstayid', 'ts'], inplace=True)
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T20:30:05.978941Z", "iopub.status.idle": "2020-02-26T20:56:11.534659Z", "iopub.execute_input": "2020-02-26T20:30:05.979158Z", "shell.execute_reply.started": "2020-02-26T20:30:05.979112Z", "shell.execute_reply": "2020-02-26T20:56:11.534004Z"}}
eICU_df = eICU_df.join(vital_prdc_df, how='outer')
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
eICU_df.reset_index(inplace=True)
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"execution": {"iopub.status.busy": "2020-02-26T20:56:11.535600Z", "iopub.status.idle": "2020-02-26T20:25:41.072051Z", "iopub.execute_input": "2020-02-26T20:56:11.535828Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_post_joining_vital_prdc.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with lab data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-27T00:58:11.955786Z", "iopub.status.idle": "2020-02-27T01:10:36.152614Z", "iopub.execute_input": "2020-02-27T00:58:11.956018Z", "shell.execute_reply.started": "2020-02-27T00:58:11.955976Z", "shell.execute_reply": "2020-02-27T01:10:36.151782Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_post_joining_vital_prdc.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-02-27T01:10:36.154356Z", "iopub.status.idle": "2020-02-27T01:14:18.564700Z", "iopub.execute_input": "2020-02-27T01:10:36.154590Z", "shell.execute_reply.started": "2020-02-27T01:10:36.154548Z", "shell.execute_reply": "2020-02-27T01:14:18.563808Z"}}
lab_df = lab_df[lab_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-02-27T01:14:18.566086Z", "iopub.status.idle": "2020-02-27T01:15:30.746679Z", "iopub.execute_input": "2020-02-27T01:14:18.566341Z", "shell.execute_reply.started": "2020-02-27T01:14:18.566301Z", "shell.execute_reply": "2020-02-27T01:15:30.745909Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(lab_df.patientunitstayid.unique())]

# + {"execution": {"iopub.status.busy": "2020-02-27T05:20:54.131063Z", "iopub.execute_input": "2020-02-27T05:20:54.131275Z", "iopub.status.idle": "2020-02-27T05:21:15.499582Z", "shell.execute_reply.started": "2020-02-27T05:20:54.131238Z", "shell.execute_reply": "2020-02-27T05:21:15.498801Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false"}
eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
lab_df.set_index(['patientunitstayid', 'ts'], inplace=True)
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-27T01:15:30.747792Z", "iopub.execute_input": "2020-02-27T01:15:30.748015Z", "iopub.status.idle": "2020-02-27T02:11:51.960090Z", "shell.execute_reply.started": "2020-02-27T01:15:30.747973Z", "shell.execute_reply": "2020-02-27T02:11:51.959315Z"}}
eICU_df = eICU_df.join(lab_df, how='outer')
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
eICU_df.reset_index(inplace=True)
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"execution": {"iopub.status.busy": "2020-02-27T02:11:51.961181Z", "iopub.status.idle": "2020-02-27T05:20:54.129974Z", "iopub.execute_input": "2020-02-27T02:11:51.961398Z", "shell.execute_reply.started": "2020-02-27T02:11:51.961359Z", "shell.execute_reply": "2020-02-27T05:20:54.129277Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_post_joining.ftr')

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:16:36.275096Z", "iopub.status.idle": "2020-03-02T02:16:52.692400Z", "iopub.execute_input": "2020-03-02T02:16:36.275391Z", "shell.execute_reply.started": "2020-03-02T02:16:36.275334Z", "shell.execute_reply": "2020-03-02T02:16:52.691647Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_post_joining.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()
# -

eICU_df.columns

eICU_df.dtypes

# + [markdown] {"Collapsed": "false"}
# ## Cleaning the joined data

# + [markdown] {"Collapsed": "false"}
# ### Removing unit stays that are too short

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T01:53:31.536616Z", "iopub.status.idle": "2020-03-02T01:54:52.891338Z", "iopub.execute_input": "2020-03-02T01:53:31.536857Z", "shell.execute_reply.started": "2020-03-02T01:53:31.536812Z", "shell.execute_reply": "2020-03-02T01:54:52.890540Z"}}
# eICU_df.info(memory_usage='deep')

# + [markdown] {"Collapsed": "false"}
# Make sure that the dataframe is ordered by time `ts`:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:17:12.532218Z", "iopub.status.idle": "2020-03-02T02:17:39.288044Z", "iopub.execute_input": "2020-03-02T02:17:12.532569Z", "shell.execute_reply.started": "2020-03-02T02:17:12.532509Z", "shell.execute_reply": "2020-03-02T02:17:39.286198Z"}}
eICU_df = eICU_df.sort_values('ts')
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove unit stays that have less than 10 records:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:17:39.289585Z", "iopub.status.idle": "2020-03-02T02:18:15.176358Z", "iopub.execute_input": "2020-03-02T02:17:39.289871Z", "shell.execute_reply.started": "2020-03-02T02:17:39.289827Z", "shell.execute_reply": "2020-03-02T02:18:15.175455Z"}}
unit_stay_len = eICU_df.groupby('patientunitstayid').patientunitstayid.count()
unit_stay_len

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:18:15.178381Z", "iopub.status.idle": "2020-03-02T02:18:15.198989Z", "iopub.execute_input": "2020-03-02T02:18:15.178640Z", "shell.execute_reply.started": "2020-03-02T02:18:15.178591Z", "shell.execute_reply": "2020-03-02T02:18:15.198038Z"}}
unit_stay_short = set(unit_stay_len[unit_stay_len < 10].index)
unit_stay_short

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:18:15.200767Z", "iopub.status.idle": "2020-03-02T02:18:17.001752Z", "iopub.execute_input": "2020-03-02T02:18:15.201115Z", "shell.execute_reply.started": "2020-03-02T02:18:15.201047Z", "shell.execute_reply": "2020-03-02T02:18:17.000646Z"}}
len(unit_stay_short)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:18:17.003368Z", "iopub.status.idle": "2020-03-02T02:18:18.720633Z", "iopub.execute_input": "2020-03-02T02:18:17.004081Z", "shell.execute_reply.started": "2020-03-02T02:18:17.003999Z", "shell.execute_reply": "2020-03-02T02:18:18.719243Z"}}
eICU_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:18:18.721989Z", "iopub.status.idle": "2020-03-02T02:18:20.427440Z", "iopub.execute_input": "2020-03-02T02:18:18.722309Z", "shell.execute_reply.started": "2020-03-02T02:18:18.722259Z", "shell.execute_reply": "2020-03-02T02:18:20.426700Z"}}
eICU_df = eICU_df[~eICU_df.patientunitstayid.isin(unit_stay_short)]

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:18:20.428470Z", "iopub.status.idle": "2020-03-02T02:18:22.585566Z", "iopub.execute_input": "2020-03-02T02:18:20.428687Z", "shell.execute_reply.started": "2020-03-02T02:18:20.428649Z", "shell.execute_reply": "2020-03-02T02:18:22.584657Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Remove unit stays that have data that represent less than 48h:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:18:22.588007Z", "iopub.status.idle": "2020-03-02T02:19:35.059137Z", "iopub.execute_input": "2020-03-02T02:18:22.588243Z", "shell.execute_reply.started": "2020-03-02T02:18:22.588196Z", "shell.execute_reply": "2020-03-02T02:19:35.058244Z"}}
unit_stay_duration = eICU_df.groupby('patientunitstayid').ts.apply(lambda x: x.max() - x.min())
unit_stay_duration

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:19:35.060437Z", "iopub.status.idle": "2020-03-02T02:19:35.081986Z", "iopub.execute_input": "2020-03-02T02:19:35.060663Z", "shell.execute_reply.started": "2020-03-02T02:19:35.060624Z", "shell.execute_reply": "2020-03-02T02:19:35.081304Z"}}
unit_stay_short = set(unit_stay_duration[unit_stay_duration < 48*60].index)
unit_stay_short

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:19:35.083029Z", "iopub.status.idle": "2020-03-02T02:19:35.087514Z", "iopub.execute_input": "2020-03-02T02:19:35.083255Z", "shell.execute_reply.started": "2020-03-02T02:19:35.083217Z", "shell.execute_reply": "2020-03-02T02:19:35.086634Z"}}
len(unit_stay_short)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:19:35.088373Z", "iopub.status.idle": "2020-03-02T02:19:36.682445Z", "iopub.execute_input": "2020-03-02T02:19:35.088588Z", "shell.execute_reply.started": "2020-03-02T02:19:35.088551Z", "shell.execute_reply": "2020-03-02T02:19:36.681534Z"}}
eICU_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:19:36.683454Z", "iopub.status.idle": "2020-03-02T02:19:38.245825Z", "iopub.execute_input": "2020-03-02T02:19:36.683688Z", "shell.execute_reply.started": "2020-03-02T02:19:36.683649Z", "shell.execute_reply": "2020-03-02T02:19:38.245000Z"}}
eICU_df = eICU_df[~eICU_df.patientunitstayid.isin(unit_stay_short)]

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:19:38.247052Z", "iopub.execute_input": "2020-03-02T02:19:38.247368Z", "iopub.status.idle": "2020-03-02T02:19:40.054358Z", "shell.execute_reply.started": "2020-03-02T02:19:38.247313Z", "shell.execute_reply": "2020-03-02T02:19:40.053667Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining duplicate columns

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:19:40.055657Z", "iopub.status.idle": "2020-03-02T02:19:40.061775Z", "iopub.execute_input": "2020-03-02T02:19:40.055928Z", "shell.execute_reply.started": "2020-03-02T02:19:40.055877Z", "shell.execute_reply": "2020-03-02T02:19:40.061167Z"}}
set([col.split('_x')[0].split('_y')[0] for col in eICU_df.columns if col.endswith('_x') or col.endswith('_y')])

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:19:40.062791Z", "iopub.status.idle": "2020-03-02T02:19:40.801829Z", "iopub.execute_input": "2020-03-02T02:19:40.063021Z", "shell.execute_reply.started": "2020-03-02T02:19:40.062965Z", "shell.execute_reply": "2020-03-02T02:19:40.800838Z"}}
eICU_df[['drugdosage_x', 'drugdosage_y']].head(20)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:19:40.802911Z", "iopub.status.idle": "2020-03-02T02:19:41.605486Z", "iopub.execute_input": "2020-03-02T02:19:40.803378Z", "shell.execute_reply.started": "2020-03-02T02:19:40.803328Z", "shell.execute_reply": "2020-03-02T02:19:41.604550Z"}}
eICU_df[eICU_df.index == 2564878][['drugdosage_x', 'drugdosage_y']]
# -

# Convert dataframe to Pandas, as the next cells aren't working properly with Modin:

# + {"execution": {"iopub.status.busy": "2020-03-02T02:34:38.088841Z", "iopub.execute_input": "2020-03-02T02:34:38.089308Z", "iopub.status.idle": "2020-03-02T02:36:29.270540Z", "shell.execute_reply.started": "2020-03-02T02:34:38.089249Z", "shell.execute_reply": "2020-03-02T02:36:29.269639Z"}}
eICU_df, pd = du.utils.convert_dataframe(eICU_df, to='pandas', dtypes=dtype_dict)

# + {"Collapsed": "false", "pixiedust": {"displayParams": {}}, "execution": {"iopub.status.busy": "2020-03-02T02:36:29.271829Z", "iopub.status.idle": "2020-03-02T02:40:49.633520Z", "iopub.execute_input": "2020-03-02T02:36:29.272090Z", "shell.execute_reply.started": "2020-03-02T02:36:29.272048Z", "shell.execute_reply": "2020-03-02T02:40:49.632359Z"}}
eICU_df = du.data_processing.merge_columns(eICU_df, inplace=True)
eICU_df.sample(20)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:19:40.055657Z", "iopub.status.idle": "2020-03-02T02:19:40.061775Z", "iopub.execute_input": "2020-03-02T02:19:40.055928Z", "shell.execute_reply.started": "2020-03-02T02:19:40.055877Z", "shell.execute_reply": "2020-03-02T02:19:40.061167Z"}}
set([col.split('_x')[0].split('_y')[0] for col in eICU_df.columns if col.endswith('_x') or col.endswith('_y')])

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:42:58.060632Z", "iopub.status.idle": "2020-03-02T02:42:58.071196Z", "iopub.execute_input": "2020-03-02T02:42:58.062830Z", "shell.execute_reply.started": "2020-03-02T02:42:58.062000Z", "shell.execute_reply": "2020-03-02T02:42:58.070236Z"}}
eICU_df['drugdosage'].head(20)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:42:58.072944Z", "iopub.status.idle": "2020-03-02T02:42:58.109699Z", "iopub.execute_input": "2020-03-02T02:42:58.073200Z", "shell.execute_reply.started": "2020-03-02T02:42:58.073152Z", "shell.execute_reply": "2020-03-02T02:42:58.108599Z"}}
eICU_df[eICU_df.index == 2564878][['drugdosage']]

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:42:58.111378Z", "iopub.status.idle": "2020-03-02T02:44:09.853833Z", "iopub.execute_input": "2020-03-02T02:42:58.111630Z", "shell.execute_reply.started": "2020-03-02T02:42:58.111587Z", "shell.execute_reply": "2020-03-02T02:44:09.852640Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_post_merge_duplicate_cols.ftr')

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T04:30:00.844962Z", "iopub.status.idle": "2020-03-02T04:30:11.572222Z", "iopub.execute_input": "2020-03-02T04:30:00.845255Z", "shell.execute_reply.started": "2020-03-02T04:30:00.845203Z", "shell.execute_reply": "2020-03-02T04:30:11.571436Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_post_merge_duplicate_cols.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Removing unit stays with too many missing values
#
# Consider removing all unit stays that have, combining rows and columns, a very high percentage of missing values.
# -

# Reconvert dataframe to Modin:

# + {"execution": {"iopub.status.busy": "2020-03-02T04:33:08.150355Z", "iopub.status.idle": "2020-03-02T04:33:08.150854Z"}}
eICU_df, pd = du.utils.convert_dataframe(vital_prdc_df, to='modin', dtypes=dtype_dict)

# + {"execution": {"iopub.status.busy": "2020-03-02T04:33:08.152336Z", "iopub.status.idle": "2020-03-02T04:33:08.153242Z", "iopub.execute_input": "2020-03-02T04:30:22.532646Z", "shell.execute_reply.started": "2020-03-02T04:30:22.532603Z", "shell.execute_reply": "2020-03-02T04:30:22.536895Z"}}
n_features = len(eICU_df.columns)
n_features

# + [markdown] {"Collapsed": "false"}
# Create a temporary column that counts each row's number of missing values:

# + {"execution": {"iopub.status.busy": "2020-03-02T04:33:08.154574Z", "iopub.status.idle": "2020-03-02T04:33:08.155051Z", "iopub.execute_input": "2020-03-02T04:30:23.160440Z", "shell.execute_reply.started": "2020-03-02T04:30:23.160388Z", "shell.execute_reply": "2020-03-02T04:30:27.325740Z"}}
eICU_df['row_msng_val'] = eICU_df.isnull().sum(axis=1)
eICU_df[['patientunitstayid', 'ts', 'row_msng_val']].head()

# + [markdown] {"Collapsed": "false"}
# Check each unit stay's percentage of missing data points:

# + {"execution": {"iopub.status.busy": "2020-03-02T04:33:08.156401Z", "iopub.status.idle": "2020-03-02T04:33:08.156886Z"}}
# Number of possible data points in each unit stay
n_data_points = eICU_df.groupby('patientunitstayid').ts.count() * n_features
n_data_points

# + {"execution": {"iopub.status.busy": "2020-03-02T04:33:08.158235Z", "iopub.status.idle": "2020-03-02T04:33:08.159229Z"}}
# Number of missing values in each unit stay
n_msng_val = eICU_df.groupby('patientunitstayid').row_msng_val.sum()
n_msng_val

# + {"execution": {"iopub.status.busy": "2020-03-02T04:33:08.160468Z", "iopub.status.idle": "2020-03-02T04:33:08.161302Z"}}
# Percentage of missing values in each unit stay
msng_val_prct = (n_msng_val / n_data_points) * 100
msng_val_prct

# + {"execution": {"iopub.status.busy": "2020-03-02T04:33:08.162698Z", "iopub.status.idle": "2020-03-02T04:33:08.163590Z"}}
msng_val_prct.describe()

# + [markdown] {"Collapsed": "false"}
# Remove unit stays that have too many missing values (>70% of their respective data points):

# + {"execution": {"iopub.status.busy": "2020-02-26T17:30:52.831884Z", "iopub.status.idle": "2020-02-26T17:30:52.832151Z"}}
unit_stay_high_msgn = set(msng_val_prct[msng_val_prct > 70].index)
unit_stay_high_msgn

# + {"execution": {"iopub.status.busy": "2020-02-26T17:30:52.833186Z", "iopub.status.idle": "2020-02-26T17:30:52.833692Z"}}
eICU_df.patientunitstayid.nunique()
# -

eICU_df = eICU_df[~eICU_df.patientunitstayid.isin(unit_stay_high_msgn)]

eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Removing columns with too many missing values
#
# We should remove features that have too many missing values (in this case, those that have more than 40% of missing values). Without enough data, it's even risky to do imputation, as it's unlikely for the imputation to correctly model the missing feature.
# -

du.search_explore.dataframe_missing_values(eICU_df)

prev_features = eICU_df.columns
len(prev_features)

eICU_df = du.data_processing.remove_cols_with_many_nans(eICU_df, nan_percent_thrsh=70, inplace=True)

features = eICU_df.columns
len(features)

# + [markdown] {"Collapsed": "false"}
# Removed features:
# -

set(prev_features) - set(features)

eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Performing imputation
# -

du.search_explore.dataframe_missing_values(eICU_df)

# Imputate `gender`:

# Forward fill
eICU_df.gender = eICU_df.groupby(id_column).gender.fillna(method='ffill')
# Backward fill
eICU_df.gender = eICU_df.groupby(id_column).gender.fillna(method='bfill')
# Replace remaining missing values with zero
eICU_df.gender = eICU_df.gender.fillna(value=0)

# Imputate the remaining features:

eICU_df = du.data_processing.missing_values_imputation(eICU_df, method='interpolation',
                                                       id_column='patientunitstayid', inplace=True)
eICU_df.head()

du.search_explore.dataframe_missing_values(eICU_df)

# ### Rearranging columns
#
# For ease of use and for better intuition, we should make sure that the ID columns (`patientunitstayid` and `ts`) are the first ones in the dataframe.

columns = list(eICU_df.columns)
columns

columns.remove('patientunitstayid')
columns.remove('ts')

columns = ['patientunitstayid', 'ts'] + columns
columns

eICU_df = eICU_df[columns]
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ## Setting the label
#
# Define the label column considering the desired time window on which we want to predict mortality (0, 24h, 48h, 72h, etc).
# -

time_window_h = 24

eICU_df['label'] = eICU_df[eICU_df.death_ts - eICU_df.ts <= time_window_h * 60]
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ## Creating a single encoding dictionary for the complete dataframe
#
# Combine the one hot encoding dictionaries of all tables, having in account the converged ones, into a single dictionary representative of all the categorical features in the resulting dataframe.

# + {"execution": {"iopub.status.busy": "2020-03-02T04:33:08.143209Z", "iopub.status.idle": "2020-03-02T04:33:08.143643Z"}}
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

# + {"execution": {"iopub.status.busy": "2020-03-02T04:33:08.144659Z", "iopub.status.idle": "2020-03-02T04:33:08.145016Z"}}
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

# + {"execution": {"iopub.status.busy": "2020-03-02T04:33:08.145908Z", "iopub.status.idle": "2020-03-02T04:33:08.146300Z"}}
cat_feat_ohe = du.utils.merge_dicts([cat_feat_ohe_adms_drug, cat_feat_ohe_inf_drug,
                                     cat_feat_ohe_med, cat_feat_ohe_treat,
                                     cat_feat_ohe_diag, cat_feat_ohe_alrg,
                                     cat_feat_ohe_past_hist, cat_feat_ohe_lab,
                                     cat_feat_ohe_patient, cat_feat_ohe_notes])

# + {"execution": {"iopub.status.busy": "2020-03-02T04:33:08.146990Z", "iopub.status.idle": "2020-03-02T04:33:08.147354Z"}}
list(cat_feat_ohe.keys())
# -

# Clean the one hot encoded column names, as they are in the final dataframe:

for key, val in cat_feat_ohe.items():
    cat_feat_ohe[key] = du.data_processing.clean_naming(cat_feat_ohe[key], lower_case=False)

# + [markdown] {"Collapsed": "false"}
# Save the final encoding dictionary:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T04:33:08.148525Z", "iopub.status.idle": "2020-03-02T04:33:08.148811Z"}}
stream = open(f'{data_path}/cleaned/cat_feat_ohe_eICU.yaml', 'w')
yaml.dump(cat_feat_ohe, stream, default_flow_style=False)

# + [markdown] {"Collapsed": "false"}
# ## Creating a single normalization dictionary for the complete dataframe
#
# Combine the normalization stats dictionaries of all tables into a single dictionary representative of all the continuous features in the resulting dataframe.

# + {"execution": {"iopub.status.busy": "2020-03-02T04:33:08.143209Z", "iopub.status.idle": "2020-03-02T04:33:08.143643Z"}}
stream_adms_drug = open(f'{data_path}admissionDrug_norm_stats.yaml', 'r')
stream_inf_drug = open(f'{data_path}infusionDrug_norm_stats.yaml', 'r')
stream_med = open(f'{data_path}medication_norm_stats.yaml', 'r')
# stream_in_out = open(f'{data_path}cat_embed_feat_enum_in_out.yaml', 'r')
stream_lab = open(f'{data_path}lab_norm_stats.yaml', 'r')
stream_patient = open(f'{data_path}patient_norm_stats.yaml', 'r')
# stream_vital_aprdc = open(f'{data_path}vitalAperiodic_norm_stats.yaml', 'r')
# stream_vital_prdc = open(f'{data_path}vitalPeriodic_norm_stats.yaml', 'r')

# + {"execution": {"iopub.status.busy": "2020-03-02T04:33:08.144659Z", "iopub.status.idle": "2020-03-02T04:33:08.145016Z"}}
norm_stats_adms_drug = yaml.load(stream_adms_drug, Loader=yaml.FullLoader)
norm_stats_inf_drug = yaml.load(stream_inf_drug, Loader=yaml.FullLoader)
norm_stats_med = yaml.load(stream_med, Loader=yaml.FullLoader)
# norm_stats__in_out = yaml.load(stream_in_out, Loader=yaml.FullLoader)
norm_stats_lab = yaml.load(stream_lab, Loader=yaml.FullLoader)
norm_stats_patient = yaml.load(stream_patient, Loader=yaml.FullLoader)
# norm_stats_vital_aprdc = yaml.load(stream_vital_aprdc, Loader=yaml.FullLoader)
# norm_stats_vital_prdc = yaml.load(stream_vital_prdc, Loader=yaml.FullLoader)

# + {"execution": {"iopub.status.busy": "2020-03-02T04:33:08.145908Z", "iopub.status.idle": "2020-03-02T04:33:08.146300Z"}}
norm_stats = du.utils.merge_dicts([norm_stats_adms_drug, norm_stats_inf_drug,
                                   norm_stats_med,
#                                    norm_stats_in_out,
                                   norm_stats_lab, norm_stats_patient,
                                   norm_stats_vital_aprdc, norm_stats_vital_prdc])

# + {"execution": {"iopub.status.busy": "2020-03-02T04:33:08.146990Z", "iopub.status.idle": "2020-03-02T04:33:08.147354Z"}}
list(norm_stats.keys())

# + [markdown] {"Collapsed": "false"}
# Save the final encoding dictionary:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T04:33:08.148525Z", "iopub.status.idle": "2020-03-02T04:33:08.148811Z"}}
stream = open(f'{data_path}/cleaned/eICU_norm_stats.yaml', 'w')
yaml.dump(norm_stats, stream, default_flow_style=False)
