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
# # eICU Data Joining
# ---
#
# Reading and joining all preprocessed parts of the eICU dataset from MIT with the data from over 139k patients collected in the US.
#
# The main goal of this notebook is to prepare a single parquet document that contains all the relevant data to be used when training a machine learning model that predicts mortality, joining tables, filtering useless columns and performing imputation.

# + [markdown] {"Collapsed": "false", "colab_type": "text", "id": "KOdmFzXqF7nq"}
# ## Importing the necessary packages

# + {"Collapsed": "false", "colab": {}, "colab_type": "code", "execution": {"iopub.execute_input": "2020-03-20T03:50:08.537999Z", "iopub.status.busy": "2020-03-20T03:50:08.537668Z", "iopub.status.idle": "2020-03-20T03:50:08.586184Z", "shell.execute_reply": "2020-03-20T03:50:08.585395Z", "shell.execute_reply.started": "2020-03-20T03:50:08.537965Z"}, "execution_event_id": "deb57b39-6a79-4b3a-95ed-02f8089ff593", "id": "G5RrWE9R_Nkl", "last_executed_text": "import os                                  # os handles directory/workspace changes\nimport numpy as np                         # NumPy to handle numeric and NaN operations\nimport yaml                                # Save and load YAML files", "persistent_id": "522745b5-b5bf-479f-b697-5c7e9e12fc33"}
import os                                  # os handles directory/workspace changes
import numpy as np                         # NumPy to handle numeric and NaN operations
import yaml                                # Save and load YAML files

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-20T03:50:08.912855Z", "iopub.status.busy": "2020-03-20T03:50:08.912538Z", "iopub.status.idle": "2020-03-20T03:50:09.579112Z", "shell.execute_reply": "2020-03-20T03:50:09.578152Z", "shell.execute_reply.started": "2020-03-20T03:50:08.912823Z"}, "execution_event_id": "fa33a2f7-7127-49c6-bbe9-f89555b1f2be", "last_executed_text": "# Debugging packages\nimport pixiedust                           # Debugging in Jupyter Notebook cells", "persistent_id": "02accdbf-be7e-415c-ba11-165906e66c50"}
# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-20T03:50:09.580960Z", "iopub.status.busy": "2020-03-20T03:50:09.580721Z", "iopub.status.idle": "2020-03-20T03:50:09.584998Z", "shell.execute_reply": "2020-03-20T03:50:09.584234Z", "shell.execute_reply.started": "2020-03-20T03:50:09.580930Z"}, "execution_event_id": "baeb346a-1c34-42d1-a501-7ae37369255e", "last_executed_text": "# Change to parent directory (presumably \"Documents\")\nos.chdir(\"../../..\")\n\n# Path to the parquet dataset files\ndata_path = 'Documents/Datasets/Thesis/eICU/uncompressed/'\n\n# Path to the code files\nproject_path = 'Documents/GitHub/eICU-mortality-prediction/'", "persistent_id": "a1f6ee7f-36d4-489d-b2dd-ec2a38f15d11"}
# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# Path to the parquet dataset files
data_path = 'data/eICU/cleaned/'
# Path to the code files
project_path = 'code/eICU-mortality-prediction/'

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-20T03:50:09.643441Z", "iopub.status.busy": "2020-03-20T03:50:09.643122Z", "iopub.status.idle": "2020-03-20T03:50:09.899790Z", "shell.execute_reply": "2020-03-20T03:50:09.898577Z", "shell.execute_reply.started": "2020-03-20T03:50:09.643409Z"}}
# Make sure that every large operation can be handled, by using the disk as an overflow for the memory
# !export MODIN_OUT_OF_CORE=true
# Another trick to do with Pandas so as to be able to allocate bigger objects to memory
# !sudo bash -c 'echo 1 > /proc/sys/vm/overcommit_memory'

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-20T03:50:09.901993Z", "iopub.status.busy": "2020-03-20T03:50:09.901736Z", "iopub.status.idle": "2020-03-20T03:50:11.875679Z", "shell.execute_reply": "2020-03-20T03:50:11.874853Z", "shell.execute_reply.started": "2020-03-20T03:50:09.901959Z"}, "execution_event_id": "82ef68be-443a-4bb8-8abd-7457a7005b4d", "last_executed_text": "import modin.pandas as pd                  # Optimized distributed version of Pandas\nimport data_utils as du                    # Data science and machine learning relevant methods", "persistent_id": "c0c2e356-d4f4-4a9d-bec2-88bdf9eb6a38"}
import modin.pandas as mpd                  # Optimized distributed version of Pandas
import pandas as pd
import data_utils as du                    # Data science and machine learning relevant methods

# + [markdown] {"Collapsed": "false"}
# Allow pandas to show more columns:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-20T03:50:11.877401Z", "iopub.status.busy": "2020-03-20T03:50:11.877169Z", "iopub.status.idle": "2020-03-20T03:50:11.881088Z", "shell.execute_reply": "2020-03-20T03:50:11.880412Z", "shell.execute_reply.started": "2020-03-20T03:50:11.877363Z"}}
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

# + [markdown] {"Collapsed": "false"}
# Set the random seed for reproducibility

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-20T03:50:11.882280Z", "iopub.status.busy": "2020-03-20T03:50:11.882063Z", "iopub.status.idle": "2020-03-20T03:50:11.888553Z", "shell.execute_reply": "2020-03-20T03:50:11.887844Z", "shell.execute_reply.started": "2020-03-20T03:50:11.882252Z"}, "execution_event_id": "29ab85ce-b7fd-4c5a-a110-5841e741c369", "last_executed_text": "du.set_random_seed(42)", "persistent_id": "39b552cd-6948-4ec8-ac04-42f850c1e05a"}
du.set_random_seed(42)

# + [markdown] {"Collapsed": "false"}
# ## Initializing variables

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-20T03:50:11.889883Z", "iopub.status.busy": "2020-03-20T03:50:11.889655Z", "iopub.status.idle": "2020-03-20T03:50:11.899913Z", "shell.execute_reply": "2020-03-20T03:50:11.899158Z", "shell.execute_reply.started": "2020-03-20T03:50:11.889855Z"}}
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

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-20T03:50:12.323253Z", "iopub.status.busy": "2020-03-20T03:50:12.322946Z", "iopub.status.idle": "2020-03-20T03:50:12.329326Z", "shell.execute_reply": "2020-03-20T03:50:12.328621Z", "shell.execute_reply.started": "2020-03-20T03:50:12.323221Z"}}
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

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-20T03:50:12.653612Z", "iopub.status.busy": "2020-03-20T03:50:12.653305Z", "iopub.status.idle": "2020-03-20T03:50:12.829736Z", "shell.execute_reply": "2020-03-20T03:50:12.828956Z", "shell.execute_reply.started": "2020-03-20T03:50:12.653581Z"}}
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

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-20T03:50:12.983216Z", "iopub.status.busy": "2020-03-20T03:50:12.982944Z", "iopub.status.idle": "2020-03-20T03:50:12.987008Z", "shell.execute_reply": "2020-03-20T03:50:12.986300Z", "shell.execute_reply.started": "2020-03-20T03:50:12.983185Z"}}
cat_feat_ohe = du.utils.merge_dicts([cat_feat_ohe_med, cat_feat_ohe_treat,
                                     cat_feat_ohe_diag,
                                     cat_feat_ohe_past_hist, cat_feat_ohe_lab,
                                     cat_feat_ohe_patient, cat_feat_ohe_notes])

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-20T03:50:13.494089Z", "iopub.status.busy": "2020-03-20T03:50:13.493788Z", "iopub.status.idle": "2020-03-20T03:50:13.499204Z", "shell.execute_reply": "2020-03-20T03:50:13.498288Z", "shell.execute_reply.started": "2020-03-20T03:50:13.494058Z"}}
ohe_columns = du.utils.merge_lists(list(cat_feat_ohe.values()))
ohe_columns = du.data_processing.clean_naming(ohe_columns, lower_case=False)

# + [markdown] {"Collapsed": "false"}
# Add the one hot encoded columns to the dtypes dictionary, specifying them with type `UInt8`

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-20T03:50:14.365745Z", "iopub.status.busy": "2020-03-20T03:50:14.365436Z", "iopub.status.idle": "2020-03-20T03:50:14.370297Z", "shell.execute_reply": "2020-03-20T03:50:14.369485Z", "shell.execute_reply.started": "2020-03-20T03:50:14.365712Z"}}
for col in ohe_columns:
    dtype_dict[col] = 'UInt8'

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T15:42:28.695715Z", "iopub.status.busy": "2020-03-19T15:42:28.695505Z", "iopub.status.idle": "2020-03-19T15:42:28.729242Z", "shell.execute_reply": "2020-03-19T15:42:28.727839Z", "shell.execute_reply.started": "2020-03-19T15:42:28.695689Z"}}
dtype_dict

# + [markdown] {"Collapsed": "false"}
# ## Loading the data

# + [markdown] {"Collapsed": "false"}
# ### Patient information

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-17T05:05:20.998608Z", "iopub.status.busy": "2020-03-17T05:05:20.998312Z", "iopub.status.idle": "2020-03-17T05:05:22.129647Z", "shell.execute_reply": "2020-03-17T05:05:22.128691Z", "shell.execute_reply.started": "2020-03-17T05:05:20.998563Z"}}
patient_df = mpd.read_csv(f'{data_path}normalized/ohe/patient.csv', dtype=dtype_dict)
patient_df = du.utils.convert_dataframe(patient_df, to='pandas', return_library=False, dtypes=dtype_dict)
patient_df = patient_df.drop(columns='Unnamed: 0')
patient_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-17T02:12:19.851527Z", "iopub.status.busy": "2020-03-17T05:04:22.707095Z", "iopub.status.idle": "2020-03-17T05:04:22.707700Z", "shell.execute_reply": "2020-03-17T02:12:20.044451Z", "shell.execute_reply.started": "2020-03-17T02:12:19.851463Z"}}
note_df = mpd.read_csv(f'{data_path}normalized/ohe/note.csv', dtype=dtype_dict)
note_df = du.utils.convert_dataframe(note_df, to='pandas', return_library=False, dtypes=dtype_dict)
note_df = note_df.drop(columns='Unnamed: 0')
note_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Diagnosis

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-17T02:18:33.721778Z", "iopub.status.busy": "2020-03-17T02:18:33.721453Z", "iopub.status.idle": "2020-03-17T02:21:02.390931Z", "shell.execute_reply": "2020-03-17T02:21:02.390002Z", "shell.execute_reply.started": "2020-03-17T02:18:33.721720Z"}}
diagns_df = mpd.read_csv(f'{data_path}normalized/ohe/diagnosis.csv', dtype=dtype_dict)
diagns_df = du.utils.convert_dataframe(diagns_df, to='pandas', return_library=False, dtypes=dtype_dict)
diagns_df = diagns_df.drop(columns='Unnamed: 0')
diagns_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:00:14.051698Z", "iopub.status.busy": "2020-03-16T19:00:14.051303Z", "iopub.status.idle": "2020-03-16T19:00:21.933263Z", "shell.execute_reply": "2020-03-16T19:00:21.932045Z", "shell.execute_reply.started": "2020-03-16T19:00:14.051642Z"}}
# alrg_df = mpd.read_csv(f'{data_path}normalized/ohe/allergy.csv', dtype=dtype_dict)
# alrg_df = du.utils.convert_dataframe(alrg_df, to='pandas', return_library=False, dtypes=dtype_dict)
# alrg_df = alrg_df.drop(columns='Unnamed: 0')
# alrg_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:00:21.934850Z", "iopub.status.busy": "2020-03-16T19:00:21.934520Z", "iopub.status.idle": "2020-03-16T19:00:33.637145Z", "shell.execute_reply": "2020-03-16T19:00:33.636199Z", "shell.execute_reply.started": "2020-03-16T19:00:21.934806Z"}}
past_hist_df = mpd.read_csv(f'{data_path}normalized/ohe/pastHistory.csv', dtype=dtype_dict)
past_hist_df = du.utils.convert_dataframe(past_hist_df, to='pandas', return_library=False, dtypes=dtype_dict)
past_hist_df = past_hist_df.drop(columns='Unnamed: 0')
past_hist_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Treatments

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:00:33.658038Z", "iopub.status.busy": "2020-03-16T19:00:33.657779Z", "iopub.status.idle": "2020-03-16T19:01:32.449135Z", "shell.execute_reply": "2020-03-16T19:01:32.447991Z", "shell.execute_reply.started": "2020-03-16T19:00:33.657989Z"}}
treat_df = mpd.read_csv(f'{data_path}normalized/ohe/treatment.csv', dtype=dtype_dict)
treat_df = du.utils.convert_dataframe(treat_df, to='pandas', return_library=False, dtypes=dtype_dict)
treat_df = treat_df.drop(columns='Unnamed: 0')
treat_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-18T23:55:16.707578Z", "iopub.status.busy": "2020-03-18T23:55:16.707277Z", "iopub.status.idle": "2020-03-18T23:55:20.090803Z", "shell.execute_reply": "2020-03-18T23:55:20.090014Z", "shell.execute_reply.started": "2020-03-18T23:55:16.707548Z"}}
adms_drug_df = mpd.read_csv(f'{data_path}normalized/ohe/admissionDrug.csv', dtype=dtype_dict)
adms_drug_df = du.utils.convert_dataframe(adms_drug_df, to='pandas', return_library=False, dtypes=dtype_dict)
adms_drug_df = adms_drug_df.drop(columns='Unnamed: 0')
adms_drug_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:19:22.199248Z", "iopub.status.busy": "2020-03-16T19:19:22.198988Z", "iopub.status.idle": "2020-03-16T19:20:27.980609Z", "shell.execute_reply": "2020-03-16T19:20:27.978951Z", "shell.execute_reply.started": "2020-03-16T19:19:22.199195Z"}}
# inf_drug_df = mpd.read_csv(f'{data_path}normalized/ohe/infusionDrug.csv', dtype=dtype_dict)
# inf_drug_df = du.utils.convert_dataframe(inf_drug_df, to='pandas', return_library=False, dtypes=dtype_dict)
# inf_drug_df = inf_drug_df.drop(columns='Unnamed: 0')
# inf_drug_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-18T23:55:20.092277Z", "iopub.status.busy": "2020-03-18T23:55:20.092051Z", "iopub.status.idle": "2020-03-19T00:03:44.110286Z", "shell.execute_reply": "2020-03-19T00:03:44.109302Z", "shell.execute_reply.started": "2020-03-18T23:55:20.092251Z"}}
med_df = mpd.read_csv(f'{data_path}normalized/ohe/medication.csv', dtype=dtype_dict)
med_df = du.utils.convert_dataframe(med_df, to='pandas', return_library=False, dtypes=dtype_dict)
med_df = med_df.drop(columns='Unnamed: 0')
med_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-15T14:14:52.587453Z", "iopub.status.busy": "2020-03-15T14:14:52.586985Z", "iopub.status.idle": "2020-03-15T14:17:00.683297Z", "shell.execute_reply": "2020-03-15T14:17:00.682629Z", "shell.execute_reply.started": "2020-03-15T14:14:52.587405Z"}}
# in_out_df = mpd.read_csv(f'{data_path}normalized/intakeOutput.csv', dtype=dtype_dict)
# in_out_df = du.utils.convert_dataframe(in_out_df, to='pandas', return_library=False, dtypes=dtype_dict)
# in_out_df = in_out_df.drop(columns='Unnamed: 0')
# in_out_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Nursing data

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-15T02:14:05.712354Z", "iopub.status.busy": "2020-03-15T04:51:14.978567Z", "iopub.status.idle": "2020-03-15T04:51:14.978880Z", "shell.execute_reply": "2020-03-15T02:14:05.714353Z", "shell.execute_reply.started": "2020-03-15T02:14:05.712316Z"}}
# nurse_care_df = mpd.read_csv(f'{data_path}normalized/ohe/nurseCare.csv')
# nurse_care_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-15T02:14:05.716226Z", "iopub.status.busy": "2020-03-15T04:51:14.979722Z", "iopub.status.idle": "2020-03-15T04:51:14.980057Z", "shell.execute_reply": "2020-03-15T02:14:05.718753Z", "shell.execute_reply.started": "2020-03-15T02:14:05.716191Z"}}
# nurse_assess_df = mpd.read_csv(f'{data_path}normalized/ohe/nurseAssessment.csv')
# nurse_assess_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-15T02:14:05.720500Z", "iopub.status.busy": "2020-03-15T04:51:14.980824Z", "iopub.status.idle": "2020-03-15T04:51:14.981136Z", "shell.execute_reply": "2020-03-15T02:14:05.723574Z", "shell.execute_reply.started": "2020-03-15T02:14:05.720462Z"}}
# nurse_care_df = nurse_care_df.drop(columns='Unnamed: 0')
# nurse_assess_df = nurse_assess_df.drop(columns='Unnamed: 0')

# + [markdown] {"Collapsed": "false"}
# ### Respiratory data

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T14:27:35.765653Z", "iopub.status.busy": "2020-03-19T14:27:35.765374Z", "iopub.status.idle": "2020-03-19T14:27:36.582532Z", "shell.execute_reply": "2020-03-19T14:27:36.581453Z", "shell.execute_reply.started": "2020-03-19T14:27:35.765624Z"}}
resp_care_df = mpd.read_csv(f'{data_path}normalized/ohe/respiratoryCare.csv', dtype=dtype_dict)
resp_care_df = du.utils.convert_dataframe(resp_care_df, to='pandas', return_library=False, dtypes=dtype_dict)
resp_care_df = resp_care_df.drop(columns='Unnamed: 0')
resp_care_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Vital signals

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T23:43:12.338139Z", "iopub.status.busy": "2020-03-19T23:43:12.337684Z", "iopub.status.idle": "2020-03-19T23:43:30.706928Z", "shell.execute_reply": "2020-03-19T23:43:30.705990Z", "shell.execute_reply.started": "2020-03-19T23:43:12.338103Z"}}
vital_aprdc_df = mpd.read_csv(f'{data_path}normalized/vitalAperiodic.csv', dtype=dtype_dict)
vital_aprdc_df = du.utils.convert_dataframe(vital_aprdc_df, to='pandas', return_library=False, dtypes=dtype_dict)
vital_aprdc_df = vital_aprdc_df.drop(columns='Unnamed: 0')
vital_aprdc_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-20T03:50:20.672954Z", "iopub.status.busy": "2020-03-20T03:50:20.672670Z", "iopub.status.idle": "2020-03-20T03:51:52.675735Z", "shell.execute_reply": "2020-03-20T03:51:52.674861Z", "shell.execute_reply.started": "2020-03-20T03:50:20.672923Z"}}
vital_prdc_df = mpd.read_csv(f'{data_path}normalized/ohe/vitalPeriodic.csv', dtype=dtype_dict)
vital_prdc_df = du.utils.convert_dataframe(vital_prdc_df, to='pandas', return_library=False, dtypes=dtype_dict)
vital_prdc_df = vital_prdc_df.drop(columns='Unnamed: 0')
vital_prdc_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Exams data

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-15T02:19:23.890554Z", "iopub.status.busy": "2020-03-15T02:19:23.890361Z", "iopub.status.idle": "2020-03-15T02:21:58.259579Z", "shell.execute_reply": "2020-03-15T02:21:58.258747Z", "shell.execute_reply.started": "2020-03-15T02:19:23.890518Z"}}
lab_df = mpd.read_csv(f'{data_path}normalized/ohe/lab.csv', dtype=dtype_dict)
lab_df = du.utils.convert_dataframe(lab_df, to='pandas', return_library=False, dtypes=dtype_dict)
lab_df = lab_df.drop(columns='Unnamed: 0')
lab_df.head()

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": false}
# ## Joining dataframes

# + [markdown] {"Collapsed": "false"}
# ### Checking the matching of unit stays IDs

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:39.678558Z", "iopub.status.busy": "2020-02-26T17:06:39.678332Z", "iopub.status.idle": "2020-02-26T17:06:40.093070Z", "shell.execute_reply": "2020-02-26T17:06:40.092337Z", "shell.execute_reply.started": "2020-02-26T17:06:39.678520Z"}}
full_stays_list = set(patient_df.patientunitstayid.unique())

# + [markdown] {"Collapsed": "false"}
# Total number of unit stays:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:40.094749Z", "iopub.status.busy": "2020-02-26T17:06:40.094402Z", "iopub.status.idle": "2020-02-26T17:06:40.099961Z", "shell.execute_reply": "2020-02-26T17:06:40.099190Z", "shell.execute_reply.started": "2020-02-26T17:06:40.094687Z"}}
len(full_stays_list)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:40.101929Z", "iopub.status.busy": "2020-02-26T17:06:40.101255Z", "iopub.status.idle": "2020-02-26T17:06:40.226963Z", "shell.execute_reply": "2020-02-26T17:06:40.225874Z", "shell.execute_reply.started": "2020-02-26T17:06:40.101857Z"}}
note_stays_list = set(note_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:40.229023Z", "iopub.status.busy": "2020-02-26T17:06:40.228664Z", "iopub.status.idle": "2020-02-26T17:06:40.234264Z", "shell.execute_reply": "2020-02-26T17:06:40.233560Z", "shell.execute_reply.started": "2020-02-26T17:06:40.228961Z"}}
len(note_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have note data:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:40.235905Z", "iopub.status.busy": "2020-02-26T17:06:40.235590Z", "iopub.status.idle": "2020-02-26T17:06:40.253443Z", "shell.execute_reply": "2020-02-26T17:06:40.252738Z", "shell.execute_reply.started": "2020-02-26T17:06:40.235847Z"}}
len(set.intersection(full_stays_list, note_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:40.254923Z", "iopub.status.busy": "2020-02-26T17:06:40.254692Z", "iopub.status.idle": "2020-02-26T17:06:40.648925Z", "shell.execute_reply": "2020-02-26T17:06:40.648116Z", "shell.execute_reply.started": "2020-02-26T17:06:40.254868Z"}}
diagns_stays_list = set(diagns_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:40.650198Z", "iopub.status.busy": "2020-02-26T17:06:40.649995Z", "iopub.status.idle": "2020-02-26T17:06:40.654306Z", "shell.execute_reply": "2020-02-26T17:06:40.653650Z", "shell.execute_reply.started": "2020-02-26T17:06:40.650161Z"}}
len(diagns_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have diagnosis data:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:40.655413Z", "iopub.status.busy": "2020-02-26T17:06:40.655215Z", "iopub.status.idle": "2020-02-26T17:06:40.704945Z", "shell.execute_reply": "2020-02-26T17:06:40.704010Z", "shell.execute_reply.started": "2020-02-26T17:06:40.655377Z"}}
len(set.intersection(full_stays_list, diagns_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:40.706193Z", "iopub.status.busy": "2020-02-26T17:06:40.705977Z", "iopub.status.idle": "2020-02-26T17:06:40.880141Z", "shell.execute_reply": "2020-02-26T17:06:40.879409Z", "shell.execute_reply.started": "2020-02-26T17:06:40.706156Z"}}
# alrg_stays_list = set(alrg_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:40.881405Z", "iopub.status.busy": "2020-02-26T17:06:40.881198Z", "iopub.status.idle": "2020-02-26T17:06:40.885978Z", "shell.execute_reply": "2020-02-26T17:06:40.885401Z", "shell.execute_reply.started": "2020-02-26T17:06:40.881369Z"}}
# len(alrg_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have allergy data:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:40.887083Z", "iopub.status.busy": "2020-02-26T17:06:40.886881Z", "iopub.status.idle": "2020-02-26T17:06:40.910256Z", "shell.execute_reply": "2020-02-26T17:06:40.909552Z", "shell.execute_reply.started": "2020-02-26T17:06:40.887046Z"}}
# len(set.intersection(full_stays_list, alrg_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:40.911618Z", "iopub.status.busy": "2020-02-26T17:06:40.911370Z", "iopub.status.idle": "2020-02-26T17:06:41.116675Z", "shell.execute_reply": "2020-02-26T17:06:41.115974Z", "shell.execute_reply.started": "2020-02-26T17:06:40.911579Z"}}
past_hist_stays_list = set(past_hist_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:41.117964Z", "iopub.status.busy": "2020-02-26T17:06:41.117749Z", "iopub.status.idle": "2020-02-26T17:06:41.122087Z", "shell.execute_reply": "2020-02-26T17:06:41.121497Z", "shell.execute_reply.started": "2020-02-26T17:06:41.117925Z"}}
len(past_hist_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have past history data:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:41.123220Z", "iopub.status.busy": "2020-02-26T17:06:41.123031Z", "iopub.status.idle": "2020-02-26T17:06:41.155381Z", "shell.execute_reply": "2020-02-26T17:06:41.154649Z", "shell.execute_reply.started": "2020-02-26T17:06:41.123186Z"}}
len(set.intersection(full_stays_list, past_hist_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:41.156784Z", "iopub.status.busy": "2020-02-26T17:06:41.156558Z", "iopub.status.idle": "2020-02-26T17:06:41.531138Z", "shell.execute_reply": "2020-02-26T17:06:41.530382Z", "shell.execute_reply.started": "2020-02-26T17:06:41.156745Z"}}
treat_stays_list = set(treat_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:41.532436Z", "iopub.status.busy": "2020-02-26T17:06:41.532226Z", "iopub.status.idle": "2020-02-26T17:06:41.536740Z", "shell.execute_reply": "2020-02-26T17:06:41.536123Z", "shell.execute_reply.started": "2020-02-26T17:06:41.532398Z"}}
len(treat_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have treatment data:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:41.538002Z", "iopub.status.busy": "2020-02-26T17:06:41.537742Z", "iopub.status.idle": "2020-02-26T17:06:41.577114Z", "shell.execute_reply": "2020-02-26T17:06:41.576445Z", "shell.execute_reply.started": "2020-02-26T17:06:41.537957Z"}}
len(set.intersection(full_stays_list, treat_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:41.578290Z", "iopub.status.busy": "2020-02-26T17:06:41.578084Z", "iopub.status.idle": "2020-02-26T17:06:41.743253Z", "shell.execute_reply": "2020-02-26T17:06:41.742093Z", "shell.execute_reply.started": "2020-02-26T17:06:41.578253Z"}}
adms_drug_stays_list = set(adms_drug_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:41.744412Z", "iopub.status.busy": "2020-02-26T17:06:41.744209Z", "iopub.status.idle": "2020-02-26T17:06:41.748862Z", "shell.execute_reply": "2020-02-26T17:06:41.748142Z", "shell.execute_reply.started": "2020-02-26T17:06:41.744375Z"}}
len(adms_drug_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have admission drug data:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:41.749931Z", "iopub.status.busy": "2020-02-26T17:06:41.749745Z", "iopub.status.idle": "2020-02-26T17:06:41.764151Z", "shell.execute_reply": "2020-02-26T17:06:41.763601Z", "shell.execute_reply.started": "2020-02-26T17:06:41.749897Z"}}
len(set.intersection(full_stays_list, adms_drug_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:41.765156Z", "iopub.status.busy": "2020-02-26T17:06:41.764961Z", "iopub.status.idle": "2020-02-26T17:06:42.309915Z", "shell.execute_reply": "2020-02-26T17:06:42.309105Z", "shell.execute_reply.started": "2020-02-26T17:06:41.765122Z"}}
# inf_drug_stays_list = set(inf_drug_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:42.311311Z", "iopub.status.busy": "2020-02-26T17:06:42.311041Z", "iopub.status.idle": "2020-02-26T17:06:42.315850Z", "shell.execute_reply": "2020-02-26T17:06:42.315150Z", "shell.execute_reply.started": "2020-02-26T17:06:42.311265Z"}}
# len(inf_drug_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have infusion drug data:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:42.317160Z", "iopub.status.busy": "2020-02-26T17:06:42.316935Z", "iopub.status.idle": "2020-02-26T17:06:42.341538Z", "shell.execute_reply": "2020-02-26T17:06:42.340907Z", "shell.execute_reply.started": "2020-02-26T17:06:42.317119Z"}}
# len(set.intersection(full_stays_list, inf_drug_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:42.342753Z", "iopub.status.busy": "2020-02-26T17:06:42.342546Z", "iopub.status.idle": "2020-02-26T17:06:43.217103Z", "shell.execute_reply": "2020-02-26T17:06:43.216344Z", "shell.execute_reply.started": "2020-02-26T17:06:42.342716Z"}}
med_stays_list = set(med_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:43.218322Z", "iopub.status.busy": "2020-02-26T17:06:43.218112Z", "iopub.status.idle": "2020-02-26T17:06:43.222301Z", "shell.execute_reply": "2020-02-26T17:06:43.221665Z", "shell.execute_reply.started": "2020-02-26T17:06:43.218285Z"}}
len(med_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have medication data:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:43.223466Z", "iopub.status.busy": "2020-02-26T17:06:43.223261Z", "iopub.status.idle": "2020-02-26T17:06:43.268485Z", "shell.execute_reply": "2020-02-26T17:06:43.267653Z", "shell.execute_reply.started": "2020-02-26T17:06:43.223429Z"}}
len(set.intersection(full_stays_list, med_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:43.269688Z", "iopub.status.busy": "2020-02-26T17:06:43.269468Z", "iopub.status.idle": "2020-02-26T17:06:44.458225Z", "shell.execute_reply": "2020-02-26T17:06:44.457486Z", "shell.execute_reply.started": "2020-02-26T17:06:43.269651Z"}}
# in_out_stays_list = set(in_out_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:44.459571Z", "iopub.status.busy": "2020-02-26T17:06:44.459345Z", "iopub.status.idle": "2020-02-26T17:06:44.464285Z", "shell.execute_reply": "2020-02-26T17:06:44.463605Z", "shell.execute_reply.started": "2020-02-26T17:06:44.459530Z"}}
# len(in_out_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have intake and output data:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:44.465431Z", "iopub.status.busy": "2020-02-26T17:06:44.465226Z", "iopub.status.idle": "2020-02-26T17:06:44.842575Z", "shell.execute_reply": "2020-02-26T17:06:44.841735Z", "shell.execute_reply.started": "2020-02-26T17:06:44.465394Z"}}
# len(set.intersection(full_stays_list, in_out_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:44.844142Z", "iopub.status.busy": "2020-02-26T17:06:44.843620Z", "iopub.status.idle": "2020-02-26T17:06:44.846905Z", "shell.execute_reply": "2020-02-26T17:06:44.846256Z", "shell.execute_reply.started": "2020-02-26T17:06:44.844096Z"}}
# nurse_care_stays_list = set(nurse_care_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:44.847965Z", "iopub.status.busy": "2020-02-26T17:06:44.847765Z", "iopub.status.idle": "2020-02-26T17:06:44.852734Z", "shell.execute_reply": "2020-02-26T17:06:44.852149Z", "shell.execute_reply.started": "2020-02-26T17:06:44.847929Z"}}
# len(nurse_care_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have nurse care data:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:44.853706Z", "iopub.status.busy": "2020-02-26T17:06:44.853502Z", "iopub.status.idle": "2020-02-26T17:06:44.857298Z", "shell.execute_reply": "2020-02-26T17:06:44.856683Z", "shell.execute_reply.started": "2020-02-26T17:06:44.853668Z"}}
# len(set.intersection(full_stays_list, nurse_care_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:44.858259Z", "iopub.status.busy": "2020-02-26T17:06:44.858075Z", "iopub.status.idle": "2020-02-26T17:06:44.861888Z", "shell.execute_reply": "2020-02-26T17:06:44.861335Z", "shell.execute_reply.started": "2020-02-26T17:06:44.858226Z"}}
# nurse_assess_stays_list = set(nurse_assess_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:44.881939Z", "iopub.status.busy": "2020-02-26T17:06:44.881703Z", "iopub.status.idle": "2020-02-26T17:06:44.884822Z", "shell.execute_reply": "2020-02-26T17:06:44.884167Z", "shell.execute_reply.started": "2020-02-26T17:06:44.881899Z"}}
# len(nurse_assess_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have nurse assessment data:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:44.887349Z", "iopub.status.busy": "2020-02-26T17:06:44.887122Z", "iopub.status.idle": "2020-02-26T17:06:44.890836Z", "shell.execute_reply": "2020-02-26T17:06:44.889727Z", "shell.execute_reply.started": "2020-02-26T17:06:44.887299Z"}}
# len(set.intersection(full_stays_list, nurse_assess_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:44.891859Z", "iopub.status.busy": "2020-02-26T17:06:44.891668Z", "iopub.status.idle": "2020-02-26T17:06:45.008553Z", "shell.execute_reply": "2020-02-26T17:06:45.007454Z", "shell.execute_reply.started": "2020-02-26T17:06:44.891825Z"}}
resp_care_stays_list = set(resp_care_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:45.009763Z", "iopub.status.busy": "2020-02-26T17:06:45.009556Z", "iopub.status.idle": "2020-02-26T17:06:45.013955Z", "shell.execute_reply": "2020-02-26T17:06:45.013334Z", "shell.execute_reply.started": "2020-02-26T17:06:45.009726Z"}}
len(resp_care_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have respiratory care data:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:45.015043Z", "iopub.status.busy": "2020-02-26T17:06:45.014823Z", "iopub.status.idle": "2020-02-26T17:06:45.032842Z", "shell.execute_reply": "2020-02-26T17:06:45.032224Z", "shell.execute_reply.started": "2020-02-26T17:06:45.015006Z"}}
len(set.intersection(full_stays_list, resp_care_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:45.033892Z", "iopub.status.busy": "2020-02-26T17:06:45.033691Z", "iopub.status.idle": "2020-02-26T17:06:48.311737Z", "shell.execute_reply": "2020-02-26T17:06:48.310937Z", "shell.execute_reply.started": "2020-02-26T17:06:45.033849Z"}}
vital_aprdc_stays_list = set(vital_aprdc_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:48.313221Z", "iopub.status.busy": "2020-02-26T17:06:48.312988Z", "iopub.status.idle": "2020-02-26T17:06:48.317433Z", "shell.execute_reply": "2020-02-26T17:06:48.316834Z", "shell.execute_reply.started": "2020-02-26T17:06:48.313180Z"}}
len(vital_aprdc_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have vital aperiodic data:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:48.318552Z", "iopub.status.busy": "2020-02-26T17:06:48.318331Z", "iopub.status.idle": "2020-02-26T17:06:48.373587Z", "shell.execute_reply": "2020-02-26T17:06:48.372779Z", "shell.execute_reply.started": "2020-02-26T17:06:48.318516Z"}}
len(set.intersection(full_stays_list, vital_aprdc_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:06:48.374897Z", "iopub.status.busy": "2020-02-26T17:06:48.374671Z", "iopub.status.idle": "2020-02-26T17:07:13.514154Z", "shell.execute_reply": "2020-02-26T17:07:13.513440Z", "shell.execute_reply.started": "2020-02-26T17:06:48.374855Z"}}
vital_prdc_stays_list = set(vital_prdc_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:07:13.515395Z", "iopub.status.busy": "2020-02-26T17:07:13.515189Z", "iopub.status.idle": "2020-02-26T17:07:13.519472Z", "shell.execute_reply": "2020-02-26T17:07:13.518805Z", "shell.execute_reply.started": "2020-02-26T17:07:13.515357Z"}}
len(vital_prdc_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have vital periodic data:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:07:13.520519Z", "iopub.status.busy": "2020-02-26T17:07:13.520320Z", "iopub.status.idle": "2020-02-26T17:07:22.498386Z", "shell.execute_reply": "2020-02-26T17:07:22.497724Z", "shell.execute_reply.started": "2020-02-26T17:07:13.520484Z"}}
len(set.intersection(full_stays_list, vital_prdc_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:07:22.500158Z", "iopub.status.busy": "2020-02-26T17:07:22.499818Z", "iopub.status.idle": "2020-02-26T17:07:26.687838Z", "shell.execute_reply": "2020-02-26T17:07:26.687020Z", "shell.execute_reply.started": "2020-02-26T17:07:22.500117Z"}}
lab_stays_list = set(lab_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:07:26.689281Z", "iopub.status.busy": "2020-02-26T17:07:26.688879Z", "iopub.status.idle": "2020-02-26T17:07:26.693687Z", "shell.execute_reply": "2020-02-26T17:07:26.693035Z", "shell.execute_reply.started": "2020-02-26T17:07:26.689058Z"}}
len(lab_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have lab data:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:07:26.694807Z", "iopub.status.busy": "2020-02-26T17:07:26.694609Z", "iopub.status.idle": "2020-02-26T17:07:27.104262Z", "shell.execute_reply": "2020-02-26T17:07:27.103549Z", "shell.execute_reply.started": "2020-02-26T17:07:26.694770Z"}}
len(set.intersection(full_stays_list, lab_stays_list))

# + [markdown] {"Collapsed": "false"}
# ### Joining patient with note data

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-17T02:12:32.588262Z", "iopub.status.busy": "2020-03-17T02:12:32.587992Z", "iopub.status.idle": "2020-03-17T02:12:32.643325Z", "shell.execute_reply": "2020-03-17T02:12:32.642479Z", "shell.execute_reply.started": "2020-03-17T02:12:32.588222Z"}}
eICU_df = patient_df
eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
note_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# + [markdown] {"Collapsed": "false"}
# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-17T02:12:34.917662Z", "iopub.status.busy": "2020-03-17T02:12:34.917364Z", "iopub.status.idle": "2020-03-17T02:12:35.706182Z", "shell.execute_reply": "2020-03-17T02:12:35.705426Z", "shell.execute_reply.started": "2020-03-17T02:12:34.917611Z"}}
eICU_df = eICU_df.join(note_df, how='left')
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
eICU_df.reset_index(inplace=True)
eICU_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-17T02:12:54.059203Z", "iopub.status.busy": "2020-03-17T02:12:54.058760Z", "iopub.status.idle": "2020-03-17T02:12:54.153541Z", "shell.execute_reply": "2020-03-17T02:12:54.152508Z", "shell.execute_reply.started": "2020-03-17T02:12:54.059139Z"}}
eICU_df.groupby(['patientunitstayid', 'ts']).size().sort_values()

# + {"Collapsed": "false"}
eICU_df.dtypes

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-17T02:13:07.956140Z", "iopub.status.busy": "2020-03-17T02:13:07.955827Z", "iopub.status.idle": "2020-03-17T02:13:08.250772Z", "shell.execute_reply": "2020-03-17T02:13:08.249833Z", "shell.execute_reply.started": "2020-03-17T02:13:07.956089Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_diagns.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with diagnosis data

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-17T02:21:02.530354Z", "iopub.status.busy": "2020-03-17T02:21:02.530014Z", "iopub.status.idle": "2020-03-17T02:21:02.570799Z", "shell.execute_reply": "2020-03-17T02:21:02.569848Z", "shell.execute_reply.started": "2020-03-17T02:21:02.530289Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_diagns.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Filter to the unit stays that also have data in the other tables:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-17T02:21:02.572327Z", "iopub.status.busy": "2020-03-17T02:21:02.572058Z", "iopub.status.idle": "2020-03-17T02:21:03.733294Z", "shell.execute_reply": "2020-03-17T02:21:03.732468Z", "shell.execute_reply.started": "2020-03-17T02:21:02.572267Z"}}
diagns_df = diagns_df[diagns_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

# + [markdown] {"Collapsed": "false"}
# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-17T02:21:03.734674Z", "iopub.status.busy": "2020-03-17T02:21:03.734428Z", "iopub.status.idle": "2020-03-17T02:21:03.779961Z", "shell.execute_reply": "2020-03-17T02:21:03.779224Z", "shell.execute_reply.started": "2020-03-17T02:21:03.734627Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(diagns_df.patientunitstayid.unique())]

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-17T02:21:03.781508Z", "iopub.status.busy": "2020-03-17T02:21:03.781208Z", "iopub.status.idle": "2020-03-17T02:21:03.792554Z", "shell.execute_reply": "2020-03-17T02:21:03.791579Z", "shell.execute_reply.started": "2020-03-17T02:21:03.781449Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-17T02:21:03.793838Z", "iopub.status.busy": "2020-03-17T02:21:03.793552Z", "iopub.status.idle": "2020-03-17T02:21:03.908162Z", "shell.execute_reply": "2020-03-17T02:21:03.907312Z", "shell.execute_reply.started": "2020-03-17T02:21:03.793792Z"}}
eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
diagns_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# + [markdown] {"Collapsed": "false"}
# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-17T02:21:03.910430Z", "iopub.status.busy": "2020-03-17T02:21:03.910110Z", "iopub.status.idle": "2020-03-17T02:21:18.726951Z", "shell.execute_reply": "2020-03-17T02:21:18.726037Z", "shell.execute_reply.started": "2020-03-17T02:21:03.910350Z"}}
eICU_df = eICU_df.join(diagns_df, how='outer')
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
eICU_df.reset_index(inplace=True)
eICU_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-17T02:21:19.272995Z", "iopub.status.busy": "2020-03-17T02:21:19.272691Z", "iopub.status.idle": "2020-03-17T02:21:20.789968Z", "shell.execute_reply": "2020-03-17T02:21:20.789231Z", "shell.execute_reply.started": "2020-03-17T02:21:19.272938Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_past_hist.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with allergy data

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-17T02:21:35.032721Z", "iopub.status.busy": "2020-03-17T02:21:35.032410Z", "iopub.status.idle": "2020-03-17T02:21:37.106768Z", "shell.execute_reply": "2020-03-17T02:21:37.105903Z", "shell.execute_reply.started": "2020-03-17T02:21:35.032645Z"}}
# eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_alrg.ftr')
# eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Filter to the unit stays that also have data in the other tables:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:01:52.209502Z", "iopub.status.busy": "2020-03-16T19:01:52.209225Z", "iopub.status.idle": "2020-03-16T19:01:52.311109Z", "shell.execute_reply": "2020-03-16T19:01:52.310204Z", "shell.execute_reply.started": "2020-03-16T19:01:52.209446Z"}}
# alrg_df = alrg_df[alrg_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:01:52.312681Z", "iopub.status.busy": "2020-03-16T19:01:52.312436Z", "iopub.status.idle": "2020-03-16T19:01:52.363719Z", "shell.execute_reply": "2020-03-16T19:01:52.362786Z", "shell.execute_reply.started": "2020-03-16T19:01:52.312641Z"}}
# eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:01:52.365354Z", "iopub.status.busy": "2020-03-16T19:01:52.365093Z", "iopub.status.idle": "2020-03-16T19:01:52.714861Z", "shell.execute_reply": "2020-03-16T19:01:52.713691Z", "shell.execute_reply.started": "2020-03-16T19:01:52.365304Z"}}
# eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
# alrg_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# + [markdown] {"Collapsed": "false"}
# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:01:52.716575Z", "iopub.status.busy": "2020-03-16T19:01:52.716196Z", "iopub.status.idle": "2020-03-16T19:01:56.577042Z", "shell.execute_reply": "2020-03-16T19:01:56.575856Z", "shell.execute_reply.started": "2020-03-16T19:01:52.716521Z"}}
# eICU_df = eICU_df.join(alrg_df, how='outer')
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
# eICU_df.reset_index(inplace=True)
# eICU_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-15T06:24:26.414647Z", "iopub.status.busy": "2020-03-15T06:24:26.414433Z", "iopub.status.idle": "2020-03-15T01:58:25.351098Z", "shell.execute_reply": "2020-02-26T18:02:20.126190Z", "shell.execute_reply.started": "2020-02-26T17:33:36.720857Z"}}
# eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_past_hist.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with past history data

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-15T14:12:15.123155Z", "iopub.status.busy": "2020-03-15T14:12:15.122958Z", "iopub.status.idle": "2020-03-15T14:14:52.585252Z", "shell.execute_reply": "2020-03-15T14:14:52.584305Z", "shell.execute_reply.started": "2020-03-15T14:12:15.123117Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_past_hist.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Filter to the unit stays that also have data in the other tables:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:01:56.705542Z", "iopub.status.busy": "2020-03-16T19:01:56.705317Z", "iopub.status.idle": "2020-03-16T19:01:56.860152Z", "shell.execute_reply": "2020-03-16T19:01:56.859095Z", "shell.execute_reply.started": "2020-03-16T19:01:56.705504Z"}}
past_hist_df = past_hist_df[past_hist_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

# + [markdown] {"Collapsed": "false"}
# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:01:56.861539Z", "iopub.status.busy": "2020-03-16T19:01:56.861285Z", "iopub.status.idle": "2020-03-16T19:01:57.049920Z", "shell.execute_reply": "2020-03-16T19:01:57.048938Z", "shell.execute_reply.started": "2020-03-16T19:01:56.861485Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(past_hist_df.patientunitstayid.unique())]

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:01:57.051925Z", "iopub.status.busy": "2020-03-16T19:01:57.051671Z", "iopub.status.idle": "2020-03-16T19:01:57.114886Z", "shell.execute_reply": "2020-03-16T19:01:57.113386Z", "shell.execute_reply.started": "2020-03-16T19:01:57.051877Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:01:57.117189Z", "iopub.status.busy": "2020-03-16T19:01:57.116774Z", "iopub.status.idle": "2020-03-16T19:01:57.316047Z", "shell.execute_reply": "2020-03-16T19:01:57.315119Z", "shell.execute_reply.started": "2020-03-16T19:01:57.117066Z"}}
eICU_df.set_index('patientunitstayid', inplace=True)
past_hist_df.set_index('patientunitstayid', inplace=True)

# + [markdown] {"Collapsed": "false"}
# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:01:57.318295Z", "iopub.status.busy": "2020-03-16T19:01:57.317417Z", "iopub.status.idle": "2020-03-16T19:01:57.323594Z", "shell.execute_reply": "2020-03-16T19:01:57.322766Z", "shell.execute_reply.started": "2020-03-16T19:01:57.318083Z"}}
len(eICU_df)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:01:57.328065Z", "iopub.status.busy": "2020-03-16T19:01:57.327544Z", "iopub.status.idle": "2020-03-16T19:02:01.158458Z", "shell.execute_reply": "2020-03-16T19:02:01.157591Z", "shell.execute_reply.started": "2020-03-16T19:01:57.327999Z"}}
eICU_df = eICU_df.join(past_hist_df, how='outer')
eICU_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:02:01.160449Z", "iopub.status.busy": "2020-03-16T19:02:01.160188Z", "iopub.status.idle": "2020-03-16T19:02:01.166005Z", "shell.execute_reply": "2020-03-16T19:02:01.164803Z", "shell.execute_reply.started": "2020-03-16T19:02:01.160404Z"}}
len(eICU_df)

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
eICU_df.reset_index(inplace=True)
eICU_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-15T06:24:26.414647Z", "iopub.status.busy": "2020-03-15T06:24:26.414433Z", "iopub.status.idle": "2020-03-15T01:58:25.351098Z", "shell.execute_reply": "2020-02-26T18:02:20.126190Z", "shell.execute_reply.started": "2020-02-26T17:33:36.720857Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_treat.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with treatment data

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-15T14:12:15.123155Z", "iopub.status.busy": "2020-03-15T14:12:15.122958Z", "iopub.status.idle": "2020-03-15T14:14:52.585252Z", "shell.execute_reply": "2020-03-15T14:14:52.584305Z", "shell.execute_reply.started": "2020-03-15T14:12:15.123117Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_treat.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Filter to the unit stays that also have data in the other tables:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:02:01.282244Z", "iopub.status.busy": "2020-03-16T19:02:01.281986Z", "iopub.status.idle": "2020-03-16T19:02:01.467518Z", "shell.execute_reply": "2020-03-16T19:02:01.465827Z", "shell.execute_reply.started": "2020-03-16T19:02:01.282194Z"}}
treat_df = treat_df[treat_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

# + [markdown] {"Collapsed": "false"}
# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:02:01.469269Z", "iopub.status.busy": "2020-03-16T19:02:01.468921Z", "iopub.status.idle": "2020-03-16T19:02:01.924822Z", "shell.execute_reply": "2020-03-16T19:02:01.923876Z", "shell.execute_reply.started": "2020-03-16T19:02:01.469198Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(treat_df.patientunitstayid.unique())]

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:02:01.926212Z", "iopub.status.busy": "2020-03-16T19:02:01.925955Z", "iopub.status.idle": "2020-03-16T19:02:01.992041Z", "shell.execute_reply": "2020-03-16T19:02:01.990767Z", "shell.execute_reply.started": "2020-03-16T19:02:01.926158Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:02:01.993516Z", "iopub.status.busy": "2020-03-16T19:02:01.993262Z", "iopub.status.idle": "2020-03-16T19:02:02.915521Z", "shell.execute_reply": "2020-03-16T19:02:02.914574Z", "shell.execute_reply.started": "2020-03-16T19:02:01.993457Z"}}
eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
treat_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# + [markdown] {"Collapsed": "false"}
# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:02:02.917356Z", "iopub.status.busy": "2020-03-16T19:02:02.917017Z", "iopub.status.idle": "2020-03-16T19:02:22.808392Z", "shell.execute_reply": "2020-03-16T19:02:22.807300Z", "shell.execute_reply.started": "2020-03-16T19:02:02.917291Z"}}
eICU_df = eICU_df.join(treat_df, how='outer')
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
eICU_df.reset_index(inplace=True)
eICU_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-15T06:24:26.414647Z", "iopub.status.busy": "2020-03-15T06:24:26.414433Z", "iopub.status.idle": "2020-03-15T01:58:25.351098Z", "shell.execute_reply": "2020-02-26T18:02:20.126190Z", "shell.execute_reply.started": "2020-02-26T17:33:36.720857Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_med.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with infusion drug data

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:20:27.994911Z", "iopub.status.busy": "2020-03-16T19:20:27.994634Z", "iopub.status.idle": "2020-03-16T19:20:34.030648Z", "shell.execute_reply": "2020-03-16T19:20:34.029788Z", "shell.execute_reply.started": "2020-03-16T19:20:27.994860Z"}}
# eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_inf_drug.ftr')
# eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
# eICU_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:12:55.553777Z", "iopub.status.busy": "2020-03-16T19:12:55.552875Z", "iopub.status.idle": "2020-03-16T19:12:55.560627Z", "shell.execute_reply": "2020-03-16T19:12:55.559734Z", "shell.execute_reply.started": "2020-03-16T19:12:55.553708Z"}}
# eICU_df.dtypes

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:12:56.646154Z", "iopub.status.busy": "2020-03-16T19:12:56.645538Z", "iopub.status.idle": "2020-03-16T19:12:56.710660Z", "shell.execute_reply": "2020-03-16T19:12:56.709564Z", "shell.execute_reply.started": "2020-03-16T19:12:56.646095Z"}}
# eICU_df.astype({'patientunitstayid': 'uint16'}).dtypes

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:13:11.222575Z", "iopub.status.busy": "2020-03-16T19:13:11.222179Z", "iopub.status.idle": "2020-03-16T19:13:11.227481Z", "shell.execute_reply": "2020-03-16T19:13:11.226485Z", "shell.execute_reply.started": "2020-03-16T19:13:11.222516Z"}}
# df_columns = list(eICU_df.columns)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:13:12.467784Z", "iopub.status.busy": "2020-03-16T19:13:12.467473Z", "iopub.status.idle": "2020-03-16T19:13:12.473863Z", "shell.execute_reply": "2020-03-16T19:13:12.472659Z", "shell.execute_reply.started": "2020-03-16T19:13:12.467723Z"}}
# tmp_dict = dict()
# for key, val in dtype_dict.items():
#     if key in df_columns:
#         tmp_dict[key] = dtype_dict[key]

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:13:14.164914Z", "iopub.status.busy": "2020-03-16T19:13:14.164630Z", "iopub.status.idle": "2020-03-16T19:13:14.171708Z", "shell.execute_reply": "2020-03-16T19:13:14.170807Z", "shell.execute_reply.started": "2020-03-16T19:13:14.164873Z"}}
# tmp_dict

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:13:16.899528Z", "iopub.status.busy": "2020-03-16T19:13:16.899070Z", "iopub.status.idle": "2020-03-16T19:13:16.956361Z", "shell.execute_reply": "2020-03-16T19:13:16.954960Z", "shell.execute_reply.started": "2020-03-16T19:13:16.899468Z"}}
# # eICU_df = eICU_df.astype(tmp_dict, copy=False)
# eICU_df = eICU_df.astype(tmp_dict)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:13:17.943337Z", "iopub.status.busy": "2020-03-16T19:13:17.943038Z", "iopub.status.idle": "2020-03-16T19:13:17.953315Z", "shell.execute_reply": "2020-03-16T19:13:17.952113Z", "shell.execute_reply.started": "2020-03-16T19:13:17.943294Z"}}
# eICU_df.dtypes

# + [markdown] {"Collapsed": "false"}
# Filter to the unit stays that also have data in the other tables:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:20:34.032347Z", "iopub.status.busy": "2020-03-16T19:20:34.032065Z", "iopub.status.idle": "2020-03-16T19:20:34.488917Z", "shell.execute_reply": "2020-03-16T19:20:34.487923Z", "shell.execute_reply.started": "2020-03-16T19:20:34.032296Z"}}
# inf_drug_df = inf_drug_df[inf_drug_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:20:34.490565Z", "iopub.status.busy": "2020-03-16T19:20:34.490228Z", "iopub.status.idle": "2020-03-16T19:20:34.542088Z", "shell.execute_reply": "2020-03-16T19:20:34.540951Z", "shell.execute_reply.started": "2020-03-16T19:20:34.490504Z"}}
# eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:20:34.544114Z", "iopub.status.busy": "2020-03-16T19:20:34.543767Z", "iopub.status.idle": "2020-03-16T19:20:36.948907Z", "shell.execute_reply": "2020-03-16T19:20:36.948045Z", "shell.execute_reply.started": "2020-03-16T19:20:34.544042Z"}}
# eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
# inf_drug_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# + [markdown] {"Collapsed": "false"}
# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:20:36.950356Z", "iopub.status.busy": "2020-03-16T19:20:36.950096Z", "iopub.status.idle": "2020-03-16T19:21:53.629422Z", "shell.execute_reply": "2020-03-16T19:21:53.628433Z", "shell.execute_reply.started": "2020-03-16T19:20:36.950305Z"}}
# eICU_df = eICU_df.join(inf_drug_df, how='outer')
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
# eICU_df.reset_index(inplace=True)
# eICU_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:22:05.277916Z", "iopub.status.busy": "2020-03-16T19:22:05.277569Z", "iopub.status.idle": "2020-03-16T19:22:05.578952Z", "shell.execute_reply": "2020-03-16T19:22:05.578030Z", "shell.execute_reply.started": "2020-03-16T19:22:05.277778Z"}}
# eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_med.ftr')

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": true, "toc-nb-collapsed": true}
# ### Joining with medication data

# + [markdown] {"Collapsed": "false"}
# #### Joining medication and admission drug data

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T00:03:44.112990Z", "iopub.status.busy": "2020-03-19T00:03:44.112540Z", "iopub.status.idle": "2020-03-19T00:03:44.234664Z", "shell.execute_reply": "2020-03-19T00:03:44.233779Z", "shell.execute_reply.started": "2020-03-19T00:03:44.112943Z"}}
med_df.set_index(['patientunitstayid', 'ts'], inplace=True)
adms_drug_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# + [markdown] {"Collapsed": "false"}
# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T00:03:44.236681Z", "iopub.status.busy": "2020-03-19T00:03:44.236305Z", "iopub.status.idle": "2020-03-19T00:07:40.517643Z", "shell.execute_reply": "2020-03-19T00:07:40.516832Z", "shell.execute_reply.started": "2020-03-19T00:03:44.236635Z"}}
med_df = med_df.join(adms_drug_df, how='outer', lsuffix='_x', rsuffix='_y')
med_df.head()

# + [markdown] {"Collapsed": "false"}
# #### Merging duplicate columns

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T00:07:40.519036Z", "iopub.status.busy": "2020-03-19T00:07:40.518812Z", "iopub.status.idle": "2020-03-19T00:07:40.529221Z", "shell.execute_reply": "2020-03-19T00:07:40.527602Z", "shell.execute_reply.started": "2020-03-19T00:07:40.519009Z"}}
set([col.split('_x')[0].split('_y')[0] for col in med_df.columns if col.endswith('_x') or col.endswith('_y')])

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T00:07:40.530809Z", "iopub.status.busy": "2020-03-19T00:07:40.530575Z", "iopub.status.idle": "2020-03-19T00:07:46.565507Z", "shell.execute_reply": "2020-03-19T00:07:46.564659Z", "shell.execute_reply.started": "2020-03-19T00:07:40.530779Z"}}
med_df[['drugadmitfrequency_twice_a_day_x', 'drugadmitfrequency_twice_a_day_y',
        'drughiclseqno_10321_x', 'drughiclseqno_10321_y']].head(20)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T00:07:46.612778Z", "iopub.status.busy": "2020-03-19T00:07:46.612397Z", "iopub.status.idle": "2020-03-18T23:32:05.303941Z", "shell.execute_reply": "2020-03-18T23:32:05.302535Z", "shell.execute_reply.started": "2020-03-18T23:31:36.174277Z"}, "pixiedust": {"displayParams": {}}}
med_df = du.data_processing.merge_columns(med_df, inplace=True)
med_df.sample(20)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T13:13:11.271940Z", "iopub.status.busy": "2020-03-19T13:13:11.271648Z", "iopub.status.idle": "2020-03-19T13:13:11.277436Z", "shell.execute_reply": "2020-03-19T13:13:11.276216Z", "shell.execute_reply.started": "2020-03-19T13:13:11.271910Z"}}
print('Hello world!')

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T13:14:31.910756Z", "iopub.status.busy": "2020-03-19T13:14:31.909860Z", "iopub.status.idle": "2020-03-19T13:14:31.916980Z", "shell.execute_reply": "2020-03-19T13:14:31.916321Z", "shell.execute_reply.started": "2020-03-19T13:14:31.910702Z"}}
set([col.split('_x')[0].split('_y')[0] for col in med_df.columns if col.endswith('_x') or col.endswith('_y')])

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T13:14:48.378233Z", "iopub.status.busy": "2020-03-19T13:14:48.377953Z", "iopub.status.idle": "2020-03-19T13:14:48.405250Z", "shell.execute_reply": "2020-03-19T13:14:48.404256Z", "shell.execute_reply.started": "2020-03-19T13:14:48.378204Z"}}
med_df[['drugadmitfrequency_twice_a_day', 'drughiclseqno_10321']].head(20)

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T13:14:53.188178Z", "iopub.status.busy": "2020-03-19T13:14:53.187879Z", "iopub.status.idle": "2020-03-19T13:14:56.189680Z", "shell.execute_reply": "2020-03-19T13:14:56.188734Z", "shell.execute_reply.started": "2020-03-19T13:14:53.188147Z"}}
med_df.reset_index(inplace=True)
med_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
# med_df = du.utils.convert_pyarrow_dtypes(med_df, inplace=True)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T13:14:58.357252Z", "iopub.status.busy": "2020-03-19T13:14:58.356838Z", "iopub.status.idle": "2020-03-19T13:15:11.529921Z", "shell.execute_reply": "2020-03-19T13:15:11.529010Z", "shell.execute_reply.started": "2020-03-19T13:14:58.357221Z"}}
med_df.to_feather(f'{data_path}normalized/ohe/med_and_adms_drug.ftr')

# + [markdown] {"Collapsed": "false"}
# #### Joining with the rest of the eICU data

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T13:17:00.301997Z", "iopub.status.busy": "2020-03-19T13:17:00.301707Z", "iopub.status.idle": "2020-03-19T13:17:17.860418Z", "shell.execute_reply": "2020-03-19T13:17:17.859468Z", "shell.execute_reply.started": "2020-03-19T13:17:00.301969Z"}}
med_drug_df = pd.read_feather(f'{data_path}normalized/ohe/med_and_adms_drug.ftr')
med_drug_df = du.utils.convert_dtypes(med_drug_df, dtypes=dtype_dict, inplace=True)
med_drug_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T13:17:17.862017Z", "iopub.status.busy": "2020-03-19T13:17:17.861801Z", "iopub.status.idle": "2020-03-19T13:17:30.568197Z", "shell.execute_reply": "2020-03-19T13:17:30.567376Z", "shell.execute_reply.started": "2020-03-19T13:17:17.861992Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_med.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Filter to the unit stays that also have data in the other tables:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T13:17:30.570050Z", "iopub.status.busy": "2020-03-19T13:17:30.569820Z", "iopub.status.idle": "2020-03-19T13:17:34.342066Z", "shell.execute_reply": "2020-03-19T13:17:34.341208Z", "shell.execute_reply.started": "2020-03-19T13:17:30.570023Z"}}
med_drug_df = med_drug_df[med_drug_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

# + [markdown] {"Collapsed": "false"}
# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T13:17:34.343921Z", "iopub.status.busy": "2020-03-19T13:17:34.343610Z", "iopub.status.idle": "2020-03-19T13:17:36.463676Z", "shell.execute_reply": "2020-03-19T13:17:36.462708Z", "shell.execute_reply.started": "2020-03-19T13:17:34.343882Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(med_drug_df.patientunitstayid.unique())]

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T13:17:36.465382Z", "iopub.status.busy": "2020-03-19T13:17:36.465070Z", "iopub.status.idle": "2020-03-19T13:17:36.477206Z", "shell.execute_reply": "2020-03-19T13:17:36.476128Z", "shell.execute_reply.started": "2020-03-19T13:17:36.465342Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T13:17:36.478910Z", "iopub.status.busy": "2020-03-19T13:17:36.478576Z", "iopub.status.idle": "2020-03-19T13:17:36.589484Z", "shell.execute_reply": "2020-03-19T13:17:36.588649Z", "shell.execute_reply.started": "2020-03-19T13:17:36.478869Z"}}
eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
med_drug_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# + [markdown] {"Collapsed": "false"}
# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T13:17:36.590907Z", "iopub.status.busy": "2020-03-19T13:17:36.590676Z", "iopub.status.idle": "2020-03-19T13:17:36.595743Z", "shell.execute_reply": "2020-03-19T13:17:36.594924Z", "shell.execute_reply.started": "2020-03-19T13:17:36.590880Z"}}
len(eICU_df)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T13:17:36.598583Z", "iopub.status.busy": "2020-03-19T13:17:36.598362Z", "iopub.status.idle": "2020-03-19T13:20:29.159142Z", "shell.execute_reply": "2020-03-19T13:20:29.158270Z", "shell.execute_reply.started": "2020-03-19T13:17:36.598558Z"}}
eICU_df = eICU_df.join(med_drug_df, how='outer')
eICU_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T13:20:29.160832Z", "iopub.status.busy": "2020-03-19T13:20:29.160594Z", "iopub.status.idle": "2020-03-19T13:20:29.166586Z", "shell.execute_reply": "2020-03-19T13:20:29.165807Z", "shell.execute_reply.started": "2020-03-19T13:20:29.160804Z"}}
len(eICU_df)

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T13:20:29.168143Z", "iopub.status.busy": "2020-03-19T13:20:29.167843Z", "iopub.status.idle": "2020-03-19T13:20:47.747291Z", "shell.execute_reply": "2020-03-19T13:20:47.746527Z", "shell.execute_reply.started": "2020-03-19T13:20:29.168115Z"}}
eICU_df.reset_index(inplace=True)
eICU_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T13:20:47.748614Z", "iopub.status.busy": "2020-03-19T13:20:47.748366Z", "iopub.status.idle": "2020-03-19T13:20:47.752015Z", "shell.execute_reply": "2020-03-19T13:20:47.751366Z", "shell.execute_reply.started": "2020-03-19T13:20:47.748588Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T13:20:47.753335Z", "iopub.status.busy": "2020-03-19T13:20:47.753063Z", "iopub.status.idle": "2020-03-19T13:21:03.998939Z", "shell.execute_reply": "2020-03-19T13:21:03.998042Z", "shell.execute_reply.started": "2020-03-19T13:20:47.753302Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_resp_care.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with intake outake data

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-15T14:12:15.123155Z", "iopub.status.busy": "2020-03-15T14:12:15.122958Z", "iopub.status.idle": "2020-03-15T14:14:52.585252Z", "shell.execute_reply": "2020-03-15T14:14:52.584305Z", "shell.execute_reply.started": "2020-03-15T14:12:15.123117Z"}}
# eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_in_out.ftr')
# eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Filter to the unit stays that also have data in the other tables:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-15T14:17:00.692386Z", "iopub.status.busy": "2020-03-15T14:17:00.692190Z", "iopub.status.idle": "2020-03-15T14:17:02.377755Z", "shell.execute_reply": "2020-03-15T14:17:02.376997Z", "shell.execute_reply.started": "2020-03-15T14:17:00.692351Z"}}
# in_out_df = in_out_df[in_out_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

# + [markdown] {"Collapsed": "false"}
# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-15T14:17:02.379120Z", "iopub.status.busy": "2020-03-15T14:17:02.378883Z", "iopub.status.idle": "2020-03-15T14:17:12.519271Z", "shell.execute_reply": "2020-03-15T14:17:12.518572Z", "shell.execute_reply.started": "2020-03-15T14:17:02.379080Z"}}
# eICU_df = eICU_df[eICU_df.patientunitstayid.isin(in_out_df.patientunitstayid.unique())]

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-27T05:20:54.131275Z", "iopub.status.busy": "2020-02-27T05:20:54.131063Z", "iopub.status.idle": "2020-02-27T05:21:15.499582Z", "shell.execute_reply": "2020-02-27T05:21:15.498801Z", "shell.execute_reply.started": "2020-02-27T05:20:54.131238Z"}}
# eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false"}
# eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
# in_out_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# + [markdown] {"Collapsed": "false"}
# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-15T14:17:12.520423Z", "iopub.status.busy": "2020-03-15T14:17:12.520211Z", "iopub.status.idle": "2020-03-15T15:14:28.597564Z", "shell.execute_reply": "2020-03-15T15:14:28.596911Z", "shell.execute_reply.started": "2020-03-15T14:17:12.520386Z"}}
# eICU_df = eICU_df.join(in_out_df, how='outer')
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
# eICU_df.reset_index(inplace=True)
# eICU_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-15T15:14:28.822215Z", "iopub.status.busy": "2020-03-15T15:14:28.822010Z", "iopub.status.idle": "2020-03-15T01:58:25.351098Z", "shell.execute_reply": "2020-02-26T18:02:20.126190Z", "shell.execute_reply.started": "2020-02-26T17:33:36.720857Z"}}
# eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_resp_care.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with nurse care data

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:13:35.213020Z", "iopub.status.busy": "2020-03-15T04:51:14.996900Z", "iopub.status.idle": "2020-03-15T04:51:14.997207Z", "shell.execute_reply": "2020-02-26T17:13:35.215332Z", "shell.execute_reply.started": "2020-02-26T17:13:35.212983Z"}}
# eICU_df = pd.merge(eICU_df, nurse_care_df, how='outer', on=['patientunitstayid', 'ts'], copy=False)
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Joining with nurse assessment data

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:13:35.217016Z", "iopub.status.busy": "2020-03-15T04:51:14.997878Z", "iopub.status.idle": "2020-03-15T04:51:14.998163Z", "shell.execute_reply": "2020-02-26T17:13:37.232810Z", "shell.execute_reply.started": "2020-02-26T17:13:35.216967Z"}}
# eICU_df = pd.merge(eICU_df, nurse_assess_df, how='outer', on=['patientunitstayid', 'ts'], copy=False)
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Joining with nurse charting data

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T17:13:37.234780Z", "iopub.status.busy": "2020-03-15T04:51:14.998878Z", "iopub.status.idle": "2020-03-15T04:51:14.999189Z", "shell.execute_reply": "2020-02-26T17:13:37.238206Z", "shell.execute_reply.started": "2020-02-26T17:13:37.234737Z"}}
# eICU_df = pd.merge(eICU_df, nurse_chart_df, how='outer', on=['patientunitstayid', 'ts'], copy=False)
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Joining with respiratory care data

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T14:27:53.299188Z", "iopub.status.busy": "2020-03-19T14:27:53.298913Z", "iopub.status.idle": "2020-03-19T14:28:52.520050Z", "shell.execute_reply": "2020-03-19T14:28:52.518516Z", "shell.execute_reply.started": "2020-03-19T14:27:53.299160Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_resp_care.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Filter to the unit stays that also have data in the other tables:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T14:28:52.522047Z", "iopub.status.busy": "2020-03-19T14:28:52.521757Z", "iopub.status.idle": "2020-03-19T14:28:52.553698Z", "shell.execute_reply": "2020-03-19T14:28:52.552193Z", "shell.execute_reply.started": "2020-03-19T14:28:52.522017Z"}}
resp_care_df = resp_care_df[resp_care_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T14:28:52.556654Z", "iopub.status.busy": "2020-03-19T14:28:52.556124Z", "iopub.status.idle": "2020-03-19T14:28:52.575762Z", "shell.execute_reply": "2020-03-19T14:28:52.574354Z", "shell.execute_reply.started": "2020-03-19T14:28:52.556617Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T14:28:52.579321Z", "iopub.status.busy": "2020-03-19T14:28:52.578294Z", "iopub.status.idle": "2020-03-19T14:28:52.715784Z", "shell.execute_reply": "2020-03-19T14:28:52.714523Z", "shell.execute_reply.started": "2020-03-19T14:28:52.579258Z"}}
eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
resp_care_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# + [markdown] {"Collapsed": "false"}
# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T14:28:52.717236Z", "iopub.status.busy": "2020-03-19T14:28:52.716986Z", "iopub.status.idle": "2020-03-19T14:33:06.863748Z", "shell.execute_reply": "2020-03-19T14:33:06.862706Z", "shell.execute_reply.started": "2020-03-19T14:28:52.717207Z"}}
eICU_df = eICU_df.join(resp_care_df, how='outer')
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T14:33:06.865383Z", "iopub.status.busy": "2020-03-19T14:33:06.865015Z", "iopub.status.idle": "2020-03-19T14:33:14.757677Z", "shell.execute_reply": "2020-03-19T14:33:14.756925Z", "shell.execute_reply.started": "2020-03-19T14:33:06.865353Z"}}
eICU_df.reset_index(inplace=True)
eICU_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T14:33:14.759046Z", "iopub.status.busy": "2020-03-19T14:33:14.758817Z", "iopub.status.idle": "2020-03-19T14:33:37.038801Z", "shell.execute_reply": "2020-03-19T14:33:37.037886Z", "shell.execute_reply.started": "2020-03-19T14:33:14.759021Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_vital_aprdc.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with aperiodic vital signals data

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T23:43:42.720265Z", "iopub.status.busy": "2020-03-19T23:43:42.719731Z", "iopub.status.idle": "2020-03-19T23:45:40.464425Z", "shell.execute_reply": "2020-03-19T23:45:40.463578Z", "shell.execute_reply.started": "2020-03-19T23:43:42.720233Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_vital_aprdc.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Filter to the unit stays that also have data in the other tables:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T23:45:40.465936Z", "iopub.status.busy": "2020-03-19T23:45:40.465704Z", "iopub.status.idle": "2020-03-19T23:45:46.101360Z", "shell.execute_reply": "2020-03-19T23:45:46.100147Z", "shell.execute_reply.started": "2020-03-19T23:45:40.465907Z"}}
vital_aprdc_df = vital_aprdc_df[vital_aprdc_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

# + [markdown] {"Collapsed": "false"}
# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T23:45:46.103192Z", "iopub.status.busy": "2020-03-19T23:45:46.102944Z", "iopub.status.idle": "2020-03-19T23:45:56.846940Z", "shell.execute_reply": "2020-03-19T23:45:56.846134Z", "shell.execute_reply.started": "2020-03-19T23:45:46.103164Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(vital_aprdc_df.patientunitstayid.unique())]

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T23:45:56.848364Z", "iopub.status.busy": "2020-03-19T23:45:56.848151Z", "iopub.status.idle": "2020-03-19T23:45:56.863914Z", "shell.execute_reply": "2020-03-19T23:45:56.862661Z", "shell.execute_reply.started": "2020-03-19T23:45:56.848337Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T23:45:56.865048Z", "iopub.status.busy": "2020-03-19T23:45:56.864826Z", "iopub.status.idle": "2020-03-19T23:45:57.597711Z", "shell.execute_reply": "2020-03-19T23:45:57.596888Z", "shell.execute_reply.started": "2020-03-19T23:45:56.865022Z"}}
eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
vital_aprdc_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# + [markdown] {"Collapsed": "false"}
# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T23:45:57.598967Z", "iopub.status.busy": "2020-03-19T23:45:57.598740Z", "iopub.status.idle": "2020-03-20T00:33:16.573587Z", "shell.execute_reply": "2020-03-20T00:33:16.572871Z", "shell.execute_reply.started": "2020-03-19T23:45:57.598938Z"}}
eICU_df = eICU_df.join(vital_aprdc_df, how='outer')
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-20T00:33:16.575060Z", "iopub.status.busy": "2020-03-20T00:33:16.574829Z", "iopub.status.idle": "2020-03-20T00:34:25.453213Z", "shell.execute_reply": "2020-03-20T00:34:25.452479Z", "shell.execute_reply.started": "2020-03-20T00:33:16.575032Z"}}
eICU_df.reset_index(inplace=True)
eICU_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-20T00:34:25.455069Z", "iopub.status.busy": "2020-03-20T00:34:25.454839Z", "iopub.status.idle": "2020-03-20T00:34:25.458218Z", "shell.execute_reply": "2020-03-20T00:34:25.457437Z", "shell.execute_reply.started": "2020-03-20T00:34:25.455040Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-20T00:34:25.459501Z", "iopub.status.busy": "2020-03-20T00:34:25.459232Z", "iopub.status.idle": "2020-03-20T00:40:17.831128Z", "shell.execute_reply": "2020-03-20T00:40:17.830232Z", "shell.execute_reply.started": "2020-03-20T00:34:25.459456Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_before_joining_vital_prdc.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with periodic vital signals data

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-20T03:52:24.667594Z", "iopub.status.busy": "2020-03-20T03:52:24.667242Z", "iopub.status.idle": "2020-03-20T03:52:38.671225Z", "shell.execute_reply": "2020-03-20T03:52:38.669765Z", "shell.execute_reply.started": "2020-03-20T03:52:24.667561Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_before_joining_vital_prdc.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Filter to the unit stays that also have data in the other tables:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T20:27:30.651538Z", "iopub.status.busy": "2020-03-20T03:52:38.672096Z", "iopub.status.idle": "2020-03-20T03:52:38.672429Z", "shell.execute_reply": "2020-02-26T20:29:03.629871Z", "shell.execute_reply.started": "2020-02-26T20:27:30.651486Z"}}
vital_prdc_df = vital_prdc_df[vital_prdc_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

# + [markdown] {"Collapsed": "false"}
# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T20:29:03.632286Z", "iopub.status.busy": "2020-03-20T03:52:38.673348Z", "iopub.status.idle": "2020-03-20T03:52:38.673658Z", "shell.execute_reply": "2020-02-26T20:30:05.977157Z", "shell.execute_reply.started": "2020-02-26T20:29:03.632247Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(vital_prdc_df.patientunitstayid.unique())]

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-27T05:20:54.131275Z", "iopub.status.busy": "2020-03-20T03:52:38.674481Z", "iopub.status.idle": "2020-03-20T03:52:38.674782Z", "shell.execute_reply": "2020-02-27T05:21:15.498801Z", "shell.execute_reply.started": "2020-02-27T05:20:54.131238Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-20T03:52:38.675601Z", "iopub.status.idle": "2020-03-20T03:52:38.676183Z"}}
eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
vital_prdc_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# + [markdown] {"Collapsed": "false"}
# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T20:30:05.979158Z", "iopub.status.busy": "2020-03-20T03:52:38.677178Z", "iopub.status.idle": "2020-03-20T03:52:38.677487Z", "shell.execute_reply": "2020-02-26T20:56:11.534004Z", "shell.execute_reply.started": "2020-02-26T20:30:05.979112Z"}}
eICU_df = eICU_df.join(vital_prdc_df, how='outer')
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-20T03:52:38.678350Z", "iopub.status.idle": "2020-03-20T03:52:38.678654Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
eICU_df.reset_index(inplace=True)
eICU_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-20T03:52:38.679497Z", "iopub.status.idle": "2020-03-20T03:52:38.679834Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-26T20:56:11.535828Z", "iopub.status.busy": "2020-03-20T03:52:38.680841Z", "iopub.status.idle": "2020-03-20T03:52:38.681144Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_post_joining_vital_prdc.ftr')

# + [markdown] {"Collapsed": "false"}
# ### Joining with lab data

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-27T00:58:11.956018Z", "iopub.status.busy": "2020-02-27T00:58:11.955786Z", "iopub.status.idle": "2020-02-27T01:10:36.152614Z", "shell.execute_reply": "2020-02-27T01:10:36.151782Z", "shell.execute_reply.started": "2020-02-27T00:58:11.955976Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_post_joining_vital_prdc.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Filter to the unit stays that also have data in the other tables:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-27T01:10:36.154590Z", "iopub.status.busy": "2020-02-27T01:10:36.154356Z", "iopub.status.idle": "2020-02-27T01:14:18.564700Z", "shell.execute_reply": "2020-02-27T01:14:18.563808Z", "shell.execute_reply.started": "2020-02-27T01:10:36.154548Z"}}
lab_df = lab_df[lab_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]

# + [markdown] {"Collapsed": "false"}
# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-27T01:14:18.566341Z", "iopub.status.busy": "2020-02-27T01:14:18.566086Z", "iopub.status.idle": "2020-02-27T01:15:30.746679Z", "shell.execute_reply": "2020-02-27T01:15:30.745909Z", "shell.execute_reply.started": "2020-02-27T01:14:18.566301Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(lab_df.patientunitstayid.unique())]

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-27T05:20:54.131275Z", "iopub.status.busy": "2020-02-27T05:20:54.131063Z", "iopub.status.idle": "2020-02-27T05:21:15.499582Z", "shell.execute_reply": "2020-02-27T05:21:15.498801Z", "shell.execute_reply.started": "2020-02-27T05:20:54.131238Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Set `patientunitstayid` and `ts` as indeces, for faster data merging:

# + {"Collapsed": "false"}
eICU_df.set_index(['patientunitstayid', 'ts'], inplace=True)
lab_df.set_index(['patientunitstayid', 'ts'], inplace=True)

# + [markdown] {"Collapsed": "false"}
# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-27T01:15:30.748015Z", "iopub.status.busy": "2020-02-27T01:15:30.747792Z", "iopub.status.idle": "2020-02-27T02:11:51.960090Z", "shell.execute_reply": "2020-02-27T02:11:51.959315Z", "shell.execute_reply.started": "2020-02-27T01:15:30.747973Z"}}
eICU_df = eICU_df.join(lab_df, how='outer')
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
eICU_df.reset_index(inplace=True)
eICU_df.head()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-02-27T02:11:51.961398Z", "iopub.status.busy": "2020-02-27T02:11:51.961181Z", "iopub.status.idle": "2020-02-27T05:20:54.129974Z", "shell.execute_reply": "2020-02-27T05:20:54.129277Z", "shell.execute_reply.started": "2020-02-27T02:11:51.961359Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_post_joining.ftr')

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:16:36.275391Z", "iopub.status.busy": "2020-03-02T02:16:36.275096Z", "iopub.status.idle": "2020-03-02T02:16:52.692400Z", "shell.execute_reply": "2020-03-02T02:16:52.691647Z", "shell.execute_reply.started": "2020-03-02T02:16:36.275334Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_post_joining.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()

# + {"Collapsed": "false"}
eICU_df.columns

# + {"Collapsed": "false"}
eICU_df.dtypes

# + [markdown] {"Collapsed": "false"}
# ## Cleaning the joined data

# + [markdown] {"Collapsed": "false"}
# ### Removing unit stays that are too short

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T01:53:31.536857Z", "iopub.status.busy": "2020-03-02T01:53:31.536616Z", "iopub.status.idle": "2020-03-02T01:54:52.891338Z", "shell.execute_reply": "2020-03-02T01:54:52.890540Z", "shell.execute_reply.started": "2020-03-02T01:53:31.536812Z"}}
# eICU_df.info(memory_usage='deep')

# + [markdown] {"Collapsed": "false"}
# Make sure that the dataframe is ordered by time `ts`:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:17:12.532569Z", "iopub.status.busy": "2020-03-02T02:17:12.532218Z", "iopub.status.idle": "2020-03-02T02:17:39.288044Z", "shell.execute_reply": "2020-03-02T02:17:39.286198Z", "shell.execute_reply.started": "2020-03-02T02:17:12.532509Z"}}
eICU_df = eICU_df.sort_values('ts')
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove unit stays that have less than 10 records:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:17:39.289871Z", "iopub.status.busy": "2020-03-02T02:17:39.289585Z", "iopub.status.idle": "2020-03-02T02:18:15.176358Z", "shell.execute_reply": "2020-03-02T02:18:15.175455Z", "shell.execute_reply.started": "2020-03-02T02:17:39.289827Z"}}
unit_stay_len = eICU_df.groupby('patientunitstayid').patientunitstayid.count()
unit_stay_len

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:18:15.178640Z", "iopub.status.busy": "2020-03-02T02:18:15.178381Z", "iopub.status.idle": "2020-03-02T02:18:15.198989Z", "shell.execute_reply": "2020-03-02T02:18:15.198038Z", "shell.execute_reply.started": "2020-03-02T02:18:15.178591Z"}}
unit_stay_short = set(unit_stay_len[unit_stay_len < 10].index)
unit_stay_short

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:18:15.201115Z", "iopub.status.busy": "2020-03-02T02:18:15.200767Z", "iopub.status.idle": "2020-03-02T02:18:17.001752Z", "shell.execute_reply": "2020-03-02T02:18:17.000646Z", "shell.execute_reply.started": "2020-03-02T02:18:15.201047Z"}}
len(unit_stay_short)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:18:17.004081Z", "iopub.status.busy": "2020-03-02T02:18:17.003368Z", "iopub.status.idle": "2020-03-02T02:18:18.720633Z", "shell.execute_reply": "2020-03-02T02:18:18.719243Z", "shell.execute_reply.started": "2020-03-02T02:18:17.003999Z"}}
eICU_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:18:18.722309Z", "iopub.status.busy": "2020-03-02T02:18:18.721989Z", "iopub.status.idle": "2020-03-02T02:18:20.427440Z", "shell.execute_reply": "2020-03-02T02:18:20.426700Z", "shell.execute_reply.started": "2020-03-02T02:18:18.722259Z"}}
eICU_df = eICU_df[~eICU_df.patientunitstayid.isin(unit_stay_short)]

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:18:20.428687Z", "iopub.status.busy": "2020-03-02T02:18:20.428470Z", "iopub.status.idle": "2020-03-02T02:18:22.585566Z", "shell.execute_reply": "2020-03-02T02:18:22.584657Z", "shell.execute_reply.started": "2020-03-02T02:18:20.428649Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Remove unit stays that have data that represent less than 48h:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:18:22.588243Z", "iopub.status.busy": "2020-03-02T02:18:22.588007Z", "iopub.status.idle": "2020-03-02T02:19:35.059137Z", "shell.execute_reply": "2020-03-02T02:19:35.058244Z", "shell.execute_reply.started": "2020-03-02T02:18:22.588196Z"}}
unit_stay_duration = eICU_df.groupby('patientunitstayid').ts.apply(lambda x: x.max() - x.min())
unit_stay_duration

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:19:35.060663Z", "iopub.status.busy": "2020-03-02T02:19:35.060437Z", "iopub.status.idle": "2020-03-02T02:19:35.081986Z", "shell.execute_reply": "2020-03-02T02:19:35.081304Z", "shell.execute_reply.started": "2020-03-02T02:19:35.060624Z"}}
unit_stay_short = set(unit_stay_duration[unit_stay_duration < 48*60].index)
unit_stay_short

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:19:35.083255Z", "iopub.status.busy": "2020-03-02T02:19:35.083029Z", "iopub.status.idle": "2020-03-02T02:19:35.087514Z", "shell.execute_reply": "2020-03-02T02:19:35.086634Z", "shell.execute_reply.started": "2020-03-02T02:19:35.083217Z"}}
len(unit_stay_short)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:19:35.088588Z", "iopub.status.busy": "2020-03-02T02:19:35.088373Z", "iopub.status.idle": "2020-03-02T02:19:36.682445Z", "shell.execute_reply": "2020-03-02T02:19:36.681534Z", "shell.execute_reply.started": "2020-03-02T02:19:35.088551Z"}}
eICU_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:19:36.683688Z", "iopub.status.busy": "2020-03-02T02:19:36.683454Z", "iopub.status.idle": "2020-03-02T02:19:38.245825Z", "shell.execute_reply": "2020-03-02T02:19:38.245000Z", "shell.execute_reply.started": "2020-03-02T02:19:36.683649Z"}}
eICU_df = eICU_df[~eICU_df.patientunitstayid.isin(unit_stay_short)]

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:19:38.247368Z", "iopub.status.busy": "2020-03-02T02:19:38.247052Z", "iopub.status.idle": "2020-03-02T02:19:40.054358Z", "shell.execute_reply": "2020-03-02T02:19:40.053667Z", "shell.execute_reply.started": "2020-03-02T02:19:38.247313Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining duplicate columns

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:19:40.055928Z", "iopub.status.busy": "2020-03-02T02:19:40.055657Z", "iopub.status.idle": "2020-03-02T02:19:40.061775Z", "shell.execute_reply": "2020-03-02T02:19:40.061167Z", "shell.execute_reply.started": "2020-03-02T02:19:40.055877Z"}}
set([col.split('_x')[0].split('_y')[0] for col in eICU_df.columns if col.endswith('_x') or col.endswith('_y')])

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:19:40.063021Z", "iopub.status.busy": "2020-03-02T02:19:40.062791Z", "iopub.status.idle": "2020-03-02T02:19:40.801829Z", "shell.execute_reply": "2020-03-02T02:19:40.800838Z", "shell.execute_reply.started": "2020-03-02T02:19:40.062965Z"}}
eICU_df[['drugdosage_x', 'drugdosage_y']].head(20)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:19:40.803378Z", "iopub.status.busy": "2020-03-02T02:19:40.802911Z", "iopub.status.idle": "2020-03-02T02:19:41.605486Z", "shell.execute_reply": "2020-03-02T02:19:41.604550Z", "shell.execute_reply.started": "2020-03-02T02:19:40.803328Z"}}
eICU_df[eICU_df.index == 2564878][['drugdosage_x', 'drugdosage_y']]

# + [markdown] {"Collapsed": "false"}
# Convert dataframe to Pandas, as the next cells aren't working properly with Modin:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:34:38.089308Z", "iopub.status.busy": "2020-03-02T02:34:38.088841Z", "iopub.status.idle": "2020-03-02T02:36:29.270540Z", "shell.execute_reply": "2020-03-02T02:36:29.269639Z", "shell.execute_reply.started": "2020-03-02T02:34:38.089249Z"}}
eICU_df, pd = du.utils.convert_dataframe(eICU_df, to='pandas', dtypes=dtype_dict)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:36:29.272090Z", "iopub.status.busy": "2020-03-02T02:36:29.271829Z", "iopub.status.idle": "2020-03-02T02:40:49.633520Z", "shell.execute_reply": "2020-03-02T02:40:49.632359Z", "shell.execute_reply.started": "2020-03-02T02:36:29.272048Z"}, "pixiedust": {"displayParams": {}}}
eICU_df = du.data_processing.merge_columns(eICU_df, inplace=True)
eICU_df.sample(20)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:19:40.055928Z", "iopub.status.busy": "2020-03-02T02:19:40.055657Z", "iopub.status.idle": "2020-03-02T02:19:40.061775Z", "shell.execute_reply": "2020-03-02T02:19:40.061167Z", "shell.execute_reply.started": "2020-03-02T02:19:40.055877Z"}}
set([col.split('_x')[0].split('_y')[0] for col in eICU_df.columns if col.endswith('_x') or col.endswith('_y')])

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:42:58.062830Z", "iopub.status.busy": "2020-03-02T02:42:58.060632Z", "iopub.status.idle": "2020-03-02T02:42:58.071196Z", "shell.execute_reply": "2020-03-02T02:42:58.070236Z", "shell.execute_reply.started": "2020-03-02T02:42:58.062000Z"}}
eICU_df['drugdosage'].head(20)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:42:58.073200Z", "iopub.status.busy": "2020-03-02T02:42:58.072944Z", "iopub.status.idle": "2020-03-02T02:42:58.109699Z", "shell.execute_reply": "2020-03-02T02:42:58.108599Z", "shell.execute_reply.started": "2020-03-02T02:42:58.073152Z"}}
eICU_df[eICU_df.index == 2564878][['drugdosage']]

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-16T19:21:53.631001Z", "iopub.status.busy": "2020-03-16T19:21:53.630729Z", "iopub.status.idle": "2020-03-16T19:22:05.275341Z", "shell.execute_reply": "2020-03-16T19:22:05.274154Z", "shell.execute_reply.started": "2020-03-16T19:21:53.630955Z"}}
# eICU_df = du.utils.convert_pyarrow_dtypes(eICU_df, inplace=True)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T02:42:58.111630Z", "iopub.status.busy": "2020-03-02T02:42:58.111378Z", "iopub.status.idle": "2020-03-02T02:44:09.853833Z", "shell.execute_reply": "2020-03-02T02:44:09.852640Z", "shell.execute_reply.started": "2020-03-02T02:42:58.111587Z"}}
eICU_df.to_feather(f'{data_path}normalized/ohe/eICU_post_merge_duplicate_cols.ftr')

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T04:30:00.845255Z", "iopub.status.busy": "2020-03-02T04:30:00.844962Z", "iopub.status.idle": "2020-03-02T04:30:11.572222Z", "shell.execute_reply": "2020-03-02T04:30:11.571436Z", "shell.execute_reply.started": "2020-03-02T04:30:00.845203Z"}}
eICU_df = pd.read_feather(f'{data_path}normalized/ohe/eICU_post_merge_duplicate_cols.ftr')
eICU_df = du.utils.convert_dtypes(eICU_df, dtypes=dtype_dict, inplace=True)
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Removing unit stays with too many missing values
#
# Consider removing all unit stays that have, combining rows and columns, a very high percentage of missing values.

# + [markdown] {"Collapsed": "false"}
# Reconvert dataframe to Modin:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T04:33:08.150355Z", "iopub.status.idle": "2020-03-02T04:33:08.150854Z"}}
eICU_df, pd = du.utils.convert_dataframe(vital_prdc_df, to='modin', dtypes=dtype_dict)

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T04:30:22.532646Z", "iopub.status.busy": "2020-03-02T04:33:08.152336Z", "iopub.status.idle": "2020-03-02T04:33:08.153242Z", "shell.execute_reply": "2020-03-02T04:30:22.536895Z", "shell.execute_reply.started": "2020-03-02T04:30:22.532603Z"}}
n_features = len(eICU_df.columns)
n_features

# + [markdown] {"Collapsed": "false"}
# Create a temporary column that counts each row's number of missing values:

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-02T04:30:23.160440Z", "iopub.status.busy": "2020-03-02T04:33:08.154574Z", "iopub.status.idle": "2020-03-02T04:33:08.155051Z", "shell.execute_reply": "2020-03-02T04:30:27.325740Z", "shell.execute_reply.started": "2020-03-02T04:30:23.160388Z"}}
eICU_df['row_msng_val'] = eICU_df.isnull().sum(axis=1)
eICU_df[['patientunitstayid', 'ts', 'row_msng_val']].head()

# + [markdown] {"Collapsed": "false"}
# Check each unit stay's percentage of missing data points:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T04:33:08.156401Z", "iopub.status.idle": "2020-03-02T04:33:08.156886Z"}}
# Number of possible data points in each unit stay
n_data_points = eICU_df.groupby('patientunitstayid').ts.count() * n_features
n_data_points

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T04:33:08.158235Z", "iopub.status.idle": "2020-03-02T04:33:08.159229Z"}}
# Number of missing values in each unit stay
n_msng_val = eICU_df.groupby('patientunitstayid').row_msng_val.sum()
n_msng_val

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T04:33:08.160468Z", "iopub.status.idle": "2020-03-02T04:33:08.161302Z"}}
# Percentage of missing values in each unit stay
msng_val_prct = (n_msng_val / n_data_points) * 100
msng_val_prct

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T04:33:08.162698Z", "iopub.status.idle": "2020-03-02T04:33:08.163590Z"}}
msng_val_prct.describe()

# + [markdown] {"Collapsed": "false"}
# Remove unit stays that have too many missing values (>70% of their respective data points):

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:30:52.831884Z", "iopub.status.idle": "2020-02-26T17:30:52.832151Z"}}
unit_stay_high_msgn = set(msng_val_prct[msng_val_prct > 70].index)
unit_stay_high_msgn

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:30:52.833186Z", "iopub.status.idle": "2020-02-26T17:30:52.833692Z"}}
eICU_df.patientunitstayid.nunique()

# + {"Collapsed": "false"}
eICU_df = eICU_df[~eICU_df.patientunitstayid.isin(unit_stay_high_msgn)]

# + {"Collapsed": "false"}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
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

# + [markdown] {"Collapsed": "false"}
# Removed features:

# + {"Collapsed": "false"}
set(prev_features) - set(features)

# + {"Collapsed": "false"}
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Performing imputation

# + {"Collapsed": "false"}
du.search_explore.dataframe_missing_values(eICU_df)

# + [markdown] {"Collapsed": "false"}
# Imputate `gender`:

# + {"Collapsed": "false"}
# Forward fill
eICU_df.gender = eICU_df.groupby(id_column).gender.fillna(method='ffill')
# Backward fill
eICU_df.gender = eICU_df.groupby(id_column).gender.fillna(method='bfill')
# Replace remaining missing values with zero
eICU_df.gender = eICU_df.gender.fillna(value=0)

# + [markdown] {"Collapsed": "false"}
# Imputate the remaining features:

# + {"Collapsed": "false"}
eICU_df = du.data_processing.missing_values_imputation(eICU_df, method='interpolation',
                                                       id_column='patientunitstayid', inplace=True)
eICU_df.head()

# + {"Collapsed": "false"}
du.search_explore.dataframe_missing_values(eICU_df)

# + [markdown] {"Collapsed": "false"}
# ### Rearranging columns
#
# For ease of use and for better intuition, we should make sure that the ID columns (`patientunitstayid` and `ts`) are the first ones in the dataframe.

# + {"Collapsed": "false"}
columns = list(eICU_df.columns)
columns

# + {"Collapsed": "false"}
columns.remove('patientunitstayid')
columns.remove('ts')

# + {"Collapsed": "false"}
columns = ['patientunitstayid', 'ts'] + columns
columns

# + {"Collapsed": "false"}
eICU_df = eICU_df[columns]
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ## Setting the label
#
# Define the label column considering the desired time window on which we want to predict mortality (0, 24h, 48h, 72h, etc).

# + {"Collapsed": "false"}
time_window_h = 24

# + {"Collapsed": "false"}
eICU_df['label'] = eICU_df[eICU_df.death_ts - eICU_df.ts <= time_window_h * 60]
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ## Creating a single encoding dictionary for the complete dataframe
#
# Combine the one hot encoding dictionaries of all tables, having in account the converged ones, into a single dictionary representative of all the categorical features in the resulting dataframe.

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T04:33:08.143209Z", "iopub.status.idle": "2020-03-02T04:33:08.143643Z"}}
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

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T04:33:08.144659Z", "iopub.status.idle": "2020-03-02T04:33:08.145016Z"}}
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

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T04:33:08.145908Z", "iopub.status.idle": "2020-03-02T04:33:08.146300Z"}}
cat_feat_ohe = du.utils.merge_dicts([cat_feat_ohe_adms_drug, cat_feat_ohe_inf_drug,
                                     cat_feat_ohe_med, cat_feat_ohe_treat,
                                     cat_feat_ohe_diag, cat_feat_ohe_alrg,
                                     cat_feat_ohe_past_hist, cat_feat_ohe_lab,
                                     cat_feat_ohe_patient, cat_feat_ohe_notes])

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T04:33:08.146990Z", "iopub.status.idle": "2020-03-02T04:33:08.147354Z"}}
list(cat_feat_ohe.keys())

# + [markdown] {"Collapsed": "false"}
# Clean the one hot encoded column names, as they are in the final dataframe:

# + {"Collapsed": "false"}
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

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T04:33:08.143209Z", "iopub.status.idle": "2020-03-02T04:33:08.143643Z"}}
stream_adms_drug = open(f'{data_path}admissionDrug_norm_stats.yaml', 'r')
stream_inf_drug = open(f'{data_path}infusionDrug_norm_stats.yaml', 'r')
stream_med = open(f'{data_path}medication_norm_stats.yaml', 'r')
# stream_in_out = open(f'{data_path}cat_embed_feat_enum_in_out.yaml', 'r')
stream_lab = open(f'{data_path}lab_norm_stats.yaml', 'r')
stream_patient = open(f'{data_path}patient_norm_stats.yaml', 'r')
# stream_vital_aprdc = open(f'{data_path}vitalAperiodic_norm_stats.yaml', 'r')
# stream_vital_prdc = open(f'{data_path}vitalPeriodic_norm_stats.yaml', 'r')

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T04:33:08.144659Z", "iopub.status.idle": "2020-03-02T04:33:08.145016Z"}}
norm_stats_adms_drug = yaml.load(stream_adms_drug, Loader=yaml.FullLoader)
norm_stats_inf_drug = yaml.load(stream_inf_drug, Loader=yaml.FullLoader)
norm_stats_med = yaml.load(stream_med, Loader=yaml.FullLoader)
# norm_stats__in_out = yaml.load(stream_in_out, Loader=yaml.FullLoader)
norm_stats_lab = yaml.load(stream_lab, Loader=yaml.FullLoader)
norm_stats_patient = yaml.load(stream_patient, Loader=yaml.FullLoader)
# norm_stats_vital_aprdc = yaml.load(stream_vital_aprdc, Loader=yaml.FullLoader)
# norm_stats_vital_prdc = yaml.load(stream_vital_prdc, Loader=yaml.FullLoader)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T04:33:08.145908Z", "iopub.status.idle": "2020-03-02T04:33:08.146300Z"}}
norm_stats = du.utils.merge_dicts([norm_stats_adms_drug, norm_stats_inf_drug,
                                   norm_stats_med,
#                                    norm_stats_in_out,
                                   norm_stats_lab, norm_stats_patient,
                                   norm_stats_vital_aprdc, norm_stats_vital_prdc])

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T04:33:08.146990Z", "iopub.status.idle": "2020-03-02T04:33:08.147354Z"}}
list(norm_stats.keys())

# + [markdown] {"Collapsed": "false"}
# Save the final encoding dictionary:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T04:33:08.148525Z", "iopub.status.idle": "2020-03-02T04:33:08.148811Z"}}
stream = open(f'{data_path}/cleaned/eICU_norm_stats.yaml', 'w')
yaml.dump(norm_stats, stream, default_flow_style=False)
