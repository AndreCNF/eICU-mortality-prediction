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
# The main goal of this notebook is to prepare a single CSV document that contains all the relevant data to be used when training a machine learning model that predicts mortality, joining tables, filtering useless columns and performing imputation.

# + [markdown] {"colab_type": "text", "id": "KOdmFzXqF7nq", "toc-hr-collapsed": true, "Collapsed": "false"}
# ## Importing the necessary packages

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl", "Collapsed": "false", "persistent_id": "522745b5-b5bf-479f-b697-5c7e9e12fc33", "last_executed_text": "import os                                  # os handles directory/workspace changes\nimport numpy as np                         # NumPy to handle numeric and NaN operations\nimport yaml                                # Save and load YAML files", "execution_event_id": "deb57b39-6a79-4b3a-95ed-02f8089ff593", "execution": {"iopub.status.busy": "2020-03-15T04:49:51.191853Z", "iopub.execute_input": "2020-03-15T04:49:51.192140Z", "iopub.status.idle": "2020-03-15T04:49:51.216886Z", "shell.execute_reply.started": "2020-03-15T04:49:51.192097Z", "shell.execute_reply": "2020-03-15T04:49:51.216198Z"}}
import os                                  # os handles directory/workspace changes
import numpy as np                         # NumPy to handle numeric and NaN operations
import yaml                                # Save and load YAML files

# + {"Collapsed": "false", "persistent_id": "02accdbf-be7e-415c-ba11-165906e66c50", "last_executed_text": "# Debugging packages\nimport pixiedust                           # Debugging in Jupyter Notebook cells", "execution_event_id": "fa33a2f7-7127-49c6-bbe9-f89555b1f2be", "execution": {"iopub.status.busy": "2020-03-15T04:49:51.493796Z", "iopub.execute_input": "2020-03-15T04:49:51.494095Z", "iopub.status.idle": "2020-03-15T04:49:52.370977Z", "shell.execute_reply.started": "2020-03-15T04:49:51.494041Z", "shell.execute_reply": "2020-03-15T04:49:52.370228Z"}}
# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# + {"Collapsed": "false", "persistent_id": "a1f6ee7f-36d4-489d-b2dd-ec2a38f15d11", "last_executed_text": "# Change to parent directory (presumably \"Documents\")\nos.chdir(\"../../..\")\n\n# Path to the CSV dataset files\ndata_path = 'Documents/Datasets/Thesis/eICU/uncompressed/'\n\n# Path to the code files\nproject_path = 'Documents/GitHub/eICU-mortality-prediction/'", "execution_event_id": "baeb346a-1c34-42d1-a501-7ae37369255e", "execution": {"iopub.status.busy": "2020-03-15T04:49:52.372112Z", "iopub.execute_input": "2020-03-15T04:49:52.372329Z", "iopub.status.idle": "2020-03-15T04:49:52.375484Z", "shell.execute_reply.started": "2020-03-15T04:49:52.372285Z", "shell.execute_reply": "2020-03-15T04:49:52.374863Z"}}
# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# Path to the CSV dataset files
data_path = 'data/eICU/uncompressed/cleaned/'
# Path to the code files
project_path = 'code/eICU-mortality-prediction/'

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:49:52.376590Z", "iopub.execute_input": "2020-03-15T04:49:52.376782Z", "iopub.status.idle": "2020-03-15T04:49:52.623336Z", "shell.execute_reply.started": "2020-03-15T04:49:52.376748Z", "shell.execute_reply": "2020-03-15T04:49:52.622427Z"}}
# Make sure that every large operation can be handled, by using the disk as an overflow for the memory
# !export MODIN_OUT_OF_CORE=true
# Another trick to do with Pandas so as to be able to allocate bigger objects to memory
# !sudo bash -c 'echo 1 > /proc/sys/vm/overcommit_memory'

# + {"Collapsed": "false", "persistent_id": "c0c2e356-d4f4-4a9d-bec2-88bdf9eb6a38", "last_executed_text": "import modin.pandas as pd                  # Optimized distributed version of Pandas\nimport data_utils as du                    # Data science and machine learning relevant methods", "execution_event_id": "82ef68be-443a-4bb8-8abd-7457a7005b4d", "execution": {"iopub.status.busy": "2020-03-15T04:49:52.624693Z", "iopub.execute_input": "2020-03-15T04:49:52.624899Z", "iopub.status.idle": "2020-03-15T04:49:54.369451Z", "shell.execute_reply.started": "2020-03-15T04:49:52.624863Z", "shell.execute_reply": "2020-03-15T04:49:54.368672Z"}}
import modin.pandas as pd                  # Optimized distributed version of Pandas
# import pandas as pd
import data_utils as du                    # Data science and machine learning relevant methods
# -

# Allow pandas to show more columns:

# + {"execution": {"iopub.status.busy": "2020-03-15T04:49:54.370566Z", "iopub.execute_input": "2020-03-15T04:49:54.370805Z", "iopub.status.idle": "2020-03-15T04:49:54.377848Z", "shell.execute_reply.started": "2020-03-15T04:49:54.370756Z", "shell.execute_reply": "2020-03-15T04:49:54.377274Z"}}
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

# + [markdown] {"Collapsed": "false"}
# Set the random seed for reproducibility

# + {"Collapsed": "false", "persistent_id": "39b552cd-6948-4ec8-ac04-42f850c1e05a", "last_executed_text": "du.set_random_seed(42)", "execution_event_id": "29ab85ce-b7fd-4c5a-a110-5841e741c369", "execution": {"iopub.status.busy": "2020-03-15T04:49:54.379280Z", "iopub.execute_input": "2020-03-15T04:49:54.379494Z", "iopub.status.idle": "2020-03-15T04:49:54.383654Z", "shell.execute_reply.started": "2020-03-15T04:49:54.379457Z", "shell.execute_reply": "2020-03-15T04:49:54.383111Z"}}
du.set_random_seed(42)
# -

# ## Initializing variables

# + {"execution": {"iopub.status.busy": "2020-03-15T04:49:54.686816Z", "iopub.execute_input": "2020-03-15T04:49:54.687089Z", "iopub.status.idle": "2020-03-15T04:49:54.696780Z", "shell.execute_reply.started": "2020-03-15T04:49:54.687049Z", "shell.execute_reply": "2020-03-15T04:49:54.696078Z"}}
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

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:49:54.956236Z", "iopub.execute_input": "2020-03-15T04:49:54.956634Z", "iopub.status.idle": "2020-03-15T04:49:54.963070Z", "shell.execute_reply.started": "2020-03-15T04:49:54.956575Z", "shell.execute_reply": "2020-03-15T04:49:54.962367Z"}}
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

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:49:55.150972Z", "iopub.status.idle": "2020-03-15T04:49:55.383910Z", "iopub.execute_input": "2020-03-15T04:49:55.151238Z", "shell.execute_reply.started": "2020-03-15T04:49:55.151197Z", "shell.execute_reply": "2020-03-15T04:49:55.383255Z"}}
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

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:49:55.384953Z", "iopub.status.idle": "2020-03-15T04:49:55.388967Z", "iopub.execute_input": "2020-03-15T04:49:55.385163Z", "shell.execute_reply.started": "2020-03-15T04:49:55.385122Z", "shell.execute_reply": "2020-03-15T04:49:55.388368Z"}}
cat_feat_ohe = du.utils.merge_dicts([cat_feat_ohe_adms_drug, cat_feat_ohe_inf_drug,
                                     cat_feat_ohe_med, cat_feat_ohe_treat,
                                     cat_feat_ohe_diag, cat_feat_ohe_alrg,
                                     cat_feat_ohe_past_hist, cat_feat_ohe_lab,
                                     cat_feat_ohe_patient, cat_feat_ohe_notes])

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:49:55.691107Z", "iopub.status.idle": "2020-03-15T04:49:55.696722Z", "iopub.execute_input": "2020-03-15T04:49:55.691492Z", "shell.execute_reply.started": "2020-03-15T04:49:55.691436Z", "shell.execute_reply": "2020-03-15T04:49:55.696051Z"}}
ohe_columns = du.utils.merge_lists(list(cat_feat_ohe.values()))
ohe_columns = du.data_processing.clean_naming(ohe_columns, lower_case=False)

# + [markdown] {"Collapsed": "false"}
# Add the one hot encoded columns to the dtypes dictionary, specifying them with type `UInt8`

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:49:56.010680Z", "iopub.status.idle": "2020-03-15T04:49:56.015123Z", "iopub.execute_input": "2020-03-15T04:49:56.010942Z", "shell.execute_reply.started": "2020-03-15T04:49:56.010901Z", "shell.execute_reply": "2020-03-15T04:49:56.014371Z"}}
for col in ohe_columns:
    dtype_dict[col] = 'UInt8'

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:49:56.110129Z", "iopub.execute_input": "2020-03-15T04:49:56.110390Z", "iopub.status.idle": "2020-03-15T04:49:56.131677Z", "shell.execute_reply.started": "2020-03-15T04:49:56.110348Z", "shell.execute_reply": "2020-03-15T04:49:56.130885Z"}}
dtype_dict

# + [markdown] {"Collapsed": "false"}
# ## Loading the data

# + [markdown] {"Collapsed": "false"}
# ### Patient information

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:10:30.090963Z", "iopub.execute_input": "2020-03-15T02:10:30.091263Z", "iopub.status.idle": "2020-03-15T02:10:31.790190Z", "shell.execute_reply.started": "2020-03-15T02:10:30.091226Z", "shell.execute_reply": "2020-03-15T02:10:31.789527Z"}}
patient_df = pd.read_csv(f'{data_path}normalized/ohe/patient.csv', dtype=dtype_dict)
patient_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:10:31.791273Z", "iopub.status.idle": "2020-03-15T02:10:32.072233Z", "iopub.execute_input": "2020-03-15T02:10:31.791507Z", "shell.execute_reply.started": "2020-03-15T02:10:31.791463Z", "shell.execute_reply": "2020-03-15T02:10:32.071354Z"}}
du.search_explore.dataframe_missing_values(patient_df)

# + [markdown] {"Collapsed": "false"}
# Remove rows that don't identify the patient:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:10:32.073387Z", "iopub.status.idle": "2020-03-15T02:10:32.379151Z", "iopub.execute_input": "2020-03-15T02:10:32.073605Z", "shell.execute_reply.started": "2020-03-15T02:10:32.073565Z", "shell.execute_reply": "2020-03-15T02:10:32.378367Z"}}
patient_df = patient_df[~patient_df.patientunitstayid.isnull()]

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:10:32.380286Z", "iopub.status.idle": "2020-03-15T02:10:32.712363Z", "iopub.execute_input": "2020-03-15T02:10:32.380511Z", "shell.execute_reply.started": "2020-03-15T02:10:32.380471Z", "shell.execute_reply": "2020-03-15T02:10:32.711786Z"}}
du.search_explore.dataframe_missing_values(patient_df)

# + {"execution": {"iopub.status.busy": "2020-03-15T02:10:32.713291Z", "iopub.execute_input": "2020-03-15T02:10:32.713512Z", "iopub.status.idle": "2020-03-15T02:10:32.718550Z", "shell.execute_reply.started": "2020-03-15T02:10:32.713475Z", "shell.execute_reply": "2020-03-15T02:10:32.717957Z"}}
patient_df.dtypes

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:10:32.719392Z", "iopub.status.idle": "2020-03-15T02:10:33.022886Z", "iopub.execute_input": "2020-03-15T02:10:32.719876Z", "shell.execute_reply.started": "2020-03-15T02:10:32.719830Z", "shell.execute_reply": "2020-03-15T02:10:33.022017Z"}}
note_df = pd.read_csv(f'{data_path}normalized/ohe/note.csv', dtype=dtype_dict)
note_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:10:33.023924Z", "iopub.status.idle": "2020-03-15T02:10:33.034847Z", "iopub.execute_input": "2020-03-15T02:10:33.024287Z", "shell.execute_reply.started": "2020-03-15T02:10:33.024239Z", "shell.execute_reply": "2020-03-15T02:10:33.033952Z"}}
patient_df = patient_df.drop(columns='Unnamed: 0')
note_df = note_df.drop(columns='Unnamed: 0')

# + [markdown] {"Collapsed": "false"}
# ### Diagnosis

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:10:33.035884Z", "iopub.execute_input": "2020-03-15T02:10:33.036085Z", "iopub.status.idle": "2020-03-15T02:10:39.516842Z", "shell.execute_reply.started": "2020-03-15T02:10:33.036050Z", "shell.execute_reply": "2020-03-15T02:10:39.516156Z"}}
diagns_df = pd.read_csv(f'{data_path}normalized/ohe/diagnosis.csv', dtype=dtype_dict)
diagns_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:10:39.517755Z", "iopub.execute_input": "2020-03-15T02:10:39.517956Z", "iopub.status.idle": "2020-03-15T02:10:41.396823Z", "shell.execute_reply.started": "2020-03-15T02:10:39.517920Z", "shell.execute_reply": "2020-03-15T02:10:41.396139Z"}}
alrg_df = pd.read_csv(f'{data_path}normalized/ohe/allergy.csv', dtype=dtype_dict)
alrg_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:10:41.399907Z", "iopub.execute_input": "2020-03-15T02:10:41.400115Z", "iopub.status.idle": "2020-03-15T02:10:43.528800Z", "shell.execute_reply.started": "2020-03-15T02:10:41.400079Z", "shell.execute_reply": "2020-03-15T02:10:43.528139Z"}}
past_hist_df = pd.read_csv(f'{data_path}normalized/ohe/pastHistory.csv', dtype=dtype_dict)
past_hist_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:10:43.530531Z", "iopub.execute_input": "2020-03-15T02:10:43.530747Z", "iopub.status.idle": "2020-03-15T02:10:43.557398Z", "shell.execute_reply.started": "2020-03-15T02:10:43.530707Z", "shell.execute_reply": "2020-03-15T02:10:43.556808Z"}}
diagns_df = diagns_df.drop(columns='Unnamed: 0')
alrg_df = alrg_df.drop(columns='Unnamed: 0')
past_hist_df = past_hist_df.drop(columns='Unnamed: 0')

# + [markdown] {"Collapsed": "false"}
# ### Treatments

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:10:43.558286Z", "iopub.execute_input": "2020-03-15T02:10:43.558504Z", "iopub.status.idle": "2020-03-15T02:10:55.524222Z", "shell.execute_reply.started": "2020-03-15T02:10:43.558466Z", "shell.execute_reply": "2020-03-15T02:10:55.523635Z"}}
treat_df = pd.read_csv(f'{data_path}normalized/ohe/treatment.csv', dtype=dtype_dict)
treat_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:10:55.525172Z", "iopub.execute_input": "2020-03-15T02:10:55.525383Z", "iopub.status.idle": "2020-03-15T02:10:56.958663Z", "shell.execute_reply.started": "2020-03-15T02:10:55.525342Z", "shell.execute_reply": "2020-03-15T02:10:56.958027Z"}}
adms_drug_df = pd.read_csv(f'{data_path}normalized/ohe/admissionDrug.csv', dtype=dtype_dict)
adms_drug_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:10:56.959678Z", "iopub.execute_input": "2020-03-15T02:10:56.959894Z", "iopub.status.idle": "2020-03-15T02:11:09.978939Z", "shell.execute_reply.started": "2020-03-15T02:10:56.959855Z", "shell.execute_reply": "2020-03-15T02:11:09.978286Z"}}
inf_drug_df = pd.read_csv(f'{data_path}normalized/ohe/infusionDrug.csv', dtype=dtype_dict)
inf_drug_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:50:21.760599Z", "iopub.execute_input": "2020-03-15T04:50:21.760857Z", "iopub.status.idle": "2020-03-15T04:51:14.019476Z", "shell.execute_reply.started": "2020-03-15T04:50:21.760816Z", "shell.execute_reply": "2020-03-15T04:51:14.018692Z"}}
med_df = pd.read_csv(f'{data_path}normalized/ohe/medication.csv', dtype=dtype_dict)
med_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:14.020865Z", "iopub.execute_input": "2020-03-15T04:51:14.021114Z", "iopub.status.idle": "2020-03-15T04:51:14.922767Z", "shell.execute_reply.started": "2020-03-15T04:51:14.021069Z", "shell.execute_reply": "2020-03-15T04:51:14.922014Z"}}
in_out_df = pd.read_csv(f'{data_path}normalized/intakeOutput.csv', dtype=dtype_dict)
in_out_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:27.533275Z", "iopub.status.idle": "2020-03-15T04:51:27.552233Z", "iopub.execute_input": "2020-03-15T04:51:27.533689Z", "shell.execute_reply.started": "2020-03-15T04:51:27.533642Z", "shell.execute_reply": "2020-03-15T04:51:27.551633Z"}}
treat_df = treat_df.drop(columns='Unnamed: 0')
adms_drug_df = adms_drug_df.drop(columns='Unnamed: 0')
inf_drug_df = inf_drug_df.drop(columns='Unnamed: 0')
med_df = med_df.drop(columns='Unnamed: 0')
in_out_df = in_out_df.drop(columns='Unnamed: 0')

# + [markdown] {"Collapsed": "false"}
# ### Nursing data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:14.978567Z", "iopub.status.idle": "2020-03-15T04:51:14.978880Z", "iopub.execute_input": "2020-03-15T02:14:05.712354Z", "shell.execute_reply.started": "2020-03-15T02:14:05.712316Z", "shell.execute_reply": "2020-03-15T02:14:05.714353Z"}}
# nurse_care_df = pd.read_csv(f'{data_path}normalized/ohe/nurseCare.csv')
# nurse_care_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:14.979722Z", "iopub.status.idle": "2020-03-15T04:51:14.980057Z", "iopub.execute_input": "2020-03-15T02:14:05.716226Z", "shell.execute_reply.started": "2020-03-15T02:14:05.716191Z", "shell.execute_reply": "2020-03-15T02:14:05.718753Z"}}
# nurse_assess_df = pd.read_csv(f'{data_path}normalized/ohe/nurseAssessment.csv')
# nurse_assess_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:14.980824Z", "iopub.status.idle": "2020-03-15T04:51:14.981136Z", "iopub.execute_input": "2020-03-15T02:14:05.720500Z", "shell.execute_reply.started": "2020-03-15T02:14:05.720462Z", "shell.execute_reply": "2020-03-15T02:14:05.723574Z"}}
# nurse_care_df = nurse_care_df.drop(columns='Unnamed: 0')
# nurse_assess_df = nurse_assess_df.drop(columns='Unnamed: 0')

# + [markdown] {"Collapsed": "false"}
# ### Respiratory data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:32.668297Z", "iopub.status.idle": "2020-03-15T04:51:32.941511Z", "iopub.execute_input": "2020-03-15T04:51:32.668581Z", "shell.execute_reply.started": "2020-03-15T04:51:32.668539Z", "shell.execute_reply": "2020-03-15T04:51:32.940836Z"}}
resp_care_df = pd.read_csv(f'{data_path}normalized/ohe/respiratoryCare.csv', dtype=dtype_dict)
resp_care_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:33.144248Z", "iopub.status.idle": "2020-03-15T04:51:33.151548Z", "iopub.execute_input": "2020-03-15T04:51:33.144599Z", "shell.execute_reply.started": "2020-03-15T04:51:33.144547Z", "shell.execute_reply": "2020-03-15T04:51:33.150672Z"}}
resp_care_df = resp_care_df.drop(columns='Unnamed: 0')

# + [markdown] {"Collapsed": "false"}
# ### Vital signals

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:33.857984Z", "iopub.status.idle": "2020-03-15T04:51:36.793767Z", "iopub.execute_input": "2020-03-15T04:51:33.858330Z", "shell.execute_reply.started": "2020-03-15T04:51:33.858281Z", "shell.execute_reply": "2020-03-15T04:51:36.792914Z"}}
vital_aprdc_df = pd.read_csv(f'{data_path}normalized/vitalAperiodic.csv', dtype=dtype_dict)
vital_aprdc_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:36.795313Z", "iopub.status.idle": "2020-03-15T04:52:12.341001Z", "iopub.execute_input": "2020-03-15T04:51:36.795586Z", "shell.execute_reply.started": "2020-03-15T04:51:36.795517Z", "shell.execute_reply": "2020-03-15T04:52:12.340358Z"}}
vital_prdc_df = pd.read_csv(f'{data_path}normalized/vitalPeriodic.csv', dtype=dtype_dict)
vital_prdc_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:52:12.342764Z", "iopub.status.idle": "2020-03-15T04:53:15.990436Z", "iopub.execute_input": "2020-03-15T04:52:12.342991Z", "shell.execute_reply.started": "2020-03-15T04:52:12.342953Z", "shell.execute_reply": "2020-03-15T04:53:15.989677Z"}}
vital_aprdc_df = vital_aprdc_df.drop(columns='Unnamed: 0')
vital_prdc_df = vital_prdc_df.drop(columns='Unnamed: 0')

# + [markdown] {"Collapsed": "false"}
# ### Exams data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:19:23.890361Z", "iopub.status.idle": "2020-03-15T02:21:58.259579Z", "iopub.execute_input": "2020-03-15T02:19:23.890554Z", "shell.execute_reply.started": "2020-03-15T02:19:23.890518Z", "shell.execute_reply": "2020-03-15T02:21:58.258747Z"}}
lab_df = pd.read_csv(f'{data_path}normalized/ohe/lab.csv', dtype=dtype_dict)
lab_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:21:58.260709Z", "iopub.status.idle": "2020-03-15T02:21:58.275069Z", "iopub.execute_input": "2020-03-15T02:21:58.260919Z", "shell.execute_reply.started": "2020-03-15T02:21:58.260882Z", "shell.execute_reply": "2020-03-15T02:21:58.274431Z"}}
lab_df = lab_df.drop(columns='Unnamed: 0')

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
alrg_stays_list = set(alrg_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:40.881198Z", "iopub.status.idle": "2020-02-26T17:06:40.885978Z", "iopub.execute_input": "2020-02-26T17:06:40.881405Z", "shell.execute_reply.started": "2020-02-26T17:06:40.881369Z", "shell.execute_reply": "2020-02-26T17:06:40.885401Z"}}
len(alrg_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have allergy data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:40.886881Z", "iopub.status.idle": "2020-02-26T17:06:40.910256Z", "iopub.execute_input": "2020-02-26T17:06:40.887083Z", "shell.execute_reply.started": "2020-02-26T17:06:40.887046Z", "shell.execute_reply": "2020-02-26T17:06:40.909552Z"}}
len(set.intersection(full_stays_list, alrg_stays_list))

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
inf_drug_stays_list = set(inf_drug_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:42.311041Z", "iopub.status.idle": "2020-02-26T17:06:42.315850Z", "iopub.execute_input": "2020-02-26T17:06:42.311311Z", "shell.execute_reply.started": "2020-02-26T17:06:42.311265Z", "shell.execute_reply": "2020-02-26T17:06:42.315150Z"}}
len(inf_drug_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have infusion drug data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:42.316935Z", "iopub.status.idle": "2020-02-26T17:06:42.341538Z", "iopub.execute_input": "2020-02-26T17:06:42.317160Z", "shell.execute_reply.started": "2020-02-26T17:06:42.317119Z", "shell.execute_reply": "2020-02-26T17:06:42.340907Z"}}
len(set.intersection(full_stays_list, inf_drug_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:42.342546Z", "iopub.status.idle": "2020-02-26T17:06:43.217103Z", "iopub.execute_input": "2020-02-26T17:06:42.342753Z", "shell.execute_reply.started": "2020-02-26T17:06:42.342716Z", "shell.execute_reply": "2020-02-26T17:06:43.216344Z"}}
med_stays_list = set(med_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:43.218112Z", "iopub.status.idle": "2020-02-26T17:06:43.222301Z", "iopub.execute_input": "2020-02-26T17:06:43.218322Z", "shell.execute_reply.started": "2020-02-26T17:06:43.218285Z", "shell.execute_reply": "2020-02-26T17:06:43.221665Z"}}
len(med_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have medication data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:43.223261Z", "iopub.status.idle": "2020-02-26T17:06:43.268485Z", "iopub.execute_input": "2020-02-26T17:06:43.223466Z", "shell.execute_reply.started": "2020-02-26T17:06:43.223429Z", "shell.execute_reply": "2020-02-26T17:06:43.267653Z"}}
len(set.intersection(full_stays_list, med_stays_list))

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:43.269468Z", "iopub.status.idle": "2020-02-26T17:06:44.458225Z", "iopub.execute_input": "2020-02-26T17:06:43.269688Z", "shell.execute_reply.started": "2020-02-26T17:06:43.269651Z", "shell.execute_reply": "2020-02-26T17:06:44.457486Z"}}
in_out_stays_list = set(in_out_df.patientunitstayid.unique())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:44.459345Z", "iopub.status.idle": "2020-02-26T17:06:44.464285Z", "iopub.execute_input": "2020-02-26T17:06:44.459571Z", "shell.execute_reply.started": "2020-02-26T17:06:44.459530Z", "shell.execute_reply": "2020-02-26T17:06:44.463605Z"}}
len(in_out_stays_list)

# + [markdown] {"Collapsed": "false"}
# Number of unit stays that have intake and output data:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:06:44.465226Z", "iopub.status.idle": "2020-02-26T17:06:44.842575Z", "iopub.execute_input": "2020-02-26T17:06:44.465431Z", "shell.execute_reply.started": "2020-02-26T17:06:44.465394Z", "shell.execute_reply": "2020-02-26T17:06:44.841735Z"}}
len(set.intersection(full_stays_list, in_out_stays_list))

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

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:21:58.276006Z", "iopub.status.idle": "2020-03-15T02:22:01.969672Z", "iopub.execute_input": "2020-03-15T02:21:58.276217Z", "shell.execute_reply.started": "2020-03-15T02:21:58.276180Z", "shell.execute_reply": "2020-03-15T02:22:01.969124Z"}}
eICU_df = pd.merge(patient_df, note_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-15T02:22:01.970558Z", "iopub.execute_input": "2020-03-15T02:22:01.970759Z", "iopub.status.idle": "2020-03-15T02:22:02.145392Z", "shell.execute_reply.started": "2020-03-15T02:22:01.970724Z", "shell.execute_reply": "2020-03-15T02:22:02.144802Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining with diagnosis data
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-15T02:22:02.146269Z", "iopub.status.idle": "2020-03-15T02:22:02.847544Z", "iopub.execute_input": "2020-03-15T02:22:02.146486Z", "shell.execute_reply.started": "2020-03-15T02:22:02.146445Z", "shell.execute_reply": "2020-03-15T02:22:02.846771Z"}}
diagns_df = diagns_df[diagns_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-03-15T02:22:02.848671Z", "iopub.status.idle": "2020-03-15T02:22:03.941351Z", "iopub.execute_input": "2020-03-15T02:22:02.848880Z", "shell.execute_reply.started": "2020-03-15T02:22:02.848843Z", "shell.execute_reply": "2020-03-15T02:22:03.940575Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(diagns_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:22:03.942691Z", "iopub.status.idle": "2020-03-15T02:22:48.069681Z", "iopub.execute_input": "2020-03-15T02:22:03.942912Z", "shell.execute_reply.started": "2020-03-15T02:22:03.942874Z", "shell.execute_reply": "2020-03-15T02:22:48.069040Z"}}
eICU_df = pd.merge(eICU_df, diagns_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-15T02:22:48.070580Z", "iopub.execute_input": "2020-03-15T02:22:48.070790Z", "iopub.status.idle": "2020-03-15T02:22:48.240287Z", "shell.execute_reply.started": "2020-03-15T02:22:48.070752Z", "shell.execute_reply": "2020-03-15T02:22:48.239548Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining with allergy data
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-15T02:22:48.241238Z", "iopub.status.idle": "2020-03-15T02:22:48.928448Z", "iopub.execute_input": "2020-03-15T02:22:48.241453Z", "shell.execute_reply.started": "2020-03-15T02:22:48.241411Z", "shell.execute_reply": "2020-03-15T02:22:48.927796Z"}}
alrg_df = alrg_df[alrg_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:22:48.929354Z", "iopub.execute_input": "2020-03-15T02:22:48.929571Z", "iopub.status.idle": "2020-03-15T02:24:07.805023Z", "shell.execute_reply.started": "2020-03-15T02:22:48.929536Z", "shell.execute_reply": "2020-03-15T02:24:07.804366Z"}}
eICU_df = pd.merge(eICU_df, alrg_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-15T02:24:07.805974Z", "iopub.execute_input": "2020-03-15T02:24:07.806173Z", "iopub.status.idle": "2020-03-15T02:24:07.983354Z", "shell.execute_reply.started": "2020-03-15T02:24:07.806137Z", "shell.execute_reply": "2020-03-15T02:24:07.982791Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining with past history data
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-15T02:24:07.984312Z", "iopub.status.idle": "2020-03-15T02:24:08.686720Z", "iopub.execute_input": "2020-03-15T02:24:07.984523Z", "shell.execute_reply.started": "2020-03-15T02:24:07.984487Z", "shell.execute_reply": "2020-03-15T02:24:08.685962Z"}}
past_hist_df = past_hist_df[past_hist_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-03-15T02:24:08.687691Z", "iopub.status.idle": "2020-03-15T02:24:09.500161Z", "iopub.execute_input": "2020-03-15T02:24:08.687897Z", "shell.execute_reply.started": "2020-03-15T02:24:08.687861Z", "shell.execute_reply": "2020-03-15T02:24:09.499422Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(past_hist_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:24:09.501351Z", "iopub.execute_input": "2020-03-15T02:24:09.501587Z", "iopub.status.idle": "2020-03-15T02:24:09.505948Z", "shell.execute_reply.started": "2020-03-15T02:24:09.501542Z", "shell.execute_reply": "2020-03-15T02:24:09.505361Z"}}
len(eICU_df)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:24:09.506920Z", "iopub.status.idle": "2020-03-15T02:26:54.694986Z", "iopub.execute_input": "2020-03-15T02:24:09.507122Z", "shell.execute_reply.started": "2020-03-15T02:24:09.507079Z", "shell.execute_reply": "2020-03-15T02:26:54.694353Z"}}
eICU_df = pd.merge(eICU_df, past_hist_df, how='outer', on='patientunitstayid')
eICU_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:26:54.695929Z", "iopub.execute_input": "2020-03-15T02:26:54.696137Z", "iopub.status.idle": "2020-03-15T02:26:54.700379Z", "shell.execute_reply.started": "2020-03-15T02:26:54.696097Z", "shell.execute_reply": "2020-03-15T02:26:54.699813Z"}}
len(eICU_df)

# + {"execution": {"iopub.status.busy": "2020-03-15T02:26:54.701219Z", "iopub.execute_input": "2020-03-15T02:26:54.701413Z", "iopub.status.idle": "2020-03-15T02:26:54.906099Z", "shell.execute_reply.started": "2020-03-15T02:26:54.701379Z", "shell.execute_reply": "2020-03-15T02:26:54.905315Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining with treatment data
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-15T02:26:54.907030Z", "iopub.status.idle": "2020-03-15T02:26:55.708474Z", "iopub.execute_input": "2020-03-15T02:26:54.907231Z", "shell.execute_reply.started": "2020-03-15T02:26:54.907191Z", "shell.execute_reply": "2020-03-15T02:26:55.707821Z"}}
treat_df = treat_df[treat_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-03-15T02:26:55.709462Z", "iopub.status.idle": "2020-03-15T02:26:56.581919Z", "iopub.execute_input": "2020-03-15T02:26:55.709683Z", "shell.execute_reply.started": "2020-03-15T02:26:55.709645Z", "shell.execute_reply": "2020-03-15T02:26:56.581202Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(treat_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:26:56.582763Z", "iopub.status.idle": "2020-03-15T02:30:46.397722Z", "iopub.execute_input": "2020-03-15T02:26:56.582962Z", "shell.execute_reply.started": "2020-03-15T02:26:56.582927Z", "shell.execute_reply": "2020-03-15T02:30:46.397039Z"}}
eICU_df = pd.merge(eICU_df, treat_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-15T02:30:46.398741Z", "iopub.execute_input": "2020-03-15T02:30:46.398953Z", "iopub.status.idle": "2020-03-15T02:30:46.563629Z", "shell.execute_reply.started": "2020-03-15T02:30:46.398915Z", "shell.execute_reply": "2020-03-15T02:30:46.563053Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining with admission drug data
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-15T02:30:46.566931Z", "iopub.status.idle": "2020-03-15T02:30:47.260285Z", "iopub.execute_input": "2020-03-15T02:30:46.567146Z", "shell.execute_reply.started": "2020-03-15T02:30:46.567109Z", "shell.execute_reply": "2020-03-15T02:30:47.259525Z"}}
adms_drug_df = adms_drug_df[adms_drug_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:30:47.261399Z", "iopub.status.idle": "2020-03-15T02:35:46.830440Z", "iopub.execute_input": "2020-03-15T02:30:47.261602Z", "shell.execute_reply.started": "2020-03-15T02:30:47.261565Z", "shell.execute_reply": "2020-03-15T02:35:46.829829Z"}}
eICU_df = pd.merge(eICU_df, adms_drug_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-15T02:35:46.831381Z", "iopub.execute_input": "2020-03-15T02:35:46.831648Z", "iopub.status.idle": "2020-03-15T02:35:47.029235Z", "shell.execute_reply.started": "2020-03-15T02:35:46.831532Z", "shell.execute_reply": "2020-03-15T02:35:47.028584Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-15T01:58:25.350816Z", "iopub.execute_input": "2020-02-26T17:33:36.720916Z", "iopub.status.idle": "2020-03-15T01:58:25.351098Z", "shell.execute_reply.started": "2020-02-26T17:33:36.720857Z", "shell.execute_reply": "2020-02-26T18:02:20.126190Z"}}
eICU_df.to_csv(f'{data_path}normalized/ohe/eICU_before_joining_inf_drug.csv')

# + [markdown] {"Collapsed": "false"}
# ### Joining with infusion drug data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T20:27:00.928018Z", "iopub.status.idle": "2020-02-26T20:27:30.650260Z", "iopub.execute_input": "2020-02-26T20:27:00.928261Z", "shell.execute_reply.started": "2020-02-26T20:27:00.928221Z", "shell.execute_reply": "2020-02-26T20:27:30.649669Z"}}
eICU_df = pd.read_csv(f'{data_path}normalized/ohe/eICU_before_joining_inf_drug.csv',
                      dtype=dtype_dict)
eICU_df = eICU_df.drop(columns=['Unnamed: 0'])
eICU_df.head()
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-15T02:35:47.030161Z", "iopub.status.idle": "2020-03-15T02:35:48.049022Z", "iopub.execute_input": "2020-03-15T02:35:47.030371Z", "shell.execute_reply.started": "2020-03-15T02:35:47.030334Z", "shell.execute_reply": "2020-03-15T02:35:48.048273Z"}}
inf_drug_df = inf_drug_df[inf_drug_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T02:35:48.050039Z", "iopub.status.idle": "2020-03-15T01:18:54.618773Z", "iopub.execute_input": "2020-03-15T02:35:48.050259Z", "shell.execute_reply.started": "2020-03-15T01:08:18.661930Z", "shell.execute_reply": "2020-03-15T01:18:54.618017Z"}}
eICU_df = pd.merge(eICU_df, inf_drug_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-15T01:18:54.619795Z", "iopub.execute_input": "2020-03-15T01:18:54.620054Z", "iopub.status.idle": "2020-03-15T01:18:54.766448Z", "shell.execute_reply.started": "2020-03-15T01:18:54.620001Z", "shell.execute_reply": "2020-03-15T01:18:54.765445Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-15T01:58:25.350816Z", "iopub.execute_input": "2020-02-26T17:33:36.720916Z", "iopub.status.idle": "2020-03-15T01:58:25.351098Z", "shell.execute_reply.started": "2020-02-26T17:33:36.720857Z", "shell.execute_reply": "2020-02-26T18:02:20.126190Z"}}
eICU_df.to_csv(f'{data_path}normalized/ohe/eICU_before_joining_med.csv')

# + [markdown] {"Collapsed": "false"}
# ### Joining with medication data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:53:15.991659Z", "iopub.status.idle": "2020-03-15T04:54:36.669943Z", "iopub.execute_input": "2020-03-15T04:53:15.991968Z", "shell.execute_reply.started": "2020-03-15T04:53:15.991894Z", "shell.execute_reply": "2020-03-15T04:54:36.669239Z"}}
eICU_df = pd.read_csv(f'{data_path}normalized/ohe/eICU_before_joining_med.csv',
                      dtype=dtype_dict)
eICU_df = eICU_df.drop(columns=['Unnamed: 0'])
eICU_df.head()
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-15T04:54:36.670934Z", "iopub.status.idle": "2020-03-15T04:54:38.410779Z", "iopub.execute_input": "2020-03-15T04:54:36.671170Z", "shell.execute_reply.started": "2020-03-15T04:54:36.671125Z", "shell.execute_reply": "2020-03-15T04:54:38.409859Z"}}
med_df = med_df[med_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-03-15T04:54:38.412063Z", "iopub.status.idle": "2020-03-15T04:55:04.016518Z", "iopub.execute_input": "2020-03-15T04:54:38.412295Z", "shell.execute_reply.started": "2020-03-15T04:54:38.412255Z", "shell.execute_reply": "2020-03-15T04:55:04.015745Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(med_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:55:04.017681Z", "iopub.status.idle": "2020-03-15T04:51:14.990635Z", "iopub.execute_input": "2020-03-15T04:55:04.017911Z", "shell.execute_reply.started": "2020-03-15T01:18:59.692493Z", "shell.execute_reply": "2020-03-15T01:58:25.322254Z"}}
eICU_df = pd.merge(eICU_df, med_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-15T04:51:14.991311Z", "iopub.execute_input": "2020-02-26T17:09:43.593979Z", "iopub.status.idle": "2020-03-15T04:51:14.991639Z", "shell.execute_reply.started": "2020-02-26T17:09:43.593941Z", "shell.execute_reply": "2020-02-26T17:09:45.491055Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-15T01:58:25.350816Z", "iopub.execute_input": "2020-02-26T17:33:36.720916Z", "iopub.status.idle": "2020-03-15T01:58:25.351098Z", "shell.execute_reply.started": "2020-02-26T17:33:36.720857Z", "shell.execute_reply": "2020-02-26T18:02:20.126190Z"}}
eICU_df.to_csv(f'{data_path}normalized/ohe/eICU_before_joining_in_out.csv')

# + [markdown] {"Collapsed": "false"}
# ### Joining with intake outake data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:53:15.991659Z", "iopub.status.idle": "2020-03-15T04:54:36.669943Z", "iopub.execute_input": "2020-03-15T04:53:15.991968Z", "shell.execute_reply.started": "2020-03-15T04:53:15.991894Z", "shell.execute_reply": "2020-03-15T04:54:36.669239Z"}}
eICU_df = pd.read_csv(f'{data_path}normalized/ohe/eICU_before_joining_in_out.csv',
                      dtype=dtype_dict)
eICU_df = eICU_df.drop(columns=['Unnamed: 0'])
eICU_df.head()
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-15T04:51:14.992310Z", "iopub.status.idle": "2020-03-15T04:51:14.992596Z", "iopub.execute_input": "2020-02-26T17:09:45.492911Z", "shell.execute_reply.started": "2020-02-26T17:09:45.492872Z", "shell.execute_reply": "2020-02-26T17:10:04.992656Z"}}
in_out_df = in_out_df[in_out_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-03-15T04:51:14.993285Z", "iopub.status.idle": "2020-03-15T04:51:14.993604Z", "iopub.execute_input": "2020-02-26T17:10:04.995131Z", "shell.execute_reply.started": "2020-02-26T17:10:04.995059Z", "shell.execute_reply": "2020-02-26T17:10:15.488283Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(in_out_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:14.994314Z", "iopub.status.idle": "2020-03-15T04:51:14.994601Z", "iopub.execute_input": "2020-02-26T17:10:15.490324Z", "shell.execute_reply.started": "2020-02-26T17:10:15.490284Z", "shell.execute_reply": "2020-02-26T17:13:31.422848Z"}}
eICU_df = pd.merge(eICU_df, in_out_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-15T04:51:14.995830Z", "iopub.execute_input": "2020-02-26T17:13:31.424888Z", "iopub.status.idle": "2020-03-15T04:51:14.996159Z", "shell.execute_reply.started": "2020-02-26T17:13:31.424842Z", "shell.execute_reply": "2020-02-26T17:13:35.211138Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-15T01:58:25.350816Z", "iopub.execute_input": "2020-02-26T17:33:36.720916Z", "iopub.status.idle": "2020-03-15T01:58:25.351098Z", "shell.execute_reply.started": "2020-02-26T17:33:36.720857Z", "shell.execute_reply": "2020-02-26T18:02:20.126190Z"}}
eICU_df.to_csv(f'{data_path}normalized/ohe/eICU_before_joining_resp_care.csv')

# + [markdown] {"Collapsed": "false"}
# ### Joining with nurse care data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:14.996900Z", "iopub.status.idle": "2020-03-15T04:51:14.997207Z", "iopub.execute_input": "2020-02-26T17:13:35.213020Z", "shell.execute_reply.started": "2020-02-26T17:13:35.212983Z", "shell.execute_reply": "2020-02-26T17:13:35.215332Z"}}
# eICU_df = pd.merge(eICU_df, nurse_care_df, how='outer', on=['patientunitstayid', 'ts'])
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Joining with nurse assessment data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:14.997878Z", "iopub.status.idle": "2020-03-15T04:51:14.998163Z", "iopub.execute_input": "2020-02-26T17:13:35.217016Z", "shell.execute_reply.started": "2020-02-26T17:13:35.216967Z", "shell.execute_reply": "2020-02-26T17:13:37.232810Z"}}
# eICU_df = pd.merge(eICU_df, nurse_assess_df, how='outer', on=['patientunitstayid', 'ts'])
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Joining with nurse charting data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:14.998878Z", "iopub.status.idle": "2020-03-15T04:51:14.999189Z", "iopub.execute_input": "2020-02-26T17:13:37.234780Z", "shell.execute_reply.started": "2020-02-26T17:13:37.234737Z", "shell.execute_reply": "2020-02-26T17:13:37.238206Z"}}
# eICU_df = pd.merge(eICU_df, nurse_chart_df, how='outer', on=['patientunitstayid', 'ts'])
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Joining with respiratory care data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:53:15.991659Z", "iopub.status.idle": "2020-03-15T04:54:36.669943Z", "iopub.execute_input": "2020-03-15T04:53:15.991968Z", "shell.execute_reply.started": "2020-03-15T04:53:15.991894Z", "shell.execute_reply": "2020-03-15T04:54:36.669239Z"}}
eICU_df = pd.read_csv(f'{data_path}normalized/ohe/eICU_before_joining_resp_care.csv',
                      dtype=dtype_dict)
eICU_df = eICU_df.drop(columns=['Unnamed: 0'])
eICU_df.head()
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-15T04:51:14.999894Z", "iopub.status.idle": "2020-03-15T04:51:15.000181Z", "iopub.execute_input": "2020-02-26T17:13:37.240721Z", "shell.execute_reply.started": "2020-02-26T17:13:37.240650Z", "shell.execute_reply": "2020-02-26T17:13:58.348266Z"}}
resp_care_df = resp_care_df[resp_care_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:15.000859Z", "iopub.status.idle": "2020-03-15T04:51:15.001147Z", "iopub.execute_input": "2020-02-26T17:13:58.355014Z", "shell.execute_reply.started": "2020-02-26T17:13:58.354969Z", "shell.execute_reply": "2020-02-26T17:16:41.972423Z"}}
eICU_df = pd.merge(eICU_df, resp_care_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-15T04:51:15.001893Z", "iopub.execute_input": "2020-02-26T17:16:41.974351Z", "iopub.status.idle": "2020-03-15T04:51:15.002201Z", "shell.execute_reply.started": "2020-02-26T17:16:41.974313Z", "shell.execute_reply": "2020-02-26T17:16:46.320677Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-15T01:58:25.350816Z", "iopub.execute_input": "2020-02-26T17:33:36.720916Z", "iopub.status.idle": "2020-03-15T01:58:25.351098Z", "shell.execute_reply.started": "2020-02-26T17:33:36.720857Z", "shell.execute_reply": "2020-02-26T18:02:20.126190Z"}}
eICU_df.to_csv(f'{data_path}normalized/ohe/eICU_before_joining_vital_aprdc.csv')

# + [markdown] {"Collapsed": "false"}
# ### Joining with aperiodic vital signals data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:53:15.991659Z", "iopub.status.idle": "2020-03-15T04:54:36.669943Z", "iopub.execute_input": "2020-03-15T04:53:15.991968Z", "shell.execute_reply.started": "2020-03-15T04:53:15.991894Z", "shell.execute_reply": "2020-03-15T04:54:36.669239Z"}}
eICU_df = pd.read_csv(f'{data_path}normalized/ohe/eICU_before_joining_vital_aprdc.csv',
                      dtype=dtype_dict)
eICU_df = eICU_df.drop(columns=['Unnamed: 0'])
eICU_df.head()
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-15T04:51:15.002899Z", "iopub.status.idle": "2020-03-15T04:51:15.003197Z", "iopub.execute_input": "2020-02-26T17:16:46.322564Z", "shell.execute_reply.started": "2020-02-26T17:16:46.322523Z", "shell.execute_reply": "2020-02-26T17:17:09.891771Z"}}
vital_aprdc_df = vital_aprdc_df[vital_aprdc_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-03-15T04:51:15.003918Z", "iopub.status.idle": "2020-03-15T04:51:15.004206Z", "iopub.execute_input": "2020-02-26T17:17:09.894188Z", "shell.execute_reply.started": "2020-02-26T17:17:09.894121Z", "shell.execute_reply": "2020-02-26T17:17:17.745172Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(vital_aprdc_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-15T04:51:15.004886Z", "iopub.status.idle": "2020-03-15T04:51:15.005169Z", "iopub.execute_input": "2020-02-26T17:17:17.747040Z", "shell.execute_reply.started": "2020-02-26T17:17:17.747001Z", "shell.execute_reply": "2020-02-26T17:29:31.450604Z"}}
eICU_df = pd.merge(eICU_df, vital_aprdc_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-03-15T04:51:15.006303Z", "iopub.execute_input": "2020-02-26T17:29:31.453698Z", "iopub.status.idle": "2020-03-15T04:51:15.006605Z", "shell.execute_reply.started": "2020-02-26T17:29:31.453655Z", "shell.execute_reply": "2020-02-26T17:29:48.824158Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-03-15T04:51:15.007280Z", "iopub.execute_input": "2020-02-26T17:33:36.720916Z", "iopub.status.idle": "2020-03-15T04:51:15.007573Z", "shell.execute_reply.started": "2020-02-26T17:33:36.720857Z", "shell.execute_reply": "2020-02-26T18:02:20.126190Z"}}
eICU_df.to_csv(f'{data_path}normalized/ohe/eICU_before_joining_vital_prdc.csv')

# + [markdown] {"Collapsed": "false"}
# ### Joining with periodic vital signals data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T20:27:00.928018Z", "iopub.status.idle": "2020-02-26T20:27:30.650260Z", "iopub.execute_input": "2020-02-26T20:27:00.928261Z", "shell.execute_reply.started": "2020-02-26T20:27:00.928221Z", "shell.execute_reply": "2020-02-26T20:27:30.649669Z"}}
eICU_df = pd.read_csv(f'{data_path}normalized/ohe/eICU_before_joining_vital_prdc.csv',
                      dtype=dtype_dict)
eICU_df = eICU_df.drop(columns=['Unnamed: 0'])
eICU_df.head()
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-03-15T01:58:25.348497Z", "iopub.status.idle": "2020-03-15T01:58:25.348812Z", "iopub.execute_input": "2020-02-26T20:27:30.651538Z", "shell.execute_reply.started": "2020-02-26T20:27:30.651486Z", "shell.execute_reply": "2020-02-26T20:29:03.629871Z"}}
vital_prdc_df = vital_prdc_df[vital_prdc_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-03-15T01:58:25.349824Z", "iopub.status.idle": "2020-03-15T01:58:25.350118Z", "iopub.execute_input": "2020-02-26T20:29:03.632286Z", "shell.execute_reply.started": "2020-02-26T20:29:03.632247Z", "shell.execute_reply": "2020-02-26T20:30:05.977157Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(vital_prdc_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T20:30:05.978941Z", "iopub.status.idle": "2020-02-26T20:56:11.534659Z", "iopub.execute_input": "2020-02-26T20:30:05.979158Z", "shell.execute_reply.started": "2020-02-26T20:30:05.979112Z", "shell.execute_reply": "2020-02-26T20:56:11.534004Z"}}
eICU_df = pd.merge(eICU_df, vital_prdc_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-02-26T20:56:11.535600Z", "iopub.status.idle": "2020-02-26T20:25:41.072051Z", "iopub.execute_input": "2020-02-26T20:56:11.535828Z"}}
eICU_df.to_csv(f'{data_path}normalized/ohe/eICU_post_joining_vital_prdc.csv')

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-27T00:58:11.955786Z", "iopub.status.idle": "2020-02-27T01:10:36.152614Z", "iopub.execute_input": "2020-02-27T00:58:11.956018Z", "shell.execute_reply.started": "2020-02-27T00:58:11.955976Z", "shell.execute_reply": "2020-02-27T01:10:36.151782Z"}}
eICU_df = pd.read_csv(f'{data_path}normalized/ohe/eICU_post_joining_vital_prdc.csv',
                      dtype=dtype_dict)
eICU_df = eICU_df.drop(columns=['Unnamed: 0'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-02-27T00:15:58.511997Z", "iopub.execute_input": "2020-02-27T00:15:58.512226Z", "iopub.status.idle": "2020-02-27T00:16:13.852541Z", "shell.execute_reply.started": "2020-02-27T00:15:58.512184Z", "shell.execute_reply": "2020-02-27T00:16:13.851428Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining with lab data
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-02-27T01:10:36.154356Z", "iopub.status.idle": "2020-02-27T01:14:18.564700Z", "iopub.execute_input": "2020-02-27T01:10:36.154590Z", "shell.execute_reply.started": "2020-02-27T01:10:36.154548Z", "shell.execute_reply": "2020-02-27T01:14:18.563808Z"}}
lab_df = lab_df[lab_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-02-27T01:14:18.566086Z", "iopub.status.idle": "2020-02-27T01:15:30.746679Z", "iopub.execute_input": "2020-02-27T01:14:18.566341Z", "shell.execute_reply.started": "2020-02-27T01:14:18.566301Z", "shell.execute_reply": "2020-02-27T01:15:30.745909Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(lab_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-27T01:15:30.747792Z", "iopub.execute_input": "2020-02-27T01:15:30.748015Z", "iopub.status.idle": "2020-02-27T02:11:51.960090Z", "shell.execute_reply.started": "2020-02-27T01:15:30.747973Z", "shell.execute_reply": "2020-02-27T02:11:51.959315Z"}}
eICU_df = pd.merge(eICU_df, lab_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-02-27T02:11:51.961181Z", "iopub.status.idle": "2020-02-27T05:20:54.129974Z", "iopub.execute_input": "2020-02-27T02:11:51.961398Z", "shell.execute_reply.started": "2020-02-27T02:11:51.961359Z", "shell.execute_reply": "2020-02-27T05:20:54.129277Z"}}
eICU_df.to_csv(f'{data_path}normalized/ohe/eICU_post_joining.csv')

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:16:36.275096Z", "iopub.status.idle": "2020-03-02T02:16:52.692400Z", "iopub.execute_input": "2020-03-02T02:16:36.275391Z", "shell.execute_reply.started": "2020-03-02T02:16:36.275334Z", "shell.execute_reply": "2020-03-02T02:16:52.691647Z"}}
eICU_df = pd.read_csv(f'{data_path}normalized/ohe/eICU_post_joining_0.csv', dtype=dtype_dict)
eICU_df = eICU_df.drop(columns=['Unnamed: 0'])
eICU_df.head()
# -

eICU_df.columns

eICU_df.dtypes

# + {"execution": {"iopub.status.busy": "2020-02-27T05:20:54.131063Z", "iopub.execute_input": "2020-02-27T05:20:54.131275Z", "iopub.status.idle": "2020-02-27T05:21:15.499582Z", "shell.execute_reply.started": "2020-02-27T05:20:54.131238Z", "shell.execute_reply": "2020-02-27T05:21:15.498801Z"}}
eICU_df.patientunitstayid.nunique()

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

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T02:42:58.111378Z", "iopub.status.idle": "2020-03-02T02:44:09.853833Z", "iopub.execute_input": "2020-03-02T02:42:58.111630Z", "shell.execute_reply.started": "2020-03-02T02:42:58.111587Z", "shell.execute_reply": "2020-03-02T02:44:09.852640Z"}}
eICU_df.to_csv(f'{data_path}normalized/ohe/eICU_post_merge_duplicate_cols.csv')

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-03-02T04:30:00.844962Z", "iopub.status.idle": "2020-03-02T04:30:11.572222Z", "iopub.execute_input": "2020-03-02T04:30:00.845255Z", "shell.execute_reply.started": "2020-03-02T04:30:00.845203Z", "shell.execute_reply": "2020-03-02T04:30:11.571436Z"}}
eICU_df = pd.read_csv(f'{data_path}normalized/ohe/eICU_post_merge_duplicate_cols.csv', dtype=dtype_dict)
eICU_df = eICU_df.drop(columns=['Unnamed: 0'])
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
                                                       id_column='patientunitstay', inplace=True)
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
