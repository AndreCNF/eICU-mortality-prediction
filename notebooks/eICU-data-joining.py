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

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl", "Collapsed": "false", "persistent_id": "522745b5-b5bf-479f-b697-5c7e9e12fc33", "last_executed_text": "import os                                  # os handles directory/workspace changes\nimport numpy as np                         # NumPy to handle numeric and NaN operations\nimport yaml                                # Save and load YAML files", "execution_event_id": "deb57b39-6a79-4b3a-95ed-02f8089ff593", "execution": {"iopub.status.busy": "2020-02-28T18:53:32.095433Z", "iopub.execute_input": "2020-02-28T18:53:32.095748Z", "iopub.status.idle": "2020-02-28T18:53:32.121877Z", "shell.execute_reply.started": "2020-02-28T18:53:32.095698Z", "shell.execute_reply": "2020-02-28T18:53:32.121052Z"}}
import os                                  # os handles directory/workspace changes
import numpy as np                         # NumPy to handle numeric and NaN operations
import yaml                                # Save and load YAML files

# + {"Collapsed": "false", "persistent_id": "02accdbf-be7e-415c-ba11-165906e66c50", "last_executed_text": "# Debugging packages\nimport pixiedust                           # Debugging in Jupyter Notebook cells", "execution_event_id": "fa33a2f7-7127-49c6-bbe9-f89555b1f2be", "execution": {"iopub.status.busy": "2020-02-28T18:53:32.337974Z", "iopub.execute_input": "2020-02-28T18:53:32.338269Z", "iopub.status.idle": "2020-02-28T18:53:35.015658Z", "shell.execute_reply.started": "2020-02-28T18:53:32.338225Z", "shell.execute_reply": "2020-02-28T18:53:35.014298Z"}}
# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# + {"Collapsed": "false", "persistent_id": "a1f6ee7f-36d4-489d-b2dd-ec2a38f15d11", "last_executed_text": "# Change to parent directory (presumably \"Documents\")\nos.chdir(\"../../..\")\n\n# Path to the CSV dataset files\ndata_path = 'Documents/Datasets/Thesis/eICU/uncompressed/'\n\n# Path to the code files\nproject_path = 'Documents/GitHub/eICU-mortality-prediction/'", "execution_event_id": "baeb346a-1c34-42d1-a501-7ae37369255e", "execution": {"iopub.status.busy": "2020-02-28T18:53:35.017309Z", "iopub.execute_input": "2020-02-28T18:53:35.017575Z", "iopub.status.idle": "2020-02-28T18:53:35.022012Z", "shell.execute_reply.started": "2020-02-28T18:53:35.017532Z", "shell.execute_reply": "2020-02-28T18:53:35.020863Z"}}
# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# Path to the CSV dataset files
data_path = 'data/eICU/uncompressed/cleaned/'
# Path to the code files
project_path = 'code/eICU-mortality-prediction/'

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T18:53:35.023307Z", "iopub.execute_input": "2020-02-28T18:53:35.023601Z", "iopub.status.idle": "2020-02-28T18:53:35.306929Z", "shell.execute_reply.started": "2020-02-28T18:53:35.023557Z", "shell.execute_reply": "2020-02-28T18:53:35.305611Z"}}
# Make sure that every large operation can be handled, by using the disk as an overflow for the memory
# !export MODIN_OUT_OF_CORE=true
# Another trick to do with Pandas so as to be able to allocate bigger objects to memory
# !sudo bash -c 'echo 1 > /proc/sys/vm/overcommit_memory'

# + {"Collapsed": "false", "persistent_id": "c0c2e356-d4f4-4a9d-bec2-88bdf9eb6a38", "last_executed_text": "import modin.pandas as pd                  # Optimized distributed version of Pandas\nimport data_utils as du                    # Data science and machine learning relevant methods", "execution_event_id": "82ef68be-443a-4bb8-8abd-7457a7005b4d", "execution": {"iopub.status.busy": "2020-02-28T18:53:35.308749Z", "iopub.execute_input": "2020-02-28T18:53:35.309054Z", "iopub.status.idle": "2020-02-28T18:53:45.266976Z", "shell.execute_reply.started": "2020-02-28T18:53:35.308980Z", "shell.execute_reply": "2020-02-28T18:53:45.265806Z"}}
import modin.pandas as pd                  # Optimized distributed version of Pandas
# import pandas as pd
import data_utils as du                    # Data science and machine learning relevant methods
# -

# Allow pandas to show more columns:

# + {"execution": {"iopub.status.busy": "2020-02-28T18:53:45.268418Z", "iopub.execute_input": "2020-02-28T18:53:45.268692Z", "iopub.status.idle": "2020-02-28T18:53:45.276792Z", "shell.execute_reply.started": "2020-02-28T18:53:45.268640Z", "shell.execute_reply": "2020-02-28T18:53:45.275805Z"}}
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

# + [markdown] {"Collapsed": "false"}
# Set the random seed for reproducibility

# + {"Collapsed": "false", "persistent_id": "39b552cd-6948-4ec8-ac04-42f850c1e05a", "last_executed_text": "du.set_random_seed(42)", "execution_event_id": "29ab85ce-b7fd-4c5a-a110-5841e741c369", "execution": {"iopub.status.busy": "2020-02-28T18:53:45.278096Z", "iopub.execute_input": "2020-02-28T18:53:45.278324Z", "iopub.status.idle": "2020-02-28T18:53:45.305888Z", "shell.execute_reply.started": "2020-02-28T18:53:45.278282Z", "shell.execute_reply": "2020-02-28T18:53:45.304931Z"}}
du.set_random_seed(42)

# + [markdown] {"Collapsed": "false"}
# ## Loading the data

# + [markdown] {"Collapsed": "false"}
# ### Patient information

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:46.211820Z", "iopub.execute_input": "2020-02-26T16:59:46.212044Z", "iopub.status.idle": "2020-02-26T16:59:46.856149Z", "shell.execute_reply.started": "2020-02-26T16:59:46.212002Z", "shell.execute_reply": "2020-02-26T16:59:46.855343Z"}}
patient_df = pd.read_csv(f'{data_path}normalized/patient.csv')
patient_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:46.857495Z", "iopub.status.idle": "2020-02-26T16:59:46.980426Z", "iopub.execute_input": "2020-02-26T16:59:46.857707Z", "shell.execute_reply.started": "2020-02-26T16:59:46.857668Z", "shell.execute_reply": "2020-02-26T16:59:46.979780Z"}}
du.search_explore.dataframe_missing_values(patient_df)

# + [markdown] {"Collapsed": "false"}
# Remove rows that don't identify the patient:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:46.981384Z", "iopub.status.idle": "2020-02-26T16:59:47.066515Z", "iopub.execute_input": "2020-02-26T16:59:46.981588Z", "shell.execute_reply.started": "2020-02-26T16:59:46.981552Z", "shell.execute_reply": "2020-02-26T16:59:47.065696Z"}}
patient_df = patient_df[~patient_df.patientunitstayid.isnull()]

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:47.067552Z", "iopub.status.idle": "2020-02-26T16:59:47.198739Z", "iopub.execute_input": "2020-02-26T16:59:47.067772Z", "shell.execute_reply.started": "2020-02-26T16:59:47.067733Z", "shell.execute_reply": "2020-02-26T16:59:47.197825Z"}}
du.search_explore.dataframe_missing_values(patient_df)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:47.199752Z", "iopub.status.idle": "2020-02-26T16:59:47.340589Z", "iopub.execute_input": "2020-02-26T16:59:47.199965Z", "shell.execute_reply.started": "2020-02-26T16:59:47.199927Z", "shell.execute_reply": "2020-02-26T16:59:47.339835Z"}}
patient_df.patientunitstayid = patient_df.patientunitstayid.astype(int)
patient_df.ts = patient_df.ts.astype(int)
patient_df.dtypes

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:47.341626Z", "iopub.status.idle": "2020-02-26T16:59:47.403437Z", "iopub.execute_input": "2020-02-26T16:59:47.341855Z", "shell.execute_reply.started": "2020-02-26T16:59:47.341801Z", "shell.execute_reply": "2020-02-26T16:59:47.402721Z"}}
note_df = pd.read_csv(f'{data_path}normalized/note.csv')
note_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:47.404401Z", "iopub.status.idle": "2020-02-26T16:59:47.411439Z", "iopub.execute_input": "2020-02-26T16:59:47.404611Z", "shell.execute_reply.started": "2020-02-26T16:59:47.404573Z", "shell.execute_reply": "2020-02-26T16:59:47.410698Z"}}
patient_df = patient_df.drop(columns='Unnamed: 0')
note_df = note_df.drop(columns='Unnamed: 0')

# + [markdown] {"Collapsed": "false"}
# ### Diagnosis

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:47.412341Z", "iopub.execute_input": "2020-02-26T16:59:47.412546Z", "iopub.status.idle": "2020-02-26T16:59:47.635581Z", "shell.execute_reply.started": "2020-02-26T16:59:47.412510Z", "shell.execute_reply": "2020-02-26T16:59:47.634774Z"}}
diagns_df = pd.read_csv(f'{data_path}normalized/diagnosis.csv')
diagns_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:47.636610Z", "iopub.execute_input": "2020-02-26T16:59:47.636829Z", "iopub.status.idle": "2020-02-26T16:59:47.715415Z", "shell.execute_reply.started": "2020-02-26T16:59:47.636789Z", "shell.execute_reply": "2020-02-26T16:59:47.714740Z"}}
alrg_df = pd.read_csv(f'{data_path}normalized/allergy.csv')
alrg_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:47.716398Z", "iopub.execute_input": "2020-02-26T16:59:47.716608Z", "iopub.status.idle": "2020-02-26T16:59:47.822680Z", "shell.execute_reply.started": "2020-02-26T16:59:47.716571Z", "shell.execute_reply": "2020-02-26T16:59:47.821486Z"}}
past_hist_df = pd.read_csv(f'{data_path}normalized/pastHistory.csv')
past_hist_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:47.823705Z", "iopub.execute_input": "2020-02-26T16:59:47.823927Z", "iopub.status.idle": "2020-02-26T16:59:47.833399Z", "shell.execute_reply.started": "2020-02-26T16:59:47.823882Z", "shell.execute_reply": "2020-02-26T16:59:47.832691Z"}}
diagns_df = diagns_df.drop(columns='Unnamed: 0')
alrg_df = alrg_df.drop(columns='Unnamed: 0')
past_hist_df = past_hist_df.drop(columns='Unnamed: 0')

# + [markdown] {"Collapsed": "false"}
# ### Treatments

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:47.834369Z", "iopub.execute_input": "2020-02-26T16:59:47.834643Z", "iopub.status.idle": "2020-02-26T16:59:48.104487Z", "shell.execute_reply.started": "2020-02-26T16:59:47.834601Z", "shell.execute_reply": "2020-02-26T16:59:48.103450Z"}}
treat_df = pd.read_csv(f'{data_path}normalized/treatment.csv')
treat_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:48.105474Z", "iopub.execute_input": "2020-02-26T16:59:48.105692Z", "iopub.status.idle": "2020-02-26T16:59:48.201610Z", "shell.execute_reply.started": "2020-02-26T16:59:48.105652Z", "shell.execute_reply": "2020-02-26T16:59:48.200678Z"}}
adms_drug_df = pd.read_csv(f'{data_path}normalized/admissionDrug.csv')
adms_drug_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:48.202951Z", "iopub.execute_input": "2020-02-26T16:59:48.203294Z", "iopub.status.idle": "2020-02-26T16:59:48.659908Z", "shell.execute_reply.started": "2020-02-26T16:59:48.203240Z", "shell.execute_reply": "2020-02-26T16:59:48.658605Z"}}
inf_drug_df = pd.read_csv(f'{data_path}normalized/infusionDrug.csv')
inf_drug_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:48.660925Z", "iopub.execute_input": "2020-02-26T16:59:48.661136Z", "iopub.status.idle": "2020-02-26T16:59:49.714974Z", "shell.execute_reply.started": "2020-02-26T16:59:48.661094Z", "shell.execute_reply": "2020-02-26T16:59:49.714159Z"}}
med_df = pd.read_csv(f'{data_path}normalized/medication.csv')
med_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:49.718351Z", "iopub.execute_input": "2020-02-26T16:59:49.718595Z", "iopub.status.idle": "2020-02-26T16:59:51.137028Z", "shell.execute_reply.started": "2020-02-26T16:59:49.718556Z", "shell.execute_reply": "2020-02-26T16:59:51.136352Z"}}
in_out_df = pd.read_csv(f'{data_path}normalized/intakeOutput.csv')
in_out_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:51.138964Z", "iopub.status.idle": "2020-02-26T16:59:51.152844Z", "iopub.execute_input": "2020-02-26T16:59:51.139172Z", "shell.execute_reply.started": "2020-02-26T16:59:51.139135Z", "shell.execute_reply": "2020-02-26T16:59:51.152164Z"}}
treat_df = treat_df.drop(columns='Unnamed: 0')
adms_drug_df = adms_drug_df.drop(columns='Unnamed: 0')
inf_drug_df = inf_drug_df.drop(columns='Unnamed: 0')
med_df = med_df.drop(columns='Unnamed: 0')
in_out_df = in_out_df.drop(columns='Unnamed: 0')

# + [markdown] {"Collapsed": "false"}
# ### Nursing data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:51.153813Z", "iopub.status.idle": "2020-02-26T16:59:51.157020Z", "iopub.execute_input": "2020-02-26T16:59:51.154014Z", "shell.execute_reply.started": "2020-02-26T16:59:51.153978Z", "shell.execute_reply": "2020-02-26T16:59:51.156340Z"}}
# nurse_care_df = pd.read_csv(f'{data_path}normalized/nurseCare.csv')
# nurse_care_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:51.157864Z", "iopub.status.idle": "2020-02-26T16:59:51.161415Z", "iopub.execute_input": "2020-02-26T16:59:51.158061Z", "shell.execute_reply.started": "2020-02-26T16:59:51.158024Z", "shell.execute_reply": "2020-02-26T16:59:51.160771Z"}}
# nurse_assess_df = pd.read_csv(f'{data_path}normalized/nurseAssessment.csv')
# nurse_assess_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:51.162399Z", "iopub.status.idle": "2020-02-26T16:59:51.165898Z", "iopub.execute_input": "2020-02-26T16:59:51.162630Z", "shell.execute_reply.started": "2020-02-26T16:59:51.162585Z", "shell.execute_reply": "2020-02-26T16:59:51.165237Z"}}
# nurse_care_df = nurse_care_df.drop(columns='Unnamed: 0')
# nurse_assess_df = nurse_assess_df.drop(columns='Unnamed: 0')

# + [markdown] {"Collapsed": "false"}
# ### Respiratory data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:51.166812Z", "iopub.status.idle": "2020-02-26T16:59:51.246979Z", "iopub.execute_input": "2020-02-26T16:59:51.167001Z", "shell.execute_reply.started": "2020-02-26T16:59:51.166968Z", "shell.execute_reply": "2020-02-26T16:59:51.246026Z"}}
resp_care_df = pd.read_csv(f'{data_path}normalized/respiratoryCare.csv')
resp_care_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:51.247949Z", "iopub.status.idle": "2020-02-26T16:59:51.253614Z", "iopub.execute_input": "2020-02-26T16:59:51.248200Z", "shell.execute_reply.started": "2020-02-26T16:59:51.248158Z", "shell.execute_reply": "2020-02-26T16:59:51.252951Z"}}
resp_care_df = resp_care_df.drop(columns='Unnamed: 0')

# + [markdown] {"Collapsed": "false"}
# ### Vital signals

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:59:51.254544Z", "iopub.status.idle": "2020-02-26T16:59:59.964661Z", "iopub.execute_input": "2020-02-26T16:59:51.254974Z", "shell.execute_reply.started": "2020-02-26T16:59:51.254914Z", "shell.execute_reply": "2020-02-26T16:59:59.963865Z"}}
vital_aprdc_df = pd.read_csv(f'{data_path}normalized/vitalAperiodic.csv')
vital_aprdc_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T20:20:10.096134Z", "iopub.status.idle": "2020-02-26T20:25:40.991309Z", "iopub.execute_input": "2020-02-26T20:20:10.096378Z", "shell.execute_reply.started": "2020-02-26T20:20:10.096338Z", "shell.execute_reply": "2020-02-26T20:25:40.990809Z"}}
vital_prdc_df = pd.read_csv(f'{data_path}normalized/vitalPeriodic.csv')
vital_prdc_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T20:25:40.992375Z", "iopub.status.idle": "2020-02-26T20:25:41.003550Z", "iopub.execute_input": "2020-02-26T20:25:40.992563Z", "shell.execute_reply.started": "2020-02-26T20:25:40.992528Z", "shell.execute_reply": "2020-02-26T20:25:41.002967Z"}}
vital_aprdc_df = vital_aprdc_df.drop(columns='Unnamed: 0')
vital_prdc_df = vital_prdc_df.drop(columns='Unnamed: 0')

# + [markdown] {"Collapsed": "false"}
# ### Exams data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-27T00:57:21.533328Z", "iopub.status.idle": "2020-02-27T00:57:26.541843Z", "iopub.execute_input": "2020-02-27T00:57:21.533548Z", "shell.execute_reply.started": "2020-02-27T00:57:21.533505Z", "shell.execute_reply": "2020-02-27T00:57:26.541224Z"}}
lab_df = pd.read_csv(f'{data_path}normalized/lab.csv')
lab_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the uneeded 'Unnamed: 0' column:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-27T00:57:26.543002Z", "iopub.status.idle": "2020-02-27T00:57:26.553962Z", "iopub.execute_input": "2020-02-27T00:57:26.543216Z", "shell.execute_reply.started": "2020-02-27T00:57:26.543179Z", "shell.execute_reply": "2020-02-27T00:57:26.553331Z"}}
lab_df = lab_df.drop(columns='Unnamed: 0')

# + [markdown] {"Collapsed": "false"}
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

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:07:27.105405Z", "iopub.status.idle": "2020-02-26T17:07:27.822208Z", "iopub.execute_input": "2020-02-26T17:07:27.105665Z", "shell.execute_reply.started": "2020-02-26T17:07:27.105607Z", "shell.execute_reply": "2020-02-26T17:07:27.821539Z"}}
eICU_df = pd.merge(patient_df, note_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-02-26T17:07:27.823222Z", "iopub.execute_input": "2020-02-26T17:07:27.823437Z", "iopub.status.idle": "2020-02-26T17:07:27.890080Z", "shell.execute_reply.started": "2020-02-26T17:07:27.823400Z", "shell.execute_reply": "2020-02-26T17:07:27.889302Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining with diagnosis data
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-02-26T17:07:27.891160Z", "iopub.status.idle": "2020-02-26T17:07:28.280247Z", "iopub.execute_input": "2020-02-26T17:07:27.891376Z", "shell.execute_reply.started": "2020-02-26T17:07:27.891332Z", "shell.execute_reply": "2020-02-26T17:07:28.279342Z"}}
diagns_df = diagns_df[diagns_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-02-26T17:07:28.281251Z", "iopub.status.idle": "2020-02-26T17:07:28.768423Z", "iopub.execute_input": "2020-02-26T17:07:28.281464Z", "shell.execute_reply.started": "2020-02-26T17:07:28.281423Z", "shell.execute_reply": "2020-02-26T17:07:28.767728Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(diagns_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:07:28.769432Z", "iopub.status.idle": "2020-02-26T17:07:30.856884Z", "iopub.execute_input": "2020-02-26T17:07:28.769634Z", "shell.execute_reply.started": "2020-02-26T17:07:28.769597Z", "shell.execute_reply": "2020-02-26T17:07:30.856071Z"}}
eICU_df = pd.merge(eICU_df, diagns_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-02-26T17:07:30.857957Z", "iopub.execute_input": "2020-02-26T17:07:30.858196Z", "iopub.status.idle": "2020-02-26T17:07:30.983722Z", "shell.execute_reply.started": "2020-02-26T17:07:30.858152Z", "shell.execute_reply": "2020-02-26T17:07:30.982952Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining with allergy data
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-02-26T17:07:30.984749Z", "iopub.status.idle": "2020-02-26T17:07:31.503355Z", "iopub.execute_input": "2020-02-26T17:07:30.984993Z", "shell.execute_reply.started": "2020-02-26T17:07:30.984946Z", "shell.execute_reply": "2020-02-26T17:07:31.502515Z"}}
alrg_df = alrg_df[alrg_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:07:31.504311Z", "iopub.execute_input": "2020-02-26T17:07:31.504517Z", "iopub.status.idle": "2020-02-26T17:07:33.873474Z", "shell.execute_reply.started": "2020-02-26T17:07:31.504474Z", "shell.execute_reply": "2020-02-26T17:07:33.872614Z"}}
eICU_df = pd.merge(eICU_df, alrg_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-02-26T17:07:33.874569Z", "iopub.execute_input": "2020-02-26T17:07:33.874789Z", "iopub.status.idle": "2020-02-26T17:07:34.047047Z", "shell.execute_reply.started": "2020-02-26T17:07:33.874751Z", "shell.execute_reply": "2020-02-26T17:07:34.046167Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining with past history data
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-02-26T17:07:34.048032Z", "iopub.status.idle": "2020-02-26T17:07:34.669423Z", "iopub.execute_input": "2020-02-26T17:07:34.048249Z", "shell.execute_reply.started": "2020-02-26T17:07:34.048211Z", "shell.execute_reply": "2020-02-26T17:07:34.668454Z"}}
past_hist_df = past_hist_df[past_hist_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-02-26T17:07:34.670722Z", "iopub.status.idle": "2020-02-26T17:07:35.040344Z", "iopub.execute_input": "2020-02-26T17:07:34.671062Z", "shell.execute_reply.started": "2020-02-26T17:07:34.671001Z", "shell.execute_reply": "2020-02-26T17:07:35.039607Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(past_hist_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:07:35.041551Z", "iopub.execute_input": "2020-02-26T17:07:35.041865Z", "iopub.status.idle": "2020-02-26T17:07:35.046235Z", "shell.execute_reply.started": "2020-02-26T17:07:35.041803Z", "shell.execute_reply": "2020-02-26T17:07:35.045648Z"}}
len(eICU_df)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:07:35.047222Z", "iopub.status.idle": "2020-02-26T17:07:40.246101Z", "iopub.execute_input": "2020-02-26T17:07:35.047420Z", "shell.execute_reply.started": "2020-02-26T17:07:35.047385Z", "shell.execute_reply": "2020-02-26T17:07:40.245382Z"}}
eICU_df = pd.merge(eICU_df, past_hist_df, how='outer', on='patientunitstayid')
eICU_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:07:40.247277Z", "iopub.execute_input": "2020-02-26T17:07:40.247495Z", "iopub.status.idle": "2020-02-26T17:07:40.252085Z", "shell.execute_reply.started": "2020-02-26T17:07:40.247451Z", "shell.execute_reply": "2020-02-26T17:07:40.251420Z"}}
len(eICU_df)

# + {"execution": {"iopub.status.busy": "2020-02-26T17:07:40.253018Z", "iopub.execute_input": "2020-02-26T17:07:40.253249Z", "iopub.status.idle": "2020-02-26T17:07:40.560396Z", "shell.execute_reply.started": "2020-02-26T17:07:40.253192Z", "shell.execute_reply": "2020-02-26T17:07:40.559660Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining with treatment data
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-02-26T17:07:40.561479Z", "iopub.status.idle": "2020-02-26T17:07:41.238902Z", "iopub.execute_input": "2020-02-26T17:07:40.561696Z", "shell.execute_reply.started": "2020-02-26T17:07:40.561658Z", "shell.execute_reply": "2020-02-26T17:07:41.238047Z"}}
treat_df = treat_df[treat_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-02-26T17:07:41.239941Z", "iopub.status.idle": "2020-02-26T17:07:41.790389Z", "iopub.execute_input": "2020-02-26T17:07:41.240144Z", "shell.execute_reply.started": "2020-02-26T17:07:41.240107Z", "shell.execute_reply": "2020-02-26T17:07:41.789618Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(treat_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:07:41.791413Z", "iopub.status.idle": "2020-02-26T17:07:47.561947Z", "iopub.execute_input": "2020-02-26T17:07:41.791687Z", "shell.execute_reply.started": "2020-02-26T17:07:41.791638Z", "shell.execute_reply": "2020-02-26T17:07:47.561334Z"}}
eICU_df = pd.merge(eICU_df, treat_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-02-26T17:07:47.563082Z", "iopub.execute_input": "2020-02-26T17:07:47.563332Z", "iopub.status.idle": "2020-02-26T17:07:47.717394Z", "shell.execute_reply.started": "2020-02-26T17:07:47.563290Z", "shell.execute_reply": "2020-02-26T17:07:47.716130Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining with admission drug data
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-02-26T17:07:47.718563Z", "iopub.status.idle": "2020-02-26T17:07:48.792166Z", "iopub.execute_input": "2020-02-26T17:07:47.718799Z", "shell.execute_reply.started": "2020-02-26T17:07:47.718758Z", "shell.execute_reply": "2020-02-26T17:07:48.791455Z"}}
adms_drug_df = adms_drug_df[adms_drug_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:07:48.793125Z", "iopub.status.idle": "2020-02-26T17:07:52.797322Z", "iopub.execute_input": "2020-02-26T17:07:48.793339Z", "shell.execute_reply.started": "2020-02-26T17:07:48.793301Z", "shell.execute_reply": "2020-02-26T17:07:52.796623Z"}}
eICU_df = pd.merge(eICU_df, adms_drug_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-02-26T17:07:52.798298Z", "iopub.execute_input": "2020-02-26T17:07:52.798578Z", "iopub.status.idle": "2020-02-26T17:07:52.989672Z", "shell.execute_reply.started": "2020-02-26T17:07:52.798523Z", "shell.execute_reply": "2020-02-26T17:07:52.988850Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining with infusion drug data
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-02-26T17:07:52.990704Z", "iopub.status.idle": "2020-02-26T17:07:54.366956Z", "iopub.execute_input": "2020-02-26T17:07:52.990918Z", "shell.execute_reply.started": "2020-02-26T17:07:52.990880Z", "shell.execute_reply": "2020-02-26T17:07:54.366220Z"}}
inf_drug_df = inf_drug_df[inf_drug_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:07:54.368025Z", "iopub.status.idle": "2020-02-26T17:08:15.190596Z", "iopub.execute_input": "2020-02-26T17:07:54.368248Z", "shell.execute_reply.started": "2020-02-26T17:07:54.368209Z", "shell.execute_reply": "2020-02-26T17:08:15.189843Z"}}
eICU_df = pd.merge(eICU_df, inf_drug_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-02-26T17:08:15.191751Z", "iopub.execute_input": "2020-02-26T17:08:15.191977Z", "iopub.status.idle": "2020-02-26T17:08:16.149318Z", "shell.execute_reply.started": "2020-02-26T17:08:15.191938Z", "shell.execute_reply": "2020-02-26T17:08:16.148363Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining with medication data
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-02-26T17:08:16.150294Z", "iopub.status.idle": "2020-02-26T17:08:21.231786Z", "iopub.execute_input": "2020-02-26T17:08:16.150530Z", "shell.execute_reply.started": "2020-02-26T17:08:16.150491Z", "shell.execute_reply": "2020-02-26T17:08:21.231033Z"}}
med_df = med_df[med_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-02-26T17:08:21.232890Z", "iopub.status.idle": "2020-02-26T17:08:23.213237Z", "iopub.execute_input": "2020-02-26T17:08:21.233125Z", "shell.execute_reply.started": "2020-02-26T17:08:21.233084Z", "shell.execute_reply": "2020-02-26T17:08:23.212449Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(med_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:08:23.214274Z", "iopub.status.idle": "2020-02-26T17:09:43.592703Z", "iopub.execute_input": "2020-02-26T17:08:23.214521Z", "shell.execute_reply.started": "2020-02-26T17:08:23.214465Z", "shell.execute_reply": "2020-02-26T17:09:43.592044Z"}}
eICU_df = pd.merge(eICU_df, med_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-02-26T17:09:43.593771Z", "iopub.execute_input": "2020-02-26T17:09:43.593979Z", "iopub.status.idle": "2020-02-26T17:09:45.491765Z", "shell.execute_reply.started": "2020-02-26T17:09:43.593941Z", "shell.execute_reply": "2020-02-26T17:09:45.491055Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining with intake outake data
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-02-26T17:09:45.492688Z", "iopub.status.idle": "2020-02-26T17:10:04.993461Z", "iopub.execute_input": "2020-02-26T17:09:45.492911Z", "shell.execute_reply.started": "2020-02-26T17:09:45.492872Z", "shell.execute_reply": "2020-02-26T17:10:04.992656Z"}}
in_out_df = in_out_df[in_out_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-02-26T17:10:04.994792Z", "iopub.status.idle": "2020-02-26T17:10:15.489016Z", "iopub.execute_input": "2020-02-26T17:10:04.995131Z", "shell.execute_reply.started": "2020-02-26T17:10:04.995059Z", "shell.execute_reply": "2020-02-26T17:10:15.488283Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(in_out_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:10:15.490034Z", "iopub.status.idle": "2020-02-26T17:13:31.423620Z", "iopub.execute_input": "2020-02-26T17:10:15.490324Z", "shell.execute_reply.started": "2020-02-26T17:10:15.490284Z", "shell.execute_reply": "2020-02-26T17:13:31.422848Z"}}
eICU_df = pd.merge(eICU_df, in_out_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-02-26T17:13:31.424657Z", "iopub.execute_input": "2020-02-26T17:13:31.424888Z", "iopub.status.idle": "2020-02-26T17:13:35.211843Z", "shell.execute_reply.started": "2020-02-26T17:13:31.424842Z", "shell.execute_reply": "2020-02-26T17:13:35.211138Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining with nurse care data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:13:35.212814Z", "iopub.status.idle": "2020-02-26T17:13:35.215962Z", "iopub.execute_input": "2020-02-26T17:13:35.213020Z", "shell.execute_reply.started": "2020-02-26T17:13:35.212983Z", "shell.execute_reply": "2020-02-26T17:13:35.215332Z"}}
# eICU_df = pd.merge(eICU_df, nurse_care_df, how='outer', on=['patientunitstayid', 'ts'])
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Joining with nurse assessment data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:13:35.216809Z", "iopub.status.idle": "2020-02-26T17:13:37.233514Z", "iopub.execute_input": "2020-02-26T17:13:35.217016Z", "shell.execute_reply.started": "2020-02-26T17:13:35.216967Z", "shell.execute_reply": "2020-02-26T17:13:37.232810Z"}}
# eICU_df = pd.merge(eICU_df, nurse_assess_df, how='outer', on=['patientunitstayid', 'ts'])
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Joining with nurse charting data

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:13:37.234561Z", "iopub.status.idle": "2020-02-26T17:13:37.238908Z", "iopub.execute_input": "2020-02-26T17:13:37.234780Z", "shell.execute_reply.started": "2020-02-26T17:13:37.234737Z", "shell.execute_reply": "2020-02-26T17:13:37.238206Z"}}
# eICU_df = pd.merge(eICU_df, nurse_chart_df, how='outer', on=['patientunitstayid', 'ts'])
# eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Joining with respiratory care data
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-02-26T17:13:37.240191Z", "iopub.status.idle": "2020-02-26T17:13:58.349081Z", "iopub.execute_input": "2020-02-26T17:13:37.240721Z", "shell.execute_reply.started": "2020-02-26T17:13:37.240650Z", "shell.execute_reply": "2020-02-26T17:13:58.348266Z"}}
resp_care_df = resp_care_df[resp_care_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:13:58.354794Z", "iopub.status.idle": "2020-02-26T17:16:41.973134Z", "iopub.execute_input": "2020-02-26T17:13:58.355014Z", "shell.execute_reply.started": "2020-02-26T17:13:58.354969Z", "shell.execute_reply": "2020-02-26T17:16:41.972423Z"}}
eICU_df = pd.merge(eICU_df, resp_care_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-02-26T17:16:41.974139Z", "iopub.execute_input": "2020-02-26T17:16:41.974351Z", "iopub.status.idle": "2020-02-26T17:16:46.321308Z", "shell.execute_reply.started": "2020-02-26T17:16:41.974313Z", "shell.execute_reply": "2020-02-26T17:16:46.320677Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining with aperiodic vital signals data
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-02-26T17:16:46.322313Z", "iopub.status.idle": "2020-02-26T17:17:09.892602Z", "iopub.execute_input": "2020-02-26T17:16:46.322564Z", "shell.execute_reply.started": "2020-02-26T17:16:46.322523Z", "shell.execute_reply": "2020-02-26T17:17:09.891771Z"}}
vital_aprdc_df = vital_aprdc_df[vital_aprdc_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-02-26T17:17:09.893681Z", "iopub.status.idle": "2020-02-26T17:17:17.745850Z", "iopub.execute_input": "2020-02-26T17:17:09.894188Z", "shell.execute_reply.started": "2020-02-26T17:17:09.894121Z", "shell.execute_reply": "2020-02-26T17:17:17.745172Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(vital_aprdc_df.patientunitstayid.unique())]
# -

# Merge the dataframes:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:17:17.746835Z", "iopub.status.idle": "2020-02-26T17:29:31.451298Z", "iopub.execute_input": "2020-02-26T17:17:17.747040Z", "shell.execute_reply.started": "2020-02-26T17:17:17.747001Z", "shell.execute_reply": "2020-02-26T17:29:31.450604Z"}}
eICU_df = pd.merge(eICU_df, vital_aprdc_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + {"execution": {"iopub.status.busy": "2020-02-26T17:29:31.453452Z", "iopub.execute_input": "2020-02-26T17:29:31.453698Z", "iopub.status.idle": "2020-02-26T17:29:48.824822Z", "shell.execute_reply.started": "2020-02-26T17:29:31.453655Z", "shell.execute_reply": "2020-02-26T17:29:48.824158Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining with periodic vital signals data
# -

# Filter to the unit stays that also have data in the other tables:

# + {"execution": {"iopub.status.busy": "2020-02-26T20:27:30.651290Z", "iopub.status.idle": "2020-02-26T20:29:03.630677Z", "iopub.execute_input": "2020-02-26T20:27:30.651538Z", "shell.execute_reply.started": "2020-02-26T20:27:30.651486Z", "shell.execute_reply": "2020-02-26T20:29:03.629871Z"}}
vital_prdc_df = vital_prdc_df[vital_prdc_df.patientunitstayid.isin(eICU_df.patientunitstayid.unique())]
# -

# Also filter only to the unit stays that have data in this new table, considering its importance:

# + {"execution": {"iopub.status.busy": "2020-02-26T20:29:03.632075Z", "iopub.status.idle": "2020-02-26T20:30:05.977870Z", "iopub.execute_input": "2020-02-26T20:29:03.632286Z", "shell.execute_reply.started": "2020-02-26T20:29:03.632247Z", "shell.execute_reply": "2020-02-26T20:30:05.977157Z"}}
eICU_df = eICU_df[eICU_df.patientunitstayid.isin(vital_prdc_df.patientunitstayid.unique())]

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-02-26T17:33:36.720239Z", "iopub.execute_input": "2020-02-26T17:33:36.720916Z", "iopub.status.idle": "2020-02-26T18:02:20.127085Z", "shell.execute_reply.started": "2020-02-26T17:33:36.720857Z", "shell.execute_reply": "2020-02-26T18:02:20.126190Z"}}
eICU_df.to_csv(f'{data_path}normalized/eICU_before_joining_vital_prdc.csv')
# -

# Merge the dataframes:

# + {"execution": {"iopub.status.busy": "2020-02-27T00:58:11.943486Z", "iopub.execute_input": "2020-02-27T00:58:11.943798Z", "iopub.status.idle": "2020-02-27T00:58:11.954565Z", "shell.execute_reply.started": "2020-02-27T00:58:11.943744Z", "shell.execute_reply": "2020-02-27T00:58:11.953754Z"}}
dtype_dict = {'patientunitstayid': int,
              'gender': float,
              'age': float,
              'ethnicity': float,
              'admissionheight': float,
              'admissionweight': float,
              'death_ts': float,
              'ts': int,
              'smoking_status': float,
              'ethanol_use': float,
              'cad': float,
              'cancer': float,
              'diagnosis_type_1': str,
              'diagnosis_disorder_2': str,
              'diagnosis_detailed_3': str,
              'allergyname': str,
              'drugallergyhiclseqno': str,
              'pasthistoryvalue': str,
              'pasthistorytype': str,
              'pasthistorydetails': str,
              'treatmenttype': str,
              'treatmenttherapy': str,
              'treatmentdetails': str,
              'drugunit_x': str,
              'drugadmitfrequency_x': str,
              'drughiclseqno_x': str,
              'drugdosage_x': float,
              'drugadmitfrequency_y': str,
              'drughiclseqno_y': str,
              'drugdosage_y': float,
              'drugunit_y': str,
              'bodyweight_(kg)': float,
              'oral_intake': float,
              'urine_output': float,
              'i.v._intake	': float,
              'saline_flush_(ml)_intake': float,
              'volume_given_ml': float,
              'stool_output': float,
              'prbcs_intake': float,
              'gastric_(ng)_output': float,
              'dialysis_output': float,
              'propofol_intake': float,
              'lr_intake': float,
              'indwellingcatheter_output': float,
              'feeding_tube_flush_ml': float,
              'patient_fluid_removal': float,
              'fentanyl_intake': float,
              'norepinephrine_intake': float,
              'crystalloids_intake': float,
              'voided_amount': float,
              'nutrition_total_intake': float,
              'nutrition': str,
              'nurse_treatments': str,
              'hygiene/adls': str,
              'activity': str,
              'pupils': str,
              'neurologic': str,
              'secretions': str,
              'cough': str,
              'priorvent': float,
              'onvent': float,
              'noninvasivesystolic': float,
              'noninvasivediastolic': float,
              'noninvasivemean': float,
              'paop': float,
              'cardiacoutput': float,
              'cardiacinput': float,
              'svr': float,
              'svri': float,
              'pvr': float,
              'pvri': float}

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T20:27:00.928018Z", "iopub.status.idle": "2020-02-26T20:27:30.650260Z", "iopub.execute_input": "2020-02-26T20:27:00.928261Z", "shell.execute_reply.started": "2020-02-26T20:27:00.928221Z", "shell.execute_reply": "2020-02-26T20:27:30.649669Z"}}
eICU_df = pd.read_csv(f'{data_path}normalized/eICU_before_joining_vital_prdc.csv',
                      dtype=dtype_dict)
eICU_df = eICU_df.drop(columns=['Unnamed: 0'])
eICU_df.head()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T20:30:05.978941Z", "iopub.status.idle": "2020-02-26T20:56:11.534659Z", "iopub.execute_input": "2020-02-26T20:30:05.979158Z", "shell.execute_reply.started": "2020-02-26T20:30:05.979112Z", "shell.execute_reply": "2020-02-26T20:56:11.534004Z"}}
eICU_df = pd.merge(eICU_df, vital_prdc_df, how='outer', on=['patientunitstayid', 'ts'])
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"execution": {"iopub.status.busy": "2020-02-26T20:56:11.535600Z", "iopub.status.idle": "2020-02-26T20:25:41.072051Z", "iopub.execute_input": "2020-02-26T20:56:11.535828Z"}}
eICU_df.to_csv(f'{data_path}normalized/eICU_post_joining_vital_prdc.csv')

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-27T00:58:11.955786Z", "iopub.status.idle": "2020-02-27T01:10:36.152614Z", "iopub.execute_input": "2020-02-27T00:58:11.956018Z", "shell.execute_reply.started": "2020-02-27T00:58:11.955976Z", "shell.execute_reply": "2020-02-27T01:10:36.151782Z"}}
eICU_df = pd.read_csv(f'{data_path}normalized/eICU_post_joining_vital_prdc.csv',
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
eICU_df.to_csv(f'{data_path}normalized/eICU_post_joining.csv')

# + {"execution": {"iopub.status.busy": "2020-02-28T18:53:45.307400Z", "iopub.execute_input": "2020-02-28T18:53:45.307681Z", "iopub.status.idle": "2020-02-28T18:53:45.322745Z", "shell.execute_reply.started": "2020-02-28T18:53:45.307632Z", "shell.execute_reply": "2020-02-28T18:53:45.321700Z"}}
dtype_dict = {'patientunitstayid': int,
              'gender': float,
              'age': float,
              'ethnicity': float,
              'admissionheight': float,
              'admissionweight': float,
              'death_ts': float,
              'ts': int,
              'smoking_status': float,
              'ethanol_use': float,
              'cad': float,
              'cancer': float,
              'diagnosis_type_1': str,
              'diagnosis_disorder_2': str,
              'diagnosis_detailed_3': str,
              'allergyname': str,
              'drugallergyhiclseqno': str,
              'pasthistoryvalue': str,
              'pasthistorytype': str,
              'pasthistorydetails': str,
              'treatmenttype': str,
              'treatmenttherapy': str,
              'treatmentdetails': str,
              'drugunit_x': str,
              'drugadmitfrequency_x': str,
              'drughiclseqno_x': str,
              'drugdosage_x': float,
              'drugadmitfrequency_y': str,
              'drughiclseqno_y': str,
              'drugdosage_y': float,
              'drugunit_y': str,
              'bodyweight_(kg)': float,
              'oral_intake': float,
              'urine_output': float,
              'i.v._intake	': float,
              'saline_flush_(ml)_intake': float,
              'volume_given_ml': float,
              'stool_output': float,
              'prbcs_intake': float,
              'gastric_(ng)_output': float,
              'dialysis_output': float,
              'propofol_intake': float,
              'lr_intake': float,
              'indwellingcatheter_output': float,
              'feeding_tube_flush_ml': float,
              'patient_fluid_removal': float,
              'fentanyl_intake': float,
              'norepinephrine_intake': float,
              'crystalloids_intake': float,
              'voided_amount': float,
              'nutrition_total_intake': float,
              'nutrition': str,
              'nurse_treatments': str,
              'hygiene/adls': str,
              'activity': str,
              'pupils': str,
              'neurologic': str,
              'secretions': str,
              'cough': str,
              'priorvent': float,
              'onvent': float,
              'noninvasivesystolic': float,
              'noninvasivediastolic': float,
              'noninvasivemean': float,
              'paop': float,
              'cardiacoutput': float,
              'cardiacinput': float,
              'svr': float,
              'svri': float,
              'pvr': float,
              'pvri': float,
              'temperature': ,
              'sao2': float,
              'heartrate': float,
              'respiration': float,
              'cvp': float,
              'etco2': float,
              'systemicsystolic': float,
              'systemicdiastolic': float,
              'systemicmean': float,
              'pasystolic': float,
              'padiastolic': float,
              'pamean': float,
              'st1': float,
              'st2': float,
              'st3': float,
              'icp': float,
              'labtypeid': str,
              'labname': str,
              'lab_units': str,
              'lab_result': float}

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T18:53:45.323720Z", "iopub.status.idle": "2020-02-28T18:53:45.324030Z", "iopub.execute_input": "2020-02-28T16:42:12.061630Z", "shell.execute_reply.started": "2020-02-28T16:42:12.061587Z", "shell.execute_reply": "2020-02-28T17:06:23.341003Z"}}
eICU_df = pd.read_csv(f'{data_path}normalized/eICU_post_joining.csv', dtype=dtype_dict)
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

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:06:23.343021Z", "iopub.status.idle": "2020-02-28T17:12:44.357768Z", "iopub.execute_input": "2020-02-28T17:06:23.343252Z", "shell.execute_reply.started": "2020-02-28T17:06:23.343207Z", "shell.execute_reply": "2020-02-28T17:12:44.355713Z"}}
eICU_df.info()

# + [markdown] {"Collapsed": "false"}
# Make sure that the dataframe is ordered by time `ts`:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:12:44.358704Z", "iopub.status.idle": "2020-02-28T17:12:44.359012Z"}}
eICU_df = eICU_df.sort_values('ts')
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove unit stays that have less than 10 records:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:12:44.359917Z", "iopub.status.idle": "2020-02-28T17:12:44.360224Z"}}
unit_stay_len = eICU_df.groupby('patientunitstayid').patientunitstayid.count()
unit_stay_len

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:12:44.361112Z", "iopub.status.idle": "2020-02-28T17:12:44.361398Z"}}
unit_stay_short = set(unit_stay_len[unit_stay_len < 10].index)
unit_stay_short

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:12:44.362353Z", "iopub.status.idle": "2020-02-28T17:12:44.362655Z"}}
len(unit_stay_short)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:12:44.363470Z", "iopub.status.idle": "2020-02-28T17:12:44.363766Z"}}
eICU_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:12:44.364747Z", "iopub.status.idle": "2020-02-28T17:12:44.365032Z"}}
eICU_df = eICU_df[~eICU_df.patientunitstayid.isin(unit_stay_short)]

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:12:44.365992Z", "iopub.status.idle": "2020-02-28T17:12:44.366292Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# Remove unit stays that have data that represent less than 48h:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:12:44.367179Z", "iopub.status.idle": "2020-02-28T17:12:44.367462Z"}}
unit_stay_duration = eICU_df.groupby('patientunitstayid').ts.apply(lambda x: x.max() - x.min())
unit_stay_duration

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:12:44.368395Z", "iopub.status.idle": "2020-02-28T17:12:44.368831Z"}}
unit_stay_short = set(unit_stay_duration[unit_stay_duration < 48*60].index)
unit_stay_short

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:12:44.369646Z", "iopub.status.idle": "2020-02-28T17:12:44.369952Z"}}
len(unit_stay_short)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:12:44.370725Z", "iopub.status.idle": "2020-02-28T17:12:44.371149Z"}}
eICU_df.patientunitstayid.nunique()

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:12:44.371915Z", "iopub.status.idle": "2020-02-28T17:12:44.372197Z"}}
eICU_df = eICU_df[~eICU_df.patientunitstayid.isin(unit_stay_short)]

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:12:44.372972Z", "iopub.execute_input": "2020-02-24T03:09:09.493994Z", "iopub.status.idle": "2020-02-28T17:12:44.373255Z", "shell.execute_reply.started": "2020-02-24T03:09:09.493953Z", "shell.execute_reply": "2020-02-24T03:09:09.508954Z"}}
eICU_df.patientunitstayid.nunique()

# + [markdown] {"Collapsed": "false"}
# ### Joining duplicate columns

# + [markdown] {"Collapsed": "false"}
# #### Continuous features

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:12:44.373995Z", "iopub.status.idle": "2020-02-28T17:12:44.374282Z"}}
set([col.split('_x')[0].split('_y')[0] for col in eICU_df.columns if col.endswith('_x') or col.endswith('_y')])

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:12:44.375054Z", "iopub.status.idle": "2020-02-28T17:12:44.375336Z"}}
eICU_df[['drugdosage_x', 'drugdosage_y']].head(20)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:12:44.376088Z", "iopub.status.idle": "2020-02-28T17:12:44.376410Z"}}
eICU_df[eICU_df.index == 2564878][['drugdosage_x', 'drugdosage_y']]

# + {"Collapsed": "false", "pixiedust": {"displayParams": {}}, "execution": {"iopub.status.busy": "2020-02-28T17:12:44.377477Z", "iopub.status.idle": "2020-02-28T17:12:44.377806Z"}}
eICU_df = du.data_processing.merge_columns(eICU_df, cols_to_merge=['drugdosage'])
eICU_df.sample(20)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:12:44.378799Z", "iopub.status.idle": "2020-02-28T17:12:44.379097Z"}}
eICU_df[['drugdosage', 'drughiclseqno']].head(20)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:12:44.379932Z", "iopub.status.idle": "2020-02-28T17:12:44.380220Z"}}
eICU_df[eICU_df.index == 2564878][['drugdosage']]

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-28T17:12:44.381185Z", "iopub.status.idle": "2020-02-28T17:12:44.381572Z"}}
eICU_df.to_csv(f'{data_path}normalized/eICU_post_merge_continuous_cols.csv')

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:30:52.807662Z", "iopub.status.idle": "2020-02-26T17:30:52.807951Z"}}
eICU_df = pd.read_csv(f'{data_path}normalized/eICU_post_merge_continuous_cols.csv')
eICU_df = eICU_df.drop(columns=['Unnamed: 0'])
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# #### Categorical features
#
# Join encodings of the same features, from different tables.

# + [markdown] {"Collapsed": "false"}
# Load encoding dictionaries:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:30:52.808934Z", "iopub.status.idle": "2020-02-26T17:30:52.809222Z"}}
stream_adms_drug = open(f'{data_path}cat_embed_feat_enum_adms_drug.yaml', 'r')
stream_med = open(f'{data_path}cat_embed_feat_enum_med.yaml', 'r')
cat_embed_feat_enum_adms_drug = yaml.load(stream_adms_drug, Loader=yaml.FullLoader)
cat_embed_feat_enum_med = yaml.load(stream_med, Loader=yaml.FullLoader)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:30:52.810075Z", "iopub.status.idle": "2020-02-26T17:30:52.810403Z"}}
eICU_df[['drugadmitfrequency_x', 'drugunit_x', 'drughiclseqno_x',
         'drugadmitfrequency_y', 'drugunit_y', 'drughiclseqno_y']].head(20)

# + [markdown] {"Collapsed": "false"}
# Standardize the encoding of similar columns:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:30:52.811134Z", "iopub.status.idle": "2020-02-26T17:30:52.811417Z"}}
list(cat_embed_feat_enum_adms_drug.keys())

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:30:52.812055Z", "iopub.status.idle": "2020-02-26T17:30:52.812387Z"}}
list(cat_embed_feat_enum_med.keys())

# + {"Collapsed": "false", "pixiedust": {"displayParams": {}}, "execution": {"iopub.status.busy": "2020-02-26T17:30:52.813323Z", "iopub.status.idle": "2020-02-26T17:30:52.813600Z"}}
eICU_df, cat_embed_feat_enum['drugadmitfrequency'] = du.embedding.converge_enum(eICU_df, cat_feat_name=['drugadmitfrequency_x',
                                                                                                        'drugadmitfrequency_y'],
                                                                                dict1=cat_embed_feat_enum_adms_drug['drugadmitfrequency'],
                                                                                dict2=cat_embed_feat_enum_med['frequency'],
                                                                                nan_value=0, sort=True, inplace=True)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:30:52.814321Z", "iopub.status.idle": "2020-02-26T17:30:52.814623Z"}}
eICU_df, cat_embed_feat_enum['drugunit'] = du.embedding.converge_enum(eICU_df, cat_feat_name=['drugunit_x',
                                                                                              'drugunit_y'],
                                                                      dict1=cat_embed_feat_enum_adms_drug['drugunit'],
                                                                      dict2=cat_embed_feat_enum_med['drugunit'],
                                                                      nan_value=0, sort=True, inplace=True)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:30:52.815558Z", "iopub.status.idle": "2020-02-26T17:30:52.815850Z"}}
eICU_df, cat_embed_feat_enum['drughiclseqno'] = du.embedding.converge_enum(eICU_df, cat_feat_name=['drughiclseqno_x',
                                                                                                   'drughiclseqno_y'],
                                                                           dict1=cat_embed_feat_enum_adms_drug['drughiclseqno'],
                                                                           dict2=cat_embed_feat_enum_med['drughiclseqno'],
                                                                           nan_value=0, sort=True, inplace=True)

# + [markdown] {"Collapsed": "false"}
# Merge the features:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:30:52.816599Z", "iopub.status.idle": "2020-02-26T17:30:52.816877Z"}}
eICU_df = du.data_processing.merge_columns(eICU_df, cols_to_merge=['drugadmitfrequency', 'drugunit', 'drughiclseqno'])
eICU_df.sample(20)

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:30:52.817588Z", "iopub.status.idle": "2020-02-26T17:30:52.817859Z"}}
eICU_df[['drugadmitfrequency', 'drugunit', 'drughiclseqno']].head(20)

# + [markdown] {"Collapsed": "false"}
# Save the current dataframe:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:30:52.818528Z", "iopub.status.idle": "2020-02-26T17:30:52.818804Z"}}
eICU_df.to_csv(f'{data_path}normalized/eICU_post_merge_categorical_cols.csv')

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T16:29:55.464340Z", "iopub.status.idle": "2020-02-26T16:29:55.464637Z"}}
eICU_df = pd.read_csv(f'{data_path}normalized/eICU_post_merge_categorical_cols.csv')
eICU_df = eICU_df.drop(columns=['Unnamed: 0'])
eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# ### Creating a single encoding dictionary for the complete dataframe
#
# Combine the encoding dictionaries of all tables, having in account the converged ones, into a single dictionary representative of all the categorical features in the resulting dataframe.

# + {"execution": {"iopub.status.busy": "2020-02-26T17:30:52.819895Z", "iopub.status.idle": "2020-02-26T17:30:52.820175Z"}}
stream_adms_drug = open(f'{data_path}cat_embed_feat_enum_adms_drug.yaml', 'r')
stream_inf_drug = open(f'{data_path}cat_embed_feat_enum_inf_drug.yaml', 'r')
stream_med = open(f'{data_path}cat_embed_feat_enum_med.yaml', 'r')
stream_treat = open(f'{data_path}cat_embed_feat_enum_treat.yaml', 'r')
stream_in_out = open(f'{data_path}cat_embed_feat_enum_in_out.yaml', 'r')
stream_diag = open(f'{data_path}cat_embed_feat_enum_diag.yaml', 'r')
stream_alrg = open(f'{data_path}cat_embed_feat_enum_alrg.yaml', 'r')
stream_past_hist = open(f'{data_path}cat_embed_feat_enum_past_hist.yaml', 'r')
stream_resp_care = open(f'{data_path}cat_embed_feat_enum_resp_care.yaml', 'r')
# stream_nurse_care = open(f'{data_path}cat_embed_feat_enum_nurse_care.yaml', 'r')
# stream_nurse_assess = open(f'{data_path}cat_embed_feat_enum_nurse_assess.yaml', 'r')
stream_lab = open(f'{data_path}cat_embed_feat_enum_lab.yaml', 'r')
stream_patient = open(f'{data_path}cat_embed_feat_enum_patient.yaml', 'r')
stream_notes = open(f'{data_path}cat_embed_feat_enum_notes.yaml', 'r')

# + {"execution": {"iopub.status.busy": "2020-02-26T17:30:52.820991Z", "iopub.status.idle": "2020-02-26T17:30:52.821270Z"}}
cat_embed_feat_enum_adms_drug = yaml.load(stream_adms_drug, Loader=yaml.FullLoader)
cat_embed_feat_enum_inf_drug = yaml.load(stream_inf_drug, Loader=yaml.FullLoader)
cat_embed_feat_enum_med = yaml.load(stream_med, Loader=yaml.FullLoader)
cat_embed_feat_enum_treat = yaml.load(stream_treat, Loader=yaml.FullLoader)
cat_embed_feat_enum_in_out = yaml.load(stream_in_out, Loader=yaml.FullLoader)
cat_embed_feat_enum_diag = yaml.load(stream_diag, Loader=yaml.FullLoader)
cat_embed_feat_enum_alrg = yaml.load(stream_alrg, Loader=yaml.FullLoader)
cat_embed_feat_enum_past_hist = yaml.load(stream_past_hist, Loader=yaml.FullLoader)
cat_embed_feat_enum_resp_care = yaml.load(stream_resp_care, Loader=yaml.FullLoader)
# cat_embed_feat_enum_nurse_care = yaml.load(stream_nurse_care, Loader=yaml.FullLoader)
# cat_embed_feat_enum_nurse_assess = yaml.load(stream_nurse_assess, Loader=yaml.FullLoader)
cat_embed_feat_enum_lab = yaml.load(stream_lab, Loader=yaml.FullLoader)
cat_embed_feat_enum_patient = yaml.load(stream_patient, Loader=yaml.FullLoader)
cat_embed_feat_enum_notes = yaml.load(stream_notes, Loader=yaml.FullLoader)

# + {"execution": {"iopub.status.busy": "2020-02-26T17:30:52.822108Z", "iopub.status.idle": "2020-02-26T17:30:52.822389Z"}}
cat_embed_feat_enum = du.utils.merge_dicts([cat_embed_feat_enum_adms_drug, cat_embed_feat_enum_inf_drug,
                                            cat_embed_feat_enum_med, cat_embed_feat_enum_treat,
                                            cat_embed_feat_enum_in_out, cat_embed_feat_enum_diag,
                                            cat_embed_feat_enum_alrg, cat_embed_feat_enum_past_hist,
                                            cat_embed_feat_enum_resp_care, cat_embed_feat_enum_lab,
                                            cat_embed_feat_enum_patient, cat_embed_feat_enum_notes,
                                            cat_embed_feat_enum])

# + {"execution": {"iopub.status.busy": "2020-02-26T17:30:52.823078Z", "iopub.status.idle": "2020-02-26T17:30:52.823403Z"}}
list(cat_embed_feat_enum.keys())

# + [markdown] {"Collapsed": "false"}
# Save the final encoding dictionary:

# + {"Collapsed": "false", "execution": {"iopub.status.busy": "2020-02-26T17:30:52.824221Z", "iopub.status.idle": "2020-02-26T17:30:52.824505Z"}}
stream = open(f'{data_path}/cleaned/cat_embed_feat_enum_eICU.yaml', 'w')
yaml.dump(cat_embed_feat_enum, stream, default_flow_style=False)

# + [markdown] {"Collapsed": "false"}
# ### Removing unit stays with too many missing values
#
# Consider removing all unit stays that have, combining rows and columns, a very high percentage of missing values.

# + {"execution": {"iopub.status.busy": "2020-02-26T17:30:52.825304Z", "iopub.status.idle": "2020-02-26T17:30:52.825570Z"}}
n_features = len(eICU_df.columns)
n_features

# + [markdown] {"Collapsed": "false"}
# Create a temporary column that counts each row's number of missing values:

# + {"execution": {"iopub.status.busy": "2020-02-26T17:30:52.826344Z", "iopub.status.idle": "2020-02-26T17:30:52.826648Z"}}
eICU_df['row_msng_val'] = eICU_df.isnull().sum(axis=1)
eICU_df[['patientunitstay', 'ts', 'row_msng_val']].head()

# + [markdown] {"Collapsed": "false"}
# Check each unit stay's percentage of missing data points:

# + {"execution": {"iopub.status.busy": "2020-02-26T17:30:52.827348Z", "iopub.status.idle": "2020-02-26T17:30:52.827618Z"}}
# Number of possible data points in each unit stay
n_data_points = eICU_df.groupby('patientunitstayid').ts.count() * n_features
n_data_points

# + {"execution": {"iopub.status.busy": "2020-02-26T17:30:52.828368Z", "iopub.status.idle": "2020-02-26T17:30:52.828643Z"}}
# Number of missing values in each unit stay
n_msng_val = eICU_df.groupby('patientunitstayid').row_msng_val.sum()
n_msng_val

# + {"execution": {"iopub.status.busy": "2020-02-26T17:30:52.829547Z", "iopub.status.idle": "2020-02-26T17:30:52.829909Z"}}
# Percentage of missing values in each unit stay
msng_val_prct = (n_msng_val / n_data_points) * 100
msng_val_prct

# + {"execution": {"iopub.status.busy": "2020-02-26T17:30:52.830831Z", "iopub.status.idle": "2020-02-26T17:30:52.831172Z"}}
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
# ### Removing rows with too many missing values
#
# This actually might not make sense to do, as some tables, such as `patient`, are set in a timestamp that is unlikely to have matches in other tables, although it's still useful to add.
# -

len(eICU_df)

n_features = len(eICU_df.columns)
n_features

eICU_df = eICU_df[eICU_df.isnull().sum(axis=1) < 0.5 * n_features]

len(eICU_df)

# + [markdown] {"Collapsed": "false"}
# ### Performing imputation
# -

du.search_explore.dataframe_missing_values(eICU_df)

# [TODO] Be careful to avoid interpolating categorical features (e.g. `drugunit`); these must only
# be imputated through zero filling
eICU_df = du.data_processing.missing_values_imputation(eICU_df, method='interpolation',
                                                       id_column='patientunitstay', inplace=True)
eICU_df.head()

du.search_explore.dataframe_missing_values(eICU_df)

# + [markdown] {"Collapsed": "false"}
# ## Setting the label
#
# Define the label column considering the desired time window on which we want to predict mortality (0, 24h, 48h, 72h, etc).
# -

time_window_h = 24

eICU_df['label'] = eICU_df[eICU_df.death_ts - eICU_df.ts <= time_window_h * 60]
eICU_df.head()
