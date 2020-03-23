# -*- coding: utf-8 -*-
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

# + [markdown] {"Collapsed": "false"}
# # eICU Model training
# ---
#
# Training models on the preprocessed the eICU dataset from MIT with the data from over 139k patients collected in the US.

# + [markdown] {"colab_type": "text", "id": "KOdmFzXqF7nq", "toc-hr-collapsed": true, "Collapsed": "false"}
# ## Importing the necessary packages

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl", "Collapsed": "false"}
import os                                  # os handles directory/workspace changes
import comet_ml                            # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                               # PyTorch to create and apply deep learning models
import xgboost as xgb                      # Gradient boosting trees models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
import joblib                              # Save scikit-learn models in disk
from datetime import datetime              # datetime to use proper date and time formats
import sys

# + {"Collapsed": "false"}
# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# + {"Collapsed": "false"}
# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# Path to the parquet dataset files
data_path = 'data/eICU/cleaned/'
# Path to the code files
project_path = 'code/eICU-mortality-prediction/'

# + {"Collapsed": "false"}
import modin.pandas as pd                  # Optimized distributed version of Pandas
import data_utils as du                    # Data science and machine learning relevant methods
import Models                              # Machine learning models

# + [markdown] {"Collapsed": "false"}
# Allow pandas to show more columns:

# + {"Collapsed": "false"}
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

# + [markdown] {"Collapsed": "false"}
# Set the random seed for reproducibility:

# + {"Collapsed": "false"}
du.set_random_seed(42)

# + [markdown] {"Collapsed": "false"}
# ## Loading the data

# + {"Collapsed": "false"}
eICU_df = pd.read_csv(f'{data_path}normalized/ohe/eICU.csv')
eICU_df.head()

# + {"Collapsed": "false"}
eICU_df.info()

# + {"Collapsed": "false"}
eICU_df.patientunitstayid.nunique()

# + {"Collapsed": "false"}
len(eICU_df)

# + {"Collapsed": "false"}
eICU_df.dtypes

# + [markdown] {"Collapsed": "false"}
# Convert to a PyTorch tensor:

# + {"Collapsed": "false"}
eICU_tsr = du.padding.dataframe_to_padded_tensor(eICU_df, id_column='patientunitstayid',
                                                 ts_column='ts', padding_value=999999,
                                                 inplace=True)
eICU_tsr

# + [markdown] {"Collapsed": "false"}
# Create a dataset:

# + {"Collapsed": "false"}
ohe_dataset = du.datasets.Time_Series_Dataset(eICU_df, arr=eICU_tsr,
                                              id_column='patientunitstayid',
                                              ts_column='ts')

# + {"Collapsed": "false"}
len(ohe_dataset)

# + {"Collapsed": "false"}
ohe_dataset.label_column

# + {"Collapsed": "false"}
ohe_dataset.y

# + [markdown] {"Collapsed": "false"}
# Get the train, validation and test sets data loaders, which will allow loading batches:

# + {"Collapsed": "false"}
batch_size = 32

# + {"Collapsed": "false"}
train_dataloader, val_dataloader, test_dataloader = du.machine_learning.create_train_sets(ohe_dataset, test_train_ratio=0.2, validation_ratio=0.1,
                                                                                          batch_size=batch_size, get_indeces=False)

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
embed_features_names = [du.data_processing.clean_naming(feat_list, lower_case=False)
                        for feat_list in list(cat_feat_ohe.values())]
ohe_columns = du.utils.merge_lists(embed_features_names)

# + [markdown] {"Collapsed": "false"}
# Add the one hot encoded columns to the dtypes dictionary, specifying them with type `UInt8`

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-20T03:50:14.365745Z", "iopub.status.busy": "2020-03-20T03:50:14.365436Z", "iopub.status.idle": "2020-03-20T03:50:14.370297Z", "shell.execute_reply": "2020-03-20T03:50:14.369485Z", "shell.execute_reply.started": "2020-03-20T03:50:14.365712Z"}}
for col in ohe_columns:
    dtype_dict[col] = 'UInt8'

# + {"Collapsed": "false", "execution": {"iopub.execute_input": "2020-03-19T15:42:28.695715Z", "iopub.status.busy": "2020-03-19T15:42:28.695505Z", "iopub.status.idle": "2020-03-19T15:42:28.729242Z", "shell.execute_reply": "2020-03-19T15:42:28.727839Z", "shell.execute_reply.started": "2020-03-19T15:42:28.695689Z"}}
dtype_dict

# + [markdown] {"Collapsed": "false"}
# ## Training models

# + [markdown] {"Collapsed": "false"}
# Training hyperparameters:

# + {"Collapsed": "false"}
n_epochs = 20                                   # Number of epochs
lr = 0.001                                      # Learning rate

# + [markdown] {"Collapsed": "false"}
# ### Models with one hot encoded features

# + [markdown] {"Collapsed": "false"}
# ### Models with embedding layers

# + {"Collapsed": "false"}
# Subtracting 1 because of the removed label column, which was before these columns
embed_features = [[du.search_explore.find_col_idx(eICU_df, col)-2 for col in feat_list]
                   for feat_list in embed_features_names]
embed_features

# + {"Collapsed": "false"}
n_embeddings = []
[n_embeddings.append(len(feat_list) + 1) for feat_list in embed_features]
n_embeddings

# + {"Collapsed": "false"}
# [TODO] Join the columns that in fact belong to the same concept, such as the drughiclseqno

# + [markdown] {"Collapsed": "false"}
# #### LSTM with embedding layer

# + [markdown] {"Collapsed": "false"}
# #### Normal training

# + [markdown] {"Collapsed": "false"}
# Model hyperparameters:

# + {"Collapsed": "false"}
n_inputs = len(eICU_df.columns)               # Number of input features
n_hidden = 100                                # Number of hidden units
n_outputs = 1                                 # Number of outputs
n_layers = 2                                  # Number of MLP layers
p_dropout = 0.2                               # Probability of dropout
embedding_dim = [3, 3]                        # Embedding dimensions for each categorical feature

# + {"Collapsed": "false"}
eICU_tsr[:, embed_features]

# + [markdown] {"Collapsed": "false"}
# Initializing the model:

# + {"Collapsed": "false"}
model = Models.VanillaLSTM(n_inputs-3, n_hidden, n_outputs, n_layers, p_dropout,
                           embed_features, n_embeddings, embedding_dim)
model

# + [markdown] {"Collapsed": "false"}
# Training and testing:

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false"}
model, val_loss_min = du.machine_learning.train(model, train_dataloader, val_dataloader, cols_to_remove=0,
                                                model_type='mlp', batch_size=batch_size, n_epochs=n_epochs,
                                                lr=lr, models_path=f'{models_path}mlp/',
                                                ModelClass=Models.MLP, do_test=True, log_comet_ml=True,
                                                comet_ml_api_key='jiDa6SsGNoyddaLPZESuAO6qi',
                                                comet_ml_project_name='tcga-tumor-classification',
                                                comet_ml_workspace='andrecnf',
                                                comet_ml_save_model=True, features_list=list(eICU_df.columns),
                                                get_val_loss_min=True)
print(f'Minimium validation loss: {val_loss_min}')
# + [markdown] {"Collapsed": "false"}
# #### Hyperparameter optimization

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false"}
# # %%pixie_debugger
du.machine_learning.optimize_hyperparameters(Models.MLP, du.datasets.Tabular_Dataset, eICU_df,
                                             config_name='tcga_hyperparameter_optimization_config.yaml',
                                             comet_ml_api_key='jiDa6SsGNoyddaLPZESuAO6qi',
                                             comet_ml_project_name='tcga-tumor-classification',
                                             comet_ml_workspace='andrecnf',
                                             n_inputs=len(eICU_df.columns)-2,
                                             id_column=0, label_column=du.search_explore.find_col_idx(eICU_df, feature='tumor_type_label'),
                                             n_outputs=1, model_type='mlp', models_path='models/',
                                             ModelClass=None, array_param=None,
                                             config_path=hyper_opt_config_path, var_seq=False, clip_value=0.5,
                                             batch_size=32, n_epochs=20, lr=0.001,
                                             test_train_ratio=0.2, validation_ratio=0.1,
                                             comet_ml_save_model=True)

# + [markdown] {"Collapsed": "false"}
# #### XGBoost

# + [markdown] {"Collapsed": "false"}
# #### Adapting the data to XGBoost and Scikit-Learn

# + [markdown] {"Collapsed": "false"}
# Make a copy of the dataframe:

# + {"Collapsed": "false"}
sckt_eICU_df = eICU_df.copy()
sckt_eICU_df

# + [markdown] {"Collapsed": "false"}
# Convert categorical columns to string type:

# + {"Collapsed": "false"}
sckt_eICU_df.race = sckt_eICU_df.race.astype(str)
sckt_eICU_df.ajcc_pathologic_tumor_stage = sckt_eICU_df.ajcc_pathologic_tumor_stage.astype(str)

# + [markdown] {"Collapsed": "false"}
# One hot encode categorical features:

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false"}
sckt_eICU_df, new_cols= du.data_processing.one_hot_encoding_dataframe(sckt_eICU_df, columns=['race', 'ajcc_pathologic_tumor_stage'],
                                                                      clean_name=False, clean_missing_values=False,
                                                                      has_nan=False, join_rows=False,
                                                                      get_new_column_names=True, inplace=True)
new_cols

# + {"Collapsed": "false"}
sckt_eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the ID column:

# + {"Collapsed": "false"}
sckt_eICU_df = sckt_eICU_df.drop(columns='sample_id')
sckt_eICU_df.head()

# + [markdown] {"Collapsed": "false"}
# Convert to a PyTorch tensor:

# + {"Collapsed": "false"}
sckt_eICU_tsr = torch.from_numpy(sckt_eICU_df.to_numpy())
sckt_eICU_tsr

# + [markdown] {"Collapsed": "false"}
# Create a dataset:

# + {"Collapsed": "false"}
dataset = du.datasets.Tabular_Dataset(sckt_eICU_tsr, sckt_eICU_df)

# + {"Collapsed": "false"}
len(dataset)

# + {"Collapsed": "false"}
dataset.label_column

# + {"Collapsed": "false"}
dataset.y

# + [markdown] {"Collapsed": "false"}
# Get the train, validation and test sets data loaders, which will allow loading batches:

# + {"Collapsed": "false"}
train_dataloader, val_dataloader, test_dataloader = du.machine_learning.create_train_sets(dataset, test_train_ratio=0.2, validation_ratio=0.1,
                                                                                          batch_size=len(dataset), get_indeces=False)

# + [markdown] {"Collapsed": "false"}
# Get the full tensors with all the data from each set:

# + {"Collapsed": "false"}
train_features, train_labels = next(iter(train_dataloader))
val_features, val_labels = next(iter(val_dataloader))
test_features, test_labels = next(iter(test_dataloader))

# + {"Collapsed": "false"}
val_features

# + {"Collapsed": "false"}
len(train_features)

# + [markdown] {"Collapsed": "false"}
# #### Normal training

# + [markdown] {"Collapsed": "false"}
# Model hyperparameters:

# + {"Collapsed": "false"}
n_class = eICU_df.tumor_type_label.nunique()    # Number of classes
lr = 0.001                                      # Learning rate
objective = 'multi:softmax'                     # Objective function to minimize (in this case, softmax)
eval_metric = 'mlogloss'                        # Metric to analyze (in this case, multioutput negative log likelihood loss)

# + [markdown] {"Collapsed": "false"}
# Initializing the model:

# + {"Collapsed": "false"}
xgb_model = xgb.XGBClassifier(objective=objective, eval_metric='mlogloss', learning_rate=lr,
                              num_class=n_class, random_state=du.random_seed, seed=du.random_seed)
xgb_model

# + [markdown] {"Collapsed": "false"}
# Training with early stopping (stops training if the evaluation metric doesn't improve on 5 consequetive iterations):

# + {"Collapsed": "false"}
xgb_model.fit(train_features, train_labels, early_stopping_rounds=5, eval_set=[(val_features, val_labels)])

# + [markdown] {"Collapsed": "false"}
# Save the model:

# + {"Collapsed": "false"}
# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
model_filename = f'{models_path}xgb/checkpoint_{current_datetime}.model'
# Save the model
joblib.dump(xgb_model, model_filename)

# + {"Collapsed": "false"}
# xgb_model = joblib.load(f'{models_path}xgb/checkpoint_16_12_2019_11_39.model')
xgb_model = joblib.load(model_filename)
xgb_model

# + [markdown] {"Collapsed": "false"}
# Train until the best iteration:

# + {"Collapsed": "false"}
xgb_model = xgb.XGBClassifier(objective=objective, eval_metric='mlogloss', learning_rate=lr,
                              num_class=n_class, random_state=du.random_seed, seed=du.random_seed)
xgb_model

# + {"Collapsed": "false"}
xgb_model.fit(train_features, train_labels, early_stopping_rounds=5, num_boost_round=xgb_model.best_iteration)

# + [markdown] {"Collapsed": "false"}
# Evaluate on the test set:

# + {"Collapsed": "false"}
pred = xgb_model.predict(test_features)

# + {"Collapsed": "false"}
acc = accuracy_score(test_labels, pred)
acc

# + {"Collapsed": "false"}
f1 = f1_score(test_labels, pred, average='weighted')
f1

# + {"Collapsed": "false"}
pred_proba = xgb_model.predict_proba(test_features)

# + {"Collapsed": "false"}
loss = log_loss(test_labels, pred_proba)
loss

# + {"Collapsed": "false"}
auc = roc_auc_score(test_labels, pred_proba, multi_class='ovr', average='weighted')
auc

# + [markdown] {"Collapsed": "false"}
# #### Hyperparameter optimization
# + {"Collapsed": "false"}



# + [markdown] {"Collapsed": "false"}
# #### Logistic Regression

# + [markdown] {"Collapsed": "false"}
# #### Normal training

# + [markdown] {"Collapsed": "false"}
# Model hyperparameters:

# + {"Collapsed": "false"}
multi_class = 'multinomial'
solver = 'lbfgs'
penalty = 'l2'
C = 1
max_iter = 1000

# + [markdown] {"Collapsed": "false"}
# Initializing the model:

# + {"Collapsed": "false"}
logreg_model = LogisticRegression(multi_class=multi_class, solver=solver, penalty=penalty, C=C, max_iter=max_iter, random_state=du.random_seed)
logreg_model

# + [markdown] {"Collapsed": "false"}
# Training and testing:

# + {"Collapsed": "false"}
logreg_model.fit(train_features, train_labels)

# + [markdown] {"Collapsed": "false"}
# Save the model:

# + {"Collapsed": "false"}
# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
model_filename = f'{models_path}logreg/checkpoint_{current_datetime}.model'
# Save the model
joblib.dump(logreg_model, model_filename)

# + {"Collapsed": "false"}
# logreg_model = joblib.load(f'{models_path}logreg/checkpoint_16_12_2019_02_27.model')
logreg_model = joblib.load(model_filename)
logreg_model

# + [markdown] {"Collapsed": "false"}
# Evaluate on the test set:

# + {"Collapsed": "false"}
acc = logreg_model.score(test_features, test_labels)
acc

# + {"Collapsed": "false"}
pred = logreg_model.predict(test_features)

# + {"Collapsed": "false"}
f1 = f1_score(test_labels, pred, average='weighted')
f1

# + {"Collapsed": "false"}
pred_proba = logreg_model.predict_proba(test_features)

# + {"Collapsed": "false"}
loss = log_loss(test_labels, pred_proba)
loss

# + {"Collapsed": "false"}
auc = roc_auc_score(test_labels, pred_proba, multi_class='ovr', average='weighted')
auc

# + [markdown] {"Collapsed": "false"}
# #### Hyperparameter optimization
# + {"Collapsed": "false"}



# + [markdown] {"Collapsed": "false"}
# #### SVM

# + [markdown] {"Collapsed": "false"}
# #### Normal training

# + [markdown] {"Collapsed": "false"}
# Model hyperparameters:

# + {"Collapsed": "false"}
decision_function_shape = 'ovo'
C = 1
kernel = 'rbf'
max_iter = 100

# + [markdown] {"Collapsed": "false"}
# Initializing the model:

# + {"Collapsed": "false"}
svm_model = SVC(kernel=kernel, decision_function_shape=decision_function_shape, C=C,
                max_iter=max_iter, probability=True, random_state=du.random_seed)
svm_model

# + [markdown] {"Collapsed": "false"}
# Training and testing:

# + {"Collapsed": "false"}
svm_model.fit(train_features, train_labels)

# + [markdown] {"Collapsed": "false"}
# Save the model:

# + {"Collapsed": "false"}
# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
model_filename = f'{models_path}svm/checkpoint_{current_datetime}.model'
# Save the model
joblib.dump(svm_model, model_filename)

# + {"Collapsed": "false"}
# svm_model = joblib.load(f'{models_path}svm/checkpoint_16_12_2019_05_51.model')
svm_model = joblib.load(model_filename)
svm_model

# + [markdown] {"Collapsed": "false"}
# Evaluate on the test set:

# + {"Collapsed": "false"}
acc = logreg_model.score(test_features, test_labels)
acc

# + {"Collapsed": "false"}
pred = logreg_model.predict(test_features)

# + {"Collapsed": "false"}
f1 = f1_score(test_labels, pred, average='weighted')
f1

# + {"Collapsed": "false"}
pred_proba = logreg_model.predict_proba(test_features)

# + {"Collapsed": "false"}
loss = log_loss(test_labels, pred_proba)
loss

# + {"Collapsed": "false"}
auc = roc_auc_score(test_labels, pred_proba, multi_class='ovr', average='weighted')
auc

# + [markdown] {"Collapsed": "false"}
# #### Hyperparameter optimization
# + {"Collapsed": "false"}
