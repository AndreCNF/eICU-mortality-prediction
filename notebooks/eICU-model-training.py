# -*- coding: utf-8 -*-
# # eICU Model training
# ---
#
# Training models on the preprocessed the eICU dataset from MIT, which has data from over 139k patients collected in the US.

# ## Importing the necessary packages

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

# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# Path to the parquet dataset files
data_path = 'data/eICU/cleaned/'
# Path to the code files
project_path = 'code/eICU-mortality-prediction/'

# Change to the scripts directory
os.chdir("../../scripts/")
import Models                              # Machine learning models
# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# import modin.pandas as pd                  # Optimized distributed version of Pandas
import pandas as pd                        # Pandas to load and handle the data
import data_utils as du                    # Data science and machine learning relevant methods

du.set_pandas_library(lib='pandas')

# Allow pandas to show more columns:

pd.set_option('display.max_columns', 3000)
pd.set_option('display.max_rows', 3000)

# Set the random seed for reproducibility:

du.set_random_seed(42)

# ## Initializing variables

# Data types:

stream_dtypes = open(f'{data_path}eICU_dtype_dict.yml', 'r')

dtype_dict = yaml.load(stream_dtypes, Loader=yaml.FullLoader)
dtype_dict

# Dataset parameters:

time_window_h = 24                         # Number of hours on which we want to predict mortality
id_column = 'patientunitstayid'            # Name of the sequence ID column
ts_column = 'ts'                           # Name of the timestamp column
n_inputs = 2093                            # Number of input features
n_outputs = 1                              # Number of outputs
padding_value = 999999                     # Padding value used to fill in sequences up to the maximum sequence length

# Training parameters:

test_train_ratio = 0.25                    # Percentage of the data which will be used as a test set
validation_ratio = 0.1                     # Percentage of the data from the training set which is used for validation purposes
batch_size = 32                            # Number of unit stays in a mini batch
n_epochs = 10                              # Number of epochs
lr = 0.001                                 # Learning rate

# ## Loading the data

eICU_df = du.data_processing.load_chunked_data(file_name='eICU', n_chunks=8,
                                               data_path=f'{data_path}normalized/ohe/', dtypes=dtype_dict)
eICU_df.head()

# Number of unit stays:

eICU_df[id_column].nunique()

# Number of rows:

len(eICU_df)

eICU_df.dtypes

if len(eICU_df.columns) != n_inputs:
    n_inputs = len(eICU_df.columns)
    print(f'Changed the number of inputs to {n_inputs}')

# +
# eICU_df.info()
# -

# ## Setting the label
#
# Define the label column considering the desired time window on which we want to predict mortality (0, 24h, 48h, 72h, etc).

eICU_df['label'] = eICU_df.death_ts - eICU_df.ts <= time_window_h * 60
eICU_df.head()

# Remove the now unneeded `death_ts` column:

eICU_df.drop(columns=death_ts, inplace=True)

# ## Preparing the dataset

# ### Padding
#
# Pad the data so that all sequences have the same length (so that it can be converted to a PyTorch tensor).

seq_len_dict = du.padding.get_sequence_length_dict(eICU_df, id_column=id_column, ts_column=ts_column)
seq_len_dict

data = du.padding.dataframe_to_padded_tensor(eICU_df, seq_len_dict=seq_len_dict,
                                             id_column=id_column, padding_value=padding_value)
data

data.shape

data[0]

# ### Dataset object

dataset = du.datasets.Time_Series_Dataset(eICU_df, data)

# ### Separating into train and validation sets
#
# Since this notebook is only for experimentation purposes, with a very small dummy dataset, we'll not be using a test set.

# Get the train and validation sets data loaders, which will allow loading batches
train_dataloader, val_dataloader, test_dataloader = du.machine_learning.create_train_sets(dataset,
                                                                                          test_train_ratio=test_train_ratio,
                                                                                          validation_ratio=validation_ratio,
                                                                                          batch_size=batch_size,
                                                                                          get_indeces=False)

next(iter(train_dataloader))[0]

next(iter(val_dataloader))[0]

next(iter(test_dataloader))[0]

dataset.__len__()

# ## Training models

# ### Models with one hot encoded features




# ### Models with embedding layers

# Subtracting 1 because of the removed label column, which was before these columns
embed_features = [[du.search_explore.find_col_idx(eICU_df, col)-2 for col in feat_list]
                   for feat_list in embed_features_names]
embed_features

n_embeddings = []
[n_embeddings.append(len(feat_list) + 1) for feat_list in embed_features]
n_embeddings

# [TODO] Join the columns that in fact belong to the same concept, such as the drughiclseqno

# #### LSTM with embedding layers

# ##### Creating the model

# Model parameters:

n_ids = dmy_norm_df.subject_id.nunique()      # Total number of sequences
n_inputs = len(dmy_norm_df.columns)           # Number of input features
n_hidden = 10                                 # Number of hidden units
n_outputs = 1                                 # Number of outputs
n_layers = 2                                  # Number of LSTM layers
p_dropout = 0.2                               # Probability of dropout
embed_features = [du.search_explore.find_col_idx(dmy_norm_df, col) for col in ohe_columns] # Indeces fo the features to be emebedded
embed_features.sort()
embedding_dim = 2                             # Number of outputs of the embedding layr

# Instantiating the model:

embed_features

model = Models.VanillaLSTM(n_inputs-3, n_hidden, n_outputs, n_layers, p_dropout,
                           embed_features=embed_features, embedding_dim=embedding_dim)
model

model.n_embeddings

# ##### Training the model

next(model.lstm.parameters())

next(model.embed_layers.parameters())

# +
# model = du.deep_learning.train(model, train_dataloader_df, val_dataloader_df, seq_len_dict=seq_len_dict,
#                                batch_size=batch_size, n_epochs=n_epochs, lr=lr, models_path='models/',
#                                padding_value=padding_value, do_test=False, log_comet_ml=False,
#                                already_embedded=True)
# -

model = du.deep_learning.train(model, train_dataloader, val_dataloader, seq_len_dict=seq_len_dict,
                               batch_size=batch_size, n_epochs=n_epochs, lr=lr, models_path='models/',
                               padding_value=padding_value, do_test=False, log_comet_ml=False,
                               already_embedded=False)

next(model.lstm.parameters())

next(model.embed_layers.parameters())

# ##### Testing the model

output, metrics = du.deep_learning.model_inference(model, dataloader=val_dataloader,
                                                   metrics=['loss', 'accuracy', 'AUC'],
                                                   seq_len_dict=seq_len_dict, padding_value=padding_value,
                                                   output_rounded=False, set_name='test',
                                                   already_embedded=False,
                                                   cols_to_remove=[du.search_explore.find_col_idx(dmy_norm_df, feature)
                                                                   for feature in ['subject_id', 'ts']])
output

metrics

# #### LSTM with embedding layers and time interval handling

# ##### Adding the time difference feature

dmy_df['delta_ts'] = dmy_df.groupby('subject_id').ts.diff()
dmy_df

# ##### Normalizing the features

dmy_df.describe().transpose()

dmy_df.dtypes

dmy_norm_df = du.data_processing.normalize_data(dmy_df, id_columns=['subject_id', 'ts'],
                                                see_progress=False)
dmy_norm_df

dmy_norm_df.describe().transpose()

# ##### Padding
#
# Pad the data so that all sequences have the same length (so that it can be converted to a PyTorch tensor).

padding_value = 999999

seq_len_dict = du.padding.get_sequence_length_dict(dmy_norm_df, id_column='subject_id', ts_column='ts')
seq_len_dict

data = du.padding.dataframe_to_padded_tensor(dmy_norm_df, seq_len_dict=seq_len_dict,
                                             id_column='subject_id', padding_value=padding_value)
data

# ##### Dataset object

dataset = du.datasets.Time_Series_Dataset(dmy_norm_df, data)

# ##### Separating into train and validation sets
#
# Since this notebook is only for experimentation purposes, with a very small dummy dataset, we'll not be using a test set.

# Get the train and validation sets data loaders, which will allow loading batches
train_dataloader, val_dataloader, test_dataloader = du.machine_learning.create_train_sets(dataset,
                                                                                          test_train_ratio=test_train_ratio,
                                                                                          validation_ratio=validation_ratio,
                                                                                          batch_size=batch_size,
                                                                                          get_indeces=False)

next(iter(train_dataloader))[0]

next(iter(val_dataloader))[0]

next(iter(test_dataloader))[0]

dataset.__len__()

# ##### Creating the model

# Model parameters:

n_ids = dmy_norm_df.subject_id.nunique()      # Total number of sequences
n_inputs = len(dmy_norm_df.columns)           # Number of input features
n_hidden = 10                                 # Number of hidden units
n_outputs = 1                                 # Number of outputs
n_layers = 2                                  # Number of LSTM layers
p_dropout = 0.2                               # Probability of dropout
embed_features = [du.search_explore.find_col_idx(dmy_norm_df, col) for col in ohe_columns] # Indeces fo the features to be emebedded
embed_features.sort()
embedding_dim = 2                             # Number of outputs of the embedding layer

# Instantiating the model:

model = Models.VanillaLSTM(n_inputs-3, n_hidden, n_outputs, n_layers, p_dropout,
                           embed_features=embed_features, embedding_dim=embedding_dim)
model

# ##### Training the model

next(model.parameters())

next(model.embed_layers.parameters())

model = du.deep_learning.train(model, train_dataloader, val_dataloader, seq_len_dict=seq_len_dict,
                               batch_size=batch_size, n_epochs=n_epochs, lr=lr, models_path='models/',
                               padding_value=padding_value, do_test=False, log_comet_ml=False)

next(model.parameters())

next(model.embed_layers.parameters())

# ##### Testing the model

output, metrics = du.deep_learning.model_inference(model, dataloader=val_dataloader,
                                                   metrics=['loss', 'accuracy', 'AUC'],
                                                   seq_len_dict=seq_len_dict, padding_value=padding_value,
                                                   output_rounded=False, set_name='test',
                                                   cols_to_remove=[du.search_explore.find_col_idx(dmy_norm_df, feature)
                                                                   for feature in ['subject_id', 'ts']])
output

metrics

# ##### Hyperparameter optimization
if do_hyperparam_optim:
    val_loss_min, exp_name_min = du.machine_learning.optimize_hyperparameters(Models.TLSTM, df=dmy_norm_df,
                                                                              config_name='TLSTM_hyperparameter_optimization_config.yaml',
                                                                              comet_ml_api_key='jiDa6SsGNoyddaLPZESuAO6qi',
                                                                              comet_ml_project_name='models-dummy-tests',
                                                                              comet_ml_workspace='andrecnf',
                                                                              n_inputs=n_inputs-4, id_column='subject_id',
                                                                              label_column='label', inst_column='ts',
                                                                              n_outputs=1, model_type='multivariate_rnn',
                                                                              is_custom=True, models_path='models/', array_param=None,
                                                                              config_path='notebooks/sandbox/', var_seq=True,
                                                                              clip_value=0.5, padding_value=padding_value,
                                                                              batch_size=batch_size, n_epochs=n_epochs,
                                                                              lr=lr, test_train_ratio=0, validation_ratio=0.25,
                                                                              comet_ml_save_model=True, embed_features=embed_features)

if do_hyperparam_optim:
    exp_name_min

# #### T-LSTM
#
# Implementation of the [_Patient Subtyping via Time-Aware LSTM Networks_](http://biometrics.cse.msu.edu/Publications/MachineLearning/Baytasetal_PatientSubtypingViaTimeAwareLSTMNetworks.pdf) paper.

# ##### Creating the model

# Model parameters:

n_ids = dmy_norm_df.subject_id.nunique()      # Total number of sequences
n_inputs = len(dmy_norm_df.columns)           # Number of input features
n_hidden = 10                                 # Number of hidden units
n_outputs = 1                                 # Number of outputs
n_rnn_layers = 4                              # Number of TLSTM layers
p_dropout = 0.2                               # Probability of dropout
embed_features = [du.search_explore.find_col_idx(dmy_norm_df, col) for col in ohe_columns] # Indeces fo the features to be emebedded
embed_features.sort()
embedding_dim = 2                             # Number of outputs of the embedding layr
# delta_ts_col = du.search_explore.find_col_idx(dmy_norm_df, 'delta_ts')   # Number of the delta_ts column
elapsed_time = 'small'                                                   # Indicates if the elapsed time between events is small or long; influences how to discount elapsed time

n_inputs

dmy_norm_df.columns

embed_features

# Instantiating the model:

model = Models.TLSTM(n_inputs-4, n_hidden, n_outputs, n_rnn_layers, p_dropout,
                     embed_features=embed_features, embedding_dim=embedding_dim,
                     elapsed_time=elapsed_time)
model

model.rnn_layers[0].cell.input_size

model.rnn_layers[0].cell.hidden_size

model.rnn_layers[0].cell.weight_ih.shape

model.rnn_layers[0].cell.delta_ts_col

model.rnn_layers[1].cell.delta_ts_col

# ##### Training the model

next(model.parameters())

next(model.embed_layers.parameters())

model = du.deep_learning.train(model, train_dataloader, val_dataloader, seq_len_dict=seq_len_dict,
                               batch_size=batch_size, n_epochs=n_epochs, lr=lr, models_path='models/',
                               padding_value=padding_value, do_test=False, log_comet_ml=False,
                               is_custom=True)

next(model.parameters())

next(model.embed_layers.parameters())

# ##### Testing the model

output, metrics = du.deep_learning.model_inference(model, dataloader=val_dataloader,
                                                   metrics=['loss', 'accuracy', 'AUC'],
                                                   seq_len_dict=seq_len_dict, padding_value=padding_value,
                                                   output_rounded=False, set_name='test',
                                                   is_custom=True,
                                                   cols_to_remove=[du.search_explore.find_col_idx(dmy_norm_df, feature)
                                                                   for feature in ['subject_id', 'ts']])
output

metrics

# ##### Hyperparameter optimization
if do_hyperparam_optim:
    val_loss_min, exp_name_min = du.machine_learning.optimize_hyperparameters(Models.TLSTM, df=dmy_norm_df,
                                                                              config_name='TLSTM_hyperparameter_optimization_config.yaml',
                                                                              comet_ml_api_key='jiDa6SsGNoyddaLPZESuAO6qi',
                                                                              comet_ml_project_name='models-dummy-tests',
                                                                              comet_ml_workspace='andrecnf',
                                                                              n_inputs=n_inputs-4, id_column='subject_id',
                                                                              label_column='label', inst_column='ts',
                                                                              n_outputs=1, model_type='multivariate_rnn',
                                                                              is_custom=True, models_path='models/', array_param=None,
                                                                              config_path='notebooks/sandbox/', var_seq=True,
                                                                              clip_value=0.5, padding_value=padding_value,
                                                                              batch_size=batch_size, n_epochs=n_epochs,
                                                                              lr=lr, test_train_ratio=0, validation_ratio=0.25,
                                                                              comet_ml_save_model=True, embed_features=embed_features)

if do_hyperparam_optim:
    exp_name_min

# #### MF1-LSTM
#
# Implementation of the [_Predicting healthcare trajectories from medical records: A deep learning approach_](https://doi.org/10.1016/j.jbi.2017.04.001) paper, time decay version.

# ##### Creating the model

# Model parameters:

n_ids = dmy_norm_df.subject_id.nunique()      # Total number of sequences
n_inputs = len(dmy_norm_df.columns)           # Number of input features
n_hidden = 10                                 # Number of hidden units
n_outputs = 1                                 # Number of outputs
n_rnn_layers = 4                              # Number of TLSTM layers
p_dropout = 0.2                               # Probability of dropout
embed_features = [du.search_explore.find_col_idx(dmy_norm_df, col) for col in ohe_columns] # Indeces fo the features to be emebedded
embed_features.sort()
embedding_dim = 2                             # Number of outputs of the embedding layr
# delta_ts_col = du.search_explore.find_col_idx(dmy_norm_df, 'delta_ts')   # Number of the delta_ts column
elapsed_time = 'small'                                                   # Indicates if the elapsed time between events is small or long; influences how to discount elapsed time

n_inputs

dmy_norm_df.columns

embed_features

# Instantiating the model:

model = Models.MF1LSTM(n_inputs-4, n_hidden, n_outputs, n_rnn_layers, p_dropout,
                       embed_features=embed_features, embedding_dim=embedding_dim,
                       elapsed_time=elapsed_time)
model

model.rnn_layers[0].cell.input_size

model.rnn_layers[0].cell.hidden_size

model.rnn_layers[0].cell.weight_ih.shape

model.rnn_layers[0].cell.delta_ts_col

model.rnn_layers[1].cell.delta_ts_col

# ##### Training the model

next(model.parameters())

next(model.embed_layers.parameters())

model = du.deep_learning.train(model, train_dataloader, val_dataloader, seq_len_dict=seq_len_dict,
                               batch_size=batch_size, n_epochs=n_epochs, lr=lr, models_path='models/',
                               ModelClass=Models.MF1LSTM, padding_value=padding_value, do_test=False,
                               log_comet_ml=False, is_custom=True)

next(model.parameters())

next(model.embed_layers.parameters())

# ##### Testing the model

output, metrics = du.deep_learning.model_inference(model, dataloader=val_dataloader,
                                                   metrics=['loss', 'accuracy', 'AUC'],
                                                   seq_len_dict=seq_len_dict, padding_value=padding_value,
                                                   output_rounded=False, set_name='test',
                                                   is_custom=True,
                                                   cols_to_remove=[du.search_explore.find_col_idx(dmy_norm_df, feature)
                                                                   for feature in ['subject_id', 'ts']])
output

metrics

# ##### Hyperparameter optimization
if do_hyperparam_optim:
    val_loss_min, exp_name_min = du.machine_learning.optimize_hyperparameters(Models.MF1LSTM, df=dmy_norm_df,
                                                                              config_name='TLSTM_hyperparameter_optimization_config.yaml',
                                                                              comet_ml_api_key='jiDa6SsGNoyddaLPZESuAO6qi',
                                                                              comet_ml_project_name='models-dummy-tests',
                                                                              comet_ml_workspace='andrecnf',
                                                                              n_inputs=n_inputs-4, id_column='subject_id',
                                                                              label_column='label', inst_column='ts',
                                                                              n_outputs=1, model_type='multivariate_rnn',
                                                                              is_custom=True, models_path='models/', array_param=None,
                                                                              config_path='notebooks/sandbox/', var_seq=True,
                                                                              clip_value=0.5, padding_value=padding_value,
                                                                              batch_size=batch_size, n_epochs=n_epochs,
                                                                              lr=lr, test_train_ratio=0, validation_ratio=0.25,
                                                                              comet_ml_save_model=True, embed_features=embed_features)

if do_hyperparam_optim:
    exp_name_min

# #### MF2-LSTM
#
# Implementation of the [_Predicting healthcare trajectories from medical records: A deep learning approach_](https://doi.org/10.1016/j.jbi.2017.04.001) paper, parametric time version.

# ##### Creating the model

# Model parameters:

n_ids = dmy_norm_df.subject_id.nunique()      # Total number of sequences
n_inputs = len(dmy_norm_df.columns)           # Number of input features
n_hidden = 10                                 # Number of hidden units
n_outputs = 1                                 # Number of outputs
n_rnn_layers = 4                              # Number of TLSTM layers
p_dropout = 0.2                               # Probability of dropout
embed_features = [du.search_explore.find_col_idx(dmy_norm_df, col) for col in ohe_columns] # Indeces fo the features to be emebedded
embed_features.sort()
embedding_dim = 2                             # Number of outputs of the embedding layr
# delta_ts_col = du.search_explore.find_col_idx(dmy_norm_df, 'delta_ts')   # Number of the delta_ts column
elapsed_time = 'small'                                                   # Indicates if the elapsed time between events is small or long; influences how to discount elapsed time

n_inputs

dmy_norm_df.columns

embed_features

# Instantiating the model:

model = Models.MF2LSTM(n_inputs-4, n_hidden, n_outputs, n_rnn_layers, p_dropout,
                       embed_features=embed_features, embedding_dim=embedding_dim,
                       elapsed_time=elapsed_time)
model

model.rnn_layers[0].cell.input_size

model.rnn_layers[0].cell.hidden_size

model.rnn_layers[0].cell.weight_ih.shape

model.rnn_layers[0].cell.delta_ts_col

model.rnn_layers[1].cell.delta_ts_col

# ##### Training the model

next(model.parameters())

next(model.embed_layers.parameters())

model = du.deep_learning.train(model, train_dataloader, val_dataloader, seq_len_dict=seq_len_dict,
                               batch_size=batch_size, n_epochs=n_epochs, lr=lr, models_path='models/',
                               ModelClass=Models.MF2LSTM, padding_value=padding_value, do_test=False,
                               log_comet_ml=False, is_custom=True)

next(model.parameters())

next(model.embed_layers.parameters())

# ##### Testing the model

output, metrics = du.deep_learning.model_inference(model, dataloader=val_dataloader,
                                                   metrics=['loss', 'accuracy', 'AUC'],
                                                   seq_len_dict=seq_len_dict, padding_value=padding_value,
                                                   output_rounded=False, set_name='test',
                                                   is_custom=True,
                                                   cols_to_remove=[du.search_explore.find_col_idx(dmy_norm_df, feature)
                                                                   for feature in ['subject_id', 'ts']])
output

metrics

# ##### Hyperparameter optimization
if do_hyperparam_optim:
    val_loss_min, exp_name_min = du.machine_learning.optimize_hyperparameters(Models.MF2LSTM, df=dmy_norm_df,
                                                                              config_name='TLSTM_hyperparameter_optimization_config.yaml',
                                                                              comet_ml_api_key='jiDa6SsGNoyddaLPZESuAO6qi',
                                                                              comet_ml_project_name='models-dummy-tests',
                                                                              comet_ml_workspace='andrecnf',
                                                                              n_inputs=n_inputs-4, id_column='subject_id',
                                                                              label_column='label', inst_column='ts',
                                                                              n_outputs=1, model_type='multivariate_rnn',
                                                                              is_custom=True, models_path='models/', array_param=None,
                                                                              config_path='notebooks/sandbox/', var_seq=True,
                                                                              clip_value=0.5, padding_value=padding_value,
                                                                              batch_size=batch_size, n_epochs=n_epochs,
                                                                              lr=lr, test_train_ratio=0, validation_ratio=0.25,
                                                                              comet_ml_save_model=True, embed_features=embed_features)

if do_hyperparam_optim:
    exp_name_min

# #### Adapting the data to XGBoost and Scikit-Learn

# Make a copy of the dataframe:

sckt_eICU_df = eICU_df.copy()
sckt_eICU_df

# Convert categorical columns to string type:

sckt_eICU_df.race = sckt_eICU_df.race.astype(str)
sckt_eICU_df.ajcc_pathologic_tumor_stage = sckt_eICU_df.ajcc_pathologic_tumor_stage.astype(str)

# One hot encode categorical features:

sckt_eICU_df, new_cols= du.data_processing.one_hot_encoding_dataframe(sckt_eICU_df, columns=['race', 'ajcc_pathologic_tumor_stage'],
                                                                      clean_name=False, clean_missing_values=False,
                                                                      has_nan=False, join_rows=False,
                                                                      get_new_column_names=True, inplace=True)
new_cols

sckt_eICU_df.head()

# Remove the ID column:

sckt_eICU_df = sckt_eICU_df.drop(columns='sample_id')
sckt_eICU_df.head()

# Convert to a PyTorch tensor:

sckt_eICU_tsr = torch.from_numpy(sckt_eICU_df.to_numpy())
sckt_eICU_tsr

# Create a dataset:

dataset = du.datasets.Tabular_Dataset(sckt_eICU_tsr, sckt_eICU_df)

len(dataset)

dataset.label_column

dataset.y

# Get the train, validation and test sets data loaders, which will allow loading batches:

train_dataloader, val_dataloader, test_dataloader = du.machine_learning.create_train_sets(dataset, test_train_ratio=0.2, validation_ratio=0.1,
                                                                                          batch_size=len(dataset), get_indeces=False)

# Get the full tensors with all the data from each set:

train_features, train_labels = next(iter(train_dataloader))
val_features, val_labels = next(iter(val_dataloader))
test_features, test_labels = next(iter(test_dataloader))

val_features

len(train_features)

# #### XGBoost

# ##### Normal training

# Model hyperparameters:

n_class = eICU_df.tumor_type_label.nunique()    # Number of classes
lr = 0.001                                      # Learning rate
objective = 'multi:softmax'                     # Objective function to minimize (in this case, softmax)
eval_metric = 'mlogloss'                        # Metric to analyze (in this case, multioutput negative log likelihood loss)

# Initializing the model:

xgb_model = xgb.XGBClassifier(objective=objective, eval_metric='mlogloss', learning_rate=lr,
                              num_class=n_class, random_state=du.random_seed, seed=du.random_seed)
xgb_model

# Training with early stopping (stops training if the evaluation metric doesn't improve on 5 consequetive iterations):

xgb_model.fit(train_features, train_labels, early_stopping_rounds=5, eval_set=[(val_features, val_labels)])

# Save the model:

# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
model_filename = f'{models_path}xgb/checkpoint_{current_datetime}.model'
# Save the model
joblib.dump(xgb_model, model_filename)

# xgb_model = joblib.load(f'{models_path}xgb/checkpoint_16_12_2019_11_39.model')
xgb_model = joblib.load(model_filename)
xgb_model

# Train until the best iteration:

xgb_model = xgb.XGBClassifier(objective=objective, eval_metric='mlogloss', learning_rate=lr,
                              num_class=n_class, random_state=du.random_seed, seed=du.random_seed)
xgb_model

xgb_model.fit(train_features, train_labels, early_stopping_rounds=5, num_boost_round=xgb_model.best_iteration)

# Evaluate on the test set:

pred = xgb_model.predict(test_features)

acc = accuracy_score(test_labels, pred)
acc

f1 = f1_score(test_labels, pred, average='weighted')
f1

pred_proba = xgb_model.predict_proba(test_features)

loss = log_loss(test_labels, pred_proba)
loss

auc = roc_auc_score(test_labels, pred_proba, multi_class='ovr', average='weighted')
auc

# ##### Hyperparameter optimization




# #### Logistic Regression

# ##### Normal training

# Model hyperparameters:

multi_class = 'multinomial'
solver = 'lbfgs'
penalty = 'l2'
C = 1
max_iter = 1000

# Initializing the model:

logreg_model = LogisticRegression(multi_class=multi_class, solver=solver, penalty=penalty, C=C, max_iter=max_iter, random_state=du.random_seed)
logreg_model

# Training and testing:

logreg_model.fit(train_features, train_labels)

# Save the model:

# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
model_filename = f'{models_path}logreg/checkpoint_{current_datetime}.model'
# Save the model
joblib.dump(logreg_model, model_filename)

# logreg_model = joblib.load(f'{models_path}logreg/checkpoint_16_12_2019_02_27.model')
logreg_model = joblib.load(model_filename)
logreg_model

# Evaluate on the test set:

acc = logreg_model.score(test_features, test_labels)
acc

pred = logreg_model.predict(test_features)

f1 = f1_score(test_labels, pred, average='weighted')
f1

pred_proba = logreg_model.predict_proba(test_features)

loss = log_loss(test_labels, pred_proba)
loss

auc = roc_auc_score(test_labels, pred_proba, multi_class='ovr', average='weighted')
auc

# ##### Hyperparameter optimization



# #### SVM

# ##### Normal training

# Model hyperparameters:

decision_function_shape = 'ovo'
C = 1
kernel = 'rbf'
max_iter = 100

# Initializing the model:

svm_model = SVC(kernel=kernel, decision_function_shape=decision_function_shape, C=C,
                max_iter=max_iter, probability=True, random_state=du.random_seed)
svm_model

# Training and testing:

svm_model.fit(train_features, train_labels)

# Save the model:

# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
model_filename = f'{models_path}svm/checkpoint_{current_datetime}.model'
# Save the model
joblib.dump(svm_model, model_filename)

# svm_model = joblib.load(f'{models_path}svm/checkpoint_16_12_2019_05_51.model')
svm_model = joblib.load(model_filename)
svm_model

# Evaluate on the test set:

acc = logreg_model.score(test_features, test_labels)
acc

pred = logreg_model.predict(test_features)

f1 = f1_score(test_labels, pred, average='weighted')
f1

pred_proba = logreg_model.predict_proba(test_features)

loss = log_loss(test_labels, pred_proba)
loss

auc = roc_auc_score(test_labels, pred_proba, multi_class='ovr', average='weighted')
auc

# ##### Hyperparameter optimization
