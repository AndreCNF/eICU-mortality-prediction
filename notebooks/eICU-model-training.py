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
import yaml                                # Save and load YAML files
import getpass                             # Get password or similar private inputs
from ipywidgets import interact            # Display selectors and sliders

# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# Path to the parquet dataset files
data_path = 'data/eICU/cleaned/'
# Path to the code files
project_path = 'code/eICU-mortality-prediction/'

# Change to the scripts directory
os.chdir("../scripts/")
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

# Comet ML settings:

comet_ml_project_name = input('Comet ML project name:')
comet_ml_workspace = input('Comet ML workspace:')
comet_ml_api_key = getpass.getpass('Comet ML API key')

# Dataset parameters:

dataset_mode = None                        # The mode in which we'll use the data, either one hot encoded or pre-embedded
ml_core = None                             # The core machine learning type we'll use; either traditional ML or DL
use_delta_ts = None                        # Indicates if we'll use time variation info
time_window_h = None                       # Number of hours on which we want to predict mortality
@interact
def get_dataset_mode(data_mode=['one hot encoded', 'embedded'], ml_or_dl=['machine learning', 'deep learning'],
                     use_delta=[False, 'normalized', 'raw'], window_h=(0, 96, 24)):
    global dataset_mode, ml_core, use_delta_ts, time_window_h
    dataset_mode, ml_core, use_delta_ts, time_window_h = data_mode, ml_or_dl, use_delta, window_h
    already_embedded = dataset_mode == 'embedded'


id_column = 'patientunitstayid'            # Name of the sequence ID column
ts_column = 'ts'                           # Name of the timestamp column
label_column = 'label'                     # Name of the label column
n_ids = 7000                               # Total number of sequences
n_inputs = 2090                            # Number of input features
n_outputs = 1                              # Number of outputs
padding_value = 999999                     # Padding value used to fill in sequences up to the maximum sequence length

# Data types:

stream_dtypes = open(f'{data_path}eICU_dtype_dict.yml', 'r')

dtype_dict = yaml.load(stream_dtypes, Loader=yaml.FullLoader)
dtype_dict

# One hot encoding columns categorization:

stream_cat_feat_ohe = open(f'{data_path}eICU_cat_feat_ohe.yml', 'r')

cat_feat_ohe = yaml.load(stream_cat_feat_ohe, Loader=yaml.FullLoader)
cat_feat_ohe

list(cat_feat_ohe.keys())

# Training parameters:

test_train_ratio = 0.25                    # Percentage of the data which will be used as a test set
validation_ratio = 0.1                     # Percentage of the data from the training set which is used for validation purposes
batch_size = 32                            # Number of unit stays in a mini batch
n_epochs = 10                              # Number of epochs
lr = 0.001                                 # Learning rate

# Testing parameters:

metrics = ['loss', 'accuracy', 'AUC', 'AUC_weighted']

# ## Loading the data

eICU_df = du.data_processing.load_chunked_data(file_name='eICU', n_chunks=8,
                                               data_path=f'{data_path}normalized/ohe/', dtypes=dtype_dict)
eICU_df.head()

# Number of unit stays:

eICU_df[id_column].nunique()

# Number of rows:

len(eICU_df)

eICU_df.dtypes

if eICU_df[id_column].nunique() != n_ids:
    n_ids = eICU_df[id_column].nunique()
    print(f'Changed the number of IDs to {n_ids}')

# +
# eICU_df.info()
# -

# ## Setting the label
#
# Define the label column considering the desired time window on which we want to predict mortality (0, 24h, 48h, 72h, etc).

eICU_df['label'] = eICU_df.death_ts - eICU_df.ts <= time_window_h * 60
eICU_df.head()

# Remove the now unneeded `death_ts` column:

eICU_df.drop(columns='death_ts', inplace=True)

# ## Preparing the dataset

# ### Embedding of the categorical features

if dataset_mode == 'embedded':
    # [TODO] Add code to pre-embed the categorical features
    # Don't train any new embedding layer
    embed_features = None
    n_embeddings = None
else:
    # Find the indeces of the features that will be embedded,
    # as well as the total number of categories per categorical feature
    # Subtracting 2 because of the ID and ts columns
    embed_features = [[du.search_explore.find_col_idx(eICU_df, col)-2 for col in feat_list]
                       for feat_list in cat_feat_ohe]
    n_embeddings = list()
    [n_embeddings.append(len(feat_list) + 1) for feat_list in cat_feat_ohe]
    print(f'Embedding features: {embed_features}')
    print(f'Number of embeddings: {n_embeddings}')

# **Note:** We need to discard 3 columns from the number of inputs as the models won't use the ID and ts columns directly and obviously the label isn't part of the inputs.

# Make sure that we discard the ID, timestamp and label columns
if n_inputs != len(eICU_df.columns) - 3:
    n_inputs = len(eICU_df.columns) - 3
    print(f'Changed the number of inputs to {n_inputs}')

# ### Adding a time variation feature

# Create the `delta_ts` features:

if use_delta_ts is not False:
    eICU['delta_ts'] = eICU.groupby(id_column).ts.diff()
    eICU[[id_column, ts_column, 'delta_ts']].head()

# Normalize `delta_ts`:

# **Note:** When using the MF2-LSTM model, since it assumes that the time variation is in minutes, we shouldn't normalize `delta_ts` with this model.

if use_delta_ts == 'normalized':
    eICU['delta_ts'] = (eICU['delta_ts'] - eICU['delta_ts'].mean()) / eICU['delta_ts'].std()

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

if ml_core == 'deep learning':
    dataset = du.datasets.Time_Series_Dataset(eICU_df, data, label_name=label_column)
else:
    dataset = du.datasets.Tabular_Dataset(eICU_df, data, label_name=label_column)

dataset.__len__()

# ### Separating into train and validation sets

(train_dataloader, val_dataloader, test_dataloader
train_indeces, val_indeces, test_indeces) = du.machine_learning.create_train_sets(dataset,
                                                                                  test_train_ratio=test_train_ratio,
                                                                                  validation_ratio=validation_ratio,
                                                                                  batch_size=batch_size,
                                                                                  get_indeces=True)

if ml_core == 'deep learning':
    # Ignore the indeces, we only care about the dataloaders when using neural networks
    del train_indeces
    del val_indeces
    del test_indeces
else:
    # Get the full arrays of each set
    train_features, train_labels = dataset.X[train_indeces], dataset.y[train_indeces]
    val_features, val_labels = dataset.X[val_indeces], dataset.y[val_indeces]
    test_features, test_labels = dataset.X[test_indeces], dataset.y[test_indeces]
    # Ignore the dataloaders, we only care about the full arrays when using scikit-learn or XGBoost
    del train_dataloaders
    del val_dataloaders
    del test_dataloaders

if ml_core == 'deep learning':
    next(iter(train_dataloader))[0]
else:
    train_features[:32]

if ml_core == 'deep learning':
    next(iter(val_dataloader))[0]
else:
    val_features[:32]

if ml_core == 'deep learning':
    next(iter(test_dataloader))[0]
else:
    test_features[:32]

# ## Training models

# ### Vanilla RNN

# #### Creating the model

# Model parameters:

n_hidden = 100                             # Number of hidden units
n_layers = 2                               # Number of LSTM layers
p_dropout = 0.2                            # Probability of dropout

if use_delta_ts == 'normalized':
    # Count the delta_ts column as another feature, only ignore ID, timestamp and label columns
    n_inputs = len(eICU_df.columns) - 3
elif use_delta_ts == 'raw':
    raise Exception('ERROR: When using a model of type Vanilla RNN, we can\'t use raw delta_ts. Please either normalize it (use_delta_ts = "normalized") or discard it (use_delta_ts = False).')

# Instantiating the model:

model = Models.VanillaRNN(n_inputs, n_hidden, n_outputs, n_layers, p_dropout,
                          embed_features=embed_features, embedding_dim=embedding_dim)
model

# Define the name that will be given to the models that will be saved:

model_name = 'rnn'
if dataset_mode == 'embedded':
    model_name.append('_embedded')
elif dataset_mode == 'one hot encoded' and embed_features is not None:
    model_name.append('_with_embedding')
elif dataset_mode == 'one hot encoded' and embed_features is None:
    model_name.append('_one_hot_encoded')
if use_delta_ts is not False:
    model_name.append('_delta_ts')

# #### Training and testing the model

next(model.parameters())

model = du.deep_learning.train(model, train_dataloader, val_dataloader, test_dataloader, seq_len_dict=seq_len_dict,
                               padding_value=padding_value, batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                               models_path=f'{project_path}models/', model_name=model_name, ModelClass=Models.VanillaRNN,
                               is_custom=False, do_test=True, metrics=metrics, log_comet_ml=True,
                               comet_ml_api_key=comet_ml_api_key, comet_ml_project_name=comet_ml_project_name,
                               comet_ml_workspace=comet_ml_workspace, comet_ml_save_model=True,
                               already_embedded=already_embedded)

next(model.parameters())

# #### Hyperparameter optimization

config_name = input('Hyperparameter optimization configuration file name:')

val_loss_min, exp_name_min = du.machine_learning.optimize_hyperparameters(Models.VanillaRNN, df=dmy_norm_df,
                                                                          config_name=config_name,
                                                                          comet_ml_api_key=comet_ml_api_key,
                                                                          comet_ml_project_name=comet_ml_project_name,
                                                                          comet_ml_workspace=comet_ml_workspace,
                                                                          n_inputs=n_inputs, id_column=id_column,
                                                                          label_column=label_column, inst_column=inst_column,
                                                                          n_outputs=n_outputs, model_type='multivariate_rnn',
                                                                          is_custom=True, models_path='models/',
                                                                          array_param='embedding_dim',
                                                                          config_path=f'{project_path}hyperparameter_optimization/',
                                                                          var_seq=True, clip_value=0.5, padding_value=padding_value,
                                                                          batch_size=batch_size, n_epochs=n_epochs,
                                                                          lr=lr, test_train_ratio=test_train_ratio,
                                                                          validation_ratio=validation_ratio,
                                                                          comet_ml_save_model=True,
                                                                          embed_features=embed_features)

exp_name_min

# ### Vanilla LSTM

# #### Creating the model

# Model parameters:

n_hidden = 100                             # Number of hidden units
n_layers = 2                               # Number of LSTM layers
p_dropout = 0.2                            # Probability of dropout

if use_delta_ts == 'normalized':
    # Count the delta_ts column as another feature, only ignore ID, timestamp and label columns
    n_inputs = len(eICU_df.columns) - 3
elif use_delta_ts == 'raw':
    raise Exception('ERROR: When using a model of type Vanilla LSTM, we can\'t use raw delta_ts. Please either normalize it (use_delta_ts = "normalized") or discard it (use_delta_ts = False).')

# Instantiating the model:

model = Models.VanillaLSTM(n_inputs, n_hidden, n_outputs, n_layers, p_dropout,
                           embed_features=embed_features, embedding_dim=embedding_dim)
model

# Define the name that will be given to the models that will be saved:

model_name = 'lstm'
if dataset_mode == 'embedded':
    model_name.append('_embedded')
elif dataset_mode == 'one hot encoded' and embed_features is not None:
    model_name.append('_with_embedding')
elif dataset_mode == 'one hot encoded' and embed_features is None:
    model_name.append('_one_hot_encoded')
if use_delta_ts is not False:
    model_name.append('_delta_ts')

# #### Training and testing the model

next(model.parameters())

model = du.deep_learning.train(model, train_dataloader, val_dataloader, test_dataloader, seq_len_dict=seq_len_dict,
                               padding_value=padding_value, batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                               models_path=f'{project_path}models/', model_name=model_name, ModelClass=Models.VanillaLSTM,
                               is_custom=False, do_test=True, metrics=metrics, log_comet_ml=True,
                               comet_ml_api_key=comet_ml_api_key, comet_ml_project_name=comet_ml_project_name,
                               comet_ml_workspace=comet_ml_workspace, comet_ml_save_model=True,
                               already_embedded=already_embedded)

next(model.parameters())

# #### Hyperparameter optimization

config_name = input('Hyperparameter optimization configuration file name:')

val_loss_min, exp_name_min = du.machine_learning.optimize_hyperparameters(Models.VanillaLSTM, df=dmy_norm_df,
                                                                          config_name=config_name,
                                                                          comet_ml_api_key=comet_ml_api_key,
                                                                          comet_ml_project_name=comet_ml_project_name,
                                                                          comet_ml_workspace=comet_ml_workspace,
                                                                          n_inputs=n_inputs, id_column=id_column,
                                                                          label_column=label_column, inst_column=inst_column,
                                                                          n_outputs=n_outputs, model_type='multivariate_rnn',
                                                                          is_custom=True, models_path='models/',
                                                                          array_param='embedding_dim',
                                                                          config_path=f'{project_path}hyperparameter_optimization/',
                                                                          var_seq=True, clip_value=0.5, padding_value=padding_value,
                                                                          batch_size=batch_size, n_epochs=n_epochs,
                                                                          lr=lr, test_train_ratio=test_train_ratio,
                                                                          validation_ratio=validation_ratio,
                                                                          comet_ml_save_model=True,
                                                                          embed_features=embed_features)

exp_name_min

# ### T-LSTM
#
# Implementation of the [_Patient Subtyping via Time-Aware LSTM Networks_](http://biometrics.cse.msu.edu/Publications/MachineLearning/Baytasetal_PatientSubtypingViaTimeAwareLSTMNetworks.pdf) paper.

# #### Creating the model

# Model parameters:

n_hidden = 100                             # Number of hidden units
n_rnn_layers = 4                           # Number of TLSTM layers
p_dropout = 0.2                            # Probability of dropout
elapsed_time = 'small'                     # Indicates if the elapsed time between events is small or long; influences how to discount elapsed time

if use_delta_ts == 'raw':
    raise Exception('ERROR: When using a model of type TLSTM, we can\'t use raw delta_ts. Please normalize it (use_delta_ts = "normalized").')
elif use_delta_ts is False:
    raise Exception('ERROR: When using a model of type TLSTM, we must use delta_ts. Please use it, in a normalized version (use_delta_ts = "normalized").')

# Instantiating the model:

model = Models.TLSTM(n_inputs, n_hidden, n_outputs, n_rnn_layers, p_dropout,
                     embed_features=embed_features, embedding_dim=embedding_dim,
                     elapsed_time=elapsed_time)
model

# Define the name that will be given to the models that will be saved:

model_name = 'tlstm'
if dataset_mode == 'embedded':
    model_name.append('_embedded')
elif dataset_mode == 'one hot encoded' and embed_features is not None:
    model_name.append('_with_embedding')
elif dataset_mode == 'one hot encoded' and embed_features is None:
    model_name.append('_one_hot_encoded')

# #### Training and testing the model

next(model.parameters())

model = du.deep_learning.train(model, train_dataloader, val_dataloader, test_dataloader, seq_len_dict=seq_len_dict,
                               padding_value=padding_value, batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                               models_path=f'{project_path}models/', model_name=model_name, ModelClass=Models.TLSTM,
                               is_custom=True, do_test=True, metrics=metrics, log_comet_ml=True,
                               comet_ml_api_key=comet_ml_api_key, comet_ml_project_name=comet_ml_project_name,
                               comet_ml_workspace=comet_ml_workspace, comet_ml_save_model=True,
                               already_embedded=already_embedded)

next(model.parameters())

# #### Hyperparameter optimization

config_name = input('Hyperparameter optimization configuration file name:')

val_loss_min, exp_name_min = du.machine_learning.optimize_hyperparameters(Models.TLSTM, df=dmy_norm_df,
                                                                          config_name=config_name,
                                                                          comet_ml_api_key=comet_ml_api_key,
                                                                          comet_ml_project_name=comet_ml_project_name,
                                                                          comet_ml_workspace=comet_ml_workspace,
                                                                          n_inputs=n_inputs, id_column=id_column,
                                                                          label_column=label_column, inst_column=inst_column,
                                                                          n_outputs=n_outputs, model_type='multivariate_rnn',
                                                                          is_custom=True, models_path='models/',
                                                                          array_param='embedding_dim',
                                                                          config_path=f'{project_path}hyperparameter_optimization/',
                                                                          var_seq=True, clip_value=0.5, padding_value=padding_value,
                                                                          batch_size=batch_size, n_epochs=n_epochs,
                                                                          lr=lr, test_train_ratio=test_train_ratio,
                                                                          validation_ratio=validation_ratio,
                                                                          comet_ml_save_model=True,
                                                                          embed_features=embed_features)

exp_name_min

# ### MF1-LSTM
#
# Implementation of the [_Predicting healthcare trajectories from medical records: A deep learning approach_](https://doi.org/10.1016/j.jbi.2017.04.001) paper, time decay version.

# #### Creating the model

# Model parameters:

n_hidden = 100                             # Number of hidden units
n_rnn_layers = 4                           # Number of MF1-LSTM layers
p_dropout = 0.2                            # Probability of dropout
elapsed_time = 'small'                     # Indicates if the elapsed time between events is small or long; influences how to discount elapsed time

if use_delta_ts == 'raw':
    raise Exception('ERROR: When using a model of type MF1-LSTM, we can\'t use raw delta_ts. Please normalize it (use_delta_ts = "normalized").')
elif use_delta_ts is False:
    raise Exception('ERROR: When using a model of type MF1-LSTM, we must use delta_ts. Please use it, in a normalized version (use_delta_ts = "normalized").')

# Instantiating the model:

model = Models.MF1LSTM(n_inputs, n_hidden, n_outputs, n_rnn_layers, p_dropout,
                       embed_features=embed_features, embedding_dim=embedding_dim,
                       elapsed_time=elapsed_time)
model

# Define the name that will be given to the models that will be saved:

model_name = 'mf1lstm'
if dataset_mode == 'embedded':
    model_name.append('_embedded')
elif dataset_mode == 'one hot encoded' and embed_features is not None:
    model_name.append('_with_embedding')
elif dataset_mode == 'one hot encoded' and embed_features is None:
    model_name.append('_one_hot_encoded')

# #### Training and testing the model

next(model.parameters())

model = du.deep_learning.train(model, train_dataloader, val_dataloader, test_dataloader, seq_len_dict=seq_len_dict,
                               padding_value=padding_value, batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                               models_path=f'{project_path}models/', model_name=model_name, ModelClass=Models.MF1LSTM,
                               is_custom=True, do_test=True, metrics=metrics, log_comet_ml=True,
                               comet_ml_api_key=comet_ml_api_key, comet_ml_project_name=comet_ml_project_name,
                               comet_ml_workspace=comet_ml_workspace, comet_ml_save_model=True,
                               already_embedded=already_embedded)

next(model.parameters())

# #### Hyperparameter optimization

config_name = input('Hyperparameter optimization configuration file name:')

val_loss_min, exp_name_min = du.machine_learning.optimize_hyperparameters(Models.MF1LSTM, df=dmy_norm_df,
                                                                          config_name=config_name,
                                                                          comet_ml_api_key=comet_ml_api_key,
                                                                          comet_ml_project_name=comet_ml_project_name,
                                                                          comet_ml_workspace=comet_ml_workspace,
                                                                          n_inputs=n_inputs, id_column=id_column,
                                                                          label_column=label_column, inst_column=inst_column,
                                                                          n_outputs=n_outputs, model_type='multivariate_rnn',
                                                                          is_custom=True, models_path='models/',
                                                                          array_param='embedding_dim',
                                                                          config_path=f'{project_path}hyperparameter_optimization/',
                                                                          var_seq=True, clip_value=0.5, padding_value=padding_value,
                                                                          batch_size=batch_size, n_epochs=n_epochs,
                                                                          lr=lr, test_train_ratio=test_train_ratio,
                                                                          validation_ratio=validation_ratio,
                                                                          comet_ml_save_model=True,
                                                                          embed_features=embed_features)

exp_name_min

# ### MF2-LSTM
#
# Implementation of the [_Predicting healthcare trajectories from medical records: A deep learning approach_](https://doi.org/10.1016/j.jbi.2017.04.001) paper, parametric time version.

# #### Creating the model

# Model parameters:

n_hidden = 100                             # Number of hidden units
n_rnn_layers = 4                           # Number of MF2-LSTM layers
p_dropout = 0.2                            # Probability of dropout
elapsed_time = 'small'                     # Indicates if the elapsed time between events is small or long; influences how to discount elapsed time

if use_delta_ts == 'normalized':
    raise Exception('ERROR: When using a model of type MF2-LSTM, we can\'t use normalized delta_ts. Please use it raw (use_delta_ts = "raw").')
elif use_delta_ts is False:
    raise Exception('ERROR: When using a model of type MF2-LSTM, we must use delta_ts. Please use it, in a raw version (use_delta_ts = "raw").')

# Instantiating the model:

model = Models.MF2LSTM(n_inputs, n_hidden, n_outputs, n_rnn_layers, p_dropout,
                       embed_features=embed_features, embedding_dim=embedding_dim,
                       elapsed_time=elapsed_time)
model

# Define the name that will be given to the models that will be saved:

model_name = 'mf2lstm'
if dataset_mode == 'embedded':
    model_name.append('_embedded')
elif dataset_mode == 'one hot encoded' and embed_features is not None:
    model_name.append('_with_embedding')
elif dataset_mode == 'one hot encoded' and embed_features is None:
    model_name.append('_one_hot_encoded')

# #### Training and testing the model

next(model.parameters())

model = du.deep_learning.train(model, train_dataloader, val_dataloader, test_dataloader, seq_len_dict=seq_len_dict,
                               padding_value=padding_value, batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                               models_path=f'{project_path}models/', model_name=model_name, ModelClass=Models.MF2LSTM,
                               is_custom=True, do_test=True, metrics=metrics, log_comet_ml=True,
                               comet_ml_api_key=comet_ml_api_key, comet_ml_project_name=comet_ml_project_name,
                               comet_ml_workspace=comet_ml_workspace, comet_ml_save_model=True,
                               already_embedded=already_embedded)

next(model.parameters())

# #### Hyperparameter optimization

config_name = input('Hyperparameter optimization configuration file name:')

val_loss_min, exp_name_min = du.machine_learning.optimize_hyperparameters(Models.MF2LSTM, df=dmy_norm_df,
                                                                          config_name=config_name,
                                                                          comet_ml_api_key=comet_ml_api_key,
                                                                          comet_ml_project_name=comet_ml_project_name,
                                                                          comet_ml_workspace=comet_ml_workspace,
                                                                          n_inputs=n_inputs, id_column=id_column,
                                                                          label_column=label_column, inst_column=inst_column,
                                                                          n_outputs=n_outputs, model_type='multivariate_rnn',
                                                                          is_custom=True, models_path='models/',
                                                                          array_param='embedding_dim',
                                                                          config_path=f'{project_path}hyperparameter_optimization/',
                                                                          var_seq=True, clip_value=0.5, padding_value=padding_value,
                                                                          batch_size=batch_size, n_epochs=n_epochs,
                                                                          lr=lr, test_train_ratio=test_train_ratio,
                                                                          validation_ratio=validation_ratio,
                                                                          comet_ml_save_model=True,
                                                                          embed_features=embed_features)

exp_name_min

# ### XGBoost

# Model hyperparameters:

objective = 'multi:softmax'                # Objective function to minimize (in this case, softmax)
eval_metric = 'mlogloss'                   # Metric to analyze (in this case, multioutput negative log likelihood loss)

# Initializing the model:

xgb_model = xgb.XGBClassifier(objective=objective, eval_metric=eval_metric, learning_rate=lr,
                              num_class=n_output, random_state=du.random_seed, seed=du.random_seed)
xgb_model

# Training with early stopping (stops training if the evaluation metric doesn't improve on 5 consequetive iterations):

xgb_model.fit(train_features, train_labels, early_stopping_rounds=5, eval_set=[(val_features, val_labels)])

# Find the validation loss:

val_pred_proba = xgb_model.predict_proba(val_features)

val_loss = log_loss(val_labels, val_pred_proba)
val_loss

# Save the model:

# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
model_filename = f'{models_path}xgb_{val_loss:.4f}valloss_{current_datetime}.pth'
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

# #### Hyperparameter optimization




# ### Logistic Regression

# Model hyperparameters:

solver = 'lbfgs'
penalty = 'l2'
C = 1
max_iter = 1000

# Initializing the model:

logreg_model = LogisticRegression(solver=solver, penalty=penalty, C=C, max_iter=max_iter, random_state=du.random_seed)
logreg_model

# Training and testing:

logreg_model.fit(train_features, train_labels)

# Find the validation loss:

val_pred_proba = logreg_model.predict_proba(val_features)

val_loss = log_loss(val_labels, val_pred_proba)
val_loss

# Save the model:

# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
model_filename = f'{models_path}logreg_{val_loss:.4f}valloss_{current_datetime}.pth'
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

# #### Hyperparameter optimization



# ### SVM

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

# Find the validation loss:

val_pred_proba = svm_model.predict_proba(val_features)

val_loss = log_loss(val_labels, val_pred_proba)
val_loss

# Save the model:

# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
model_filename = f'{models_path}svm_{val_loss:.4f}valloss_{current_datetime}.pth'
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

# #### Hyperparameter optimization


