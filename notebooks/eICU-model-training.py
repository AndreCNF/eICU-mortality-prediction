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
import utils                               # Context specific (in this case, for the eICU data) methods
import Models                              # Deep learning models
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
already_embedded = None                    # Indicates if categorical features are already embedded when fetching a batch
@interact
def get_dataset_mode(data_mode=['one hot encoded', 'learn embedding', 'pre-embedded'],
                     ml_or_dl=['deep learning', 'machine learning'],
                     use_delta=[False, 'normalized', 'raw'], window_h=(0, 96, 24)):
    global dataset_mode, ml_core, use_delta_ts, time_window_h, already_embedded
    dataset_mode, ml_core, use_delta_ts, time_window_h = data_mode, ml_or_dl, use_delta, window_h
    already_embedded = dataset_mode == 'embedded'


id_column = 'patientunitstayid'            # Name of the sequence ID column
ts_column = 'ts'                           # Name of the timestamp column
label_column = 'label'                     # Name of the label column
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

# test_train_ratio = 0.25                    # Percentage of the data which will be used as a test set
# validation_ratio = 0.1                     # Percentage of the data from the training set which is used for validation purposes
batch_size = 32                            # Number of unit stays in a mini batch
n_epochs = 10                              # Number of epochs
lr = 0.001                                 # Learning rate

stream_tvt_sets = open(f'{data_path}eICU_tvt_sets.yml', 'r')
eICU_tvt_sets = yaml.load(stream_tvt_sets, Loader=yaml.FullLoader)
eICU_tvt_sets

# Testing parameters:

metrics = ['loss', 'accuracy', 'AUC', 'AUC_weighted']

# ## Defining the dataset object

cat_feat_ohe

[feat_list for feat_list in cat_feat_ohe.values()]

[[col for col in feat_list] for feat_list in [feat_list for feat_list in cat_feat_ohe.values()]]

dataset = du.datasets.Large_Dataset(files_name='eICU', process_pipeline=utils.eICU_process_pipeline,
                                    id_column=id_column, initial_analysis=utils.eICU_initial_analysis,
                                    files_path=data_path, dataset_mode=dataset_mode, ml_core=ml_core,
                                    use_delta_ts=use_delta_ts, time_window_h=time_window_h,
                                    padding_value=padding_value, cat_feat_ohe=cat_feat_ohe, dtype_dict=dtype_dict)

# Make sure that we discard the ID, timestamp and label columns
if n_inputs != dataset.n_inputs:
    n_inputs = dataset.n_inputs
    print(f'Changed the number of inputs to {n_inputs}')
else:
    n_inputs

if dataset_mode == 'learn embedding':
    embed_features = dataset.embed_features
    n_embeddings = dataset.n_embeddings
else:
    embed_features = None
    n_embeddings = None
print(f'Embedding features: {embed_features}')
print(f'Number of embeddings: {n_embeddings}')

dataset.__len__()

dataset.bool_feat

# ## Separating into train and validation sets

(train_dataloader, val_dataloader, test_dataloader,
train_indeces, val_indeces, test_indeces) = du.machine_learning.create_train_sets(dataset,
#                                                                                   test_train_ratio=test_train_ratio,
#                                                                                   validation_ratio=validation_ratio,
                                                                                  train_indices=eICU_tvt_sets['train_indices'],
                                                                                  val_indices=eICU_tvt_sets['val_indices'],
                                                                                  test_indices=eICU_tvt_sets['test_indices'],
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
    print(next(iter(train_dataloader))[0])
else:
    print(train_features[:32])

if ml_core == 'deep learning':
    print(next(iter(val_dataloader))[0])
else:
    print(val_features[:32])

if ml_core == 'deep learning':
    print(next(iter(test_dataloader))[0])
else:
    print(test_features[:32])

next(iter(test_dataloader))[0].shape

# ## Training models

# ### Vanilla RNN

# #### Creating the model

# Model parameters:

n_hidden = 100                             # Number of hidden units
n_layers = 2                               # Number of LSTM layers
p_dropout = 0.2                            # Probability of dropout
bidir = False                              # Sets if the RNN layer is bidirectional or not

if use_delta_ts == 'normalized':
    # Count the delta_ts column as another feature, only ignore ID, timestamp and label columns
    n_inputs = dataset.n_inputs + 1
elif use_delta_ts == 'raw':
    raise Exception('ERROR: When using a model of type Vanilla RNN, we can\'t use raw delta_ts. Please either normalize it (use_delta_ts = "normalized") or discard it (use_delta_ts = False).')

# Instantiating the model:

model = Models.VanillaRNN(n_inputs, n_hidden, n_outputs, n_layers, p_dropout,
                          embed_features=embed_features, n_embeddings=n_embeddings,
                          embedding_dim=embedding_dim, bidir=bidir)
model

# Define the name that will be given to the models that will be saved:

model_name = 'rnn'
if dataset_mode == 'pre-embedded':
    model_name = model_name + '_pre_embedded'
elif dataset_mode == 'learn embedding':
    model_name = model_name + '_with_embedding'
elif dataset_mode == 'one hot encoded':
    model_name = model_name + '_one_hot_encoded'
if use_delta_ts is not False:
    model_name = model_name + '_delta_ts'
model_name

# #### Training and testing the model

next(model.parameters())

model = du.deep_learning.train(model, train_dataloader, val_dataloader, test_dataloader, dataset=dataset,
                               padding_value=padding_value, batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                               models_path=f'{project_path}models/', model_name=model_name, ModelClass=Models.VanillaRNN,
                               is_custom=False, do_test=True, metrics=metrics, log_comet_ml=True,
                               comet_ml_api_key=comet_ml_api_key, comet_ml_project_name=comet_ml_project_name,
                               comet_ml_workspace=comet_ml_workspace, comet_ml_save_model=True,
                               already_embedded=already_embedded)

next(model.parameters())

# #### Hyperparameter optimization

config_name = input('Hyperparameter optimization configuration file name:')

val_loss_min, exp_name_min = du.machine_learning.optimize_hyperparameters(Models.VanillaRNN,
                                                                          train_dataloader=train_dataloader,
                                                                          val_dataloader=val_dataloader,
                                                                          test_dataloader=test_dataloader,
                                                                          dataset=dataset,
                                                                          config_name=config_name,
                                                                          comet_ml_api_key=comet_ml_api_key,
                                                                          comet_ml_project_name=comet_ml_project_name,
                                                                          comet_ml_workspace=comet_ml_workspace,
                                                                          n_inputs=n_inputs, id_column=id_column,
                                                                          inst_column=ts_column,
                                                                          id_columns_idx=[0, 1],
                                                                          n_outputs=n_outputs, model_type='multivariate_rnn',
                                                                          is_custom=False, models_path='models/',
                                                                          model_name=model_name,
                                                                          array_param='embedding_dim',
                                                                          metrics=metrics,
                                                                          config_path=f'{project_path}hyperparameter_optimization/',
                                                                          var_seq=True, clip_value=0.5,
                                                                          padding_value=padding_value,
                                                                          batch_size=batch_size, n_epochs=n_epochs,
                                                                          lr=lr,
                                                                          comet_ml_save_model=True,
                                                                          embed_features=embed_features,
                                                                          n_embeddings=n_embeddings)

exp_name_min

# ### Vanilla LSTM

# #### Creating the model

# Model parameters:

n_hidden = 100                             # Number of hidden units
n_layers = 2                               # Number of LSTM layers
p_dropout = 0.2                            # Probability of dropout
bidir = False                              # Sets if the RNN layer is bidirectional or not

if use_delta_ts == 'normalized':
    # Count the delta_ts column as another feature, only ignore ID, timestamp and label columns
    n_inputs = dataset.n_inputs + 1
elif use_delta_ts == 'raw':
    raise Exception('ERROR: When using a model of type Vanilla RNN, we can\'t use raw delta_ts. Please either normalize it (use_delta_ts = "normalized") or discard it (use_delta_ts = False).')

# Instantiating the model:

model = Models.VanillaLSTM(n_inputs, n_hidden, n_outputs, n_layers, p_dropout,
                           embed_features=embed_features, n_embeddings=n_embeddings,
                           embedding_dim=embedding_dim, bidir=bidir)
model

# Define the name that will be given to the models that will be saved:

model_name = 'lstm'
if dataset_mode == 'pre-embedded':
    model_name = model_name + '_pre_embedded'
elif dataset_mode == 'learn embedding':
    model_name = model_name + '_with_embedding'
elif dataset_mode == 'one hot encoded':
    model_name = model_name + '_one_hot_encoded'
if use_delta_ts is not False:
    model_name = model_name + '_delta_ts'
model_name

# #### Training and testing the model

next(model.parameters())

model = du.deep_learning.train(model, train_dataloader, val_dataloader, test_dataloader, dataset=dataset,
                               padding_value=padding_value, batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                               models_path=f'{project_path}models/', model_name=model_name, ModelClass=Models.VanillaLSTM,
                               is_custom=False, do_test=True, metrics=metrics, log_comet_ml=True,
                               comet_ml_api_key=comet_ml_api_key, comet_ml_project_name=comet_ml_project_name,
                               comet_ml_workspace=comet_ml_workspace, comet_ml_save_model=True,
                               already_embedded=already_embedded)

next(model.parameters())

# #### Hyperparameter optimization

config_name = input('Hyperparameter optimization configuration file name:')

val_loss_min, exp_name_min = du.machine_learning.optimize_hyperparameters(Models.VanillaLSTM,
                                                                          train_dataloader=train_dataloader,
                                                                          val_dataloader=val_dataloader,
                                                                          test_dataloader=test_dataloader,
                                                                          dataset=dataset,
                                                                          config_name=config_name,
                                                                          comet_ml_api_key=comet_ml_api_key,
                                                                          comet_ml_project_name=comet_ml_project_name,
                                                                          comet_ml_workspace=comet_ml_workspace,
                                                                          n_inputs=n_inputs, id_column=id_column,
                                                                          inst_column=ts_column,
                                                                          id_columns_idx=[0, 1],
                                                                          n_outputs=n_outputs, model_type='multivariate_rnn',
                                                                          is_custom=False, models_path='models/',
                                                                          model_name=model_name,
                                                                          array_param='embedding_dim',
                                                                          metrics=metrics,
                                                                          config_path=f'{project_path}hyperparameter_optimization/',
                                                                          var_seq=True, clip_value=0.5,
                                                                          padding_value=padding_value,
                                                                          batch_size=batch_size, n_epochs=n_epochs,
                                                                          lr=lr,
                                                                          comet_ml_save_model=True,
                                                                          embed_features=embed_features,
                                                                          n_embeddings=n_embeddings)

exp_name_min

# ### T-LSTM
#
# Implementation of the [_Patient Subtyping via Time-Aware LSTM Networks_](http://biometrics.cse.msu.edu/Publications/MachineLearning/Baytasetal_PatientSubtypingViaTimeAwareLSTMNetworks.pdf) paper.

# #### Creating the model

# Model parameters:

n_hidden = 100                             # Number of hidden units
n_rnn_layers = 2                           # Number of TLSTM layers
p_dropout = 0.2                            # Probability of dropout
elapsed_time = 'small'                     # Indicates if the elapsed time between events is small or long; influences how to discount elapsed time

if use_delta_ts == 'raw':
    raise Exception('ERROR: When using a model of type TLSTM, we can\'t use raw delta_ts. Please normalize it (use_delta_ts = "normalized").')
elif use_delta_ts is False:
    raise Exception('ERROR: When using a model of type TLSTM, we must use delta_ts. Please use it, in a normalized version (use_delta_ts = "normalized").')

# Instantiating the model:

model = Models.TLSTM(n_inputs, n_hidden, n_outputs, n_rnn_layers, p_dropout,
                     embed_features=embed_features, n_embeddings=n_embeddings,
                     embedding_dim=embedding_dim, elapsed_time=elapsed_time)
model

# Define the name that will be given to the models that will be saved:

model_name = 'tlstm'
if dataset_mode == 'pre-embedded':
    model_name = model_name + '_pre_embedded'
elif dataset_mode == 'learn embedding':
    model_name = model_name + '_with_embedding'
elif dataset_mode == 'one hot encoded':
    model_name = model_name + '_one_hot_encoded'
if use_delta_ts is not False:
    model_name = model_name + '_delta_ts'
model_name

# #### Training and testing the model

next(model.parameters())

model = du.deep_learning.train(model, train_dataloader, val_dataloader, test_dataloader, dataset=dataset,
                               padding_value=padding_value, batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                               models_path=f'{project_path}models/', model_name=model_name, ModelClass=Models.TLSTM,
                               is_custom=True, do_test=True, metrics=metrics, log_comet_ml=True,
                               comet_ml_api_key=comet_ml_api_key, comet_ml_project_name=comet_ml_project_name,
                               comet_ml_workspace=comet_ml_workspace, comet_ml_save_model=True,
                               already_embedded=already_embedded)

next(model.parameters())

# #### Hyperparameter optimization

config_name = input('Hyperparameter optimization configuration file name:')

val_loss_min, exp_name_min = du.machine_learning.optimize_hyperparameters(Models.TLSTM,
                                                                          train_dataloader=train_dataloader,
                                                                          val_dataloader=val_dataloader,
                                                                          test_dataloader=test_dataloader,
                                                                          dataset=dataset,
                                                                          config_name=config_name,
                                                                          comet_ml_api_key=comet_ml_api_key,
                                                                          comet_ml_project_name=comet_ml_project_name,
                                                                          comet_ml_workspace=comet_ml_workspace,
                                                                          n_inputs=n_inputs, id_column=id_column,
                                                                          inst_column=ts_column,
                                                                          id_columns_idx=[0, 1],
                                                                          n_outputs=n_outputs, model_type='multivariate_rnn',
                                                                          is_custom=True, models_path='models/',
                                                                          model_name=model_name,
                                                                          array_param='embedding_dim',
                                                                          metrics=metrics,
                                                                          config_path=f'{project_path}hyperparameter_optimization/',
                                                                          var_seq=True, clip_value=0.5,
                                                                          padding_value=padding_value,
                                                                          batch_size=batch_size, n_epochs=n_epochs,
                                                                          lr=lr,
                                                                          comet_ml_save_model=True,
                                                                          embed_features=embed_features,
                                                                          n_embeddings=n_embeddings)

exp_name_min

# ### MF1-LSTM
#
# Implementation of the [_Predicting healthcare trajectories from medical records: A deep learning approach_](https://doi.org/10.1016/j.jbi.2017.04.001) paper, time decay version.

# #### Creating the model

# Model parameters:

n_hidden = 100                             # Number of hidden units
n_rnn_layers = 2                           # Number of MF1-LSTM layers
p_dropout = 0.2                            # Probability of dropout
elapsed_time = 'small'                     # Indicates if the elapsed time between events is small or long; influences how to discount elapsed time

if use_delta_ts == 'raw':
    raise Exception('ERROR: When using a model of type MF1-LSTM, we can\'t use raw delta_ts. Please normalize it (use_delta_ts = "normalized").')
elif use_delta_ts is False:
    raise Exception('ERROR: When using a model of type MF1-LSTM, we must use delta_ts. Please use it, in a normalized version (use_delta_ts = "normalized").')

# Instantiating the model:

model = Models.MF1LSTM(n_inputs, n_hidden, n_outputs, n_rnn_layers, p_dropout,
                       embed_features=embed_features, n_embeddings=n_embeddings,
                       embedding_dim=embedding_dim, elapsed_time=elapsed_time)
model

# Define the name that will be given to the models that will be saved:

model_name = 'mf1lstm'
if dataset_mode == 'pre-embedded':
    model_name = model_name + '_pre_embedded'
elif dataset_mode == 'learn embedding':
    model_name = model_name + '_with_embedding'
elif dataset_mode == 'one hot encoded':
    model_name = model_name + '_one_hot_encoded'
if use_delta_ts is not False:
    model_name = model_name + '_delta_ts'
model_name

# #### Training and testing the model

next(model.parameters())

model = du.deep_learning.train(model, train_dataloader, val_dataloader, test_dataloader, dataset=dataset,
                               padding_value=padding_value, batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                               models_path=f'{project_path}models/', model_name=model_name, ModelClass=Models.MF1LSTM,
                               is_custom=True, do_test=True, metrics=metrics, log_comet_ml=True,
                               comet_ml_api_key=comet_ml_api_key, comet_ml_project_name=comet_ml_project_name,
                               comet_ml_workspace=comet_ml_workspace, comet_ml_save_model=True,
                               already_embedded=already_embedded)

next(model.parameters())

# #### Hyperparameter optimization

config_name = input('Hyperparameter optimization configuration file name:')

val_loss_min, exp_name_min = du.machine_learning.optimize_hyperparameters(Models.MF1LSTM,
                                                                          train_dataloader=train_dataloader,
                                                                          val_dataloader=val_dataloader,
                                                                          test_dataloader=test_dataloader,
                                                                          dataset=dataset,
                                                                          config_name=config_name,
                                                                          comet_ml_api_key=comet_ml_api_key,
                                                                          comet_ml_project_name=comet_ml_project_name,
                                                                          comet_ml_workspace=comet_ml_workspace,
                                                                          n_inputs=n_inputs, id_column=id_column,
                                                                          inst_column=ts_column,
                                                                          id_columns_idx=[0, 1],
                                                                          n_outputs=n_outputs, model_type='multivariate_rnn',
                                                                          is_custom=True, models_path='models/',
                                                                          model_name=model_name,
                                                                          array_param='embedding_dim',
                                                                          metrics=metrics,
                                                                          config_path=f'{project_path}hyperparameter_optimization/',
                                                                          var_seq=True, clip_value=0.5,
                                                                          padding_value=padding_value,
                                                                          batch_size=batch_size, n_epochs=n_epochs,
                                                                          lr=lr,
                                                                          comet_ml_save_model=True,
                                                                          embed_features=embed_features,
                                                                          n_embeddings=n_embeddings)

exp_name_min

# ### MF2-LSTM
#
# Implementation of the [_Predicting healthcare trajectories from medical records: A deep learning approach_](https://doi.org/10.1016/j.jbi.2017.04.001) paper, parametric time version.

# #### Creating the model

# Model parameters:

n_hidden = 100                             # Number of hidden units
n_rnn_layers = 2                           # Number of MF2-LSTM layers
p_dropout = 0.2                            # Probability of dropout
elapsed_time = 'small'                     # Indicates if the elapsed time between events is small or long; influences how to discount elapsed time

if use_delta_ts == 'normalized':
    raise Exception('ERROR: When using a model of type MF2-LSTM, we can\'t use normalized delta_ts. Please use it raw (use_delta_ts = "raw").')
elif use_delta_ts is False:
    raise Exception('ERROR: When using a model of type MF2-LSTM, we must use delta_ts. Please use it, in a raw version (use_delta_ts = "raw").')

# Instantiating the model:

model = Models.MF2LSTM(n_inputs, n_hidden, n_outputs, n_rnn_layers, p_dropout,
                       embed_features=embed_features, n_embeddings=n_embeddings,
                       embedding_dim=embedding_dim, elapsed_time=elapsed_time)
model

# Define the name that will be given to the models that will be saved:

model_name = 'mf2lstm'
if dataset_mode == 'pre-embedded':
    model_name = model_name + '_pre_embedded'
elif dataset_mode == 'learn embedding':
    model_name = model_name + '_with_embedding'
elif dataset_mode == 'one hot encoded':
    model_name = model_name + '_one_hot_encoded'
if use_delta_ts is not False:
    model_name = model_name + '_delta_ts'
model_name

# #### Training and testing the model

next(model.parameters())

model = du.deep_learning.train(model, train_dataloader, val_dataloader, test_dataloader, dataset=dataset,
                               padding_value=padding_value, batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                               models_path=f'{project_path}models/', model_name=model_name, ModelClass=Models.MF2LSTM,
                               is_custom=True, do_test=True, metrics=metrics, log_comet_ml=True,
                               comet_ml_api_key=comet_ml_api_key, comet_ml_project_name=comet_ml_project_name,
                               comet_ml_workspace=comet_ml_workspace, comet_ml_save_model=True,
                               already_embedded=already_embedded)

next(model.parameters())

# #### Hyperparameter optimization

config_name = input('Hyperparameter optimization configuration file name:')

val_loss_min, exp_name_min = du.machine_learning.optimize_hyperparameters(Models.MF2LSTM,
                                                                          train_dataloader=train_dataloader,
                                                                          val_dataloader=val_dataloader,
                                                                          test_dataloader=test_dataloader,
                                                                          dataset=dataset,
                                                                          config_name=config_name,
                                                                          comet_ml_api_key=comet_ml_api_key,
                                                                          comet_ml_project_name=comet_ml_project_name,
                                                                          comet_ml_workspace=comet_ml_workspace,
                                                                          n_inputs=n_inputs, id_column=id_column,
                                                                          inst_column=ts_column,
                                                                          id_columns_idx=[0, 1],
                                                                          n_outputs=n_outputs, model_type='multivariate_rnn',
                                                                          is_custom=True, models_path='models/',
                                                                          model_name=model_name,
                                                                          array_param='embedding_dim',
                                                                          metrics=metrics,
                                                                          config_path=f'{project_path}hyperparameter_optimization/',
                                                                          var_seq=True, clip_value=0.5,
                                                                          padding_value=padding_value,
                                                                          batch_size=batch_size, n_epochs=n_epochs,
                                                                          lr=lr,
                                                                          comet_ml_save_model=True,
                                                                          embed_features=embed_features,
                                                                          n_embeddings=n_embeddings)

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
