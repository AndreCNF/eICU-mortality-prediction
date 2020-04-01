# # Models dummy tests
# ---
#
# Testing models from the project defined classes, including the embedding layers and time intervals handling, on dummy datasets.

# ## Importing the necessary packages

import comet_ml                            # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import os                                  # os handles directory/workspace changes
import pandas as pd                        # Pandas to load the data initially
# import modin.pandas as pd                  # Optimized distributed version of Pandas
import numpy as np                         # Mathematical operations package, allowing also for missing values representation
import torch                               # PyTorch for tensor and deep learning operations
import plotly.graph_objs as go             # Plotly for interactive and pretty plots
import data_utils as du                    # Data science and machine learning relevant methods
from model_interpreter.model_interpreter import ModelInterpreter  # Model interpretability class
import shap                                # Model-agnostic interpretability package inspired on Shapley values

du.random_seed

du.set_random_seed(42)

du.random_seed

du.set_pandas_library(lib='pandas')

import pixiedust                           # Debugging in Jupyter Notebook cells

# Change to scripts directory
os.chdir('../../scripts')

import Models                              # Script with all the machine learning model classes

# Change to parent directory (presumably "eICU-mortality-prediction")
os.chdir('..')

# ## Initializing variables

# Comet ML settings:

comet_ml_project_name = input('Comet ML project name:')
comet_ml_workspace = input('Comet ML workspace:')
comet_ml_api_key = getpass.getpass('Comet ML API key')

# Data that we'll be using:

dmy_data = np.array([[0, 0, 23, 284, 70, 5, np.nan, 0],
                     [0, 1, 23, 284, 70, 5, 'b', 0],
                     [0, 2, 24, 270, 73, 5, 'b', 0],
                     [0, 3, 22, 290, 71, 5, 'a', 0],
                     [0, 3, 22, 290, 71, 5, 'b', 0],
                     [0, 4, 20, 288, 65, 4, 'a', 1],
                     [0, 4, 20, 288, 65, 4, 'b', 1],
                     [0, 5, 21, 297, 64, 4, 'a', 1],
                     [0, 5, 21, 297, 64, 4, 'b', 1],
                     [0, 5, 21, 297, 64, 4, 'c', 1],
                     [1, 0, 25, 300, 76, 5, 'a', 0],
                     [1, 1, 19, 283, 70, 5, 'c', 0],
                     [1, 2, 19, 306, 59, 5, 'a', 1],
                     [1, 2, 19, 306, 59, 5, 'c', 1],
                     [1, 3, 18, 298, 55, 3, 'c', 1],
                     [2, 0, 20, 250, 70, 5, 'c', 0],
                     [2, 1, 20, 254, 68, 4, 'a', 1],
                     [2, 1, 20, 254, 68, 4, 'c', 1],
                     [2, 2, 19, 244, 70, 3, 'a', 1],
                     [3, 0, 27, 264, 78, 4, 'b', 0],
                     [3, 1, 22, 293, 67, 4, 'b', 1],
                     [4, 0, 28, 290, 73, 5, 'b', 0],
                     [4, 1, 29, 288, 75, 5, 'b', 0],
                     [4, 2, 28, 289, 75, 5, 'b', 0],
                     [4, 5, 26, 290, 62, 5, 'b', 0],
                     [4, 6, 25, 285, 63, 4, 'b', 0],
                     [4, 12, 23, 280, 58, 4, 'b', 0],
                     [4, 12, 23, 280, 58, 4, 'c', 0],
                     [4, 14, 21, 282, 59, 3, 'a', 0],
                     [4, 14, 21, 282, 59, 3, 'b', 0],
                     [4, 14, 21, 282, 59, 3, 'c', 0],
                     [4, 15, 22, 277, 56, 2, 'a', 1],
                     [4, 16, 20, 270, 53, 2, 'a', 1],])

dmy_data

dmy_df = pd.DataFrame(dmy_data, columns=['subject_id', 'ts', 'Var0', 'Var1', 'Var2', 'Var3', 'Var4', 'label'])
dmy_df

dmy_df.dtypes

# Fix the columns dtypes:

dmy_df['subject_id'] = dmy_df['subject_id'].astype(int)
dmy_df['ts'] = dmy_df['ts'].astype(int)
dmy_df['Var0'] = dmy_df['Var0'].astype(int)
dmy_df['Var1'] = dmy_df['Var1'].astype(int)
dmy_df['Var2'] = dmy_df['Var2'].astype(int)
dmy_df['Var3'] = dmy_df['Var3'].astype(int)
dmy_df['Var4'] = dmy_df['Var4'].astype(str)
dmy_df['label'] = dmy_df['label'].astype(int)

dmy_df.dtypes

# List of used features
dmy_cols = list(dmy_df.columns)
# Remove features that aren't used by the model to predict the label
for unused_feature in ['subject_id', 'ts', 'label']:
    dmy_cols.remove(unused_feature)

dmy_cols

dmy_df.index

dmy_df['subject_id'] == 0

dmy_df.index[dmy_df['subject_id'] == 4]

dmy_df.iloc[dmy_df.index[dmy_df['subject_id'] == 4]]

dmy_df.set_index(['subject_id', 'ts'], inplace=True)

type(dmy_df)

dmy_df

dmy_df.index

# Define if the notebook will run hyperparameter optimization on each model:

do_hyperparam_optim = False

# ## Preparing the dataset

# ### Encoding categories
#
# Converting the categorical feature `Var4` into one hot encoded columns, so that it can be used by the neural networks and by embedding layers.

# ~Encode each row's categorical value:~
#
# One hot encode the categorical feature:

# +
# dmy_df['Var4'], enum_dict = du.embedding.enum_categorical_feature(dmy_df, feature='Var4',
#                                                                   nan_value=0, forbidden_digit=0)
# dmy_df
# -

# %%time
x1 = pd.get_dummies(dmy_df, columns=['Var4'])
x1.head()

x1.dtypes

# %%time
x2 = pd.get_dummies(dmy_df, columns=['Var4'], sparse=True)
x2.head()

x2.dtypes

x2.values

dmy_df, ohe_columns = du.data_processing.one_hot_encoding_dataframe(dmy_df, columns='Var4', 
                                                                    join_rows=False, 
                                                                    get_new_column_names=True, 
                                                                    inplace=True)
dmy_df

ohe_columns

# ### Joining the rows that have the same identifiers

dmy_df = du.embedding.join_repeated_rows(dmy_df, id_columns=['subject_id', 'ts'])
dmy_df

dmy_df.info(memory_usage='deep')

# Testing the merge of boolean columns
tmp_df = dmy_df.rename(columns={'Var4_a': 'Var4_x', 'Var4_b': 'Var4_y'})
tmp_df.head()

du.data_processing.merge_columns(tmp_df, cols_to_merge='Var4')

# ### Normalizing the features

dmy_df.describe().transpose()

dmy_norm_df, mean, std = du.data_processing.normalize_data(dmy_df, id_columns=['subject_id', 'ts'],
                                                           see_progress=False, get_stats=True)
dmy_norm_df

# +
# dmy_norm_df, mean, std = du.data_processing.normalize_data(dmy_df, id_columns=['subject_id', 'ts'],
#                                                            categ_columns=['Var4'], see_progress=False,
#                                                            get_stats=True)
# dmy_norm_df

# +
# dmy_norm_df, mean, std = du.data_processing.normalize_data(dmy_df, id_columns=['subject_id', 'ts'],
#                                                            columns_to_normalize=False,
#                                                            columns_to_normalize_categ=('Var4', ['Var0', 'Var1', 'Var2', 'Var3']), 
#                                                            see_progress=False, get_stats=True)
# dmy_norm_df

# +
# dmy_norm_df, mean, std = du.data_processing.normalize_data(dmy_df, id_columns=['subject_id', 'ts'],
#                                                            columns_to_normalize=False,
#                                                            columns_to_normalize_categ=('Var4', 'Var0'), 
#                                                            see_progress=False, get_stats=True)
# dmy_norm_df
# -

stats = dict()
for key, _ in mean.items():
    stats[key] = dict()
    stats[key]['mean'] = mean[key]
    stats[key]['std'] = std[key]
stats

dmy_norm_df.describe().transpose()

# ### Padding
#
# Pad the data so that all sequences have the same length (so that it can be converted to a PyTorch tensor).

padding_value = 999999

seq_len_dict = du.padding.get_sequence_length_dict(dmy_norm_df, id_column='subject_id', ts_column='ts')
seq_len_dict

data = du.padding.dataframe_to_padded_tensor(dmy_norm_df, seq_len_dict=seq_len_dict,
                                             id_column='subject_id', padding_value=padding_value)
data

data.shape

data[0]

data_perm = data.permute(1, 0, 2)
data_perm

data_perm.shape

data_perm[0]

# ### Dataset object

dataset = du.datasets.Time_Series_Dataset(dmy_norm_df, data)

# ### Separating into train and validation sets
#
# Since this notebook is only for experimentation purposes, with a very small dummy dataset, we'll not be using a test set.

# Training parameters:

batch_size = 32                                 # Number of patients in a mini batch
n_epochs = 100                                  # Number of epochs
lr = 0.001                                      # Learning rate

# Separation in train and validation sets:

# Get the train and validation sets data loaders, which will allow loading batches
train_dataloader, val_dataloader, _ = du.machine_learning.create_train_sets(dataset, test_train_ratio=0, 
                                                                            validation_ratio=0.25,
                                                                            batch_size=4, get_indeces=False)

next(iter(train_dataloader))[0]

next(iter(val_dataloader))[0]

dataset.__len__()

# ## Models testing

# ### Vanilla LSTM

#
#
# #### Creating the model

# Model parameters:

n_ids = dmy_norm_df.subject_id.nunique()      # Total number of sequences
n_inputs = len(dmy_norm_df.columns)           # Number of input features
n_hidden = 10                                 # Number of hidden units
n_outputs = 1                                 # Number of outputs
n_layers = 2                                  # Number of LSTM layers
p_dropout = 0.2                               # Probability of dropout

# Instantiating the model:

model = Models.VanillaLSTM(n_inputs-3, n_hidden, n_outputs, n_layers, p_dropout)
model

# #### Training the model

next(model.parameters())

model = du.deep_learning.train(model, train_dataloader, val_dataloader, seq_len_dict=seq_len_dict,
                               batch_size=batch_size, n_epochs=n_epochs, lr=lr, models_path='models/',
                               padding_value=padding_value, do_test=False, log_comet_ml=False)

next(model.parameters())

# #### Testing the model

output, metrics = du.deep_learning.model_inference(model, dataloader=val_dataloader, 
                                                   metrics=['loss', 'accuracy', 'AUC'],
                                                   seq_len_dict=seq_len_dict, padding_value=padding_value, 
                                                   output_rounded=False, set_name='test', 
                                                   cols_to_remove=[du.search_explore.find_col_idx(dmy_norm_df, feature)
                                                                   for feature in ['subject_id', 'ts']])
output

metrics

# ### LSTM with embedding layers

# #### Creating the model

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

embed_features

model = Models.VanillaLSTM(n_inputs-3, n_hidden, n_outputs, n_layers, p_dropout,
                           embed_features=embed_features, embedding_dim=embedding_dim)
model

model.n_embeddings

# #### Training the model

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

# #### Testing the model

output, metrics = du.deep_learning.model_inference(model, dataloader=val_dataloader, 
                                                   metrics=['loss', 'accuracy', 'AUC'],
                                                   seq_len_dict=seq_len_dict, padding_value=padding_value, 
                                                   output_rounded=False, set_name='test', 
                                                   already_embedded=False,
                                                   cols_to_remove=[du.search_explore.find_col_idx(dmy_norm_df, feature)
                                                                   for feature in ['subject_id', 'ts']])
output

metrics

# ### LSTM with embedding layers and time interval handling

# #### Adding the time difference feature

dmy_df['delta_ts'] = dmy_df.groupby('subject_id').ts.diff()
dmy_df

# #### Normalizing the features

dmy_df.describe().transpose()

dmy_df.dtypes

dmy_norm_df = du.data_processing.normalize_data(dmy_df, id_columns=['subject_id', 'ts'],
                                                see_progress=False)
dmy_norm_df

dmy_norm_df.describe().transpose()

# #### Imputation
#
# Replace the missing time difference values with the mean (zero).

dmy_norm_df = du.data_processing.missing_values_imputation(dmy_norm_df, method='zero')
dmy_norm_df

# #### Padding
#
# Pad the data so that all sequences have the same length (so that it can be converted to a PyTorch tensor).

padding_value = 999999

seq_len_dict = du.padding.get_sequence_length_dict(dmy_norm_df, id_column='subject_id', ts_column='ts')
seq_len_dict

data = du.padding.dataframe_to_padded_tensor(dmy_norm_df, seq_len_dict=seq_len_dict,
                                             id_column='subject_id', padding_value=padding_value)
data

# #### Dataset object

dataset = du.datasets.Time_Series_Dataset(dmy_norm_df, data)

# #### Separating into train and validation sets
#
# Since this notebook is only for experimentation purposes, with a very small dummy dataset, we'll not be using a test set.

# Training parameters:

batch_size = 32                                 # Number of patients in a mini batch
n_epochs = 100                                  # Number of epochs
lr = 0.001                                      # Learning rate

# Separation in train and validation sets:

# Get the train and validation sets data loaders, which will allow loading batches
train_dataloader, val_dataloader, _ = du.machine_learning.create_train_sets(dataset, test_train_ratio=0, 
                                                                            validation_ratio=0.25,
                                                                            batch_size=4, get_indeces=False)

train_features, train_labels = next(iter(train_dataloader))
train_features

val_features, val_labels = next(iter(val_dataloader))
val_features

# #### Creating the model

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

# #### Training the model

next(model.parameters())

next(model.embed_layers.parameters())

model = du.deep_learning.train(model, train_dataloader, val_dataloader, seq_len_dict=seq_len_dict,
                               batch_size=batch_size, n_epochs=n_epochs, lr=lr, models_path='models/',
                               padding_value=padding_value, do_test=False, log_comet_ml=False)

next(model.parameters())

next(model.embed_layers.parameters())

# #### Testing the model

output, metrics = du.deep_learning.model_inference(model, dataloader=val_dataloader, 
                                                   metrics=['loss', 'accuracy', 'AUC'],
                                                   seq_len_dict=seq_len_dict, padding_value=padding_value, 
                                                   output_rounded=False, set_name='test', 
                                                   cols_to_remove=[du.search_explore.find_col_idx(dmy_norm_df, feature)
                                                                   for feature in ['subject_id', 'ts']])
output

metrics

# ### T-LSTM
#
# Implementation of the [_Patient Subtyping via Time-Aware LSTM Networks_](http://biometrics.cse.msu.edu/Publications/MachineLearning/Baytasetal_PatientSubtypingViaTimeAwareLSTMNetworks.pdf) paper.

# #### Creating the model

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

# #### Training the model

next(model.parameters())

next(model.embed_layers.parameters())

model = du.deep_learning.train(model, train_dataloader, val_dataloader, seq_len_dict=seq_len_dict,
                               batch_size=batch_size, n_epochs=n_epochs, lr=lr, models_path='models/',
                               padding_value=padding_value, do_test=False, log_comet_ml=False,
                               is_custom=True)

next(model.parameters())

next(model.embed_layers.parameters())

# #### Testing the model

output, metrics = du.deep_learning.model_inference(model, dataloader=val_dataloader, 
                                                   metrics=['loss', 'accuracy', 'AUC'],
                                                   seq_len_dict=seq_len_dict, padding_value=padding_value, 
                                                   output_rounded=False, set_name='test',
                                                   is_custom=True,
                                                   cols_to_remove=[du.search_explore.find_col_idx(dmy_norm_df, feature)
                                                                   for feature in ['subject_id', 'ts']])
output

metrics

# #### Hyperparameter optimization

if do_hyperparam_optim:
    val_loss_min, exp_name_min = du.machine_learning.optimize_hyperparameters(Models.TLSTM, df=dmy_norm_df, 
                                                                              config_name='TLSTM_hyperparameter_optimization_config.yaml', 
                                                                              comet_ml_api_key=comet_ml_api_key,
                                                                              comet_ml_project_name=comet_ml_project_name, 
                                                                              comet_ml_workspace=comet_ml_workspace, 
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

# ### MF1-LSTM
#
# Implementation of the [_Predicting healthcare trajectories from medical records: A deep learning approach_](https://doi.org/10.1016/j.jbi.2017.04.001) paper, time decay version.

# #### Creating the model

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

# #### Training the model

next(model.parameters())

next(model.embed_layers.parameters())

model = du.deep_learning.train(model, train_dataloader, val_dataloader, seq_len_dict=seq_len_dict,
                               batch_size=batch_size, n_epochs=n_epochs, lr=lr, models_path='models/',
                               ModelClass=Models.MF1LSTM, padding_value=padding_value, do_test=False, 
                               log_comet_ml=False, is_custom=True)

next(model.parameters())

next(model.embed_layers.parameters())

# #### Testing the model

output, metrics = du.deep_learning.model_inference(model, dataloader=val_dataloader, 
                                                   metrics=['loss', 'accuracy', 'AUC'],
                                                   seq_len_dict=seq_len_dict, padding_value=padding_value, 
                                                   output_rounded=False, set_name='test',
                                                   is_custom=True, 
                                                   cols_to_remove=[du.search_explore.find_col_idx(dmy_norm_df, feature)
                                                                   for feature in ['subject_id', 'ts']])
output

metrics

# #### Hyperparameter optimization

if do_hyperparam_optim:
    val_loss_min, exp_name_min = du.machine_learning.optimize_hyperparameters(Models.MF1LSTM, df=dmy_norm_df, 
                                                                              config_name='TLSTM_hyperparameter_optimization_config.yaml', 
                                                                              comet_ml_api_key=comet_ml_api_key,
                                                                              comet_ml_project_name=comet_ml_project_name, 
                                                                              comet_ml_workspace=comet_ml_workspace, 
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

# ### MF2-LSTM
#
# Implementation of the [_Predicting healthcare trajectories from medical records: A deep learning approach_](https://doi.org/10.1016/j.jbi.2017.04.001) paper, parametric time version.

# #### Creating the model

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

# #### Training the model

next(model.parameters())

next(model.embed_layers.parameters())

model = du.deep_learning.train(model, train_dataloader, val_dataloader, seq_len_dict=seq_len_dict,
                               batch_size=batch_size, n_epochs=n_epochs, lr=lr, models_path='models/',
                               ModelClass=Models.MF2LSTM, padding_value=padding_value, do_test=False,
                               log_comet_ml=False, is_custom=True)

next(model.parameters())

next(model.embed_layers.parameters())

# #### Testing the model

output, metrics = du.deep_learning.model_inference(model, dataloader=val_dataloader, 
                                                   metrics=['loss', 'accuracy', 'AUC'],
                                                   seq_len_dict=seq_len_dict, padding_value=padding_value, 
                                                   output_rounded=False, set_name='test',
                                                   is_custom=True,
                                                   cols_to_remove=[du.search_explore.find_col_idx(dmy_norm_df, feature)
                                                                   for feature in ['subject_id', 'ts']])
output

metrics

# #### Hyperparameter optimization

if do_hyperparam_optim:
    val_loss_min, exp_name_min = du.machine_learning.optimize_hyperparameters(Models.MF2LSTM, df=dmy_norm_df, 
                                                                              config_name='TLSTM_hyperparameter_optimization_config.yaml', 
                                                                              comet_ml_api_key=comet_ml_api_key,
                                                                              comet_ml_project_name=comet_ml_project_name, 
                                                                              comet_ml_workspace=comet_ml_workspace, 
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

# #### Interpreting the model

interpreter = ModelInterpreter(model, dmy_norm_df, model_type='multivariate_rnn',
                               id_column=0, inst_column=1, fast_calc=True, SHAP_bkgnd_samples=10000,
                               random_seed=du.random_seed, padding_value=padding_value, is_custom=True)

all_features = np.concatenate([train_features, val_features])
all_features

all_labels = np.concatenate([train_labels, val_labels])
all_labels

idx = 0

all_features[idx]

_ = interpreter.interpret_model(test_data=all_features, 
                                test_labels=all_labels, instance_importance=True, 
                                feature_importance='shap')

interpreter.feat_scores

interpreter.feat_scores.shape

interpreter.test_data[:, :, 2:].shape

interpreter.feat_scores.sum(axis=2)

interpreter.explainer.expected_value[0]

interpreter.feat_scores.sum(axis=2) + interpreter.explainer.expected_value[0]

idx = 0

interpreter.feat_scores.sum(axis=2)[idx] + interpreter.explainer.expected_value[0]

interpreter.test_data[idx]

model(interpreter.test_data[idx, :, 2:].unsqueeze(0))

interpreter.test_data[idx]

interpreter.explainer.subject_ids

interpreter.feat_names

interpreter.feat_scores.reshape(-1, model.n_inputs+1).shape

val_features[:, :4, 2:].numpy().reshape(-1, model.n_inputs+1).shape

# Summarize the effects of all the features
shap.summary_plot(interpreter.feat_scores.reshape(-1, model.n_inputs+1), 
                  features=interpreter.test_data[:, :4, 2:].numpy().reshape(-1, model.n_inputs+1), 
                  feature_names=interpreter.feat_names, plot_type='bar')

# +
# [TODO] Do the same bar plot as above but in plotly
# -

np.abs(interpreter.feat_scores).reshape(-1, interpreter.feat_scores.shape[-1]).shape

mean_abs_shap = np.mean(np.abs(interpreter.feat_scores).reshape(-1, interpreter.feat_scores.shape[-1]), axis=0)
mean_abs_shap

sorted_idx = np.argsort(mean_abs_shap)
sorted_idx

interpreter.feat_names

[interpreter.feat_names[idx] for idx in sorted_idx]

mean_abs_shap[sorted_idx]

figure={
    'data': [dict(
        type='bar',
        x=mean_abs_shap[sorted_idx],
        y=[interpreter.feat_names[idx] for idx in sorted_idx],
        orientation='h'
    )],
    'layout': dict(
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
        xaxis_title='mean(|SHAP value|) (average impact on model output magnitude)',
        font=dict(
                family='Roboto',
                size=14,
                color='black'
            )
    )
}

go.Figure(figure)

du.visualization.shap_summary_plot(interpreter.feat_scores, interpreter.feat_names)

# +
# # Choosing which example to use
# subject_id = 125
# patient = utils.find_subject_idx(test_features_denorm, subject_id=subject_id)
# patient

# +
# # True sequence length of the current patient's data
# seq_len = seq_len_dict[test_features_denorm[patient, 0, 0].item()]
# # Plot the explanation of the predictions for one patient
# shap.force_plot(interpreter.explainer.expected_value[0], 
#                 interpreter.feat_scores[patient, :seq_len], 
#                 features=test_features_denorm[patient, :seq_len, 2:].numpy(), 
#                 feature_names=ALS_cols)

# +
# # Init the JS visualization code
# shap.initjs()

# # Choosing which timestamp to use
# ts = 9

# # Plot the explanation of one prediction
# shap.force_plot(interpreter.explainer.expected_value[0], 
#                 interpreter.feat_scores[patient][ts], 
#                 features=test_features_denorm[patient, ts, 2:].numpy(), 
#                 feature_names=ALS_cols)
# -

pred = 0
sample = 0

[f'{feature}={val:.2e}' for (feature, val) in zip(interpreter.feat_names, interpreter.test_data[pred, sample, 2:])]

interpreter.explainer.expected_value[0]

interpreter.feat_scores.shape

interpreter.feat_scores[pred, sample].shape

len(interpreter.feat_scores[pred, sample].shape)

interpreter.feat_scores[pred, sample]

model(interpreter.test_data[pred, sample, 2:].unsqueeze(0).unsqueeze(0))

np.sum(interpreter.feat_scores[pred, sample]) + interpreter.explainer.expected_value[0]

interpreter.feat_names

interpreter.test_data[pred, sample, :].numpy()

shap.waterfall_plot(interpreter.explainer.expected_value[0], 
                    interpreter.feat_scores[pred, sample],
                    features=interpreter.test_data[pred, sample, 2:].numpy(), 
                    feature_names=interpreter.feat_names)

shap.waterfall_plot(interpreter.explainer.expected_value[0], 
                    interpreter.feat_scores[pred, sample],
                    features=interpreter.test_data[pred, sample, 2:].numpy(), 
                    feature_names=interpreter.feat_names,
                    max_display=2)

# du.visualization.shap_waterfall_plot(interpreter.explainer.expected_value[0], interpreter.feat_scores[pred, sample],
du.visualization.shap_waterfall_plot(0, interpreter.feat_scores[pred, sample],
                                     interpreter.test_data[pred, sample, 2:], interpreter.feat_names,
                                     max_display=2)

# +
fig = go.Figure()

fig.add_trace(go.Waterfall(
    y = [["initial", "q1", "q2", "q3", "total", "q1", "q2", "q3", "total"]],
    measure = ["absolute", "relative", "relative", "relative", "total", "relative", "relative", "relative", "total"],
    x = [1, 2, 3, -1, None, 1, 2, -4, None],
    base = 1000,
    orientation='h'
))

fig.add_trace(go.Waterfall(
    y = [["2016", "2017", "2017", "2017", "2017", "2018", "2018", "2018", "2018"],
        ["initial", "q1", "q2", "q3", "total", "q1", "q2", "q3", "total"]],
    measure = ["absolute", "relative", "relative", "relative", "total", "relative", "relative", "relative", "total"],
    x = [1.1, 2.2, 3.3, -1.1, None, 1.1, 2.2, -4.4, None],
    base = 1000,
    orientation='h'
))

fig.update_layout(
    waterfallgroupgap = 0.5,
)

fig.show()

# +
fig = go.Figure()

fig.add_trace(go.Waterfall(
    y = ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
#     measure = ["absolute", "relative", "relative", "relative", "total", "relative", "relative", "relative", "total"],
    x = [1, 2, 3, -1, None, 1, 2, -4, None],
    base = 1000,
    orientation='h'
))

fig.show()
# -

interpreter.feat_scores[pred, sample]

interpreter.feat_names

# +
fig = go.Figure()

fig.add_trace(go.Waterfall(
    y = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
    x = [1, 2, 1, -2, -1, 3, -4, 1],
    base = 100,
    orientation='h'
))

fig.show()

# +
fig = go.Figure()

fig.add_trace(go.Waterfall(
    y = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
    x = [-1, -2, -1, 2, 1, -3, 4, -1],
    base = 100,
    orientation='h'
))

fig.show()
# -

# ### Deep Care with parametric time
#
# Implementation of the [_Predicting healthcare trajectories from medical records: A deep learning approach_](https://doi.org/10.1016/j.jbi.2017.04.001) paper, full parametric time version.


