# # Large dataset testing
# ---
#
# Checking if the new large dataset class, which lazily loads batch files instead of diving a giant pre-loaded one, works well to train my models.

# ## Importing the necessary packages

import os                                  # os handles directory/workspace changes
import comet_ml                            # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                               # PyTorch to create and apply deep learning models
# import modin.pandas as pd                  # Optimized distributed version of Pandas
import pandas as pd                        # Pandas to load and handle the data
import numpy as np                         # NumPy to handle numeric and NaN operations
import getpass                             # Get password or similar private inputs
from ipywidgets import interact            # Display selectors and sliders
import data_utils as du                    # Data science and machine learning relevant methods

du.set_random_seed(42)

# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# Path to the parquet dataset files
data_path = 'notebooks/sandbox/dummy_data/'
# Path to the code files
project_path = ''

# Change to the scripts directory
os.chdir("../../scripts/")
import Models                              # Machine learning models
import utils                               # Context specific (in this case, for the eICU data) methods
# Change to parent directory (presumably "Documents")
os.chdir("..")

du.set_pandas_library(lib='pandas')

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
n_ids = 6                                  # Total number of sequences
n_inputs = 9                               # Number of input features
n_outputs = 1                              # Number of outputs
padding_value = 999999                     # Padding value used to fill in sequences up to the maximum sequence length

# Data types:

dtype_dict = dict(patientunitstayid='uint',
                  ts='uint',
                  int_col='Int32',
                  float_col='float32',
                  cat_1_bool_1='UInt8',
                  cat_1_bool_2='UInt8',
                  cat_2_bool_1='UInt8',
                  cat_3_bool_1='UInt8',
                  cat_3_bool_2='UInt8',
                  cat_3_bool_3='UInt8',
                  cat_3_bool_4='UInt8',
                  death_ts='Int32')

# One hot encoding columns categorization:

cat_feat_ohe = dict(cat_1=['cat_1_bool_1', 'cat_1_bool_2'], 
                    cat_2=['cat_2_bool_1'], 
                    cat_3=['cat_3_bool_1', 'cat_3_bool_2', 'cat_3_bool_3', 'cat_3_bool_4'])
cat_feat_ohe

list(cat_feat_ohe.keys())

# Training parameters:

test_train_ratio = 0.25                    # Percentage of the data which will be used as a test set
validation_ratio = 1/3                     # Percentage of the data from the training set which is used for validation purposes
batch_size = 2                             # Number of unit stays in a mini batch
n_epochs = 1                               # Number of epochs
lr = 0.001                                 # Learning rate

# Testing parameters:

metrics = ['loss', 'accuracy', 'AUC', 'AUC_weighted']

# ## Creating large dummy data

# Create each individual column as a NumPy array:

patientunitstayid_col = np.concatenate([np.repeat(1, 25), 
                                        np.repeat(2, 17), 
                                        np.repeat(3, 56), 
                                        np.repeat(4, 138), 
                                        np.repeat(5, 2000),  
                                        np.repeat(6, 4000), 
                                        np.repeat(7, 6000),
                                        np.repeat(8, 100000)])
patientunitstayid_col

ts_col = np.concatenate([np.arange(25), 
                         np.arange(17), 
                         np.arange(56), 
                         np.arange(138), 
                         np.arange(2000), 
                         np.arange(4000), 
                         np.arange(6000),
                         np.arange(100000)])
ts_col

int_col = np.random.randint(0, 50, size=(112236))
np.random.shuffle(int_col)
int_col

float_col = np.random.uniform(3, 15, size=(112236))
np.random.shuffle(float_col)
float_col

cat_1_bool_1 = np.concatenate([np.random.randint(0, 2, size=(112236))])
np.random.shuffle(cat_1_bool_1)
cat_1_bool_1

cat_1_bool_2 = np.concatenate([np.random.randint(0, 2, size=(112236))])
np.random.shuffle(cat_1_bool_2)
cat_1_bool_2

cat_2_bool_1 = np.concatenate([np.random.randint(0, 2, size=(112236))])
np.random.shuffle(cat_2_bool_1)
cat_2_bool_1

cat_3_bool_1 = np.concatenate([np.random.randint(0, 2, size=(112236))])
np.random.shuffle(cat_3_bool_1)
cat_3_bool_1

cat_3_bool_2 = np.concatenate([np.random.randint(0, 2, size=(112236))])
np.random.shuffle(cat_3_bool_2)
cat_3_bool_2

cat_3_bool_3 = np.concatenate([np.random.randint(0, 2, size=(112236))])
np.random.shuffle(cat_3_bool_3)
cat_3_bool_3

cat_3_bool_4 = np.concatenate([np.random.randint(0, 2, size=(112236))])
np.random.shuffle(cat_3_bool_4)
cat_3_bool_4

death_ts = np.concatenate([np.random.randint(0, 1000, size=(22236)), np.repeat(np.nan, 90000)])
np.random.shuffle(death_ts)
death_ts

data = np.column_stack([patientunitstayid_col, ts_col, int_col, float_col, cat_1_bool_1, 
                        cat_1_bool_2, cat_2_bool_1, cat_3_bool_1, 
                        cat_3_bool_2, cat_3_bool_3, cat_3_bool_4,
                        death_ts])
data

# Create a pandas dataframe with all the columns:

data_df = pd.DataFrame(data, columns=['patientunitstayid', 'ts', 'int_col', 'float_col', 'cat_1_bool_1', 
                                      'cat_1_bool_2', 'cat_2_bool_1', 'cat_3_bool_1', 
                                      'cat_3_bool_2', 'cat_3_bool_3', 'cat_3_bool_4',
                                      'death_ts'])
data_df

data_df.dtypes

data_df = du.utils.convert_dtypes(data_df, dtypes=dtype_dict, inplace=True)

data_df.dtypes

# Save in batch files:

du.data_processing.save_chunked_data(data_df, file_name='dmy_large_data', batch_size=1,
                                     id_column=id_column, data_path=data_path)

pd.read_feather(f'{data_path}dmy_large_data_2.ftr')

# ## Defining the dataset object

dataset = du.datasets.Large_Dataset(files_name='dmy_large_data', process_pipeline=utils.eICU_process_pipeline,
                                    id_column=id_column, initial_analysis=utils.eICU_initial_analysis, 
                                    files_path=data_path, dataset_mode=dataset_mode, ml_core=ml_core, 
                                    use_delta_ts=use_delta_ts, time_window_h=time_window_h, total_length=100000,
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
                                                                                  test_train_ratio=test_train_ratio,
                                                                                  validation_ratio=validation_ratio,
                                                                                  batch_size=batch_size,
                                                                                  get_indeces=True,
                                                                                  num_workers=2)

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

next(iter(train_dataloader))[0].shape

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

n_hidden = 10                              # Number of hidden units
n_layers = 3                               # Number of LSTM layers
p_dropout = 0.2                            # Probability of dropout
embedding_dim = [3, 2, 4]                  # List of embedding dimensions

if use_delta_ts == 'normalized':
    # Count the delta_ts column as another feature, only ignore ID, timestamp and label columns
    n_inputs = dataset.n_inputs + 1
elif use_delta_ts == 'raw':
    raise Exception('ERROR: When using a model of type Vanilla RNN, we can\'t use raw delta_ts. Please either normalize it (use_delta_ts = "normalized") or discard it (use_delta_ts = False).')

# Instantiating the model:

model = Models.VanillaRNN(n_inputs, n_hidden, n_outputs, n_layers, p_dropout,
                          embed_features=embed_features, n_embeddings=n_embeddings, 
                          embedding_dim=embedding_dim, total_length=100000)
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
                               is_custom=False, do_test=True, metrics=metrics, log_comet_ml=False,
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
                                                                          config_path=f'{project_path}notebooks/sandbox/',
                                                                          var_seq=True, clip_value=0.5, 
                                                                          padding_value=padding_value,
                                                                          batch_size=batch_size, n_epochs=n_epochs,
                                                                          lr=lr, 
                                                                          comet_ml_save_model=True,
                                                                          embed_features=embed_features,
                                                                          n_embeddings=n_embeddings)

exp_name_min
