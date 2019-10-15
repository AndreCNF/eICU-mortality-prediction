import comet_ml                                         # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                                            # PyTorch to create and apply deep learning models
from torch import nn, optim                             # nn for neural network layers and optim for training optimizers
import pandas as pd                                     # Pandas to handle the data in dataframes
from datetime import datetime                           # datetime to use proper date and time formats
import os                                               # os handles directory/workspace changes
import numpy as np                                      # NumPy to handle numeric and NaN operations
from NeuralNetwork import NeuralNetwork                 # Import the neural network model class
import utils                                            # Contains auxiliary functions
from Time_Series_Dataset import Time_Series_Dataset     # Dataset subclass which allows the creation of Dataset objects
import argparse                                         # Add command line arguments
import warnings                                         # Print warnings for bad practices

# Create and parse the command line arguments
parser = argparse.ArgumentParser(description='Parameter optimization for ALS neural network.')
parser.add_argument('--comet_ml_api_key', type=str, help='Comet.ml API key')
parser.add_argument('--comet_ml_workspace', type=str, help='Comet.ml workspace where data the experiment will be uploaded')
parser.add_argument('--comet_ml_project_name', type=str, help='Comet.ml project name where data the experiment will be uploaded')
parser.add_argument('--comet_ml_save_model', type=bool, help='Boolean to decide whether the trained models are uploaded to Comet.ml')
args = parser.parse_args()

# Only log training info to Comet.ml if the required parameters are specified
if args.comet_ml_api_key is not None and \
   args.comet_ml_project_name is not None and \
   args.comet_ml_workspace is not None:
    log_comet_ml = True
else:
    log_comet_ml = False
    warnings.warn('All necessary Comet.ml parameters (comet_ml_api_key, \
                   comet_ml_project_name, comet_ml_workspace) must be correctly \
                   specified. Otherwise, the parameter optimization won\'t work.')

# Set random seed to the specified value
np.random.seed(utils.random_seed)
torch.manual_seed(utils.random_seed)

# Change to parent directory (presumably "Documents")
os.chdir("../..")

# Path to the CSV dataset files
data_path = 'Datasets/Thesis/FCUL_ALS/cleaned/'

print('Reading the CSV data...')

# Read the cleaned dataset dataframe
ALS_df = pd.read_csv(f'{data_path}FCUL_ALS_cleaned.csv')

# Drop the unnamed index and the NIV columns
ALS_df.drop(columns=['Unnamed: 0', 'niv'], inplace=True)

# Dataset parameters
n_patients = ALS_df.subject_id.nunique()     # Total number of patients
n_inputs = len(ALS_df.columns)               # Number of input features
n_outputs = 1                                # Number of outputs

if log_comet_ml:
    # Create a Comet.ml parameter optimizer:
    param_optimizer = comet_ml.Optimizer(api_key=args.comet_ml_api_key)

    # Neural network parameters to be optimized with Comet.ml
    params = """
                    n_hidden integer [500, 2000] [1052]
                    n_layers integer [1, 4] [2]
                    p_dropout real [0, 0.5] [0.2]
             """

    # Report to the optimizer the parameters that will be optimized
    param_optimizer.set_params(params)

# Maximum number of iterations to do in the parameter optimization
max_optim_iter = 100

print('Building a dictionary containing the sequence length of each patient\'s time series...')

# Dictionary containing the sequence length (number of temporal events) of each sequence (patient)
seq_len_df = ALS_df.groupby('subject_id').ts.count().to_frame().sort_values(by='ts', ascending=False)
seq_len_dict = dict([(idx, val[0]) for idx, val in list(zip(seq_len_df.index, seq_len_df.values))])

print('Creating a padded tensor version of the dataframe...')

# Value to be used in the padding
padding_value = 999999

# Pad data (to have fixed sequence length) and convert into a PyTorch tensor
data = utils.dataframe_to_padded_tensor(ALS_df, seq_len_dict, n_patients, n_inputs, padding_value=padding_value)

# Training parameters
batch_size = 32                                 # Number of patients in a mini batch
n_epochs = 20                                   # Number of epochs
lr = 0.001                                      # Learning rate

print('Creating a dataset object...')

# Create a Dataset object from the data tensor
dataset = Time_Series_Dataset(data, ALS_df)

print('Distributing the data to train, validation and test sets and getting their data loaders...')

# Get the train, validation and test sets data loaders, which will allow loading batches
train_dataloader, val_dataloader, test_dataloader = utils.create_train_sets(dataset, test_train_ratio=0.2, validation_ratio=0.1,
                                                                            batch_size=batch_size, get_indeces=False)

# Start off with a minimum validation score of infinity
val_loss_min = np.inf

for iter in range(max_optim_iter):
    print('Starting a new parameter optimization iteration...')

    # Create a new Comet.ml experiment
    experiment = comet_ml.Experiment(api_key=args.comet_ml_api_key,
                                     project_name=args.comet_ml_project_name,
                                     workspace=args.comet_ml_workspace)

    try:
        # Get a suggestion
        suggestion = param_optimizer.get_suggestion()
    except comet_ml.NoMoreSuggestionsAvailable:
        # get_suggestion() will raise an exception when no new suggestions are available
        break

    # Instantiate the model (removing the two identifier columns and the labels from the input size)
    model = NeuralNetwork(n_inputs-3, suggestion['n_hidden'], n_outputs,
                          suggestion['n_layers'], suggestion['p_dropout'])

    # Check if GPU (CUDA) is available
    train_on_gpu = torch.cuda.is_available()

    if train_on_gpu:
        model = model.cuda()                        # Move the model to the GPU

    # Set gradient clipping to avoid exploding gradients
    clip_value = 10
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    print('Training the model...')

    # Train the model and get the minimum validation loss
    model, val_loss_min = utils.train(model, train_dataloader, val_dataloader, test_dataloader,
                                      seq_len_dict, batch_size, n_epochs, lr,
                                      model_path='models/',
                                      padding_value=padding_value, do_test=True,
                                      log_comet_ml=log_comet_ml,
                                      comet_ml_save_model=args.comet_ml_save_model,
                                      experiment=experiment,
                                      features_list=list(ALS_df.columns).remove('niv_label'),
                                      get_val_loss_min=True)

    # Report the minimum validation loss achieved so that the parameter optimizer updates the score
    suggestion.report_score('val_loss_min', val_loss_min)
