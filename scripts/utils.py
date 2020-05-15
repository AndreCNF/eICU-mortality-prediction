from comet_ml import Experiment            # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import os                                  # os handles directory/workspace changes
print(f'DEBUG: Comet ML environment variable:\n{os.environ["COMET_DISABLE_AUTO_LOGGING"]}')
import pandas as pd                        # Pandas to load and handle the data
import torch                               # PyTorch to create and apply deep learning models
from torch.nn.functional import sigmoid, softmax
import ray                                 # Distributed computing
from ray.util.sgd.utils import override    # Distributed training package
from ray.util.sgd.torch import TrainingOperator
import numpy as np                         # NumPy to handle numeric and NaN operations
from datetime import datetime              # datetime to use proper date and time formats
import inspect                             # Inspect methods and their arguments
from sklearn.metrics import roc_auc_score  # ROC AUC model performance metric
import yaml                                # Save and load YAML files
import warnings                            # Print warnings for bad practices
import data_utils as du                    # Data science and machine learning relevant methods
import sys
sys.path.append('/home/ubuntu/')           # Add the folder on which local files are uploaded
import Models                              # Deep learning models

def eICU_initial_analysis(self):
    # Load a single, so as to retrieve some basic information
    df = pd.read_feather(self.files[0])
    # Number of input features, discarding the ID, timestamp and label columns
    self.n_inputs = len(df.columns) - 3
    # Find the column indices for the ID columns
    self.id_columns_idx = [du.search_explore.find_col_idx(df, col)
                           for col in ['patientunitstayid', 'ts']]
    if self.dataset_mode != 'one hot encoded':
        # Find the indices of the features that will be embedded,
        # as well as the total number of categories per categorical feature
        # Subtracting 2 because of the ID and ts columns
        cat_feat_ohe_list = [feat_list for feat_list in self.cat_feat_ohe.values()]
        self.embed_features = [[du.search_explore.find_col_idx(df, col)
                                for col in feat_list]
                                for feat_list in cat_feat_ohe_list]
        self.n_embeddings = list()
        [self.n_embeddings.append(len(feat_list) + 1)
         for feat_list in self.embed_features]
    if self.dtype_dict is not None:
        # Convert column data types and use them to find the boolean features later
        df = du.utils.convert_dtypes(df, dtypes=self.dtype_dict, inplace=True)
        search_by_dtypes = True
    else:
        search_by_dtypes = False
    # Find all the boolean features
    self.bool_feat = du.search_explore.list_boolean_columns(df, search_by_dtypes=search_by_dtypes)
    return


def eICU_process_pipeline(self, df):
    # Set the label column
    df['label'] = df.death_ts - df.ts <= self.time_window_h * 60
    # Remove the now unneeded `death_ts` column
    df.drop(columns='death_ts', inplace=True)
    if self.use_delta_ts is not False:
        # Create a time variation column
        df['delta_ts'] = df.groupby('patientunitstayid').ts.diff()
    if self.use_delta_ts == 'normalized':
        # Normalize the time variation data
        # NOTE: When using the MF2-LSTM model, since it assumes that the time
        # variation is in minutes, we shouldn't normalize `delta_ts` with this model.
        df['delta_ts'] = (df['delta_ts'] - df['delta_ts'].mean()) / df['delta_ts'].std()
    # Find the label column's index number (i.e. it's place in the order of the columns)
    label_num = du.search_explore.find_col_idx(df, 'label')
    # Create a equence length dictionary
    self.seq_len_dict = du.padding.get_sequence_length_dict(df,
                                                            id_column='patientunitstayid',
                                                            ts_column='ts')
    # Pad the data and convert it into a PyTorch tensor
    df = du.padding.dataframe_to_padded_tensor(df, seq_len_dict=self.seq_len_dict,
                                               id_column='patientunitstayid',
                                               ts_column='ts',
                                               bool_feat=self.bool_feat,
                                               padding_value=self.padding_value,
                                               total_length=self.total_length,
                                               inplace=True)
    # Check if we need to pre-embed the categorical features
    if self.dataset_mode == 'pre-embedded':
        # Run each embedding layer on each respective feature, adding the
        # resulting embedding values to the tensor and removing the original,
        # categorical encoded columns
        x_t = du.embedding.embedding_bag_pipeline(x_t, self.embed_layers,
                                                  self.embed_features,
                                                  inplace=True)
    # Labels
    labels = df[:, label_num]
    # Features
    features = du.deep_learning.remove_tensor_column(df, label_num, inplace=True)
    return features, labels


def eICU_model_creator(config):
    """Constructor function for the model(s) to be optimized.

    You will also need to provide a custom training
    function to specify the optimization procedure for multiple models.

    Args:
        config (dict): Configuration dictionary passed into ``TorchTrainer``.

    Returns:
        One or more torch.nn.Module objects.
    """
    model_class = config.get('model', 'VanillaRNN')
    if model_class == 'VanillaRNN':
        return Models.VanillaRNN(config.get('n_inputs', 2090), config.get('n_hidden', 100),
                                 config.get('n_outputs', 1), config.get('n_rnn_layers', 2),
                                 config.get('p_dropout', 0.2), bidir=config.get('bidir', False),
                                 embed_features=config.get('embed_features', None),
                                 n_embeddings=config.get('n_embeddings', None),
                                 embedding_dim=config.get('embedding_dim', None))
    elif model_class == 'VanillaLSTM':
        return Models.VanillaLSTM(config.get('n_inputs', 2090), config.get('n_hidden', 100),
                                  config.get('n_outputs', 1), config.get('n_rnn_layers', 2),
                                  config.get('p_dropout', 0.2), bidir=config.get('bidir', False),
                                  embed_features=config.get('embed_features', None),
                                  n_embeddings=config.get('n_embeddings', None),
                                  embedding_dim=config.get('embedding_dim', None))
    elif model_class == 'TLSTM':
        return Models.TLSTM(config.get('n_inputs', 2090), config.get('n_hidden', 100),
                            config.get('n_outputs', 1), config.get('n_rnn_layers', 2),
                            config.get('p_dropout', 0.2),
                            embed_features=config.get('embed_features', None),
                            n_embeddings=config.get('n_embeddings', None),
                            embedding_dim=config.get('embedding_dim', None),
                            elapsed_time=config.get('elapsed_time', None))
    elif model_class == 'MF1LSTM':
        return Models.MF1LSTM(config.get('n_inputs', 2090), config.get('n_hidden', 100),
                              config.get('n_outputs', 1), config.get('n_rnn_layers', 2),
                              config.get('p_dropout', 0.2),
                              embed_features=config.get('embed_features', None),
                              n_embeddings=config.get('n_embeddings', None),
                              embedding_dim=config.get('embedding_dim', None),
                              elapsed_time=config.get('elapsed_time', None))
    elif model_class == 'MF2LSTM':
        return Models.MF2LSTM(config.get('n_inputs', 2090), config.get('n_hidden', 100),
                              config.get('n_outputs', 1), config.get('n_rnn_layers', 2),
                              config.get('p_dropout', 0.2),
                              embed_features=config.get('embed_features', None),
                              n_embeddings=config.get('n_embeddings', None),
                              embedding_dim=config.get('embedding_dim', None),
                              elapsed_time=config.get('elapsed_time', None))
    else:
        raise Exception(f'ERROR: {model_class} is an invalid model type. Please specify either "VanillaRNN", "VanillaLSTM", "TLSTM", "MF1LSTM" or "MF2LSTM".')


def eICU_data_creator(config):
    """Constructs Iterables for training and validation.

    Note that even though two Iterable objects can be returned,
    only one Iterable will be used for training.

    Args:
        config: Configuration dictionary passed into ``TorchTrainer``

    Returns:
        One or Two Iterable objects. If only one Iterable object is provided,
        ``trainer.validate()`` will throw a ValueError.
    """
    data_path = config.get('data_path', '')                                     # Path to the directory where the data is stored
    # Load the data types
    stream_dtypes = open(f'{data_path}eICU_dtype_dict.yml', 'r')
    dtype_dict = yaml.load(stream_dtypes, Loader=yaml.FullLoader)
    # Load the one hot encoding columns categorization information
    stream_cat_feat_ohe = open(f'{data_path}eICU_cat_feat_ohe.yml', 'r')
    cat_feat_ohe = yaml.load(stream_cat_feat_ohe, Loader=yaml.FullLoader)
    # Load the predefined train, validation and test sets' indices, if they exist
    sets_file_name = config.get('sets_file_name', None)
    if data_path is not None and sets_file_name is not None:
        print(f'DEBUG: Found indices file {data_path}{sets_file_name}.yml')
        stream_tvt_sets = open(f'{data_path}{sets_file_name}.yml', 'r')
        eICU_tvt_sets = yaml.load(stream_tvt_sets, Loader=yaml.FullLoader)
        train_indices = eICU_tvt_sets.get('train_indices', None)
        print('DEBUG: train_indices is None') if train_indices is None else print('train_indices is not None')
        val_indices = eICU_tvt_sets.get('val_indices', None)
        print('DEBUG: val_indices is None') if val_indices is None else print('val_indices is not None')
        test_indices = eICU_tvt_sets.get('test_indices', [])
        print('DEBUG: test_indices is None') if test_indices is None else print('test_indices is not None')
    else:
        print('DEBUG: Didn\'t find the indices file name in the config!')
        # No predefined train, validation and test sets' indices
        train_indices = None
        val_indices = None
        test_indices = None
    # Load the dataset parameters
    dataset_mode = config.get('dataset_mode', 'one hot encoded')                # The mode in which we'll use the data, either one hot encoded or pre-embedded
    ml_core = config.get('ml_core', 'deep learning')                            # The core machine learning type we'll use; either traditional ML or DL
    use_delta_ts = config.get('use_delta_ts', False)                            # Indicates if we'll use time variation info
    time_window_h = config.get('time_window_h', 48)                             # Number of hours on which we want to predict mortality
    padding_value = config.get('padding_value', 999999)                         # Value to use in the padding, to fill the sequences
    # Define the dataset object
    dataset = du.datasets.Large_Dataset(files_name='eICU',
                                        process_pipeline=eICU_process_pipeline,
                                        id_column='patientunitstayid',
                                        initial_analysis=eICU_initial_analysis,
                                        files_path=data_path,
                                        dataset_mode=dataset_mode,
                                        ml_core=ml_core,
                                        use_delta_ts=use_delta_ts,
                                        time_window_h=time_window_h,
                                        padding_value=padding_value,
                                        cat_feat_ohe=cat_feat_ohe,
                                        dtype_dict=dtype_dict)
    print('DEBUG: Successfully created the dataset object.')
    print(f'DEBUG: Dataset length (number of files): {dataset.__len__()}')
    # Update the embedding information
    if dataset_mode == 'learn embedding':
        config['embed_features'] = dataset.embed_features
        config['n_embeddings'] = dataset.n_embeddings
    else:
        config['embed_features'] = None
        config['n_embeddings'] = None
    # Separate into train and validation sets
    train_dataloader, val_dataloader, _ = du.machine_learning.create_train_sets(dataset,
                                                                                train_indices=train_indices,
                                                                                val_indices=val_indices,
                                                                                test_indices=test_indices,
                                                                                test_train_ratio=config.get('test_train_ratio', None),
                                                                                validation_ratio=config.get('validation_ratio', None),
                                                                                batch_size=config.get('batch_size', 32),
                                                                                get_indices=False)
    print('DEBUG: Successfully created the dataloaders.')
    return train_dataloader, val_dataloader


def eICU_optimizer_creator(model, config):
    """Constructor of one or more Torch optimizers.

    Args:
        models: The return values from ``model_creator``. This can be one
            or more torch nn modules.
        config (dict): Configuration dictionary passed into ``TorchTrainer``.

    Returns:
        One or more Torch optimizer objects.
    """
    return torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.001))


def eICU_loss_creator(config):
    """Constructs the Torch Loss object.

    Note that optionally, you can pass in a Torch Loss constructor directly
    into the TorchTrainer (i.e., ``TorchTrainer(loss_creator=nn.BCELoss, ...)``).

    Args:
        config: Configuration dictionary passed into ``TorchTrainer``

    Returns:
        Torch Loss object.
    """
    return torch.nn.CrossEntropyLoss()


class eICU_Operator(TrainingOperator):
    def setup(self, config):
        # Number of RaySGD workers
        self.num_workers = config.get('num_workers', 1)
        # Fetch the Comet ML credentials
        self.comet_ml_api_key = config['comet_ml_api_key']
        self.comet_ml_project_name = config['comet_ml_project_name']
        self.comet_ml_workspace = config['comet_ml_workspace']
        self.log_comet_ml = config.get('log_comet_ml', True)
        self.comet_ml_save_model = config.get('comet_ml_save_model', True)
        # Fetch model and dataset parameters
        self.model_class = config.get('model', 'VanillaRNN')                    # Model class
        self.dataset_mode = config.get('dataset_mode', 'one hot encoded')       # The mode in which we'll use the data, either one hot encoded or pre-embedded
        self.ml_core = config.get('ml_core', 'deep learning')                   # The core machine learning type we'll use; either traditional ML or DL
        self.use_delta_ts = config.get('use_delta_ts', False)                   # Indicates if we'll use time variation info
        self.time_window_h = config.get('time_window_h', 48)                    # Number of hours on which we want to predict mortality
        # Additional properties and relevant training information
        self.step = 0                                                           # Number of iteration steps done so far
        self.print_every = config.get('print_every', 10)                        # Steps interval where the metrics are printed
        self.val_loss_min = np.inf                                              # Start with an infinitely big minimum validation loss
        self.clip_value = config.get('clip_value', 0.5)                         # Gradient clipping value, to avoid exploiding gradients
        self.features_list = config.get('features_list', None)                  # Names of the features being used in the current pipeline
        self.model_type = config.get('model_type', 'multivariate_rnn')          # Type of model to train
        self.padding_value = config.get('padding_value', 999999)                # Value to use in the padding, to fill the sequences
        self.cols_to_remove = config.get('cols_to_remove', [0, 1])              # List of indices of columns to remove from the features before feeding to the model
        self.is_custom = config.get('is_custom', False)                         # Specifies if the model being used is a custom built one
        self.already_embedded = config.get('already_embedded', False)           # Indicates if the categorical features are already embedded when fetching a batch
        self.batch_size = config.get('batch_size', 32)                          # The number of samples used in each training, validation or test iteration
        self.n_epochs = config.get('n_epochs', 1)                               # Number of epochs, i.e. the number of times to iterate through all of the training data
        self.lr = config.get('lr', 0.001)                                       # Learning rate
        self.models_path = config.get('models_path', '')                        # Path to the directory where the models are stored
        self.see_progress = config.get('see_progress', True)                    # Sets if a progress bar is shown for each training and validation loop
        # Register all the hyperparameters
        if self.num_workers == 1:
            model = self.model
        else:
            # Get the original model, as the current one is wrapped in DistributedDataParallel
            model = self.model.module
        model_args = inspect.getfullargspec(model.__init__).args[1:]
        self.hyper_params = dict([(param, getattr(model, param))
                                  for param in model_args])
        self.hyper_params.update({'batch_size': self.batch_size,
                                  'n_epochs': self.n_epochs,
                                  'learning_rate': self.lr})
        if self.log_comet_ml is True:
            # Create a new Comet.ml experiment
            self.experiment = Experiment(api_key=self.comet_ml_api_key,
                                         project_name=self.comet_ml_project_name,
                                         workspace=self.comet_ml_workspace,
                                         auto_param_logging=False,
                                         auto_metric_logging=False,
                                         auto_output_logging=False)
            self.experiment.log_other('completed', False)
            self.experiment.log_other('random_seed', du.random_seed)
            # Report hyperparameters to Comet.ml
            self.experiment.log_parameters(self.hyper_params)
            self.experiment.log_parameters(config)
            if self.features_list is not None:
                # Log the names of the features being used
                self.experiment.log_other('features_list', self.features_list)
        if self.clip_value is not None:
            # Set gradient clipping to avoid exploding gradients
            for p in self.model.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -self.clip_value, self.clip_value))

    def set_model_filename(self, val_loss):
        # Start with the model class name
        if self.model_class == 'VanillaRNN':
            model_filename = 'rnn'
        elif self.model_class == 'VanillaLSTM':
            model_filename = 'lstm'
        elif self.model_class == 'TLSTM':
            model_filename = 'tlstm'
        elif self.model_class == 'MF1LSTM':
            model_filename = 'mf1lstm'
        elif self.model_class == 'MF2LSTM':
            model_filename = 'mf2lstm'
        else:
            raise Exception(f'ERROR: {self.model_class} is an invalid model type. Please specify either "VanillaRNN", "VanillaLSTM", "TLSTM", "MF1LSTM" or "MF2LSTM".')
        # Add dataset mode information
        if self.dataset_mode == 'pre-embedded':
            model_filename = model_filename + '_pre_embedded'
        elif self.dataset_mode == 'learn embedding':
            model_filename = model_filename + '_with_embedding'
        elif self.dataset_mode == 'one hot encoded':
            model_filename = model_filename + '_one_hot_encoded'
        # Use of time variation information
        if self.use_delta_ts is not False and (self.model_class == 'VanillaRNN' or self.model_class == 'VanillaLSTM'):
            model_filename = model_filename + '_delta_ts'
        # Add the validation loss and timestamp
        current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
        model_filename = f'{val_loss:.4f}_valloss_{model_filename}_{current_datetime}.pth'
        return model_filename

    @override(TrainingOperator)
    def validate(self, val_iterator, info):
        # Number of iteration steps done so far
        step = info.get('step', 0)
        # Initialize the validation metrics
        val_loss = 0
        val_acc = 0
        val_auc = list()
        if self.num_workers == 1:
            model = self.model
        else:
            # Get the original model, as the current one is wrapped in DistributedDataParallel
            model = self.model.module
        if model.n_outputs > 1:
            val_auc_wgt = list()
        # Loop through the validation data
        for features, labels in du.utils.iterations_loop(val_iterator, see_progress=self.see_progress, desc='Val batches'):
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                if self.is_custom is False:
                    # Find the original sequence lengths
                    seq_lengths = du.search_explore.find_seq_len(labels, padding_value=self.padding_value)
                else:
                    # No need to find the sequence lengths now
                    seq_lengths = None
                if self.use_gpu is True:
                    # Move data to GPU
                    features, labels = features.to(self.device), labels.to(self.device)
                # Do inference on the data
                if self.model_type.lower() == 'multivariate_rnn':
                    (pred, correct_pred,
                     scores, labels, loss) = (du.deep_learning.inference_iter_multi_var_rnn(self.model, features, labels,
                                                                                            padding_value=self.padding_value,
                                                                                            cols_to_remove=self.cols_to_remove, is_train=False,
                                                                                            prob_output=True, is_custom=self.is_custom,
                                                                                            already_embedded=self.already_embedded,
                                                                                            seq_lengths=seq_lengths))
                elif self.model_type.lower() == 'mlp':
                    pred, correct_pred, scores, loss = (du.deep_learning.inference_iter_mlp(self.model, features, labels,
                                                                                            self.cols_to_remove, is_train=False,
                                                                                            prob_output=True))
                else:
                    raise Exception(f'ERROR: Invalid model type. It must be "multivariate_rnn" or "mlp", not {self.model_type}.')
                val_loss += loss                                                # Add the validation loss of the current batch
                val_acc += torch.mean(correct_pred.type(torch.FloatTensor))     # Add the validation accuracy of the current batch, ignoring all padding values
                if self.use_gpu is True:
                    # Move data to CPU for performance computations
                    scores, labels = scores.cpu(), labels.cpu()
                # Add the training ROC AUC of the current batch
                if model.n_outputs == 1:
                    try:
                        val_auc.append(roc_auc_score(labels.numpy(), scores.detach().numpy()))
                    except Exception as e:
                        warnings.warn(f'Couldn\'t calculate the validation AUC on step {step}. Received exception "{str(e)}".')
                else:
                    # It might happen that not all labels are present in the current batch;
                    # as such, we must focus on the ones that appear in the batch
                    labels_in_batch = labels.unique().long()
                    try:
                        val_auc.append(roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                                     multi_class='ovr', average='macro', labels=labels_in_batch.numpy()))
                        # Also calculate a weighted version of the AUC; important for imbalanced dataset
                        val_auc_wgt.append(roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                                         multi_class='ovr', average='weighted', labels=labels_in_batch.numpy()))
                    except Exception as e:
                        warnings.warn(f'Couldn\'t calculate the validation AUC on step {step}. Received exception "{str(e)}".')
                # Remove the current features and labels from memory
                del features
                del labels
        # Calculate the average of the metrics over the batches
        val_loss = val_loss / len(val_iterator)
        val_acc = val_acc / len(val_iterator)
        val_auc = np.mean(val_auc)
        if model.n_outputs > 1:
            val_auc_wgt = np.mean(val_auc_wgt)
        # Return the validation metrics
        metrics = dict(val_loss=val_loss,
                       val_acc=val_acc,
                       val_auc=val_auc)
        if model.n_outputs > 1:
            metrics['val_auc_wgt'] = val_auc_wgt
        return metrics

    @override(TrainingOperator)
    def train_epoch(self, iterator, info):
        if self.num_workers == 1:
            model = self.model
        else:
            # Get the original model, as the current one is wrapped in DistributedDataParallel
            model = self.model.module
        print(f'DEBUG: TrainingOperator attributes:\n{vars(self)}')
        print(f'DEBUG: Model\'s attributes:\n{vars(model)}')
        # Register the current epoch
        epoch = info.get('epoch_idx', 0)
        # Number of iteration steps done so far
        step = info.get('step', 0)
        # Initialize the training metrics
        train_loss = 0
        train_acc = 0
        train_auc = list()
        if model.n_outputs > 1:
            train_auc_wgt = list()
        # try:
        # Loop through the training data
        for features, labels in du.utils.iterations_loop(iterator, see_progress=self.see_progress, desc='Steps'):
            # Activate dropout to train the model
            self.model.train()
            # Clear the gradients of all optimized variables
            self.optimizer.zero_grad()
            if self.is_custom is False:
                # Find the original sequence lengths
                seq_lengths = du.search_explore.find_seq_len(labels, padding_value=self.padding_value)
            else:
                # No need to find the sequence lengths now
                seq_lengths = None
            if self.use_gpu is True:
                # Move data to GPU
                features, labels = features.to(self.device), labels.to(self.device)
            # Do inference on the data
            if self.model_type.lower() == 'multivariate_rnn':
                (pred, correct_pred,
                    scores, labels, loss) = (du.deep_learning.inference_iter_multi_var_rnn(self.model, features, labels,
                                                                                        padding_value=self.padding_value,
                                                                                        cols_to_remove=self.cols_to_remove, is_train=True,
                                                                                        prob_output=True, optimizer=self.optimizer,
                                                                                        is_custom=self.is_custom,
                                                                                        already_embedded=self.already_embedded,
                                                                                        seq_lengths=seq_lengths))
            elif self.model_type.lower() == 'mlp':
                pred, correct_pred, scores, loss = (du.deep_learning.inference_iter_mlp(self.model, features, labels,
                                                                                        self.cols_to_remove, is_train=True,
                                                                                        prob_output=True, optimizer=self.optimizer))
            else:
                raise Exception(f'ERROR: Invalid model type. It must be "multivariate_rnn" or "mlp", not {self.model_type}.')
            # Add the training loss and accuracy of the current batch
            train_loss += loss
            train_acc += torch.mean(correct_pred.type(torch.FloatTensor))
            if self.use_gpu is True:
                # Move data to CPU for performance computations
                scores, labels = scores.cpu(), labels.cpu()
            # Add the training ROC AUC of the current batch
            if model.n_outputs == 1:
                try:
                    train_auc.append(roc_auc_score(labels.numpy(), scores.detach().numpy()))
                except Exception as e:
                    warnings.warn(f'Couldn\'t calculate the training AUC on step {step}. Received exception "{str(e)}".')
            else:
                # It might happen that not all labels are present in the current batch;
                # as such, we must focus on the ones that appear in the batch
                labels_in_batch = labels.unique().long()
                try:
                    train_auc.append(roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                                    multi_class='ovr', average='macro', labels=labels_in_batch.numpy()))
                    # Also calculate a weighted version of the AUC; important for imbalanced dataset
                    train_auc_wgt.append(roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                                        multi_class='ovr', average='weighted', labels=labels_in_batch.numpy()))
                except Exception as e:
                    warnings.warn(f'Couldn\'t calculate the training AUC on step {step}. Received exception "{str(e)}".')
            # Count one more iteration step
            step += 1
            info['step'] = step
            # Deactivate dropout to test the model
            self.model.eval()
            # Remove the current features and labels from memory
            del features
            del labels
            # Run the current model on the validation set
            val_metrics = self.validate(self.validation_loader, info)
            # Display validation loss
            if step % self.print_every == 0:
                print(f'Epoch {epoch} step {step}: Validation loss: {val_metrics["val_loss"]}; Validation Accuracy: {val_metrics["val_acc"]}; Validation AUC: {val_metrics["val_auc"]}')
            # Check if the performance obtained in the validation set is the best so far (lowest loss value)
            if val_metrics['val_loss'] < val_loss_min:
                print(f'New minimum validation loss: {val_loss_min} -> {val_metrics["val_loss"]}.')
                # Update the minimum validation loss
                val_loss_min = val_metrics['val_loss']
                # Filename and path where the model will be saved
                model_filename = self.set_model_filename(val_metrics['val_loss'])
                print(f'Saving model in {model_filename}')
                # Save the best performing model so far, along with additional information to implement it
                checkpoint = self.hyper_params
                checkpoint['state_dict'] = self.model.state_dict()
                # [TODO] Check if this really works locally or if it just saves in the temporary nodes
                self.save(checkpoint, f'{self.models_path}{model_filename}')
                if self.log_comet_ml is True and self.comet_ml_save_model is True:
                    # Upload the model to Comet.ml
                    self.experiment.log_asset(file_data=model_filename, overwrite=True)
        # except Exception as e:
        #     warnings.warn(f'There was a problem doing training epoch {epoch}. Ending current epoch. Original exception message: "{str(e)}"')
        # try:
        # Calculate the average of the metrics over the epoch
        train_loss = train_loss / len(iterator)
        train_acc = train_acc / len(iterator)
        train_auc = np.mean(train_auc)
        if model.n_outputs > 1:
            train_auc_wgt = np.mean(train_auc_wgt)
        # Remove attached gradients so as to be able to print the values
        train_loss, val_loss = train_loss.detach(), val_metrics['val_loss'].detach()
        if self.use_gpu is True:
            # Move metrics data to CPU
            train_loss, val_loss = train_loss.cpu(), val_loss.cpu()
        if self.log_comet_ml is True:
            # Log metrics to Comet.ml
            self.experiment.log_metric('train_loss', train_loss, step=epoch)
            self.experiment.log_metric('train_acc', train_acc, step=epoch)
            self.experiment.log_metric('train_auc', train_auc, step=epoch)
            self.experiment.log_metric('val_loss', val_loss, step=epoch)
            self.experiment.log_metric('val_acc', val_metrics['val_acc'], step=epoch)
            self.experiment.log_metric('val_auc', val_metrics['val_auc'], step=epoch)
            self.experiment.log_metric('epoch', epoch)
            self.experiment.log_epoch_end(epoch, step=step)
            if model.n_outputs > 1:
                self.experiment.log_metric('train_auc_wgt', train_auc_wgt, step=epoch)
                self.experiment.log_metric('val_auc_wgt', val_metrics['val_auc_wgt'], step=epoch)
        # Print a report of the epoch
        print(f'Epoch {epoch}: Training loss: {train_loss}; Training Accuracy: {train_acc}; Training AUC: {train_auc}; \
                Validation loss: {val_loss}; Validation Accuracy: {val_metrics["val_acc"]}; Validation AUC: {val_metrics["val_auc"]}')
        print('----------------------')
        # except Exception as e:
        #     warnings.warn(f'There was a problem printing metrics from epoch {epoch}. Original exception message: "{str(e)}"')
        # Return the training metrics
        metrics = dict(train_loss=train_loss,
                       train_acc=train_acc,
                       train_auc=train_auc)
        if model.n_outputs > 1:
            metrics['train_auc_wgt'] = train_auc_wgt
        return metrics
