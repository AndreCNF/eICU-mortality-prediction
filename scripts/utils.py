import pandas as pd                        # Pandas to load and handle the data
import torch                               # PyTorch to create and apply deep learning models
import data_utils as du                    # Data science and machine learning relevant methods
from . import Models                       # Deep learning models

def eICU_initial_analysis(self):
    # Load a single, so as to retrieve some basic information
    df = pd.read_feather(self.files[0])
    # Number of input features, discarding the ID, timestamp and label columns
    self.n_inputs = len(df.columns) - 3
    # Find the column indeces for the ID columns
    self.id_columns_idx = [du.search_explore.find_col_idx(df, col)
                           for col in ['patientunitstayid', 'ts']]
    if self.dataset_mode != 'one hot encoded':
        # Find the indeces of the features that will be embedded,
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
    labels = df[:, :, label_num]
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
    if config['model'] == 'VanillaRNN':
        return Models.VanillaRNN(config['n_inputs'], config['n_hidden'],
                                 config['n_outputs'], config['n_rnn_layers'],
                                 config['p_dropout'], bidir=config['bidir'],
                                 embed_features=config['embed_features'],
                                 n_embeddings=config['n_embeddings'],
                                 embedding_dim=config['embedding_dim'])
    elif config['model'] == 'VanillaLSTM':
        return Models.VanillaLSTM(config['n_inputs'], config['n_hidden'],
                                  config['n_outputs'], config['n_rnn_layers'],
                                  config['p_dropout'], bidir=config['bidir'],
                                  embed_features=config['embed_features'],
                                  n_embeddings=config['n_embeddings'],
                                  embedding_dim=config['embedding_dim'])
    elif config['model'] == 'TLSTM':
        return Models.TLSTM(config['n_inputs'], config['n_hidden'],
                            config['n_outputs'], config['n_rnn_layers'],
                            config['p_dropout'],
                            embed_features=config['embed_features'],
                            n_embeddings=config['n_embeddings'],
                            embedding_dim=config['embedding_dim'],
                            elapsed_time=config['elapsed_time'])
    elif config['model'] == 'MF1LSTM':
        return Models.MF1LSTM(config['n_inputs'], config['n_hidden'],
                              config['n_outputs'], config['n_rnn_layers'],
                              config['p_dropout'],
                              embed_features=config['embed_features'],
                              n_embeddings=config['n_embeddings'],
                              embedding_dim=config['embedding_dim'],
                              elapsed_time=config['elapsed_time'])
    elif config['model'] == 'MF2LSTM':
        return Models.MF2LSTM(config['n_inputs'], config['n_hidden'],
                              config['n_outputs'], config['n_rnn_layers'],
                              config['p_dropout'],
                              embed_features=config['embed_features'],
                              n_embeddings=config['n_embeddings'],
                              embedding_dim=config['embedding_dim'],
                              elapsed_time=config['elapsed_time'])
    else:
        raise Exception(f'ERROR: {config['model']} is an invalid model type. Please specify either "VanillaRNN", "VanillaLSTM", "TLSTM", "MF1LSTM" or "MF2LSTM".')


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
    data_path = config['data_path']
    # Load the data types
    stream_dtypes = open(f'{data_path}eICU_dtype_dict.yml', 'r')
    dtype_dict = yaml.load(stream_dtypes, Loader=yaml.FullLoader)
    # Load the one hot encoding columns categorization information
    stream_cat_feat_ohe = open(f'{data_path}eICU_cat_feat_ohe.yml', 'r')
    cat_feat_ohe = yaml.load(stream_cat_feat_ohe, Loader=yaml.FullLoader)
    # Define the dataset object
    dataset = du.datasets.Large_Dataset(files_name='eICU',
                                        process_pipeline=eICU_process_pipeline,
                                        id_column=config['id_column'],
                                        initial_analysis=eICU_initial_analysis,
                                        files_path=config['data_path'],
                                        dataset_mode=config['dataset_mode'],
                                        ml_core=config['ml_core'],
                                        use_delta_ts=config['use_delta_ts'],
                                        time_window_h=config['time_window_h'],
                                        padding_value=config['padding_value'],
                                        cat_feat_ohe=cat_feat_ohe,
                                        dtype_dict=dtype_dict)
    # Update the embedding information
    if dataset_mode == 'learn embedding':
        config['embed_features'] = dataset.embed_features
        config['n_embeddings'] = dataset.n_embeddings
    else:
        config['embed_features'] = None
        config['n_embeddings'] = None
    # Separate into train and validation sets
    train_dataloader, val_dataloader, _ = du.machine_learning.create_train_sets(dataset,
                                                                                test_train_ratio=config['test_train_ratio'],
                                                                                validation_ratio=config['validation_ratio'],
                                                                                batch_size=config['batch_size'],
                                                                                get_indeces=False)
    return train_loader, val_loader


def eICU_optimizer_creator(model, config):
    """Constructor of one or more Torch optimizers.

    Args:
        models: The return values from ``model_creator``. This can be one
            or more torch nn modules.
        config (dict): Configuration dictionary passed into ``TorchTrainer``.

    Returns:
        One or more Torch optimizer objects.
    """
    return torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-4))


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
