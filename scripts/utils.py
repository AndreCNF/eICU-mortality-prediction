import pandas as pd                        # Pandas to load and handle the data
import torch                               # PyTorch to create and apply deep learning models
import data_utils as du                    # Data science and machine learning relevant methods

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
