import pandas as pd                        # Pandas to load and handle the data
import torch                               # PyTorch to create and apply deep learning models
import data_utils as du                    # Data science and machine learning relevant methods

def eICU_initial_analysis(files, cat_feat_ohe):
    # Create a dictionary that will contain all the attributes to be added
    initial_info = dict()
    # Load a single, so as to retrieve some basic information
    df = pd.read_feather(files[0])
    # Number of input features, discarding the ID, timestamp and label columns
    initial_info['n_inputs'] = len(df.columns) - 3
    # Find the indeces of the features that will be embedded,
    # as well as the total number of categories per categorical feature
    # Subtracting 2 because of the ID and ts columns
    initial_info['embed_features'] = [[du.search_explore.find_col_idx(df, col)-2
                                       for col in feat_list]
                                       for feat_list in cat_feat_ohe]
    initial_info['n_embeddings'] = list()
    [initial_info['n_embeddings'].append(len(feat_list) + 1)
     for feat_list in cat_feat_ohe]
    return initial_info


def eICU_process_pipeline(df, time_window_h=48, use_delta_ts=False,
                          padding_value=999999):
    # Set the label column
    df['label'] = df.death_ts - df.ts <= time_window_h * 60
    # Fill the label's missing values with zero (i.e. considering as no death)
    df['label'] = df['label'].fillna(0)
    # Remove the now unneeded `death_ts` column
    df.drop(columns='death_ts', inplace=True)
    if use_delta_ts is not False:
        # Create a time variation column
        df['delta_ts'] = df.groupby('patientunitstayid').ts.diff()
    if use_delta_ts == 'normalized':
        # Normalize the time variation data
        # NOTE: When using the MF2-LSTM model, since it assumes that the time
        # variation is in minutes, we shouldn't normalize `delta_ts` with this model.
        df['delta_ts'] = (df['delta_ts'] - df['delta_ts'].mean()) / df['delta_ts'].std()
    # Find the label column's index number (i.e. it's place in the order of the columns)
    label_num = du.search_explore.find_col_idx(df, 'label')
    # Create a equence length dictionary
    self.seq_len_dict = du.padding.get_sequence_length_dict(df, id_column=id_column,
                                                            ts_column=ts_column)
    # Pad the data and convert it into a PyTorch tensor
    df = padding.dataframe_to_padded_tensor(df, seq_len_dict=self.seq_len_dict,
                                            id_column='patientunitstayid',
                                            ts_column='ts',
                                            padding_value=padding_value,
                                            inplace=True)
    # Check if we need to pre-embed the categorical features
    if self.embed_layers is not None and self.embed_features is not None:
        # Run each embedding layer on each respective feature, adding the
        # resulting embedding values to the tensor and removing the original,
        # categorical encoded columns
        x_t = embedding_bag_pipeline(x_t, self.embed_layers, self.embed_features,
                                     inplace=True)
    # Features
    x_t = df[:, :, :-label_num]
    # Labels
    y_t = df[:, :, label_num]
