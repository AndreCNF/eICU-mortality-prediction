import torch                            # PyTorch to create and apply deep learning models
from torch import nn                    # nn for neural network layers
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.jit as jit                 # TorchScript for faster custom models
import math                             # Useful package for logarithm operations
import numpy as np                      # Mathematical operations package, allowing also for missing values representation
from functools import partial           # Fix some parameters of a function
import data_utils as du                 # Data science and machine learning relevant methods

class BaseRNN(nn.Module):
    def __init__(self, rnn_module, n_inputs, n_hidden, n_outputs, n_rnn_layers=1,
                 p_dropout=0, embed_features=None, n_embeddings=None,
                 embedding_dim=None, bidir=False, is_lstm=True,
                 padding_value=999999):
        '''A base RNN model, to use custom TorchScript modules, with
        the option to include embedding layers.

        nn.Parameters
        ----------
        rnn_module : nn.Module or jit.ScriptModule
            Recurrent neural network module to be used in this model.
        n_inputs : int
            Number of input features.
        n_hidden : int
            Number of hidden units.
        n_outputs : int
            Number of outputs.
        n_rnn_layers : int, default 1
            Number of RNN layers.
        p_dropout : float or int, default 0
            Probability of dropout.
        embed_features : list of ints or list of lists of ints, default None
            List of features (refered to by their indeces) that need to go
            through embedding layers. One list of one hot encoded feature per
            embedding layer must be set.
        n_embeddings : list of ints, default None
            List of the total number of unique categories for the embedding
            layers. Needs to be in the same order as the embedding layers are
            described in `embed_features`.
        embedding_dim : list of ints, default None
            List of embedding dimensions. Needs to be in the same order as the
            embedding layers are described in `embed_features`.
        bidir : bool, default False
            If set to True, the RNN model will be bidirectional (have hidden
            memory flowing both forward and backwards).
        is_lstm : bool, default True
            If set to True, it means that the provided model is of type (or at
            least a variant of) LSTM. This is important so as to know if the
            hidden state has two (h and c) or just one variable (h).
        padding_value : int or float, default 999999
            Value to use in the padding, to fill the sequences.
        '''
        super().__init__()
        self.rnn_module = rnn_module
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.n_rnn_layers = n_rnn_layers
        self.p_dropout = p_dropout
        self.embed_features = embed_features
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        self.is_lstm = is_lstm
        self.padding_value = padding_value
        # Embedding layers
        if self.embed_features is not None:
            if not isinstance(self.embed_features, list):
                raise Exception(f'ERROR: The embedding features must be indicated in `embed_features` as either a list of indices or a list of lists of indices (there are multiple one hot encoded columns for every original categorical feature). The provided argument has type {type(embed_features)}.')
            if self.n_embeddings is None:
                # Find the number of embeddings based on the number of one hot encoded feature
                if all([isinstance(feature, int) for feature in self.embed_features]):
                    self.n_embeddings = len(self.embed_features) + 1
                elif (all([isinstance(feat_list, list) for feat_list in self.embed_features])
                and all([isinstance(feature, int) for feat_list in self.embed_features
                         for feature in feat_list])):
                    self.n_embeddings = []
                    [self.n_embeddings.append(len(feat_list) + 1) for feat_list in self.embed_features]
                else:
                    raise Exception(f'ERROR: The embedding features must be indicated in `embed_features` as either a single, integer index or a list of indices. The provided argument has type {type(embed_features)}.')
            else:
                if all([isinstance(feature, int) for feature in self.embed_features]):
                    if self.n_embeddings != len(self.embed_features)+1:
                        raise Exception(f'ERROR: The number of embeddings `n_embeddings` must equal the length of its corresponding embedding features `embed_features` + 1 (missing values). The provided `n_embeddings` is {self.n_embeddings} while `embed_features` has length {len(self.embed_features)}.')
                elif (all([isinstance(feat_list, list) for feat_list in self.embed_features])
                and all([isinstance(feature, int) for feat_list in self.embed_features
                         for feature in feat_list])):
                    if len(self.n_embeddings) != len(self.embed_features):
                        raise Exception(f'ERROR: The list of the number of embeddings `n_embeddings` and the embedding features `embed_features` must have the same length. The provided `n_embeddings` has length {len(self.n_embeddings)} while `embed_features` has length {len(self.embed_features)}.')
                    for i in range(len(self.n_embeddings)):
                        if self.n_embeddings[i] != len(self.embed_features[i])+1:
                            raise Exception(f'ERROR: The number of embeddings `n_embeddings` must equal the length of its corresponding embedding features `embed_features` + 1 (missing values). The provided `n_embeddings` is {self.n_embeddings[i]} while `embed_features` has length {len(self.embed_features[i])}, in embedding features set {i}.')
            if all([isinstance(feature, int) for feature in self.embed_features]):
                if embedding_dim is None:
                    # Calculate a reasonable embedding dimension for the
                    # current feature; the formula sets a minimum embedding
                    # dimension of 3, with above values being calculated as
                    # the rounded up base 5 logarithm of the number of
                    # embeddings.
                    embedding_dim = max(3, int(math.ceil(math.log(self.n_embeddings, base=5))))
                # Create a single embedding layer
                self.embed_layers = nn.EmbeddingBag(self.n_embeddings, embedding_dim)
            elif (all([isinstance(feat_list, list) for feat_list in self.embed_features])
            and all([isinstance(feature, int) for feat_list in self.embed_features
                     for feature in feat_list])):
                # Create a modules list of embedding bag layers
                self.embed_layers = nn.ModuleList()
                for i in range(len(self.embed_features)):
                    if embedding_dim is None:
                        # Calculate a reasonable embedding dimension for the
                        # current feature; the formula sets a minimum embedding
                        # dimension of 3, with above values being calculated as
                        # the rounded up base 5 logarithm of the number of
                        # embeddings.
                        embedding_dim_i = max(3, int(math.ceil(math.log(self.n_embeddings[i], base=5))))
                    else:
                        embedding_dim_i = self.embedding_dim[i]
                    # Create an embedding layer for the current feature
                    self.embed_layers.append(nn.EmbeddingBag(self.n_embeddings[i], embedding_dim_i))
            else:
                raise Exception(f'ERROR: The embedding features must be indicated in `embed_features` as either a single, integer index or a list of indices. The provided argument has type {type(embed_features)}.')
        # RNN layer(s)
        if self.embed_features is None:
            self.rnn_n_inputs = self.n_inputs
        else:
            # Have into account the new embedding columns that will be added, as
            # well as the removal of the originating categorical columns
            if all([isinstance(feature, int) for feature in self.embed_features]):
                self.rnn_n_inputs = self.n_inputs + self.embedding_dim - len(self.embed_features)
            elif (all([isinstance(feat_list, list) for feat_list in self.embed_features])
            and all([isinstance(feature, int) for feat_list in self.embed_features
                     for feature in feat_list])):
                self.rnn_n_inputs = self.n_inputs
                for i in range(len(self.embed_features)):
                    self.rnn_n_inputs = self.rnn_n_inputs + self.embedding_dim[i] - len(self.embed_features[i])
        if self.n_rnn_layers == 1:
            # Create a single RNN layer
            self.rnn_layer = self.rnn_module(self.rnn_n_inputs, self.n_hidden)
            # The output dimension of the last RNN layer
            rnn_output_dim = self.n_hidden
        else:
            # Create a list of multiple, stacked RNN layers
            self.rnn_layers = nn.ModuleList()
            # Add the first RNN layer
            self.rnn_layers.append(self.rnn_module(self.rnn_n_inputs, self.n_hidden))
            # Add the remaining RNN layers
            for i in range(1, self.n_rnn_layers):
                self.rnn_layers.append(self.rnn_module(self.n_hidden, self.n_hidden))
            # The output dimension of the last RNN layer
            rnn_output_dim = self.n_hidden
        # Fully connected layer which takes the RNN's hidden units and
        # calculates the output classification
        self.fc = nn.Linear(rnn_output_dim, self.n_outputs)
        # Dropout used between the last RNN layer and the fully connected layer
        self.dropout = nn.Dropout(p=self.p_dropout)
        if self.n_outputs == 1:
            # Use the sigmoid activation function
            self.activation = nn.Sigmoid()
            # Use the binary cross entropy function
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            # Use the sigmoid activation function
            self.activation = nn.Softmax()
            # Use the binary cross entropy function
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, hidden_state=None, get_hidden_state=False,
                prob_output=True, already_embedded=False):
        if self.embed_features is not None and already_embedded is False:
            # Run each embedding layer on each respective feature, adding the
            # resulting embedding values to the tensor and removing the original,
            # categorical encoded columns
            x = du.embedding.embedding_bag_pipeline(x, self.embed_layers, self.embed_features,
                                                    model_forward=True, inplace=True)
        # Make sure that the input data is of type float
        x = x.float()
        # Get the batch size (might not be always the same)
        batch_size = x.shape[0]
        if hidden_state is None:
            # Reset the LSTM hidden state. Must be done before you run a new
            # batch. Otherwise the LSTM will treat a new batch as a continuation
            # of a sequence.
            self.hidden = self.init_hidden(batch_size)
        else:
            # Use the specified hidden state
            self.hidden = hidden_state
        # Get the outputs and hidden states from the RNN layer(s)
        if self.n_rnn_layers == 1:
            if self.bidir is False:
                # Since there's only one layer and the model is not bidirectional,
                # we only need one set of hidden state
                if self.is_lstm is True:
                    hidden_state = (self.hidden[0][0], self.hidden[1][0])
                else:
                    hidden_state = self.hidden[0]
            rnn_output, self.hidden = self.rnn_layer(x, hidden_state)
        else:
            # List[RNNState]: One state per layer
            # output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
            if self.is_lstm is True:
                output_states = (torch.zeros(self.hidden[0].shape), torch.zeros(self.hidden[1].shape))
            else:
                output_states = torch.zeros(self.hidden.shape)
            i = 0
            # The first RNN layer's input is the original input;
            # the following layers will use their respective previous layer's
            # output as input
            rnn_output = x
            for rnn_layer in self.rnn_layers:
                if self.is_lstm is True:
                    hidden_state = (self.hidden[0][i], self.hidden[1][i])
                else:
                    hidden_state = self.hidden[i]
                rnn_output, out_state = rnn_layer(rnn_output, hidden_state)
                # Apply the dropout layer except the last layer
                if i < self.n_rnn_layers - 1:
                    rnn_output = self.dropout(rnn_output)
                if self.is_lstm is True:
                    output_states[0][i] = out_state[0]
                    output_states[1][i] = out_state[1]
                else:
                    output_states[i] = [out_state]
                i += 1
            # Update the hidden states variable
            self.hidden = output_states
        # Apply dropout to the last RNN layer
        # [TODO] Consider if it makes sense to add dropout to the last RNN layer
        # rnn_output = self.dropout(rnn_output)
        # Flatten RNN output to fit into the fully connected layer
        flat_rnn_output = rnn_output.contiguous().view(-1, self.n_hidden)
        # Apply the final fully connected layer
        output = self.fc(flat_rnn_output)
        if prob_output is True:
            # Get the outputs in the form of probabilities
            if self.n_outputs == 1:
                output = self.activation(output)
            else:
                # Normalize outputs on their last dimension
                output = self.activation(output, dim=len(output.shape)-1)
        if get_hidden_state is True:
            return output, self.hidden
        else:
            return output

    def loss(self, y_pred, y_labels):
        # Flatten the data
        y_pred = y_pred.reshape(-1)
        y_labels = y_labels.reshape(-1)
        # Find the indeces that don't correspond to padding samples
        non_pad_idx = y_labels != self.padding_value
        # Remove the padding samples
        y_labels = y_labels[non_pad_idx]
        y_pred = y_pred[non_pad_idx]
        # Compute cross entropy loss which ignores all padding values
        ce_loss = self.criterion(y_pred, y_labels)
        return ce_loss

    def init_hidden(self, batch_size):
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # Check if GPU is available
        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu is True:
            hidden = (weight.new(self.n_rnn_layers * (1 + self.bidir), batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_rnn_layers * (1 + self.bidir), batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_rnn_layers * (1 + self.bidir), batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_rnn_layers * (1 + self.bidir), batch_size, self.n_hidden).zero_())
        return hidden


class VanillaRNN(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_rnn_layers=1, p_dropout=0,
                 embed_features=None, n_embeddings=None, embedding_dim=None,
                 bidir=False, padding_value=999999, total_length=None):
        '''A vanilla RNN model, using PyTorch's predefined RNN module, with
        the option to include embedding layers.
        Parameters
        ----------
        n_inputs : int
            Number of input features.
        n_hidden : int
            Number of hidden units.
        n_outputs : int
            Number of outputs.
        n_rnn_layers : int, default 1
            Number of RNN layers.
        p_dropout : float or int, default 0
            Probability of dropout.
        embed_features : list of ints, default None
            List of features (refered to by their indeces) that need to go
            through embedding layers.
        n_embeddings : list of ints, default None
            List of the total number of unique categories for the embedding
            layers. Needs to be in the same order as the embedding layers are
            described in `embed_features`.
        embedding_dim : list of ints, default None
            List of embedding dimensions. Needs to be in the same order as the
            embedding layers are described in `embed_features`.
        bidir : bool, default False
            If set to True, the RNN model will be bidirectional (have hidden
            memory flowing both forward and backwards).
        padding_value : int or float, default 999999
            Value to use in the padding, to fill the sequences.
        total_length : int, default None
            If not None, the output will be padded to have length total_length.
            This method will throw ValueError if total_length is less than the
            max sequence length in sequence.
        '''
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.n_rnn_layers = n_rnn_layers
        self.p_dropout = p_dropout
        self.embed_features = embed_features
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        self.padding_value = padding_value
        self.total_length = total_length
        # Embedding layers
        if self.embed_features is not None:
            if not isinstance(self.embed_features, list):
                raise Exception(f'ERROR: The embedding features must be indicated in `embed_features` as either a list of indices or a list of lists of indices (there are multiple one hot encoded columns for every original categorical feature). The provided argument has type {type(embed_features)}.')
            if self.n_embeddings is None:
                # Find the number of embeddings based on the number of one hot encoded feature
                if all([isinstance(feature, int) for feature in self.embed_features]):
                    self.n_embeddings = len(self.embed_features) + 1
                elif (all([isinstance(feat_list, list) for feat_list in self.embed_features])
                and all([isinstance(feature, int) for feat_list in self.embed_features
                         for feature in feat_list])):
                    self.n_embeddings = []
                    [self.n_embeddings.append(len(feat_list) + 1) for feat_list in self.embed_features]
                else:
                    raise Exception(f'ERROR: The embedding features must be indicated in `embed_features` as either a single, integer index or a list of indices. The provided argument has type {type(embed_features)}.')
            else:
                if isinstance(self.n_embeddings, int):
                    self.n_embeddings = [self.n_embeddings]
                if all([isinstance(feature, int) for feature in self.embed_features]):
                    if self.n_embeddings != len(self.embed_features)+1:
                        raise Exception(f'ERROR: The number of embeddings `n_embeddings` must equal the length of its corresponding embedding features `embed_features` + 1 (missing values). The provided `n_embeddings` is {self.n_embeddings} while `embed_features` has length {len(self.embed_features)}.')
                elif (all([isinstance(feat_list, list) for feat_list in self.embed_features])
                and all([isinstance(feature, int) for feat_list in self.embed_features
                         for feature in feat_list])):
                    if len(self.n_embeddings) != len(self.embed_features):
                        raise Exception(f'ERROR: The list of the number of embeddings `n_embeddings` and the embedding features `embed_features` must have the same length. The provided `n_embeddings` has length {len(self.n_embeddings)} while `embed_features` has length {len(self.embed_features)}.')
                    for i in range(len(self.n_embeddings)):
                        if self.n_embeddings[i] != len(self.embed_features[i])+1:
                            raise Exception(f'ERROR: The number of embeddings `n_embeddings` must equal the length of its corresponding embedding features `embed_features` + 1 (missing values). The provided `n_embeddings` is {self.n_embeddings[i]} while `embed_features` has length {len(self.embed_features[i])}, in embedding features set {i}.')
            if all([isinstance(feature, int) for feature in self.embed_features]):
                if embedding_dim is None:
                    # Calculate a reasonable embedding dimension for the
                    # current feature; the formula sets a minimum embedding
                    # dimension of 3, with above values being calculated as
                    # the rounded up base 5 logarithm of the number of
                    # embeddings.
                    embedding_dim = max(3, int(math.ceil(math.log(self.n_embeddings, base=5))))
                # Create a single embedding layer
                self.embed_layers = nn.EmbeddingBag(self.n_embeddings, embedding_dim)
            elif (all([isinstance(feat_list, list) for feat_list in self.embed_features])
            and all([isinstance(feature, int) for feat_list in self.embed_features
                     for feature in feat_list])):
                # Create a modules list of embedding bag layers
                self.embed_layers = nn.ModuleList()
                for i in range(len(self.embed_features)):
                    if embedding_dim is None:
                        # Calculate a reasonable embedding dimension for the
                        # current feature; the formula sets a minimum embedding
                        # dimension of 3, with above values being calculated as
                        # the rounded up base 5 logarithm of the number of
                        # embeddings.
                        embedding_dim_i = max(3, int(math.ceil(math.log(self.n_embeddings[i], base=5))))
                    else:
                        embedding_dim_i = self.embedding_dim[i]
                    # Create an embedding layer for the current feature
                    self.embed_layers.append(nn.EmbeddingBag(self.n_embeddings[i], embedding_dim_i))
            else:
                raise Exception(f'ERROR: The embedding features must be indicated in `embed_features` as either a single, integer index or a list of indices. The provided argument has type {type(embed_features)}.')
        # RNN layer(s)
        if self.embed_features is None:
            self.rnn_n_inputs = self.n_inputs
        else:
            # Have into account the new embedding columns that will be added, as
            # well as the removal of the originating categorical columns
            if all([isinstance(feature, int) for feature in self.embed_features]):
                self.rnn_n_inputs = self.n_inputs + self.embedding_dim - len(self.embed_features)
            elif (all([isinstance(feat_list, list) for feat_list in self.embed_features])
            and all([isinstance(feature, int) for feat_list in self.embed_features
                     for feature in feat_list])):
                self.rnn_n_inputs = self.n_inputs
                for i in range(len(self.embed_features)):
                    self.rnn_n_inputs = self.rnn_n_inputs + self.embedding_dim[i] - len(self.embed_features[i])
        self.rnn = nn.RNN(self.rnn_n_inputs, self.n_hidden, self.n_rnn_layers,
                           batch_first=True, dropout=self.p_dropout,
                           bidirectional=self.bidir)
        # Fully connected layer which takes the RNN's hidden units and
        # calculates the output classification
        self.fc = nn.Linear(self.n_hidden, self.n_outputs)
        # Dropout used between the last RNN layer and the fully connected layer
        self.dropout = nn.Dropout(p=self.p_dropout)
        if self.n_outputs == 1:
            # Use the sigmoid activation function
            self.activation = nn.Sigmoid()
            # Use the binary cross entropy function
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            # Use the sigmoid activation function
            self.activation = nn.Softmax()
            # Use the binary cross entropy function
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, hidden_state=None, x_lengths=None, get_hidden_state=False,
                prob_output=True, already_embedded=False):
        if self.embed_features is not None and already_embedded is False:
            # Run each embedding layer on each respective feature, adding the
            # resulting embedding values to the tensor and removing the original,
            # categorical encoded columns
            x = du.embedding.embedding_bag_pipeline(x, self.embed_layers, self.embed_features,
                                                    model_forward=True, inplace=True)
        # Make sure that the input data is of type float
        x = x.float()
        # Get the batch size (might not be always the same)
        batch_size = x.shape[0]
        if hidden_state is None:
            # Reset the RNN hidden state. Must be done before you run a new
            # batch. Otherwise the RNN will treat a new batch as a continuation
            # of a sequence.
            self.hidden = self.init_hidden(batch_size)
        else:
            # Use the specified hidden state
            self.hidden = hidden_state
        if x_lengths is not None:
            # pack_padded_sequence so that padded items in the sequence won't be
            # shown to the RNN
            x = pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        # Get the outputs and hidden states from the RNN layer(s)
        rnn_output, self.hidden = self.rnn(x, self.hidden)
        if x_lengths is not None:
            # Undo the packing operation
            rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True,
                                                total_length=self.total_length)
        # Apply dropout to the last RNN layer
        rnn_output = self.dropout(rnn_output)
        # Flatten RNN output to fit into the fully connected layer
        flat_rnn_output = rnn_output.contiguous().view(-1, self.n_hidden)
        # Apply the final fully connected layer
        output = self.fc(flat_rnn_output)
        if prob_output is True:
            # Get the outputs in the form of probabilities
            if self.n_outputs == 1:
                output = self.activation(output)
            else:
                # Normalize outputs on their last dimension
                output = self.activation(output, dim=len(output.shape)-1)
        if get_hidden_state is True:
            return output, self.hidden
        else:
            return output

    def loss(self, y_pred, y_labels):
        # Flatten the data
        y_pred = y_pred.reshape(-1)
        y_labels = y_labels.reshape(-1)
        # Find the indeces that don't correspond to padding samples
        non_pad_idx = y_labels != self.padding_value
        # Remove the padding samples
        y_labels = y_labels[non_pad_idx]
        y_pred = y_pred[non_pad_idx]
        # Compute cross entropy loss which ignores all padding values
        ce_loss = self.criterion(y_pred, y_labels)
        return ce_loss

    def init_hidden(self, batch_size):
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of RNN
        weight = next(self.parameters()).data
        # Check if GPU is available
        train_on_gpu = torch.cuda.is_available()
        hidden = weight.new(self.n_rnn_layers * (1 + self.bidir), batch_size, self.n_hidden).zero_()
        if train_on_gpu is True:
            hidden = hidden.cuda()
        return hidden


class VanillaLSTM(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_lstm_layers=1, p_dropout=0,
                 embed_features=None, n_embeddings=None, embedding_dim=None,
                 bidir=False, padding_value=999999, total_length=None):
        '''A vanilla LSTM model, using PyTorch's predefined LSTM module, with
        the option to include embedding layers.
        Parameters
        ----------
        n_inputs : int
            Number of input features.
        n_hidden : int
            Number of hidden units.
        n_outputs : int
            Number of outputs.
        n_lstm_layers : int, default 1
            Number of LSTM layers.
        p_dropout : float or int, default 0
            Probability of dropout.
        embed_features : list of ints, default None
            List of features (refered to by their indeces) that need to go
            through embedding layers.
        n_embeddings : list of ints, default None
            List of the total number of unique categories for the embedding
            layers. Needs to be in the same order as the embedding layers are
            described in `embed_features`.
        embedding_dim : list of ints, default None
            List of embedding dimensions. Needs to be in the same order as the
            embedding layers are described in `embed_features`.
        bidir : bool, default False
            If set to True, the LSTM model will be bidirectional (have hidden
            memory flowing both forward and backwards).
        padding_value : int or float, default 999999
            Value to use in the padding, to fill the sequences.
        total_length : int, default None
            If not None, the output will be padded to have length total_length.
            This method will throw ValueError if total_length is less than the
            max sequence length in sequence.
        '''
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.n_lstm_layers = n_lstm_layers
        self.p_dropout = p_dropout
        self.embed_features = embed_features
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        self.padding_value = padding_value
        self.total_length = total_length
        # Embedding layers
        if self.embed_features is not None:
            if not isinstance(self.embed_features, list):
                raise Exception(f'ERROR: The embedding features must be indicated in `embed_features` as either a list of indices or a list of lists of indices (there are multiple one hot encoded columns for every original categorical feature). The provided argument has type {type(embed_features)}.')
            if self.n_embeddings is None:
                # Find the number of embeddings based on the number of one hot encoded feature
                if all([isinstance(feature, int) for feature in self.embed_features]):
                    self.n_embeddings = len(self.embed_features) + 1
                elif (all([isinstance(feat_list, list) for feat_list in self.embed_features])
                and all([isinstance(feature, int) for feat_list in self.embed_features
                         for feature in feat_list])):
                    self.n_embeddings = []
                    [self.n_embeddings.append(len(feat_list) + 1) for feat_list in self.embed_features]
                else:
                    raise Exception(f'ERROR: The embedding features must be indicated in `embed_features` as either a single, integer index or a list of indices. The provided argument has type {type(embed_features)}.')
            else:
                if isinstance(self.n_embeddings, int):
                    self.n_embeddings = [self.n_embeddings]
                if all([isinstance(feature, int) for feature in self.embed_features]):
                    if self.n_embeddings != len(self.embed_features)+1:
                        raise Exception(f'ERROR: The number of embeddings `n_embeddings` must equal the length of its corresponding embedding features `embed_features` + 1 (missing values). The provided `n_embeddings` is {self.n_embeddings} while `embed_features` has length {len(self.embed_features)}.')
                elif (all([isinstance(feat_list, list) for feat_list in self.embed_features])
                and all([isinstance(feature, int) for feat_list in self.embed_features
                         for feature in feat_list])):
                    if len(self.n_embeddings) != len(self.embed_features):
                        raise Exception(f'ERROR: The list of the number of embeddings `n_embeddings` and the embedding features `embed_features` must have the same length. The provided `n_embeddings` has length {len(self.n_embeddings)} while `embed_features` has length {len(self.embed_features)}.')
                    for i in range(len(self.n_embeddings)):
                        if self.n_embeddings[i] != len(self.embed_features[i])+1:
                            raise Exception(f'ERROR: The number of embeddings `n_embeddings` must equal the length of its corresponding embedding features `embed_features` + 1 (missing values). The provided `n_embeddings` is {self.n_embeddings[i]} while `embed_features` has length {len(self.embed_features[i])}, in embedding features set {i}.')
            if all([isinstance(feature, int) for feature in self.embed_features]):
                if embedding_dim is None:
                    # Calculate a reasonable embedding dimension for the
                    # current feature; the formula sets a minimum embedding
                    # dimension of 3, with above values being calculated as
                    # the rounded up base 5 logarithm of the number of
                    # embeddings.
                    embedding_dim = max(3, int(math.ceil(math.log(self.n_embeddings, base=5))))
                # Create a single embedding layer
                self.embed_layers = nn.EmbeddingBag(self.n_embeddings, embedding_dim)
            elif (all([isinstance(feat_list, list) for feat_list in self.embed_features])
            and all([isinstance(feature, int) for feat_list in self.embed_features
                     for feature in feat_list])):
                # Create a modules list of embedding bag layers
                self.embed_layers = nn.ModuleList()
                for i in range(len(self.embed_features)):
                    if embedding_dim is None:
                        # Calculate a reasonable embedding dimension for the
                        # current feature; the formula sets a minimum embedding
                        # dimension of 3, with above values being calculated as
                        # the rounded up base 5 logarithm of the number of
                        # embeddings.
                        embedding_dim_i = max(3, int(math.ceil(math.log(self.n_embeddings[i], base=5))))
                    else:
                        embedding_dim_i = self.embedding_dim[i]
                    # Create an embedding layer for the current feature
                    self.embed_layers.append(nn.EmbeddingBag(self.n_embeddings[i], embedding_dim_i))
            else:
                raise Exception(f'ERROR: The embedding features must be indicated in `embed_features` as either a single, integer index or a list of indices. The provided argument has type {type(embed_features)}.')
        # LSTM layer(s)
        if self.embed_features is None:
            self.lstm_n_inputs = self.n_inputs
        else:
            # Have into account the new embedding columns that will be added, as
            # well as the removal of the originating categorical columns
            if all([isinstance(feature, int) for feature in self.embed_features]):
                self.lstm_n_inputs = self.n_inputs + self.embedding_dim - len(self.embed_features)
            elif (all([isinstance(feat_list, list) for feat_list in self.embed_features])
            and all([isinstance(feature, int) for feat_list in self.embed_features
                     for feature in feat_list])):
                self.lstm_n_inputs = self.n_inputs
                for i in range(len(self.embed_features)):
                    self.lstm_n_inputs = self.lstm_n_inputs + self.embedding_dim[i] - len(self.embed_features[i])
        self.lstm = nn.LSTM(self.lstm_n_inputs, self.n_hidden, self.n_lstm_layers,
                            batch_first=True, dropout=self.p_dropout,
                            bidirectional=self.bidir)
        # Fully connected layer which takes the LSTM's hidden units and
        # calculates the output classification
        self.fc = nn.Linear(self.n_hidden, self.n_outputs)
        # Dropout used between the last LSTM layer and the fully connected layer
        self.dropout = nn.Dropout(p=self.p_dropout)
        if self.n_outputs == 1:
            # Use the sigmoid activation function
            self.activation = nn.Sigmoid()
            # Use the binary cross entropy function
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            # Use the sigmoid activation function
            self.activation = nn.Softmax()
            # Use the binary cross entropy function
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, hidden_state=None, x_lengths=None, get_hidden_state=False,
                prob_output=True, already_embedded=False):
        if self.embed_features is not None and already_embedded is False:
            # Run each embedding layer on each respective feature, adding the
            # resulting embedding values to the tensor and removing the original,
            # categorical encoded columns
            x = du.embedding.embedding_bag_pipeline(x, self.embed_layers, self.embed_features,
                                                    model_forward=True, inplace=True)
        # Make sure that the input data is of type float
        x = x.float()
        # Get the batch size (might not be always the same)
        batch_size = x.shape[0]
        if hidden_state is None:
            # Reset the LSTM hidden state. Must be done before you run a new
            # batch. Otherwise the LSTM will treat a new batch as a continuation
            # of a sequence.
            self.hidden = self.init_hidden(batch_size)
        else:
            # Use the specified hidden state
            self.hidden = hidden_state
        if x_lengths is not None:
            # pack_padded_sequence so that padded items in the sequence won't be
            # shown to the LSTM
            x = pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        # Get the outputs and hidden states from the LSTM layer(s)
        lstm_output, self.hidden = self.lstm(x, self.hidden)
        if x_lengths is not None:
            # Undo the packing operation
            lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True,
                                                 total_length=self.total_length)
        # Apply dropout to the last LSTM layer
        lstm_output = self.dropout(lstm_output)
        # Flatten LSTM output to fit into the fully connected layer
        flat_lstm_output = lstm_output.contiguous().view(-1, self.n_hidden)
        # Apply the final fully connected layer
        output = self.fc(flat_lstm_output)
        if prob_output is True:
            # Get the outputs in the form of probabilities
            if self.n_outputs == 1:
                output = self.activation(output)
            else:
                # Normalize outputs on their last dimension
                output = self.activation(output, dim=len(output.shape)-1)
        if get_hidden_state is True:
            return output, self.hidden
        else:
            return output

    def loss(self, y_pred, y_labels):
        # Flatten the data
        y_pred = y_pred.reshape(-1)
        y_labels = y_labels.reshape(-1)
        # Find the indeces that don't correspond to padding samples
        non_pad_idx = y_labels != self.padding_value
        # Remove the padding samples
        y_labels = y_labels[non_pad_idx]
        y_pred = y_pred[non_pad_idx]
        # Compute cross entropy loss which ignores all padding values
        ce_loss = self.criterion(y_pred, y_labels)
        return ce_loss

    def init_hidden(self, batch_size):
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # Check if GPU is available
        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu is True:
            hidden = (weight.new(self.n_lstm_layers * (1 + self.bidir), batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_lstm_layers * (1 + self.bidir), batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_lstm_layers * (1 + self.bidir), batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_lstm_layers * (1 + self.bidir), batch_size, self.n_hidden).zero_())
        return hidden


class CustomLSTM(BaseRNN):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_lstm_layers=1, p_dropout=0,
                 embed_features=None, n_embeddings=None, embedding_dim=None,
                 bidir=False, padding_value=999999):
        if bidir is True:
            rnn_module = lambda *cell_args: BidirLSTMLayer(LSTMCell, *cell_args)
        else:
            rnn_module = lambda *cell_args: LSTMLayer(LSTMCell, *cell_args)
        super(CustomLSTM, self).__init__(rnn_module=rnn_module, n_inputs=n_inputs,
                                         n_hidden=n_hidden, n_outputs=n_outputs,
                                         n_lstm_layers=n_lstm_layers, p_dropout=p_dropout,
                                         embed_features=embed_features,
                                         n_embeddings=n_embeddings,
                                         embedding_dim=embedding_dim,
                                         bidir=bidir, is_lstm=True,
                                         padding_value=padding_value)


class TLSTM(BaseRNN):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_rnn_layers=1, p_dropout=0,
                 embed_features=None, n_embeddings=None, embedding_dim=None,
                 bidir=False, padding_value=999999,
                 delta_ts_col=None, elapsed_time='small', no_small_delta=True):
        if delta_ts_col is None:
            if embed_features is None:
                self.delta_ts_col = n_inputs
            else:
                # Have into account the new embedding columns that will be added,
                # as well as the removal of the originating categorical columns
                # NOTE: This only works assuming that the delta_ts column is the
                # last one on the dataframe, standing to the left of all the
                # embedding features
                if all([isinstance(feature, int) for feature in embed_features]):
                    self.delta_ts_col = n_inputs - len(embed_features)
                elif (all([isinstance(feat_list, list) for feat_list in embed_features])
                and all([isinstance(feature, int) for feat_list in embed_features
                         for feature in feat_list])):
                    self.delta_ts_col = n_inputs
                    for i in range(len(embed_features)):
                        self.delta_ts_col = rnn_n_inputs - len(embed_features[i])

        else:
            self.delta_ts_col = delta_ts_col
        self.elapsed_time = elapsed_time
        self.no_small_delta = no_small_delta
        TLSTMCell_prtl = partial(TLSTMCell, delta_ts_col=self.delta_ts_col,
                                 elapsed_time=self.elapsed_time,
                                 no_small_delta=self.no_small_delta)
        if bidir is True:
            rnn_module = lambda *cell_args: BidirTLSTMLayer(TLSTMCell_prtl, *cell_args)
        else:
            rnn_module = lambda *cell_args: TLSTMLayer(TLSTMCell_prtl, *cell_args)
        super(TLSTM, self).__init__(rnn_module=rnn_module, n_inputs=n_inputs,
                                    n_hidden=n_hidden, n_outputs=n_outputs,
                                    n_rnn_layers=n_rnn_layers,
                                    p_dropout=p_dropout,
                                    embed_features=embed_features,
                                    n_embeddings=n_embeddings,
                                    embedding_dim=embedding_dim,
                                    bidir=bidir, is_lstm=True,
                                    padding_value=padding_value)

    def forward(self, x, hidden_state=None, get_hidden_state=False,
                prob_output=True, already_embedded=False):
        if self.embed_features is not None and already_embedded is False:
            # Run each embedding layer on each respective feature, adding the
            # resulting embedding values to the tensor and removing the original,
            # categorical encoded columns
            x = du.embedding.embedding_bag_pipeline(x, self.embed_layers, self.embed_features,
                                                    model_forward=True, inplace=True)
        # Make sure that the input data is of type float
        x = x.float()
        # Get the batch size (might not be always the same)
        batch_size = x.shape[0]
        # Isolate the delta_ts feature
        delta_ts = x[:, :, self.delta_ts_col].clone()
        left_to_delta = x[:, :, :self.delta_ts_col]
        right_to_delta = x[:, :, self.delta_ts_col+1:]
        x = torch.cat([left_to_delta, right_to_delta], 2)
        if hidden_state is None:
            # Reset the LSTM hidden state. Must be done before you run a new
            # batch. Otherwise the LSTM will treat a new batch as a continuation
            # of a sequence.
            self.hidden = self.init_hidden(batch_size)
        else:
            # Use the specified hidden state
            self.hidden = hidden_state
        # Make sure that the data is input in the format of (timestamp x sample x features)
        x = x.permute(1, 0, 2)
        # Get the outputs and hidden states from the RNN layer(s)
        if self.n_rnn_layers == 1:
            if self.bidir is False:
                # Since there's only one layer and the model is not bidirectional,
                # we only need one set of hidden state
                hidden_state = (self.hidden[0][0], self.hidden[1][0])
            # Run the RNN layer on the data
            rnn_output, self.hidden = self.rnn_layer(x, hidden_state, delta_ts=delta_ts)
        else:
            # List[RNNState]: One state per layer
            # output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
            output_states = (torch.zeros(self.hidden[0].shape), torch.zeros(self.hidden[1].shape))
            i = 0
            # The first RNN layer's input is the original input;
            # the following layers will use their respective previous layer's
            # output as input
            rnn_output = x
            for rnn_layer in self.rnn_layers:
                hidden_state = (self.hidden[0][i], self.hidden[1][i])
                # Run the RNN layer on the data
                rnn_output, out_state = rnn_layer(rnn_output, hidden_state, delta_ts=delta_ts)
                # Apply the dropout layer except the last layer
                if i < self.n_rnn_layers - 1:
                    rnn_output = self.dropout(rnn_output)
                output_states[0][i] = out_state[0]
                output_states[1][i] = out_state[1]
                i += 1
            # Update the hidden states variable
            self.hidden = output_states
        # Apply dropout to the last RNN layer
        # [TODO] Consider if it makes sense to add dropout to the last RNN layer
        # rnn_output = self.dropout(rnn_output)
        # Flatten RNN output to fit into the fully connected layer
        flat_rnn_output = rnn_output.contiguous().view(-1, self.n_hidden)
        # Apply the final fully connected layer
        output = self.fc(flat_rnn_output)
        if prob_output is True:
            # Get the outputs in the form of probabilities
            if self.n_outputs == 1:
                output = self.activation(output)
            else:
                # Normalize outputs on their last dimension
                output = self.activation(output, dim=len(output.shape)-1)
        if get_hidden_state is True:
            return output, self.hidden
        else:
            return output


class MF1LSTM(BaseRNN):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_rnn_layers=1, p_dropout=0,
                 embed_features=None, n_embeddings=None, embedding_dim=None,
                 bidir=False, padding_value=999999,
                 delta_ts_col=None, elapsed_time='small', no_small_delta=True):
        if delta_ts_col is None:
            if embed_features is None:
                self.delta_ts_col = n_inputs
            else:
                # Have into account the new embedding columns that will be added,
                # as well as the removal of the originating categorical columns
                # NOTE: This only works assuming that the delta_ts column is the
                # last one on the dataframe, standing to the left of all the
                # embedding features
                if all([isinstance(feature, int) for feature in embed_features]):
                    self.delta_ts_col = n_inputs - len(embed_features)
                elif (all([isinstance(feat_list, list) for feat_list in embed_features])
                and all([isinstance(feature, int) for feat_list in embed_features
                         for feature in feat_list])):
                    self.delta_ts_col = n_inputs
                    for i in range(len(embed_features)):
                        self.delta_ts_col = rnn_n_inputs - len(embed_features[i])

        else:
            self.delta_ts_col = delta_ts_col
        self.elapsed_time = elapsed_time
        self.no_small_delta = no_small_delta
        MF1LSTMCell_prtl = partial(MF1LSTMCell, delta_ts_col=self.delta_ts_col,
                                 elapsed_time=self.elapsed_time,
                                 no_small_delta=self.no_small_delta)
        if bidir is True:
            rnn_module = lambda *cell_args: BidirTLSTMLayer(MF1LSTMCell_prtl, *cell_args)
        else:
            rnn_module = lambda *cell_args: TLSTMLayer(MF1LSTMCell_prtl, *cell_args)
        super(MF1LSTM, self).__init__(rnn_module=rnn_module, n_inputs=n_inputs,
                                      n_hidden=n_hidden, n_outputs=n_outputs,
                                      n_rnn_layers=n_rnn_layers,
                                      p_dropout=p_dropout,
                                      embed_features=embed_features,
                                      n_embeddings=n_embeddings,
                                      embedding_dim=embedding_dim,
                                      bidir=bidir, is_lstm=True,
                                      padding_value=padding_value)

    def forward(self, x, hidden_state=None, get_hidden_state=False,
                prob_output=True, already_embedded=False):
        if self.embed_features is not None and already_embedded is False:
            # Run each embedding layer on each respective feature, adding the
            # resulting embedding values to the tensor and removing the original,
            # categorical encoded columns
            x = du.embedding.embedding_bag_pipeline(x, self.embed_layers, self.embed_features,
                                                    model_forward=True, inplace=True)
        # Make sure that the input data is of type float
        x = x.float()
        # Get the batch size (might not be always the same)
        batch_size = x.shape[0]
        # Isolate the delta_ts feature
        delta_ts = x[:, :, self.delta_ts_col].clone()
        left_to_delta = x[:, :, :self.delta_ts_col]
        right_to_delta = x[:, :, self.delta_ts_col+1:]
        x = torch.cat([left_to_delta, right_to_delta], 2)
        if hidden_state is None:
            # Reset the LSTM hidden state. Must be done before you run a new
            # batch. Otherwise the LSTM will treat a new batch as a continuation
            # of a sequence.
            self.hidden = self.init_hidden(batch_size)
        else:
            # Use the specified hidden state
            self.hidden = hidden_state
        # Make sure that the data is input in the format of (timestamp x sample x features)
        x = x.permute(1, 0, 2)
        # Get the outputs and hidden states from the RNN layer(s)
        if self.n_rnn_layers == 1:
            if self.bidir is False:
                # Since there's only one layer and the model is not bidirectional,
                # we only need one set of hidden state
                hidden_state = (self.hidden[0][0], self.hidden[1][0])
            # Run the RNN layer on the data
            rnn_output, self.hidden = self.rnn_layer(x, hidden_state, delta_ts=delta_ts)
        else:
            # List[RNNState]: One state per layer
            # output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
            output_states = (torch.zeros(self.hidden[0].shape), torch.zeros(self.hidden[1].shape))
            i = 0
            # The first RNN layer's input is the original input;
            # the following layers will use their respective previous layer's
            # output as input
            rnn_output = x
            for rnn_layer in self.rnn_layers:
                hidden_state = (self.hidden[0][i], self.hidden[1][i])
                # Run the RNN layer on the data
                rnn_output, out_state = rnn_layer(rnn_output, hidden_state, delta_ts=delta_ts)
                # Apply the dropout layer except the last layer
                if i < self.n_rnn_layers - 1:
                    rnn_output = self.dropout(rnn_output)
                output_states[0][i] = out_state[0]
                output_states[1][i] = out_state[1]
                i += 1
            # Update the hidden states variable
            self.hidden = output_states
        # Apply dropout to the last RNN layer
        # [TODO] Consider if it makes sense to add dropout to the last RNN layer
        # rnn_output = self.dropout(rnn_output)
        # Flatten RNN output to fit into the fully connected layer
        flat_rnn_output = rnn_output.contiguous().view(-1, self.n_hidden)
        # Apply the final fully connected layer
        output = self.fc(flat_rnn_output)
        if prob_output is True:
            # Get the outputs in the form of probabilities
            if self.n_outputs == 1:
                output = self.activation(output)
            else:
                # Normalize outputs on their last dimension
                output = self.activation(output, dim=len(output.shape)-1)
        if get_hidden_state is True:
            return output, self.hidden
        else:
            return output


class MF2LSTM(BaseRNN):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_rnn_layers=1, p_dropout=0,
                 embed_features=None, n_embeddings=None, embedding_dim=None,
                 bidir=False, padding_value=999999,
                 delta_ts_col=None, elapsed_time='small', no_small_delta=True):
        # NOTE: In the case of MF2-LSTM models, delta_ts must be in an unormalized
        # version, with each value representing time in minutes
        if delta_ts_col is None:
            if embed_features is None:
                self.delta_ts_col = n_inputs
            else:
                # Have into account the new embedding columns that will be added,
                # as well as the removal of the originating categorical columns
                # NOTE: This only works assuming that the delta_ts column is the
                # last one on the dataframe, standing to the left of all the
                # embedding features
                if all([isinstance(feature, int) for feature in embed_features]):
                    self.delta_ts_col = n_inputs - len(embed_features)
                elif (all([isinstance(feat_list, list) for feat_list in embed_features])
                and all([isinstance(feature, int) for feat_list in embed_features
                         for feature in feat_list])):
                    self.delta_ts_col = n_inputs
                    for i in range(len(embed_features)):
                        self.delta_ts_col = rnn_n_inputs - len(embed_features[i])

        else:
            self.delta_ts_col = delta_ts_col
        self.elapsed_time = elapsed_time
        self.no_small_delta = no_small_delta
        MF2LSTMCell_prtl = partial(MF2LSTMCell, delta_ts_col=self.delta_ts_col,
                                 elapsed_time=self.elapsed_time,
                                 no_small_delta=self.no_small_delta)
        if bidir is True:
            rnn_module = lambda *cell_args: BidirTLSTMLayer(MF2LSTMCell_prtl, *cell_args)
        else:
            rnn_module = lambda *cell_args: TLSTMLayer(MF2LSTMCell_prtl, *cell_args)
        super(MF2LSTM, self).__init__(rnn_module=rnn_module, n_inputs=n_inputs,
                                      n_hidden=n_hidden, n_outputs=n_outputs,
                                      n_rnn_layers=n_rnn_layers,
                                      p_dropout=p_dropout,
                                      embed_features=embed_features,
                                      n_embeddings=n_embeddings,
                                      embedding_dim=embedding_dim,
                                      bidir=bidir, is_lstm=True,
                                      padding_value=padding_value)

    def forward(self, x, hidden_state=None, get_hidden_state=False,
                prob_output=True, already_embedded=False):
        if self.embed_features is not None and already_embedded is False:
            # Run each embedding layer on each respective feature, adding the
            # resulting embedding values to the tensor and removing the original,
            # categorical encoded columns
            x = du.embedding.embedding_bag_pipeline(x, self.embed_layers, self.embed_features,
                                                    model_forward=True, inplace=True)
        # Make sure that the input data is of type float
        x = x.float()
        # Get the batch size (might not be always the same)
        batch_size = x.shape[0]
        # Isolate the delta_ts feature
        delta_ts = x[:, :, self.delta_ts_col].clone()
        left_to_delta = x[:, :, :self.delta_ts_col]
        right_to_delta = x[:, :, self.delta_ts_col+1:]
        x = torch.cat([left_to_delta, right_to_delta], 2)
        if hidden_state is None:
            # Reset the LSTM hidden state. Must be done before you run a new
            # batch. Otherwise the LSTM will treat a new batch as a continuation
            # of a sequence.
            self.hidden = self.init_hidden(batch_size)
        else:
            # Use the specified hidden state
            self.hidden = hidden_state
        # Make sure that the data is input in the format of (timestamp x sample x features)
        x = x.permute(1, 0, 2)
        # Get the outputs and hidden states from the RNN layer(s)
        if self.n_rnn_layers == 1:
            if self.bidir is False:
                # Since there's only one layer and the model is not bidirectional,
                # we only need one set of hidden state
                hidden_state = (self.hidden[0][0], self.hidden[1][0])
            # Run the RNN layer on the data
            rnn_output, self.hidden = self.rnn_layer(x, hidden_state, delta_ts=delta_ts)
        else:
            # List[RNNState]: One state per layer
            # output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
            output_states = (torch.zeros(self.hidden[0].shape), torch.zeros(self.hidden[1].shape))
            i = 0
            # The first RNN layer's input is the original input;
            # the following layers will use their respective previous layer's
            # output as input
            rnn_output = x
            for rnn_layer in self.rnn_layers:
                hidden_state = (self.hidden[0][i], self.hidden[1][i])
                # Run the RNN layer on the data
                rnn_output, out_state = rnn_layer(rnn_output, hidden_state, delta_ts=delta_ts)
                # Apply the dropout layer except the last layer
                if i < self.n_rnn_layers - 1:
                    rnn_output = self.dropout(rnn_output)
                output_states[0][i] = out_state[0]
                output_states[1][i] = out_state[1]
                i += 1
            # Update the hidden states variable
            self.hidden = output_states
        # Apply dropout to the last RNN layer
        # [TODO] Consider if it makes sense to add dropout to the last RNN layer
        # rnn_output = self.dropout(rnn_output)
        # Flatten RNN output to fit into the fully connected layer
        flat_rnn_output = rnn_output.contiguous().view(-1, self.n_hidden)
        # Apply the final fully connected layer
        output = self.fc(flat_rnn_output)
        if prob_output is True:
            # Get the outputs in the form of probabilities
            if self.n_outputs == 1:
                output = self.activation(output)
            else:
                # Normalize outputs on their last dimension
                output = self.activation(output, dim=len(output.shape)-1)
        if get_hidden_state is True:
            return output, self.hidden
        else:
            return output


class LSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class TLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, delta_ts_col=-1, elapsed_time='small',
                 no_small_delta=True):
        super(TLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.delta_ts_col = delta_ts_col
        self.elapsed_time = elapsed_time.lower()
        self.no_small_delta = no_small_delta
        if self.elapsed_time != 'small' and self.elapsed_time != 'long':
            raise Exception(f'ERROR: The parameter `elapsed_time` must either be set to "small" or "long". Received "{elapsed_time}" instead.')
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.weight_ch = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_ch = nn.Parameter(torch.randn(hidden_size))

    @jit.script_method
    def forward(self, input, state, delta_ts=torch.tensor(np.nan)):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        # Separate the elapsed time from the remaining features
        if torch.all(delta_ts.eq(torch.tensor(np.nan))):
            delta_ts = input[:, self.delta_ts_col].clone()
            left_to_delta = input[:, :self.delta_ts_col]
            right_to_delta = input[:, self.delta_ts_col+1:]
            input = torch.cat([left_to_delta, right_to_delta], 1)
        # Get the hidden state and cell memory from the state variable
        hx, cx = state
        # Perform the LSTM gates operations
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)
        # TLSTM's subspace decomposition into a short-term memory
        cs = torch.mm(cx, self.weight_ch.t()) + self.bias_ch
        # TLSTM's long-term memory
        ct = cx - cs
        if self.no_small_delta is True:
            # Set the elapsed time to have a minimum normalized value of 1, so
            # as to prevent smaller than average time differences to increase
            # hidden state values to excessively large numbers
            delta_ts = torch.max(delta_ts, torch.ones(delta_ts.shape))
        # Calculate the time decay value
        if self.elapsed_time == 'small':
            g = 1 / delta_ts
        else:
            g = 1 / torch.log(math.e * delta_ts)
        # TLSTM's discounted short-term memory
        cs = (g * cs.t()).t()
        # TLSTM's adjusted previous memory
        cx = ct + cs
        # Apply each gate's activation function
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)
        # Calculate the new cell memory
        cy = (forget_gate * cx) + (in_gate * cell_gate)
        # Calculate the new hidden state
        hy = out_gate * torch.tanh(cy)
        # Return the new output and state
        return hy, (hy, cy)


class MF1LSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, delta_ts_col=-1, elapsed_time='small',
                 no_small_delta=True):
        super(MF1LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.delta_ts_col = delta_ts_col
        self.elapsed_time = elapsed_time.lower()
        self.no_small_delta = no_small_delta
        if self.elapsed_time != 'small' and self.elapsed_time != 'long':
            raise Exception(f'ERROR: The parameter `elapsed_time` must either be set to "small" or "long". Received "{elapsed_time}" instead.')
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))

    @jit.script_method
    def forward(self, input, state, delta_ts=torch.tensor(np.nan)):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        # Separate the elapsed time from the remaining features
        if torch.all(delta_ts.eq(torch.tensor(np.nan))):
            delta_ts = input[:, self.delta_ts_col].clone()
            left_to_delta = input[:, :self.delta_ts_col]
            right_to_delta = input[:, self.delta_ts_col+1:]
            input = torch.cat([left_to_delta, right_to_delta], 1)
        # Get the hidden state and cell memory from the state variable
        hx, cx = state
        # Perform the LSTM gates operations
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)
        if self.no_small_delta is True:
            # Set the elapsed time to have a minimum normalized value of 1, so
            # as to prevent smaller than average time differences to increase
            # hidden state values to excessively large numbers
            delta_ts = torch.max(delta_ts, torch.ones(delta_ts.shape))
        # Calculate the time decay value
        if self.elapsed_time == 'small':
            g = 1 / delta_ts
        else:
            g = 1 / torch.log(math.e * delta_ts)
        # Apply each gate's activation function
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)
        # Apply MF1-LSTM's time decay
        forget_gate = (g * forget_gate.t()).t()
        # Calculate the new cell memory
        cy = (forget_gate * cx) + (in_gate * cell_gate)
        # Calculate the new hidden state
        hy = out_gate * torch.tanh(cy)
        # Return the new output and state
        return hy, (hy, cy)


class MF2LSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, delta_ts_col=-1, elapsed_time='small',
                 no_small_delta=True):
        super(MF2LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.delta_ts_col = delta_ts_col
        self.elapsed_time = elapsed_time.lower()
        self.no_small_delta = no_small_delta
        if self.elapsed_time != 'small' and self.elapsed_time != 'long':
            raise Exception(f'ERROR: The parameter `elapsed_time` must either be set to "small" or "long". Received "{elapsed_time}" instead.')
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.weight_fq = nn.Parameter(torch.randn(hidden_size, 3))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_fq = nn.Parameter(torch.randn(hidden_size))

    @jit.script_method
    def forward(self, input, state, delta_ts=torch.tensor(np.nan)):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        # Separate the elapsed time from the remaining features
        if torch.all(delta_ts.eq(torch.tensor(np.nan))):
            delta_ts = input[:, self.delta_ts_col].clone()
            left_to_delta = input[:, :self.delta_ts_col]
            right_to_delta = input[:, self.delta_ts_col+1:]
            input = torch.cat([left_to_delta, right_to_delta], 1)
        # Get the hidden state and cell memory from the state variable
        hx, cx = state
        # Perform the LSTM gates operations
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)
        if self.no_small_delta is True:
            # Set the elapsed time to have a minimum normalized value of 1, so
            # as to prevent smaller than average time differences to increase
            # hidden state values to excessively large numbers
            delta_ts = torch.max(delta_ts, torch.ones(delta_ts.shape))
        # Calculate the time decay value
        if self.elapsed_time == 'small':
            g = 1 / delta_ts
        else:
            g = 1 / torch.log(math.e * delta_ts)
        # Apply MF2-LSTM's parametric time
        g = g.view(input.shape[0], -1)
        q = torch.cat([(g / 60), (g / 720) ** 2, (g / 1440) ** 3], dim=1)
        forget_gate = forget_gate + torch.mm(q, self.weight_fq.t()) + self.bias_fq
        # Apply each gate's activation function
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)
        # Calculate the new cell memory
        cy = (forget_gate * cx) + (in_gate * cell_gate)
        # Calculate the new hidden state
        hy = out_gate * torch.tanh(cy)
        # Return the new output and state
        return hy, (hy, cy)


class LSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        # Get mini batches of data on each timestamp
        inputs = input.unbind(0)
        outputs = []
        # Run the LSTM cell on each timestamp, for multiple sequences at the same time
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class ReverseLSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(ReverseLSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        # Get mini batches of data on each timestamp
        inputs = du.utils.reverse(input.unbind(0))
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(du.utils.reverse(outputs)), state


class BidirLSTMLayer(jit.ScriptModule):
    __constants__ = ['directions']

    def __init__(self, cell, *cell_args):
        super(BidirLSTMLayer, self).__init__()
        self.directions = nn.ModuleList([
            LSTMLayer(cell, *cell_args),
            ReverseLSTMLayer(cell, *cell_args),
        ])

    @jit.script_method
    def forward(self, input, states):
        # type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: [forward LSTMState, backward LSTMState]
        outputs = jit.annotate(List[Tensor], [])
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for direction in self.directions:
            state = (states[0][i], states[1][i])
            out, out_state = direction(input, state)
            outputs += [out]
            output_states += [out_state]
            i += 1
        return torch.cat(outputs, -1), output_states


class TLSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(TLSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input, state, delta_ts=torch.tensor(np.nan)):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        # Get mini batches of data on each timestamp
        inputs = input.unbind(0)
        outputs = []
        # Run the LSTM cell on each timestamp, for multiple sequences at the same time
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state, delta_ts[:, i])
            outputs += [out]
        return torch.stack(outputs), state


class ReverseTLSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(ReverseTLSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input, state, delta_ts=torch.tensor(np.nan)):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        # Get mini batches of data on each timestamp
        inputs = du.utils.reverse(input.unbind(0))
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state, delta_ts)
            outputs += [out]
        return torch.stack(du.utils.reverse(outputs)), state


class BidirTLSTMLayer(jit.ScriptModule):
    __constants__ = ['directions']

    def __init__(self, cell, *cell_args):
        super(BidirTLSTMLayer, self).__init__()
        self.directions = nn.ModuleList([
            TLSTMLayer(cell, *cell_args),
            ReverseTLSTMLayer(cell, *cell_args),
        ])

    @jit.script_method
    def forward(self, input, states, delta_ts=torch.tensor(np.nan)):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        # List[LSTMState]: [forward LSTMState, backward LSTMState]
        outputs = jit.annotate(List[Tensor], [])
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for direction in self.directions:
            state = (states[0][i], states[1][i])
            out, out_state = direction(input, state, delta_ts)
            outputs += [out]
            output_states += [out_state]
            i += 1
        return torch.cat(outputs, -1), output_states


class StackedLSTMWithDropout(jit.ScriptModule):
    # Necessary for iterating through self.layers and dropout support
    __constants__ = ['layers', 'num_layers']

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedLSTMWithDropout, self).__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args,
                                        other_layer_args)
        # Introduces a Dropout layer on the outputs of each LSTM layer except
        # the last layer, with dropout probability = 0.4.
        self.num_layers = num_layers

        if (num_layers == 1):
            warnings.warn("dropout lstm adds dropout layers after all but last "
                          "recurrent layer, it expects num_layers greater than "
                          "1, but got num_layers = 1")

        self.dropout_layer = nn.Dropout(0.4)

    @jit.script_method
    def forward(self, input, states):
        # type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            # Apply the dropout layer except the last layer
            if i < self.num_layers - 1:
                output = self.dropout_layer(output)
            output_states += [out_state]
            i += 1
        return output, output_states


# class DeepCare(nn.Module):
