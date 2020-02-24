import torch                            # PyTorch to create and apply deep learning models
from torch import nn, optim             # nn for neural network layers and optim for training optimizers
from torch.nn import functional as F    # Module containing several activation functions
import math                             # Useful package for logarithm operations
from data_utils import embedding        # Embeddings and other categorical features handling methods

# [TODO] Create new classes for each model type and add options to include
# variants such as embedding, time decay, regularization learning, etc
class VanillaLSTM(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_layers=1, p_dropout=0,
                 embed_features=None, n_embeddings=None, embedding_dim=None,
                 bidirectional=False, padding_value=999999):
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
        n_layers : int, default 1
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
        bidirectional : bool, default False
            If set to True, the LSTM model will be bidirectional (have hidden
            memory flowing both forward and backwards).
        padding_value : int or float, default 999999
            Value to use in the padding, to fill the sequences.
        '''
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.embed_features = embed_feature
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.bidir = bidirectional
        self.padding_value = padding_value
        # Embedding layers
        if self.embed_features is not None:
            if isinstance(self.embed_features, int):
                self.embed_features = [self.embed_features]
            if self.n_embeddings is None:
                raise Exception('ERROR: If the user specifies features to be embedded, each feature\'s number of embeddings must also be specified. Received a `embed_features` argument, but not `n_embeddings`.')
            else:
                if isinstance(self.n_embeddings, int):
                    self.n_embeddings = [self.n_embeddings]
                if len(self.n_embeddings) != len(self.embed_features):
                    raise Exception(f'ERROR: The list of the number of embeddings `n_embeddings` and the embedding features `embed_features` must have the same length. The provided `n_embeddings` has length {len(self.n_embeddings)} while `embed_features` has length {len(self.embed_features)}.')
            if isinstance(self.embed_features, list):
                # Create a modules dictionary of embedding bag layers;
                # each key corresponds to a embedded feature's index
                self.embed_layers = nn.ModuleDict()
                for i in range(len(self.embed_features)):
                    if embedding_dim is None:
                        # Calculate a reasonable embedding dimension for the
                        # current feature; the formula sets a minimum embedding
                        # dimension of 3, with above values being calculated as
                        # the rounded up base 5 logarithm of the number of
                        # embeddings.
                        embedding_dim_i = max(3, int(math.ceil(math.log(self.n_embeddings[i], base=5))))
                    else:
                        if isinstance(self.embedding_dim, int):
                            self.embedding_dim = [self.embedding_dim]
                        embedding_dim_i = self.embedding_dim[i]
                    # Create an embedding layer for the current feature
                    self.embed_layers[f'embed_{self.embed_features[i]}'] = nn.EmbeddingBag(self.n_embeddings[i], embedding_dim_i)
            else:
                raise Exception(f'ERROR: The embedding features must be indicated in `embed_features` as either a single, integer index or a list of indices. The provided argument has type {type(embed_features)}.')
        # LSTM layer(s)
        if self.embed_features is None:
            self.lstm_n_inputs = self.n_inputs
        else:
            # Have into account the new embedding columns that will be added, as
            # well as the removal of the originating categorical columns
            self.lstm_n_inputs = self.n_inputs + sum(self.embedding_dim) - len(self.embedding_dim)
        self.lstm = nn.LSTM(self.lstm_n_inputs, self.n_hidden, self.n_layers,
                            batch_first=True, dropout=self.p_dropout,
                            bidirectional=self.bidir)
        # Fully connected layer which takes the LSTM's hidden units and
        # calculates the output classification
        self.fc = nn.Linear(self.n_hidden, self.n_outputs)
        # Dropout used between the last LSTM layer and the fully connected layer
        self.dropout = nn.Dropout(p=self.p_dropout)
        # Use the standard cross entropy function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, x_lengths=None, get_hidden_state=False,
                hidden_state=None, prob_output=True):
        if self.embed_features is not None:
            # Run each embedding layer on each respective feature, adding the
            # resulting embedding values to the tensor and removing the original,
            # categorical encoded columns
            x = embedding.embedding_bag_pipeline(x, self.embed_layers, self.embed_features,
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
            x = nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)
        # Get the outputs and hidden states from the LSTM layer(s)
        lstm_output, self.hidden = self.lstm(x, self.hidden)
        if x_lengths is not None:
            # Undo the packing operation
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        # Apply dropout to the last LSTM layer
        lstm_output = self.dropout(lstm_output)
        # Flatten LSTM output to fit into the fully connected layer
        flat_lstm_output = lstm_output.contiguous().view(-1, self.n_hidden)
        # Apply the final fully connected layer
        output = self.fc(flat_lstm_output)
        if prob_output is True:
            # Get the outputs in the form of probabilities
            if self.n_outputs == 1:
                output = F.sigmoid(output)
            else:
                # Normalize outputs on their last dimension
                output = F.softmax(output, dim=len(output.shape)-1)
        if get_hidden_state is True:
            return output, self.hidden
        else:
            return output

    def loss(self, y_pred, y_labels):
        # Make sure that the labels are in long format
        y_labels = y_labels.long()
        # Compute cross entropy loss which ignores all padding values
        ce_loss = self.criterion(y_pred, y_labels, ignore_index=self.padding_value)
        return ce_loss

    def init_hidden(self, batch_size):
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # Check if GPU is available
        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu is True:
            hidden = (weight.new(self.n_layers * (1 + self.bidir), batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers * (1 + self.bidir), batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers * (1 + self.bidir), batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers * (1 + self.bidir), batch_size, self.n_hidden).zero_())
        return hidden


class TLSTM(nn.Module):



class DeepCare(nn.Module):
