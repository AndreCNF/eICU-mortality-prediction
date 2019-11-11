import torch                            # PyTorch to create and apply deep learning models
from torch import nn, optim             # nn for neural network layers and optim for training optimizers
from torch.nn import functional as F    # Module containing several activation functions
import math                             # Useful package for logarithm operations
from data_utils import embedding        # Embeddings and other categorical features handling methods

# [TODO] Create new classes for each model type and add options to include
# variants such as embedding, time decay, regularization learning, etc
class VanillaLSTM(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_layers=1, p_dropout=0, embed_features=None,
                 num_embeddings=None, embedding_dim=None):
        # [TODO] Add documentation for each model class
        super().__init__()
        self.n_inputs = n_inputs                # Number of input features
        self.n_hidden = n_hidden                # Number of hidden units
        self.n_outputs = n_outputs              # Number of outputs
        self.n_layers = n_layers                # Number of LSTM layers
        self.p_dropout = p_dropout              # Probability of dropout
        self.embed_features = embed_features    # List of features that need to go through embedding layers
        self.num_embeddings = num_embeddings    # List of the total number of unique categories for the embedding layers
        self.embedding_dim = embedding_dim      # List of embedding dimensions
        # Embedding layers
        if self.embed_features is not None:
            if isinstance(self.embed_features, int):
                self.embed_features = [self.embed_features]
            if self.num_embeddings is None:
                raise Exception('ERROR: If the user specifies features to be embedded, each feature\'s number of \
                                 embeddings must also be specified. Received a `embed_features` argument, but not \
                                 `num_embeddings`.')
            else:
                if isinstance(self.num_embeddings, int):
                    self.num_embeddings = [self.num_embeddings]
                if len(self.num_embeddings) != len(self.embed_features):
                    raise Exception(f'ERROR: The list of the number of embeddings `num_embeddings` and the embedding \
                                      features `embed_features` must have the same length. The provided `num_embeddings` \
                                      has length {len(self.num_embeddings)} while `embed_features` has length {len(self.embed_features)}.')
            if isinstance(self.embed_features, list):
                # Create a modules dictionary of embedding bag layers;
                # each key corresponds to a embedded feature's index
                self.embed_layers = nn.ModuleDict()
                for i in range(len(self.embed_features)):
                    if embedding_dim is None:
                        # Calculate a reasonable embedding dimension for the current feature;
                        # the formula sets a minimum embedding dimension of 3, with above
                        # values being calculated as the rounded up base 5 logarithm of
                        # the number of embeddings
                        embedding_dim_i = max(3, int(math.ceil(math.log(self.num_embeddings[i], base=5))))
                    else:
                        if isinstance(self.embedding_dim, int):
                            self.embedding_dim = [self.embedding_dim]
                        embedding_dim_i = self.embedding_dim[i]
                    # Create an embedding layer for the current feature
                    self.embed_layers[f'embed_{self.embed_features[i]}'] = nn.EmbeddingBag(self.num_embeddings[i], embedding_dim_i)
            else:
                raise Exception(f'ERROR: The embedding features must be indicated in `embed_features` as either a \
                                  single, integer index or a list of indices. The provided argument has type {type(embed_features)}.')
        # LSTM layer(s)
        if self.embed_features is None:
            self.lstm_n_inputs = self.n_inputs
        else:
            # Have into account the new embedding columns that will be added, as well as the removal
            # of the originating categorical columns
            self.lstm_n_inputs = self.n_inputs + sum(self.embedding_dim) - len(self.embedding_dim)
        self.lstm = nn.LSTM(self.lstm_n_inputs, self.n_hidden, self.n_layers, 
                            batch_first=True, dropout=self.p_dropout)
        # Fully connected layer which takes the LSTM's hidden units and calculates the output classification
        self.fc = nn.Linear(self.n_hidden, self.n_outputs)
        # Dropout used between the last LSTM layer and the fully connected layer
        self.dropout = nn.Dropout(p=self.p_dropout)

    def forward(self, x, x_lengths=None, get_hidden_state=False, hidden_state=None):
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
            # Reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
            # a new batch as a continuation of a sequence.
            self.hidden = self.init_hidden(batch_size)
        else:
            # Use the specified hidden state
            self.hidden = hidden_state
        if x_lengths is not None:
            # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
            x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)
        # Get the outputs and hidden states from the LSTM layer(s)
        lstm_output, self.hidden = self.lstm(x, self.hidden)
        if x_lengths is not None:
            # Undo the packing operation
            lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        # Apply dropout to the last LSTM layer
        lstm_output = self.dropout(lstm_output)
        # Flatten LSTM output to fit into the fully connected layer
        flat_lstm_output = lstm_output.contiguous().view(-1, self.n_hidden)
        # Classification scores after applying the fully connected layers and softmax
        output = torch.sigmoid(self.fc(flat_lstm_output))
        if get_hidden_state is True:
            return output, self.hidden
        else:
            return output

    def loss(self, y_pred, y_labels, x_lengths):
        # Before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.
        # Flatten all the labels and make it have type long instead of float
        y_labels = y_labels.contiguous().view(-1).long()
        # Flatten all predictions
        y_pred = y_pred.view(-1, self.n_outputs)
        # Create a mask by filtering out all labels that are not a padding value
        # Also need to make it have type float to be able to multiply with y_pred
        mask = (y_labels <= 1)
        # Count how many predictions we have
        n_pred = int(torch.sum(mask).item())
        # Check if there's only one class to classify (either it belongs to that class or it doesn't)
        if self.n_outputs == 1:
            # Add a column to the predictions tensor with the probability of not being part of the
            # class being used
            y_pred_other_class = 1 - y_pred
            y_pred = torch.stack([y_pred_other_class, y_pred]).permute(1, 0, 2).squeeze()
        # Pick the values for the label and zero out the rest with the mask
        y_pred = y_pred[range(y_pred.shape[0]), y_labels * mask.long()] * mask.float()
        # I need to get the diagonal of the tensor, which represents a vector of each
        # score (y_pred) multiplied by its correct mask value
        # Otherwise we get a square matrix of every score multiplied by every mask value
        # Completely remove the padded values from the predictions using the mask
        y_pred = torch.masked_select(y_pred, mask)
        # Compute cross entropy loss which ignores all padding values
        ce_loss = -torch.sum(torch.log(y_pred)) / n_pred
        return ce_loss

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # Check if GPU is available
        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu is True:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden


class TLSTM(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_layers=1, p_dropout=0, embed_features=None,
                 num_embeddings=None, embedding_dim=None):
        # [TODO] Add documentation for each model class
        super().__init__()
        self.n_inputs = n_inputs                # Number of input features
        self.n_hidden = n_hidden                # Number of hidden units
        self.n_outputs = n_outputs              # Number of outputs
        self.n_layers = n_layers                # Number of LSTM layers
        self.p_dropout = p_dropout              # Probability of dropout
        self.embed_features = embed_features    # List of features that need to go through embedding layers
        self.num_embeddings = num_embeddings    # List of the total number of unique categories for the embedding layers
        self.embedding_dim = embedding_dim      # List of embedding dimensions
        # Embedding layers
        if self.embed_features is not None:
            if isinstance(self.embed_features, int):
                self.embed_features = [self.embed_features]
            if self.num_embeddings is None:
                raise Exception('ERROR: If the user specifies features to be embedded, each feature\'s number of \
                                 embeddings must also be specified. Received a `embed_features` argument, but not \
                                 `num_embeddings`.')
            else:
                if isinstance(self.num_embeddings, int):
                    self.num_embeddings = [self.num_embeddings]
                if len(self.num_embeddings) != len(self.embed_features):
                    raise Exception(f'ERROR: The list of the number of embeddings `num_embeddings` and the embedding \
                                      features `embed_features` must have the same length. The provided `num_embeddings` \
                                      has length {len(self.num_embeddings)} while `embed_features` has length {len(self.embed_features)}.')
            if isinstance(self.embed_features, list):
                # Create a modules dictionary of embedding bag layers;
                # each key corresponds to a embedded feature's index
                self.embed_layers = nn.ModuleDict()
                for i in range(len(self.embed_features)):
                    if embedding_dim is None:
                        # Calculate a reasonable embedding dimension for the current feature;
                        # the formula sets a minimum embedding dimension of 3, with above
                        # values being calculated as the rounded up base 5 logarithm of
                        # the number of embeddings
                        embedding_dim_i = max(3, int(math.ceil(math.log(self.num_embeddings[i], base=5))))
                    else:
                        if isinstance(self.embedding_dim, int):
                            self.embedding_dim = [self.embedding_dim]
                        embedding_dim_i = self.embedding_dim[i]
                    # Create an embedding layer for the current feature
                    self.embed_layers[f'embed_{self.embed_features[i]}'] = nn.EmbeddingBag(self.num_embeddings[i], embedding_dim_i)
            else:
                raise Exception(f'ERROR: The embedding features must be indicated in `embed_features` as either a \
                                  single, integer index or a list of indices. The provided argument has type {type(embed_features)}.')
        # LSTM layer(s)
        if self.embed_features is None:
            self.lstm_n_inputs = self.n_inputs
        else:
            # Have into account the new embedding columns that will be added, as well as the removal
            # of the originating categorical columns
            self.lstm_n_inputs = self.n_inputs + sum(self.embedding_dim) - len(self.embedding_dim)
        self.lstm = nn.LSTM(self.lstm_n_inputs, self.n_hidden, self.n_layers, 
                            batch_first=True, dropout=self.p_dropout)
        # Fully connected layer which takes the LSTM's hidden units and calculates the output classification
        self.fc = nn.Linear(self.n_hidden, self.n_outputs)
        # Dropout used between the last LSTM layer and the fully connected layer
        self.dropout = nn.Dropout(p=self.p_dropout)

    def forward(self, x, x_lengths=None, get_hidden_state=False, hidden_state=None):
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
            # Reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
            # a new batch as a continuation of a sequence.
            self.hidden = self.init_hidden(batch_size)
        else:
            # Use the specified hidden state
            self.hidden = hidden_state
        if x_lengths is not None:
            # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
            x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)
        # Get the outputs and hidden states from the LSTM layer(s)
        lstm_output, self.hidden = self.lstm(x, self.hidden)
        if x_lengths is not None:
            # Undo the packing operation
            lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        # Apply dropout to the last LSTM layer
        lstm_output = self.dropout(lstm_output)
        # Flatten LSTM output to fit into the fully connected layer
        flat_lstm_output = lstm_output.contiguous().view(-1, self.n_hidden)
        # Classification scores after applying the fully connected layers and softmax
        output = torch.sigmoid(self.fc(flat_lstm_output))
        if get_hidden_state is True:
            return output, self.hidden
        else:
            return output

    def loss(self, y_pred, y_labels, x_lengths):
        # Before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.
        # Flatten all the labels and make it have type long instead of float
        y_labels = y_labels.contiguous().view(-1).long()
        # Flatten all predictions
        y_pred = y_pred.view(-1, self.n_outputs)
        # Create a mask by filtering out all labels that are not a padding value
        # Also need to make it have type float to be able to multiply with y_pred
        mask = (y_labels <= 1)
        # Count how many predictions we have
        n_pred = int(torch.sum(mask).item())
        # Check if there's only one class to classify (either it belongs to that class or it doesn't)
        if self.n_outputs == 1:
            # Add a column to the predictions tensor with the probability of not being part of the
            # class being used
            y_pred_other_class = 1 - y_pred
            y_pred = torch.stack([y_pred_other_class, y_pred]).permute(1, 0, 2).squeeze()
        # Pick the values for the label and zero out the rest with the mask
        y_pred = y_pred[range(y_pred.shape[0]), y_labels * mask.long()] * mask.float()
        # I need to get the diagonal of the tensor, which represents a vector of each
        # score (y_pred) multiplied by its correct mask value
        # Otherwise we get a square matrix of every score multiplied by every mask value
        # Completely remove the padded values from the predictions using the mask
        y_pred = torch.masked_select(y_pred, mask)
        # Compute cross entropy loss which ignores all padding values
        ce_loss = -torch.sum(torch.log(y_pred)) / n_pred
        return ce_loss

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # Check if GPU is available
        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu is True:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden


class DeepCare(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_layers=1, p_dropout=0, embed_features=None,
                 num_embeddings=None, embedding_dim=None):
        # [TODO] Add documentation for each model class
        super().__init__()
        self.n_inputs = n_inputs                # Number of input features
        self.n_hidden = n_hidden                # Number of hidden units
        self.n_outputs = n_outputs              # Number of outputs
        self.n_layers = n_layers                # Number of LSTM layers
        self.p_dropout = p_dropout              # Probability of dropout
        self.embed_features = embed_features    # List of features that need to go through embedding layers
        self.num_embeddings = num_embeddings    # List of the total number of unique categories for the embedding layers
        self.embedding_dim = embedding_dim      # List of embedding dimensions
        # Embedding layers
        if self.embed_features is not None:
            if isinstance(self.embed_features, int):
                self.embed_features = [self.embed_features]
            if self.num_embeddings is None:
                raise Exception('ERROR: If the user specifies features to be embedded, each feature\'s number of \
                                 embeddings must also be specified. Received a `embed_features` argument, but not \
                                 `num_embeddings`.')
            else:
                if isinstance(self.num_embeddings, int):
                    self.num_embeddings = [self.num_embeddings]
                if len(self.num_embeddings) != len(self.embed_features):
                    raise Exception(f'ERROR: The list of the number of embeddings `num_embeddings` and the embedding \
                                      features `embed_features` must have the same length. The provided `num_embeddings` \
                                      has length {len(self.num_embeddings)} while `embed_features` has length {len(self.embed_features)}.')
            if isinstance(self.embed_features, list):
                # Create a modules dictionary of embedding bag layers;
                # each key corresponds to a embedded feature's index
                self.embed_layers = nn.ModuleDict()
                for i in range(len(self.embed_features)):
                    if embedding_dim is None:
                        # Calculate a reasonable embedding dimension for the current feature;
                        # the formula sets a minimum embedding dimension of 3, with above
                        # values being calculated as the rounded up base 5 logarithm of
                        # the number of embeddings
                        embedding_dim_i = max(3, int(math.ceil(math.log(self.num_embeddings[i], base=5))))
                    else:
                        if isinstance(self.embedding_dim, int):
                            self.embedding_dim = [self.embedding_dim]
                        embedding_dim_i = self.embedding_dim[i]
                    # Create an embedding layer for the current feature
                    self.embed_layers[f'embed_{self.embed_features[i]}'] = nn.EmbeddingBag(self.num_embeddings[i], embedding_dim_i)
            else:
                raise Exception(f'ERROR: The embedding features must be indicated in `embed_features` as either a \
                                  single, integer index or a list of indices. The provided argument has type {type(embed_features)}.')
        # LSTM layer(s)
        if self.embed_features is None:
            self.lstm_n_inputs = self.n_inputs
        else:
            # Have into account the new embedding columns that will be added, as well as the removal
            # of the originating categorical columns
            self.lstm_n_inputs = self.n_inputs + sum(self.embedding_dim) - len(self.embedding_dim)
        self.lstm = nn.LSTM(self.lstm_n_inputs, self.n_hidden, self.n_layers, 
                            batch_first=True, dropout=self.p_dropout)
        # Fully connected layer which takes the LSTM's hidden units and calculates the output classification
        self.fc = nn.Linear(self.n_hidden, self.n_outputs)
        # Dropout used between the last LSTM layer and the fully connected layer
        self.dropout = nn.Dropout(p=self.p_dropout)

    def forward(self, x, x_lengths=None, get_hidden_state=False, hidden_state=None):
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
            # Reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
            # a new batch as a continuation of a sequence.
            self.hidden = self.init_hidden(batch_size)
        else:
            # Use the specified hidden state
            self.hidden = hidden_state
        if x_lengths is not None:
            # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
            x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)
        # Get the outputs and hidden states from the LSTM layer(s)
        lstm_output, self.hidden = self.lstm(x, self.hidden)
        if x_lengths is not None:
            # Undo the packing operation
            lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        # Apply dropout to the last LSTM layer
        lstm_output = self.dropout(lstm_output)
        # Flatten LSTM output to fit into the fully connected layer
        flat_lstm_output = lstm_output.contiguous().view(-1, self.n_hidden)
        # Classification scores after applying the fully connected layers and softmax
        output = torch.sigmoid(self.fc(flat_lstm_output))
        if get_hidden_state is True:
            return output, self.hidden
        else:
            return output

    def loss(self, y_pred, y_labels, x_lengths):
        # Before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.
        # Flatten all the labels and make it have type long instead of float
        y_labels = y_labels.contiguous().view(-1).long()
        # Flatten all predictions
        y_pred = y_pred.view(-1, self.n_outputs)
        # Create a mask by filtering out all labels that are not a padding value
        # Also need to make it have type float to be able to multiply with y_pred
        mask = (y_labels <= 1)
        # Count how many predictions we have
        n_pred = int(torch.sum(mask).item())
        # Check if there's only one class to classify (either it belongs to that class or it doesn't)
        if self.n_outputs == 1:
            # Add a column to the predictions tensor with the probability of not being part of the
            # class being used
            y_pred_other_class = 1 - y_pred
            y_pred = torch.stack([y_pred_other_class, y_pred]).permute(1, 0, 2).squeeze()
        # Pick the values for the label and zero out the rest with the mask
        y_pred = y_pred[range(y_pred.shape[0]), y_labels * mask.long()] * mask.float()
        # I need to get the diagonal of the tensor, which represents a vector of each
        # score (y_pred) multiplied by its correct mask value
        # Otherwise we get a square matrix of every score multiplied by every mask value
        # Completely remove the padded values from the predictions using the mask
        y_pred = torch.masked_select(y_pred, mask)
        # Compute cross entropy loss which ignores all padding values
        ce_loss = -torch.sum(torch.log(y_pred)) / n_pred
        return ce_loss

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # Check if GPU is available
        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu is True:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden


class TransformerXL(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_layers=1, p_dropout=0, embed_features=None,
                 num_embeddings=None, embedding_dim=None):
        # [TODO] Add documentation for each model class
        super().__init__()
        self.n_inputs = n_inputs                # Number of input features
        self.n_hidden = n_hidden                # Number of hidden units
        self.n_outputs = n_outputs              # Number of outputs
        self.n_layers = n_layers                # Number of LSTM layers
        self.p_dropout = p_dropout              # Probability of dropout
        self.embed_features = embed_features    # List of features that need to go through embedding layers
        self.num_embeddings = num_embeddings    # List of the total number of unique categories for the embedding layers
        self.embedding_dim = embedding_dim      # List of embedding dimensions
        # Embedding layers
        if self.embed_features is not None:
            if isinstance(self.embed_features, int):
                self.embed_features = [self.embed_features]
            if self.num_embeddings is None:
                raise Exception('ERROR: If the user specifies features to be embedded, each feature\'s number of \
                                 embeddings must also be specified. Received a `embed_features` argument, but not \
                                 `num_embeddings`.')
            else:
                if isinstance(self.num_embeddings, int):
                    self.num_embeddings = [self.num_embeddings]
                if len(self.num_embeddings) != len(self.embed_features):
                    raise Exception(f'ERROR: The list of the number of embeddings `num_embeddings` and the embedding \
                                      features `embed_features` must have the same length. The provided `num_embeddings` \
                                      has length {len(self.num_embeddings)} while `embed_features` has length {len(self.embed_features)}.')
            if isinstance(self.embed_features, list):
                # Create a modules dictionary of embedding bag layers;
                # each key corresponds to a embedded feature's index
                self.embed_layers = nn.ModuleDict()
                for i in range(len(self.embed_features)):
                    if embedding_dim is None:
                        # Calculate a reasonable embedding dimension for the current feature;
                        # the formula sets a minimum embedding dimension of 3, with above
                        # values being calculated as the rounded up base 5 logarithm of
                        # the number of embeddings
                        embedding_dim_i = max(3, int(math.ceil(math.log(self.num_embeddings[i], base=5))))
                    else:
                        if isinstance(self.embedding_dim, int):
                            self.embedding_dim = [self.embedding_dim]
                        embedding_dim_i = self.embedding_dim[i]
                    # Create an embedding layer for the current feature
                    self.embed_layers[f'embed_{self.embed_features[i]}'] = nn.EmbeddingBag(self.num_embeddings[i], embedding_dim_i)
            else:
                raise Exception(f'ERROR: The embedding features must be indicated in `embed_features` as either a \
                                  single, integer index or a list of indices. The provided argument has type {type(embed_features)}.')
        # LSTM layer(s)
        if self.embed_features is None:
            self.lstm_n_inputs = self.n_inputs
        else:
            # Have into account the new embedding columns that will be added, as well as the removal
            # of the originating categorical columns
            self.lstm_n_inputs = self.n_inputs + sum(self.embedding_dim) - len(self.embedding_dim)
        self.lstm = nn.LSTM(self.lstm_n_inputs, self.n_hidden, self.n_layers, 
                            batch_first=True, dropout=self.p_dropout)
        # Fully connected layer which takes the LSTM's hidden units and calculates the output classification
        self.fc = nn.Linear(self.n_hidden, self.n_outputs)
        # Dropout used between the last LSTM layer and the fully connected layer
        self.dropout = nn.Dropout(p=self.p_dropout)

    def forward(self, x, x_lengths=None, get_hidden_state=False, hidden_state=None):
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
            # Reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
            # a new batch as a continuation of a sequence.
            self.hidden = self.init_hidden(batch_size)
        else:
            # Use the specified hidden state
            self.hidden = hidden_state
        if x_lengths is not None:
            # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
            x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)
        # Get the outputs and hidden states from the LSTM layer(s)
        lstm_output, self.hidden = self.lstm(x, self.hidden)
        if x_lengths is not None:
            # Undo the packing operation
            lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        # Apply dropout to the last LSTM layer
        lstm_output = self.dropout(lstm_output)
        # Flatten LSTM output to fit into the fully connected layer
        flat_lstm_output = lstm_output.contiguous().view(-1, self.n_hidden)
        # Classification scores after applying the fully connected layers and softmax
        output = torch.sigmoid(self.fc(flat_lstm_output))
        if get_hidden_state is True:
            return output, self.hidden
        else:
            return output

    def loss(self, y_pred, y_labels, x_lengths):
        # Before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.
        # Flatten all the labels and make it have type long instead of float
        y_labels = y_labels.contiguous().view(-1).long()
        # Flatten all predictions
        y_pred = y_pred.view(-1, self.n_outputs)
        # Create a mask by filtering out all labels that are not a padding value
        # Also need to make it have type float to be able to multiply with y_pred
        mask = (y_labels <= 1)
        # Count how many predictions we have
        n_pred = int(torch.sum(mask).item())
        # Check if there's only one class to classify (either it belongs to that class or it doesn't)
        if self.n_outputs == 1:
            # Add a column to the predictions tensor with the probability of not being part of the
            # class being used
            y_pred_other_class = 1 - y_pred
            y_pred = torch.stack([y_pred_other_class, y_pred]).permute(1, 0, 2).squeeze()
        # Pick the values for the label and zero out the rest with the mask
        y_pred = y_pred[range(y_pred.shape[0]), y_labels * mask.long()] * mask.float()
        # I need to get the diagonal of the tensor, which represents a vector of each
        # score (y_pred) multiplied by its correct mask value
        # Otherwise we get a square matrix of every score multiplied by every mask value
        # Completely remove the padded values from the predictions using the mask
        y_pred = torch.masked_select(y_pred, mask)
        # Compute cross entropy loss which ignores all padding values
        ce_loss = -torch.sum(torch.log(y_pred)) / n_pred
        return ce_loss

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # Check if GPU is available
        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu is True:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden
