import torch                            # PyTorch to create and apply deep learning models
from torch import nn                    # nn for neural network layers
import torch.jit as jit                 # TorchScript for faster custom models
import math                             # Useful package for logarithm operations
import data_utils as du                 # Data science and machine learning relevant methods

# [TODO] Create new classes for each model type and add options to include
# variants such as embedding, time decay, regularization learning, etc
class BaseRNN(nn.Module):
    def __init__(self, layer, n_inputs, n_hidden, n_outputs, n_layers=1, p_dropout=0,
                 embed_features=None, n_embeddings=None, embedding_dim=None,
                 bidir=False, padding_value=999999):
        '''A vanilla LSTM model, using PyTorch's predefined LSTM module, with
        the option to include embedding layers.

        nn.Parameters
        ----------
        layer : nn.Module or jit.ScriptModule
            Recurrent neural network layer to be used in this model.
        n_inputs : int
            Number of input features.
        n_hidden : int or list of ints
            Number of hidden units. If there are multiple RNN layers (n_layers > 1),
            this parameter must be a list of integers, with the number of hidden
            units for each layer.
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
        bidir : bool, default False
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
        self.embed_features = embed_features
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.bidir = bidir
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
        # [TODO] Replace the LSTM layer with the user defined `layer` parameter
        # [TODO] Add LSTM layers in a ModuleList, with each one having its own
        # hidden units number
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

    def forward(self, x, x_lengths=None, get_hidden_state=False,
                hidden_state=None, prob_output=True):
        if self.embed_features is not None:
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
        # Find the indeces that correspond to padding samples
        pad_idx = du.search_explore.find_val_idx(y_labels, self.padding_value)
        if pad_idx is not None:
            non_pad_idx = list(range(len(y_labels)))
            [non_pad_idx.remove(idx) for idx in pad_idx]
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
            hidden = (weight.new(self.n_layers * (1 + self.bidir), batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers * (1 + self.bidir), batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers * (1 + self.bidir), batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers * (1 + self.bidir), batch_size, self.n_hidden).zero_())
        return hidden


class VanillaLSTM(BaseRNN):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_layers=1, p_dropout=0,
                 embed_features=None, n_embeddings=None, embedding_dim=None,
                 bidir=False, padding_value=999999):
        # [TODO] Test this
        super.__init__(layer=nn.LSTM, n_inputs=n_inputs, n_hidden=n_hidden,
                       n_outputs=n_outputs, n_layers=1, p_dropout=0,
                       embed_features=None, n_embeddings=None,
                       embedding_dim=None, bidir=False, padding_value=999999)


class TLSTM(BaseRNN):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_layers=1, p_dropout=0,
                 embed_features=None, n_embeddings=None, embedding_dim=None,
                 bidir=False, padding_value=999999):
        # [TODO] Finish the TLSTM code
        # [TODO] Test this
        if bidir is True:
            TLSTMLayer = BidirLSTMLayer(TLSTMCell, args)
        else:
            TLSTMLayer = LSTMLayer(TLSTMCell, args)
        super.__init__(layer=TLSTMLayer, n_inputs=n_inputs, n_hidden=n_hidden,
                       n_outputs=n_outputs, n_layers=1, p_dropout=0,
                       embed_features=None, n_embeddings=None,
                       embedding_dim=None, bidir=False, padding_value=999999)


# [TODO]
# class MF1LSTM(BaseRNN):


# [TODO]
# class MF2LSTM(BaseRNN):


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
    def __init__(self, input_size, hidden_size, elapsed_time='small'):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.weight_ch = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_ch = nn.Parameter(torch.randn(hidden_size))
        self.elapsed_time = elapsed_time.lower()
        if self.elapsed_time != 'small' and self.elapsed_time != 'long':
            raise Exception(f'ERROR: The parameter `elapsed_time` must either be set to "small" or "long". Received "{elapsed_time}" instead.')

    @jit.script_method
    def forward(self, input, state, delta_ts_col):
        # type: (Tensor, Tuple[Tensor, Tensor], int) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        # Separate the elapsed time from the remaining features
        delta_ts = input[delta_ts_col].clone()
        input = du.deep_learning.remove_tensor_column(data, delta_ts_col, inplace=True)
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
        # Calculate the time decay value
        if self.elapsed_time == 'small':
            g = 1 / delta_ts
        else:
            g = 1 / torch.log(math.e * delta_ts)
        # TLSTM's discounted short-term memory
        cs = g * cs
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


class LSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(0)
        outputs = []
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
            state = states[i]
            out, out_state = direction(input, state)
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
