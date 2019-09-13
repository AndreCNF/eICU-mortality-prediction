from comet_ml import Experiment                         # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                                            # PyTorch to create and apply deep learning models
import numpy as np                                      # NumPy to handle numeric and NaN operations
import warnings                                         # Print warnings for bad practices
from datetime import datetime                           # datetime to use proper date and time formats
import sys                                              # Identify types of exceptions
from NeuralNetwork import NeuralNetwork                 # Import the neural network model class
from sklearn.metrics import roc_auc_score               # ROC AUC model performance metric
import utils                                            # Generic and useful methods
import padding                                          # Padding and variable sequence length related methods

# Random seed used in PyTorch and NumPy's random operations (such as weight initialization)
random_seed = utils.random_seed

if isinstance(random_seed, int):
    # Set user specified random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
else:
    # Set completely random seed from utils
    np.random.set_state(random_seed)
    torch.manual_seed(random_seed[1][0])

# Ignore Dask's 'meta' warning
warnings.filterwarnings("ignore", message="`meta` is not specified, inferred from partial data. Please provide `meta` if the result is unexpected.")

# Methods

def load_checkpoint(filepath):
    '''Load a model from a specified path and name.

    Parameters
    ----------
    filepath : str
        Path to the model being loaded, including it's own file name.

    Returns
    -------
    model : nn.Module
        The loaded model with saved weight values.
    '''
    checkpoint = torch.load(filepath)
    model = NeuralNetwork(checkpoint['n_inputs'],
                          checkpoint['n_hidden'],
                          checkpoint['n_outputs'],
                          checkpoint['n_layers'],
                          checkpoint['p_dropout'])
    model.load_state_dict(checkpoint['state_dict'])

    return model


def change_grad(grad, data, min=0, max=1):
    '''Restrict the gradients to only have valid values.

    Parameters
    ----------
    grad : torch.Tensor
        PyTorch tensor containing the gradients of the data being optimized.
    data : torch.Tensor
        PyTorch tensor containing the data being optimized.
    min : int, default 0
        Minimum valid data value.
    max : int, default 0
        Maximum valid data value.

    Returns
    -------
    grad : torch.Tensor
        PyTorch tensor containing the corrected gradients of the data being
        optimized.
    '''
    # Minimum accepted gradient value to be considered
    min_grad_val = 0.001

    for i in range(data.shape[0]):
        if (data[i] == min and grad[i] < 0) or (data[i] == max and grad[i] > 0):
            # Stop the gradient from excedding the limit
            grad[i] = 0
        elif data[i] == min and grad[i] > min_grad_val:
            # Make the gradient have a integer value
            grad[i] = 1
        elif data[i] == max and grad[i] < -min_grad_val:
            # Make the gradient have a integer value
            grad[i] = -1
        else:
            # Avoid any insignificant gradient
            grad[i] = 0

    return grad


def ts_tensor_to_np_matrix(data, feat_num=None, padding_value=999999):
    '''Convert a 3D PyTorch tensor, such as one representing multiple time series
    data, into a 2D NumPy matrix. Can be useful for applying the SHAP Kernel
    Explainer.

    Parameters
    ----------
    data : torch.Tensor
        PyTorch tensor containing the three dimensional data being converted.
    feat_num : list of int, default None
        List of the column numbers that represent the features. If not specified,
        all columns will be used.
    padding_value : numeric
        Value to use in the padding, to fill the sequences.

    Returns
    -------
    data_matrix : numpy.ndarray
        NumPy two dimensional matrix obtained from the data after conversion.
    '''
    # View as a single sequence, i.e. like a dataframe without grouping by id
    data_matrix = data.contiguous().view(-1, data.shape[2]).detach().numpy()
    # Remove rows that are filled with padding values
    if feat_num is not None:
        data_matrix = data_matrix[[not all(row == padding_value) for row in data_matrix[:, feat_num]]]
    else:
        data_matrix = data_matrix[[not all(row == padding_value) for row in data_matrix]]
    return data_matrix


def model_inference(model, seq_len_dict, dataloader=None, data=None, metrics=['loss', 'accuracy', 'AUC'],
                    padding_value=999999, output_rounded=False, experiment=None, set_name='test',
                    seq_final_outputs=False, cols_to_remove=[0, 1]):
    '''Do inference on specified data using a given model.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model which does the inference on the data.
    seq_len_dict : dict
        Dictionary containing the sequence lengths for each index of the
        original dataframe. This allows to ignore the padding done in
        the fixed sequence length tensor.
    dataloader : torch.utils.data.DataLoader, default None
        Data loader which will be used to get data batches during inference.
    data : tuple of torch.Tensor, default None
        If a data loader isn't specified, the user can input directly a
        tuple of PyTorch tensor on which inference will be done. The first
        tensor must correspond to the features tensor whe second one
        should be the labels tensor.
    metrics : list of strings, default ['loss', 'accuracy', 'AUC'],
        List of metrics to be used to evaluate the model on the infered data.
        Available metrics are cross entropy loss (loss), accuracy, AUC
        (Receiver Operating Curve Area Under the Curve), precision, recall
        and F1.
    padding_value : numeric
        Value to use in the padding, to fill the sequences.
    output_rounded : bool, default False
        If True, the output is rounded, to represent the class assigned by
        the model, instead of just probabilities (>= 0.5 rounded to 1,
        otherwise it's 0)
    experiment : comet_ml.Experiment, default None
        Represents a connection to a Comet.ml experiment to which the
        metrics performance is uploaded, if specified.
    set_name : str
        Defines what name to give to the set when uploading the metrics
        values to the specified Comet.ml experiment.
    seq_final_outputs : bool, default False
        If set to true, the function only returns the ouputs given at each
        sequence's end.
    cols_to_remove : list of ints, default [0, 1]
        List of indeces of columns to remove from the features before feeding to
        the model. This tend to be the identifier columns, such as subject_id
        and ts (timestamp).

    Returns
    -------
    output : torch.Tensor
        Contains the output scores (or classes, if output_rounded is set to
        True) for all of the input data.
    metrics_vals : dict of floats
        Dictionary containing the calculated performance on each of the
        specified metrics.
    '''
    # Guarantee that the model is in evaluation mode, so as to deactivate dropout
    model.eval()

    # Create an empty dictionary with all the possible metrics
    metrics_vals = {'loss': None,
                    'accuracy': None,
                    'AUC': None,
                    'precision': None,
                    'recall': None,
                    'F1': None}

    # Initialize the metrics
    if 'loss' in metrics:
        loss = 0
    if 'accuracy' in metrics:
        acc = 0
    if 'AUC' in metrics:
        auc = 0
    if 'precision' in metrics:
        prec = 0
    if 'recall' in metrics:
        rcl = 0
    if 'F1' in metrics:
        f1_score = 0

    # Check if the user wants to do inference directly on a PyTorch tensor
    if dataloader is None and data is not None:
        features, labels = data[0].float(), data[1].float()             # Make the data have type float instead of double, as it would cause problems
        features, labels, x_lengths = padding.sort_by_seq_len(features, seq_len_dict, labels) # Sort the data by sequence length

        # Remove unwanted columns from the data
        features_idx = list(range(features.shape[2]))
        [features_idx.remove(column) for column in cols_to_remove]
        features = features[:, :, features_idx]
        scores = model.forward(features, x_lengths)                     # Feedforward the data through the model

        # Adjust the labels so that it gets the exact same shape as the predictions
        # (i.e. sequence length = max sequence length of the current batch, not the max of all the data)
        labels = torch.nn.utils.rnn.pack_padded_sequence(labels, x_lengths, batch_first=True)
        labels, _ = torch.nn.utils.rnn.pad_packed_sequence(labels, batch_first=True, padding_value=padding_value)

        mask = (labels <= 1).view_as(scores).float()                    # Create a mask by filtering out all labels that are not a padding value
        unpadded_labels = torch.masked_select(labels.contiguous().view_as(scores), mask.byte()) # Completely remove the padded values from the labels using the mask
        unpadded_scores = torch.masked_select(scores, mask.byte())      # Completely remove the padded values from the scores using the mask
        pred = torch.round(unpadded_scores)                             # Get the predictions

        if output_rounded:
            # Get the predicted classes
            output = pred.int()
        else:
            # Get the model scores (class probabilities)
            output = unpadded_scores

        if seq_final_outputs:
            # Only get the outputs retrieved at the sequences' end
            # Cumulative sequence lengths
            final_seq_idx = np.cumsum(x_lengths) - 1

            # Get the outputs of the last instances of each sequence
            output = output[final_seq_idx]

        if any(mtrc in metrics for mtrc in ['precision', 'recall', 'F1']):
            # Calculate the number of true positives, false negatives, true negatives and false positives
            true_pos = int(sum(torch.masked_select(pred, unpadded_labels.byte())))
            false_neg = int(sum(torch.masked_select(pred == 0, unpadded_labels.byte())))
            true_neg = int(sum(torch.masked_select(pred == 0, (unpadded_labels == 0).byte())))
            false_pos = int(sum(torch.masked_select(pred, (unpadded_labels == 0).byte())))

        if 'loss' in metrics:
            metrics_vals['loss'] = model.loss(scores, labels, x_lengths).item() # Add the loss of the current batch
        if 'accuracy' in metrics:
            correct_pred = pred == unpadded_labels                          # Get the correct predictions
            metrics_vals['accuracy'] = torch.mean(correct_pred.type(torch.FloatTensor)).item() # Add the accuracy of the current batch, ignoring all padding values
        if 'AUC' in metrics:
            metrics_vals['AUC'] = roc_auc_score(unpadded_labels.numpy(), unpadded_scores.detach().numpy()) # Add the ROC AUC of the current batch
        if 'precision' in metrics:
            curr_prec = true_pos / (true_pos + false_pos)
            metrics_vals['precision'] = curr_prec                           # Add the precision of the current batch
        if 'recall' in metrics:
            curr_rcl = true_pos / (true_pos + false_neg)
            metrics_vals['recall'] = curr_rcl                               # Add the recall of the current batch
        if 'F1' in metrics:
            # Check if precision has not yet been calculated
            if 'curr_prec' not in locals():
                curr_prec = true_pos / (true_pos + false_pos)
            # Check if recall has not yet been calculated
            if 'curr_rcl' not in locals():
                curr_rcl = true_pos / (true_pos + false_neg)
            metrics_vals['F1'] = 2 * curr_prec * curr_rcl / (curr_prec + curr_rcl) # Add the F1 score of the current batch

        return output, metrics_vals

    # Initialize the output
    output = torch.tensor([]).int()

    # Evaluate the model on the set
    for features, labels in dataloader:
        # Turn off gradients, saves memory and computations
        with torch.no_grad():
            features, labels = features.float(), labels.float()             # Make the data have type float instead of double, as it would cause problems
            features, labels, x_lengths = padding.sort_by_seq_len(features, seq_len_dict, labels) # Sort the data by sequence length

            # Remove unwanted columns from the data
            features_idx = list(range(features.shape[2]))
            [features_idx.remove(column) for column in cols_to_remove]
            features = features[:, :, features_idx]
            scores = model.forward(features, x_lengths)                     # Feedforward the data through the model

            # Adjust the labels so that it gets the exact same shape as the predictions
            # (i.e. sequence length = max sequence length of the current batch, not the max of all the data)
            labels = torch.nn.utils.rnn.pack_padded_sequence(labels, x_lengths, batch_first=True)
            labels, _ = torch.nn.utils.rnn.pad_packed_sequence(labels, batch_first=True, padding_value=padding_value)

            mask = (labels <= 1).view_as(scores).float()                    # Create a mask by filtering out all labels that are not a padding value
            unpadded_labels = torch.masked_select(labels.contiguous().view_as(scores), mask.byte()) # Completely remove the padded values from the labels using the mask
            unpadded_scores = torch.masked_select(scores, mask.byte())      # Completely remove the padded values from the scores using the mask
            pred = torch.round(unpadded_scores)                             # Get the predictions

            if output_rounded:
                # Get the predicted classes
                output = torch.cat([output, pred.int()])
            else:
                # Get the model scores (class probabilities)
                output = torch.cat([output.float(), unpadded_scores])

            if seq_final_outputs:
                # Indeces at the end of each sequence
                final_seq_idx = [n_subject*features.shape[1]+x_lengths[n_subject]-1 for n_subject in range(features.shape[0])]

                # Get the outputs of the last instances of each sequence
                output = output[final_seq_idx]

            if any(mtrc in metrics for mtrc in ['precision', 'recall', 'F1']):
                # Calculate the number of true positives, false negatives, true negatives and false positives
                true_pos = int(sum(torch.masked_select(pred, unpadded_labels.byte())))
                false_neg = int(sum(torch.masked_select(pred == 0, unpadded_labels.byte())))
                true_neg = int(sum(torch.masked_select(pred == 0, (unpadded_labels == 0).byte())))
                false_pos = int(sum(torch.masked_select(pred, (unpadded_labels == 0).byte())))

            if 'loss' in metrics:
                loss += model.loss(scores, labels, x_lengths)               # Add the loss of the current batch
            if 'accuracy' in metrics:
                correct_pred = pred == unpadded_labels                      # Get the correct predictions
                acc += torch.mean(correct_pred.type(torch.FloatTensor))     # Add the accuracy of the current batch, ignoring all padding values
            if 'AUC' in metrics:
                auc += roc_auc_score(unpadded_labels.numpy(), unpadded_scores.detach().numpy()) # Add the ROC AUC of the current batch
            if 'precision' in metrics:
                curr_prec = true_pos / (true_pos + false_pos)
                prec += curr_prec                                           # Add the precision of the current batch
            if 'recall' in metrics:
                curr_rcl = true_pos / (true_pos + false_neg)
                rcl += curr_rcl                                             # Add the recall of the current batch
            if 'F1' in metrics:
                # Check if precision has not yet been calculated
                if 'curr_prec' not in locals():
                    curr_prec = true_pos / (true_pos + false_pos)
                # Check if recall has not yet been calculated
                if 'curr_rcl' not in locals():
                    curr_rcl = true_pos / (true_pos + false_neg)
                f1_score += 2 * curr_prec * curr_rcl / (curr_prec + curr_rcl) # Add the F1 score of the current batch

    # Calculate the average of the metrics over the batches
    if 'loss' in metrics:
        metrics_vals['loss'] = loss / len(dataloader)
        metrics_vals['loss'] = metrics_vals['loss'].item()                  # Get just the value, not a tensor
    if 'accuracy' in metrics:
        metrics_vals['accuracy'] = acc / len(dataloader)
        metrics_vals['accuracy'] = metrics_vals['accuracy'].item()          # Get just the value, not a tensor
    if 'AUC' in metrics:
        metrics_vals['AUC'] = auc / len(dataloader)
    if 'precision' in metrics:
        metrics_vals['precision'] = prec / len(dataloader)
    if 'recall' in metrics:
        metrics_vals['recall'] = rcl / len(dataloader)
    if 'F1' in metrics:
        metrics_vals['F1'] = f1_score / len(dataloader)

    if experiment is not None:
        # Log metrics to Comet.ml
        if 'loss' in metrics:
            experiment.log_metric(f'{set_name}_loss', metrics_vals['loss'])
        if 'accuracy' in metrics:
            experiment.log_metric(f'{set_name}_acc', metrics_vals['accuracy'])
        if 'AUC' in metrics:
            experiment.log_metric(f'{set_name}_auc', metrics_vals['AUC'])
        if 'precision' in metrics:
            experiment.log_metric(f'{set_name}_prec', metrics_vals['precision'])
        if 'recall' in metrics:
            experiment.log_metric(f'{set_name}_rcl', metrics_vals['recall'])
        if 'F1' in metrics:
            experiment.log_metric(f'{set_name}_f1_score', metrics_vals['F1'])

    return output, metrics_vals


def train(model, train_dataloader, val_dataloader, test_dataloader, seq_len_dict,
          batch_size=32, n_epochs=50, lr=0.001, model_path='models/',
          padding_value=999999, do_test=True, log_comet_ml=False,
          comet_ml_api_key=None, comet_ml_project_name=None,
          comet_ml_workspace=None, comet_ml_save_model=False, experiment=None,
          features_list=None, get_val_loss_min=False):
    '''Trains a given model on the provided data.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model which is trained on the data to perform a
        classification task.
    train_dataloader : torch.utils.data.DataLoader
        Data loader which will be used to get data batches during training.
    val_dataloader : torch.utils.data.DataLoader
        Data loader which will be used to get data batches when evaluating
        the model's performance on a validation set during training.
    test_dataloader : torch.utils.data.DataLoader
        Data loader which will be used to get data batches whe evaluating
        the model's performance on a test set, after finishing the
        training process.
    seq_len_dict : dict
        Dictionary containing the sequence lengths for each index of the
        original dataframe. This allows to ignore the padding done in
        the fixed sequence length tensor.
    batch_size : int, default 32
        Defines the batch size, i.e. the number of samples used in each
        training iteration to update the model's weights.
    n_epochs : int, default 50
        Number of epochs, i.e. the number of times the training loop
        iterates through all of the training data.
    lr : float, default 0.001
        Learning rate used in the optimization algorithm.
    model_path : string, default 'models/'
        Path where the model will be saved. By default, it saves in
        the directory named "models".
    padding_value : numeric
        Value to use in the padding, to fill the sequences.
    do_test : bool, default True
        If true, evaluates the model on the test set, after completing
        the training.
    log_comet_ml : bool, default False
        If true, makes the code upload a training report and metrics
        to comet.ml, a online platform which allows for a detailed
        version control for machine learning models.
    comet_ml_api_key : string, default None
        Comet.ml API key used when logging data to the platform.
    comet_ml_project_name : string, default None
        Name of the comet.ml project used when logging data to the
        platform.
    comet_ml_workspace : string, default None
        Name of the comet.ml workspace used when logging data to the
        platform.
    comet_ml_save_model : bool, default False
        If set to true, uploads the model with the lowest validation loss
        to comet.ml when logging data to the platform.
    experiment : comet_ml.Experiment, default None
        Defines an already existing Comet.ml experiment object to be used in the
        training. If not defined (None), a new experiment is created inside the
        method. In any case, a Comet.ml experiment is only used if log_comet_ml
        is set to True and the remaining necessary Comet.ml related parameters
        (comet_ml_api_key, comet_ml_project_name, comet_ml_workspace) are
        properly set up.
    features_list : list of strings, default None
        Names of the features being used in the current pipeline. This
        will be logged to comet.ml, if activated, in order to have a
        more detailed version control.
    get_val_loss_min : bool, default False
        If set to True, besides returning the trained model, the method also
        returns the minimum validation loss found during training.

    Returns
    -------
    model : nn.Module
        The same input model but with optimized weight values.
    val_loss_min : float
        If get_val_loss_min is set to True, the method also returns the minimum
        validation loss found during training.
    '''
    if log_comet_ml:
        if experiment is None:
            # Create a new Comet.ml experiment
            experiment = Experiment(api_key=comet_ml_api_key, project_name=comet_ml_project_name, workspace=comet_ml_workspace)
        experiment.log_other("completed", False)
        experiment.log_other("random_seed", random_seed)

        # Report hyperparameters to Comet.ml
        hyper_params = {"batch_size": batch_size,
                        "n_epochs": n_epochs,
                        "n_hidden": model.n_hidden,
                        "n_layers": model.n_layers,
                        "learning_rate": lr,
                        "p_dropout": model.p_dropout,
                        "random_seed": random_seed}
        experiment.log_parameters(hyper_params)

        if features_list is not None:
            # Log the names of the features being used
            experiment.log_other("features_list", features_list)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)                 # Adam optimization algorithm
    step = 0                                                                # Number of iteration steps done so far
    print_every = 10                                                        # Steps interval where the metrics are printed
    train_on_gpu = torch.cuda.is_available()                                # Check if GPU is available
    val_loss_min = np.inf                                                   # Start with an infinitely big minimum validation loss

    for epoch in range(1, n_epochs+1):
        # Initialize the training metrics
        train_loss = 0
        train_acc = 0
        train_auc = 0

        try:
            # Loop through the training data
            for features, labels in train_dataloader:
                model.train()                                                   # Activate dropout to train the model
                optimizer.zero_grad()                                           # Clear the gradients of all optimized variables

                if train_on_gpu:
                    features, labels = features.cuda(), labels.cuda()           # Move data to GPU

                features, labels = features.float(), labels.float()             # Make the data have type float instead of double, as it would cause problems
                features, labels, x_lengths = sort_by_seq_len(features, seq_len_dict, labels) # Sort the data by sequence length
                scores = model.forward(features[:, :, 2:], x_lengths)           # Feedforward the data through the model
                                                                                # (the 2 is there to avoid using the identifier features in the predictions)

                # Adjust the labels so that it gets the exact same shape as the predictions
                # (i.e. sequence length = max sequence length of the current batch, not the max of all the data)
                labels = torch.nn.utils.rnn.pack_padded_sequence(labels, x_lengths, batch_first=True)
                labels, _ = torch.nn.utils.rnn.pad_packed_sequence(labels, batch_first=True, padding_value=padding_value)

                loss = model.loss(scores, labels, x_lengths)                    # Calculate the cross entropy loss
                loss.backward()                                                 # Backpropagate the loss
                optimizer.step()                                                # Update the model's weights
                train_loss += loss                                              # Add the training loss of the current batch
                mask = (labels <= 1).view_as(scores).float()                    # Create a mask by filtering out all labels that are not a padding value
                unpadded_labels = torch.masked_select(labels.contiguous().view_as(scores), mask.byte()) # Completely remove the padded values from the labels using the mask
                unpadded_scores = torch.masked_select(scores, mask.byte())      # Completely remove the padded values from the scores using the mask
                pred = torch.round(unpadded_scores)                             # Get the predictions
                correct_pred = pred == unpadded_labels                          # Get the correct predictions
                train_acc += torch.mean(correct_pred.type(torch.FloatTensor))   # Add the training accuracy of the current batch, ignoring all padding values
                train_auc += roc_auc_score(unpadded_labels.numpy(), unpadded_scores.detach().numpy()) # Add the training ROC AUC of the current batch
                step += 1                                                       # Count one more iteration step
                model.eval()                                                    # Deactivate dropout to test the model

                # Initialize the validation metrics
                val_loss = 0
                val_acc = 0
                val_auc = 0

                # Loop through the validation data
                for features, labels in val_dataloader:
                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        features, labels = features.float(), labels.float()             # Make the data have type float instead of double, as it would cause problems
                        features, labels, x_lengths = sort_by_seq_len(features, seq_len_dict, labels) # Sort the data by sequence length
                        scores = model.forward(features[:, :, 2:], x_lengths)           # Feedforward the data through the model
                                                                                        # (the 2 is there to avoid using the identifier features in the predictions)

                        # Adjust the labels so that it gets the exact same shape as the predictions
                        # (i.e. sequence length = max sequence length of the current batch, not the max of all the data)
                        labels = torch.nn.utils.rnn.pack_padded_sequence(labels, x_lengths, batch_first=True)
                        labels, _ = torch.nn.utils.rnn.pad_packed_sequence(labels, batch_first=True, padding_value=padding_value)

                        val_loss += model.loss(scores, labels, x_lengths)               # Calculate and add the validation loss of the current batch
                        mask = (labels <= 1).view_as(scores).float()                    # Create a mask by filtering out all labels that are not a padding value
                        unpadded_labels = torch.masked_select(labels.contiguous().view_as(scores), mask.byte()) # Completely remove the padded values from the labels using the mask
                        unpadded_scores = torch.masked_select(scores, mask.byte())      # Completely remove the padded values from the scores using the mask
                        pred = torch.round(unpadded_scores)                             # Get the predictions
                        correct_pred = pred == unpadded_labels                          # Get the correct predictions
                        val_acc += torch.mean(correct_pred.type(torch.FloatTensor))     # Add the validation accuracy of the current batch, ignoring all padding values
                        val_auc += roc_auc_score(unpadded_labels.numpy(), unpadded_scores.detach().numpy()) # Add the validation ROC AUC of the current batch

                # Calculate the average of the metrics over the batches
                val_loss = val_loss / len(val_dataloader)
                val_acc = val_acc / len(val_dataloader)
                val_auc = val_auc / len(val_dataloader)


                # Display validation loss
                if step%print_every == 0:
                    print(f'Epoch {epoch} step {step}: Validation loss: {val_loss}; Validation Accuracy: {val_acc}; Validation AUC: {val_auc}')

                # Check if the performance obtained in the validation set is the best so far (lowest loss value)
                if val_loss < val_loss_min:
                    print(f'New minimum validation loss: {val_loss_min} -> {val_loss}.')

                    # Update the minimum validation loss
                    val_loss_min = val_loss

                    # Get the current day and time to attach to the saved model's name
                    current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')

                    # Filename and path where the model will be saved
                    model_filename = f'{model_path}checkpoint_{current_datetime}.pth'

                    print(f'Saving model in {model_filename}')

                    # Save the best performing model so far, a long with additional information to implement it
                    checkpoint = {'n_inputs': model.n_inputs,
                                  'n_hidden': model.n_hidden,
                                  'n_outputs': model.n_outputs,
                                  'n_layers': model.n_layers,
                                  'p_dropout': model.p_dropout,
                                  'state_dict': model.state_dict()}
                    torch.save(checkpoint, model_filename)

                    if log_comet_ml and comet_ml_save_model:
                        # Upload the model to Comet.ml
                        experiment.log_asset(file_data=model_filename, overwrite=True)

            # Calculate the average of the metrics over the epoch
            train_loss = train_loss / len(train_dataloader)
            train_acc = train_acc / len(train_dataloader)
            train_auc = train_auc / len(train_dataloader)

            if log_comet_ml:
                # Log metrics to Comet.ml
                experiment.log_metric("train_loss", train_loss, step=epoch)
                experiment.log_metric("train_acc", train_acc, step=epoch)
                experiment.log_metric("train_auc", train_auc, step=epoch)
                experiment.log_metric("val_loss", val_loss, step=epoch)
                experiment.log_metric("val_acc", val_acc, step=epoch)
                experiment.log_metric("val_auc", val_auc, step=epoch)
                experiment.log_metric("epoch", epoch)

            # Print a report of the epoch
            print(f'Epoch {epoch}: Training loss: {train_loss}; Training Accuracy: {train_acc}; Training AUC: {train_auc}; \
                    Validation loss: {val_loss}; Validation Accuracy: {val_acc}; Validation AUC: {val_auc}')
            print('----------------------')
        except:
            warnings.warn(f'There was a problem doing training epoch {epoch}. Ending training.')

    try:
        if do_test and model_filename is not None:
            # Load the model with the best validation performance
            model = load_checkpoint(model_filename)

            # Run inference on the test data
            model_inference(model, seq_len_dict, dataloader=test_dataloader , experiment=experiment)
    except UnboundLocalError:
        warnings.warn('Inference failed due to non existent saved models. Skipping evaluation on test set.')
    except:
        warnings.warn(f'Inference failed due to {sys.exc_info()[0]}. Skipping evaluation on test set.')

    if log_comet_ml:
        # Only report that the experiment completed successfully if it finished the training without errors
        experiment.log_other("completed", True)

    if get_val_loss_min:
        # Also return the minimum validation loss alongside the corresponding model
        return model, val_loss_min.item()

    return model
