import torch                                            # PyTorch to create and apply deep learning models
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np                                      # NumPy to handle numeric and NaN operations
import warnings                                         # Print warnings for bad practices
import utils                                            # Generic and useful methods

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

def create_train_sets(dataset, test_train_ratio=0.2, validation_ratio=0.1, batch_size=32,
                      get_indeces=True, random_seed=42, shuffle_dataset=True):
    '''Distributes the data into train, validation and test sets and returns the respective data loaders.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset object which will be used to train, validate and test the model.
    test_train_ratio : float, default 0.8
        Number from 0 to 1 which indicates the percentage of the data
        which will be used as a test set. The remaining percentage
        is used in the training and validation sets.
    validation_ratio : float, default 0.1
        Number from 0 to 1 which indicates the percentage of the data
        from the training set which is used for validation purposes.
        A value of 0.0 corresponds to not using validation.
    batch_size : int, default 32
        Defines the batch size, i.e. the number of samples used in each
        training iteration to update the model's weights.
    get_indeces : bool, default True
        If set to True, the function returns the dataloader objects of
        the train, validation and test sets and also the indices of the
        sets' data. Otherwise, it only returns the data loaders.
    random_seed : int or tuple, default 42
        Seed used when shuffling the data.
    shuffle_dataset : bool, default True
        If set to True, the data of which set is shuffled.

    Returns
    -------
    train_data : torch.Tensor
        Data which will be used during training.
    val_data : torch.Tensor
        Data which will be used to evaluate the model's performance
        on a validation set during training.
    test_data : torch.Tensor
        Data which will be used to evaluate the model's performance
        on a test set, after finishing the training process.
    '''
    # Create data indices for training and test splits
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_train_ratio * dataset_size))
    if shuffle_dataset:
        if type(random_seed) is int:
            np.random.seed(random_seed)
        elif type(random_seed) is tuple:
            np.random.set_state(random_seed)
        else:
            raise(f'ERROR: {type(random_seed)} is an incorrect random seed type. It should either be an integer or a random state tuple.')
        np.random.shuffle(indices)
    train_indices, test_indices = indices[test_split:], indices[:test_split]

    # Create data indices for training and validation splits
    train_dataset_size = len(train_indices)
    val_split = int(np.floor(validation_ratio * train_dataset_size))
    if shuffle_dataset:
        np.random.shuffle(train_indices)
    train_indices, val_indices = train_indices[val_split:], train_indices[:val_split]

    # Create data samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create dataloaders for each set, which will allow loading batches
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    if get_indeces:
        # Return the data loaders and the indices of the sets
        return train_dataloader, val_dataloader, test_dataloader, train_indices, val_indices, test_indices
    else:
        # Just return the data loaders of each set
        return train_dataloader, val_dataloader, test_dataloader


# [TODO] Create a generic train method that can train any relevant machine learning model on the input data
# [TODO] Create a generic inference method that can run inference with any relevant machine learning model on the input data