from comet_ml import Experiment                         # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                                            # PyTorch to create and apply deep learning models
import numpy as np                                      # NumPy to handle numeric and NaN operations
from tqdm.auto import tqdm                              # tqdm allows to track code execution progress
import numbers                                          # numbers allows to check if data is numeric

# [TODO] Make the random seed a user option (randomly generated or user defined)
# Random seed used in PyTorch and NumPy's random operations (such as weight initialization)
# Automatic seed
random_seed = np.random.get_state()
np.random.set_state(random_seed)
torch.manual_seed(random_seed[1][0])

# Methods

def set_random_seed(num):
    '''Set a user specified seed to use in stochastic (i.e. random) processes.
    This method should be called before importing packages which use a
    random seed.

    Parameters
    ----------
    num : int
        Number that will serve as the random seed.

    Returns
    -------
    None
    '''
    global random_seed
    random_seed = num
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    return


def apply_dict_convertion(x, conv_dict, nan_value=0):
    '''Safely apply a convertion through a dictionary.

    Parameters
    ----------
    x : anything
        Object that will be converted through the dictionary.
    conv_dict : dict
        Dictionary used to convert the input object.
    nan_value: anything
        Value or object that repressents missingness.

    Returns
    -------
    x : anything
        Converted object.
    '''
    # Check if it's a missing value (NaN)
    if isinstance(x, numbers.Number):
        if np.isnan(x):
            return nan_value
    # Must be a convertable value
    else:
        return conv_dict[x]


def invert_dict(x):
    '''Invert a dictionary, switching its keys with its values.

    Parameters
    ----------
    x : dict
        Dictionary to be inverted

    Returns
    -------
    x : dict:
        Inverted dictionary
    '''
    return {v: k for k, v in x.items()}


def is_definitely_string(x):
    '''Reports if a value is actually a real string or if it has some number in it.

    Parameters
    ----------
    x
        Any value which will be judged to be either a real string or numeric.

    Returns
    -------
    boolean
        Returns a boolean, being it True if it really is a string or False if it's
        either numeric data or a string with a number inside.
    '''
    if isinstance(x, int) or isinstance(x, float):
        return False

    try:
        float(x)
        return False

    except:
        return isinstance(x, str)


def is_string_nan(x):
    '''Indicates if a string corresponds to a missing value.

    Parameters
    ----------
    x : string
        A string that will be compared with possible missing value
        representations.

    Returns
    -------
    boolean
        Returns a boolean, being it True if the string corresponds to a missing
        value representation or False if it doesn't.
    '''
    # Considering the possibility of just 3 more random characters in NaN-like strings
    if ('other' in x.lower() and len(x) < 9) \
    or ('null' in x.lower() and len(x) < 7) \
    or ('nan' in x.lower() and len(x) < 6) \
    or ('discrepancy' in x.lower() and len(x) < 14) \
    or all([char == ' ' for char in x]) \
    or all([char == '_' for char in x]) \
    or ('unknown' in x.lower()) \
    or ('not obtainable' in x.lower()) \
    or ('not obtained' in x.lower()) \
    or ('not applicable' in x.lower()) \
    or ('not available' in x.lower()) \
    or ('not evaluated' in x.lower()):
        return True
    else:
        return False


def in_ipynb():
    '''Detect if code is running in a IPython notebook, such as in Jupyter Lab.'''
    try:
        return str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>"
    except:
        # Not on IPython if get_ipython fails
        return False


def iterations_loop(x, see_progress=True):
    '''Determine if a progress bar is shown or not.'''
    if see_progress:
        # Use a progress bar
        return tqdm(x)
    else:
        # Don't show any progress bar if see_progress is False
        return x


def set_bar_color(values, ids, seq_len, threshold=0,
                  neg_color='rgba(30,136,229,1)', pos_color='rgba(255,13,87,1)'):
    '''Determine each bar's color in a bar chart, according to the values being
    plotted and the predefined threshold.

    Parameters
    ----------
    values : numpy.Array
        Array containing the values to be plotted.
    ids : int or list of ints
        ID or list of ID's that select which time series / sequences to use in
        the color selection.
    seq_len : int or list of ints
        Single or multiple sequence lengths, which represent the true, unpadded
        size of the input sequences.
    threshold : int or float, default 0
        Value to use as a threshold in the plot's color selection. In other
        words, values that exceed this threshold will have one color while the
        remaining have a different one, as specified in the parameters.
    pos_color : string
        Color to use in the bars corresponding to threshold exceeding values.
    neg_color : string
        Color to use in the bars corresponding to values bellow the threshold.

    Returns
    -------
    colors : list of strings
        Resulting bar colors list.'''
    if type(ids) is list:
        # Create a list of lists, with the colors for each sequences' instances
        return [[pos_color if val > 0 else neg_color for val in values[id, :seq_len]]
                for id in ids]
    else:
        # Create a single list, with the colors for the sequence's instances
        return [pos_color if val > 0 else neg_color for val in values[ids, :seq_len]]
