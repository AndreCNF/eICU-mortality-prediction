import pandas as pd                                     # Pandas to handle the data in dataframes
import numpy as np                                      # NumPy to handle numeric and NaN operations
import numbers                                          # numbers allows to check if data is numeric
import warnings                                         # Print warnings for bad practices
import utils                                            # Generic and useful methods

# Random seed used in NumPy's random operations
random_seed = utils.random_seed

if isinstance(random_seed, int):
    # Set user specified random seed
    np.random.seed(random_seed)
else:
    # Set completely random seed from utils
    np.random.set_state(random_seed)

# Ignore Dask's 'meta' warning
warnings.filterwarnings("ignore", message="`meta` is not specified, inferred from partial data. Please provide `meta` if the result is unexpected.")

# Methods

def dataframe_missing_values(df, column=None):
    '''Returns a dataframe with the percentages of missing values of every column
    of the original dataframe.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Original dataframe which the user wants to analyze for missing values.
    column : string, default None
        Optional argument which, if provided, makes the function only return
        the percentage of missing values in the specified column.

    Returns
    -------
    missing_value_df : pandas.DataFrame or dask.DataFrame
        DataFrame containing the percentages of missing values for each column.
    col_percent_missing : float
        If the "column" argument is provided, the function only returns a float
        corresponfing to the percentage of missing values in the specified column.
    '''
    if column is None:
        columns = df.columns
        percent_missing = df.isnull().sum() * 100 / len(df)
        if 'dask' in str(type(df)):
            # Make sure that the values are computed, in case we're using Dask
            percent_missing = percent_missing.compute()
        missing_value_df = pd.DataFrame({'column_name': columns,
                                         'percent_missing': percent_missing})
        missing_value_df.sort_values('percent_missing', inplace=True)
        return missing_value_df
    else:
        col_percent_missing = df[column].isnull().sum() * 100 / len(df)
        return col_percent_missing


def is_one_hot_encoded_column(df, column):
    '''Checks if a given column is one hot encoded.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe that will be used, which contains the specified column.
    column : string
        Name of the column that will be checked for one hot encoding.

    Returns
    -------
    bool
        Returns true if the column is in one hot encoding format.
        Otherwise, returns false.
    '''
    n_unique_values = df[column].nunique()
    if 'dask' in str(type(df)):
        # Make sure that the number of unique values are computed, in case we're using Dask
        n_unique_values = n_unique_values.compute()
    # Check if it only has 2 possible values
    if n_unique_values == 2:
        unique_values = df[column].unique()
        if 'dask' in str(type(df)):
            # Make sure that the unique values are computed, in case we're using Dask
            unique_values = unique_values.compute()
        # Check if the possible values are all numeric
        if all([isinstance(x, numbers.Number) for x in unique_values]):
            # Check if the only possible values are 0 and 1 (and ignore NaN's)
            if (np.sort(list(set(np.nan_to_num(unique_values)))) == [0, 1]).all():
                return True
    return False


def list_one_hot_encoded_columns(df):
    '''Lists the columns in a dataframe which are in a one hot encoding format.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe that will be used checked for one hot encoded columns.

    Returns
    -------
    list of strings
        Returns a list of the column names which correspond to one hot encoded columns.
    '''
    return [col for col in df.columns if is_one_hot_encoded_column(df, col)]


def find_subject_idx(data, subject_id, subject_id_col=0):
    '''Find the index that corresponds to a given subject in a data tensor.

    Parameters
    ----------
    data : torch.Tensor
        PyTorch tensor containing the data on which the subject's index will be
        searched for.
    subject_id : int or string
        Unique identifier of the subject whose index on the data tensor one
        wants to find out.
    subject_id_col : int, default 0
        The number of the column in the data tensor that stores the subject
        identifiers.

    Returns
    -------
    idx : int
        Index where the specified subject appears in the data tensor.'''
    return (data[:, 0, subject_id_col] == subject_id).nonzero().item()


def find_row_contains_word(df, feature, words):
    '''Find if each row in a specified dataframe string feature contains some
    word from a list.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe containing the feature on which to run the words search.
    feature : string
        Name of the feature through which the method will search if strings
        contain any of the specified words.
    words : list of strings
        List of the words to search for in the feature's rows. Even if searching
        for the existence of a single word, it should be specified inside a list.

    Returns
    -------
    row_contains_word : pandas.Series or dask.Series
        Boolean series indicating for each row of the dataframe if its specified
        feature contains any of the words that the user is looking for.'''
    row_contains_word = None
    if not df[feature].dtype == 'object':
        raise Exception(f'ERROR: The specified feature should have type \'object\', not type {df[feature].dtype}.')
    if any([not isinstance(word, str) for word in words]):
        raise Exception('ERROR: All words in the specified words list should be strings.')
    if 'dask' in str(type(df)):
        row_contains_word = df[feature].apply(lambda row: any([word.lower() in row.lower() for word in words]),
                                              meta=('row', bool))
    elif 'pandas' in str(type(df)):
        row_contains_word = df[feature].apply(lambda row: any([word.lower() in row.lower() for word in words]))
    else:
        raise Exception(f'ERROR: `df` should either be a Pandas or Dask dataframe, not {type(df)}.')
    return row_contains_word


def get_element(x, n, till_the_end=False):
    '''Try to get an element from a list. Useful for nagging apply and map
    dataframe operations.

    Parameters
    ----------
    x : list or numpy.ndarray
        List from which to get an element.
    n : int
        Index of the element from the list that we want to retrieve.
    till_the_end : bool, default False
        If set to true, all elements from index n until the end of the list will
        be fetched. Otherwise, the method only returns the n'th element.

    Returns
    -------
    y : anything
        Returns the n'th element of the list or NaN if it's not found.
    '''
    try:
        if till_the_end:
            return x[n:]
        else:
            return x[n]
    except:
        return np.nan


def get_element_from_split(orig_string, n, separator='|', till_the_end=False):
    '''Split a string by a specified separator and return the n'th element of
    the obtained list of words.

    Parameters
    ----------
    orig_string : string
        Original string on which to apply the splitting and element retrieval.
    n : int
        The index of the element to return from the post-split list of words.
    separator : string, default '|'
        Symbol that concatenates each string's words, which will be used in the
        splitting.
    till_the_end : bool, default False
        If set to true, all elements from index n until the end of the list will
        be fetched. Otherwise, the method only returns the n'th element.

    Returns
    -------
    n_element : string
        The n'th element from the split string.
    '''
    # Split the string, by the specified separator, to get the list of all words
    split_list = orig_string.split(separator)
    # Get the n'th element of the list
    n_element = get_element(split_list, n, till_the_end)
    if till_the_end:
        # Rejoin the elements of the list by their separator
        n_element = separator.join(n_element)
    return n_element
