from comet_ml import Experiment                         # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                                            # PyTorch to create and apply deep learning models
from torch import nn, optim                             # nn for neural network layers and optim for training optimizers
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd                                     # Pandas to handle the data in dataframes
import dask.dataframe as dd                             # Dask to handle big data in dataframes
from datetime import datetime                           # datetime to use proper date and time formats
import os                                               # os handles directory/workspace changes
import numpy as np                                      # NumPy to handle numeric and NaN operations
from tqdm.auto import tqdm                              # tqdm allows to track code execution progress
import numbers                                          # numbers allows to check if data is numeric
from NeuralNetwork import NeuralNetwork                 # Import the neural network model class
from sklearn.metrics import roc_auc_score               # ROC AUC model performance metric
import warnings                                         # Print warnings for bad practices
import sys                                              # Identify types of exceptions
from functools import reduce                            # Parallelize functions

# [TODO] Make the random seed a user option (randomly generated or user defined)
# Random seed used in PyTorch and NumPy's random operations (such as weight initialization)
# Automatic seed
# random_seed = np.random.get_state()
# np.random.set_state(random_seed)
# torch.manual_seed(random_seed[1][0])
# Manual seed
random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Ignore Dask's 'meta' warning
warnings.filterwarnings("ignore", message="`meta` is not specified, inferred from partial data. Please provide `meta` if the result is unexpected.")

# Exceptions

class ColumnNotFoundError(Exception):
   """Raised when the column name is not found in the dataframe."""
   pass


# Auxiliary functions

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


def get_clean_label(orig_label, clean_labels, column_name=None):
    '''Gets the clean version of a given label.

    Parameters
    ----------
    orig_label : string
        Original label name that needs to be converted to the new format.
    clean_labels : dict
        Dictionary that converts each original label into a new, cleaner designation.
    column_name : string, default None
        Optional parameter to indicate a column name, which is used to specify better the
        missing values.

    Returns
    -------
    key : string
        Returns the dictionary key from clean_labels that corresponds to the translation
        given to the input label orig_label.
    '''
    for key in clean_labels:
        if orig_label in clean_labels[key]:
            return key

    # Remaining labels (or lack of one) are considered as missing data
    if column_name is not None:
        return f'{column_name}_missing_value'
    else:
        return 'missing_value'


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


def clean_naming(df, column):
    '''Change categorical values to only have lower case letters and underscores.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe that contains the column to be cleaned.
    column : string
        Name of the dataframe's column which needs to have its string values
        standardized.

    Returns
    -------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe with its string column already cleaned.
    '''
    if 'dask' in str(type(df)):
        df[column] = df[column].map(lambda x: str(x).lower().replace('  ', '') \
                                                            .replace(' ', '_') \
                                                            .replace(',', '_and'), meta=('x', str))
    else:
        df[column] = df[column].map(lambda x: str(x).lower().replace('  ', '') \
                                                            .replace(' ', '_') \
                                                            .replace(',', '_and'))
    return df


def one_hot_encoding_dataframe(df, columns, clean_name=True, has_nan=False,
                               join_rows=True, join_by=['patientunitstayid', 'ts'],
                               get_new_column_names=False):
    '''Transforms specified column(s) from a dataframe into a one hot encoding
    representation.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe that will be used, which contains the specified column.
    columns : list of strings
        Name of the column(s) that will be conveted to one hot encoding. Even if
        it's just one column, please provide inside a list.
    clean_name : bool, default True
        If set to true, changes the name of the categorical values into lower
        case, with words separated by an underscore instead of space.
    has_nan : bool, default False
        If set to true, will first fill the missing values (NaN) with the string
        f'{column}_missing_value'.
    join_rows : bool, default True
        If set to true, will group the rows created by the one hot encoding by
        summing the boolean values in the rows that have the same identifiers.
    join_by : string or list, default ['subject_id', 'ts'])
        Name of the column (or columns) which serves as a unique identifier of
        the dataframe's rows, which will be used in the groupby operation if the
        parameter join_rows is set to true. Can be a string (single column) or a
        list of strings (multiple columns).
    get_new_column_names : bool, default False
        If set to True, the names of the new columns will also be outputed.

    Raises
    ------
    ColumnNotFoundError
        Column name not found in the dataframe.

    Returns
    -------
    ohe_df : pandas.DataFrame or dask.DataFrame
        Returns a new dataframe with the specified column in a one hot encoding
        representation.
    new_column_names : list of strings
        List of the new, one hot encoded columns' names.
    '''
    data = df.copy()

    for col in columns:
        # Check if the column exists
        if col not in df.columns:
            raise ColumnNotFoundError('Column name not found in the dataframe.')

        if has_nan:
            # Fill NaN with "missing_value" name
            data[col] = data[col].fillna(value='missing_value')

        if clean_name:
            # Clean the column's string values to have the same, standard format
            data = clean_naming(data, col)

        # Cast the variable into the built in pandas Categorical data type
        if 'pandas' in str(type(data)):
            data[col] = pd.Categorical(data[col])
    if 'dask' in str(type(data)):
        data = data.categorize(columns)

    if get_new_column_names:
        # Find the previously existing column names
        old_column_names = data.columns

    # Apply the one hot encoding to the specified columns
    if 'dask' in str(type(data)):
        ohe_df = dd.get_dummies(data, columns=columns)
    else:
        ohe_df = pd.get_dummies(data, columns=columns)

    if join_rows:
        # Columns which are one hot encoded
        ohe_columns = list_one_hot_encoded_columns(ohe_df)

        # Group the rows that have the same identifiers
        ohe_df = ohe_df.groupby(join_by).sum(min_count=1).reset_index()

        # Clip the one hot encoded columns to a maximum value of 1
        # (there might be duplicates which cause values bigger than 1)
        ohe_df.loc[:, ohe_columns] = ohe_df[ohe_columns].clip(upper=1)

    if get_new_column_names:
        # Find the new column names and output them
        new_column_names = list(set(ohe_df.columns) - set(old_column_names))
        return ohe_df, new_column_names
    else:
        return ohe_df


def apply_dict_convertion(x, conv_dict, nan_value=0):
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


def category_to_feature(df, categories_feature, values_feature, min_len=None):
    '''Convert a categorical column and its corresponding values column into
    new features, one for each category.
    WARNING: Currently not working properly on a Dask dataframe. Apply .compute()
    to the dataframe to convert it to Pandas, before passing it to this method.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe on which to add the new features.
    categories_feature : string
        Name of the feature that contains the categories that will be converted
        to individual features.
    values_feature : string
        Name of the feature that has each category's corresponding value, which
        may or may not be a category on its own (e.g. it could be numeric values).
    min_len : int, default None
        If defined, only the categories that appear on at least `min_len` rows
        are converted to features.

    Returns
    -------
    data_df : pandas.DataFrame or dask.DataFrame
        Dataframe with the newly created features.
    '''
    # Copy the dataframe to avoid potentially unwanted inplace changes
    data_df = df.copy()
    # Find the unique categories
    categories = data_df[categories_feature].unique()
    if 'dask' in str(type(df)):
        categories = categories.compute()
    # Create a feature for each category
    for category in categories:
        if min_len:
            # Check if the current category has enough data to be worth it to convert to a feature
            if len(data_df[data_df[categories_feature] == category]) < min_len:
                # Ignore the current category
                continue
        # Convert category to feature
        data_df[category] = data_df.apply(lambda x: x[values_feature] if x[categories_feature] == category
                                                    else np.nan, axis=1)
    return data_df


def create_enum_dict(unique_values, nan_value=0):
    '''Enumerate all categories in a specified categorical feature, while also
    attributing a specific number to NaN and other unknown values.

    Parameters
    ----------
    unique_values : list of strings, default None
        Specifies all the unique values to be enumerated.
    nan_value : int, default 0
        Integer number that gets assigned to NaN and NaN-like values.

    Returns
    -------
    enum_dict : dict
        Dictionary containing the mapping between the original values and the
        numbering obtained.
    '''
    # Enumerate the unique values in the categorical feature and put them in a dictionary
    enum_dict = dict(enumerate(unique_values, start=1))
    # Invert the dictionary to have the unique categories as keys and the numbers as values
    enum_dict = invert_dict(enum_dict)
    # Move NaN to key 0
    enum_dict[np.nan] = nan_value
    # Search for NaN-like categories
    for key, val in enum_dict.items():
        if type(key) is str:
            # Considering the possibility of just 3 more random characters in NaN-like strings
            if ('other' in key.lower() and len(key) < 9) \
            or ('unknown' in key.lower() and len(key) < 10) \
            or ('null' in key.lower() and len(key) < 7) \
            or ('nan' in key.lower() and len(key) < 6) \
            or all([char == ' ' for char in key]) \
            or all([char == '_' for char in key]):
                # Move NaN-like key to nan_value
                enum_dict[key] = nan_value
        elif isinstance(key, numbers.Number):
            if np.isnan(key) or str(key).lower() == 'nan':
                # Move NaN-like key to nan_value
                enum_dict[key] = nan_value
    return enum_dict


def enum_categorical_feature(df, feature, nan_value=0, clean_name=True,
                             apply_on_df=True):
    '''Enumerate all categories in a specified categorical feature, while also
    attributing a specific number to NaN and other unknown values.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe which the categorical feature belongs to.
    feature : string
        Name of the categorical feature which will be enumerated.
    nan_value : int, default 0
        Integer number that gets assigned to NaN and NaN-like values.
    clean_name : bool, default True
        If set to True, the method assumes that the feature is of type string
        and it will make sure that all the feature's values are in lower case,
        to reduce duplicate information.
    unique_values : list of strings, default None
        Specifies all the unique values present in the categorical feature.
        If not specified, the method will look for them in the dataframe.

    Returns
    -------
    enum_series : pandas.Series or dask.Series
        Series corresponding to the analyzed feature, after
        enumeration.
    enum_dict : dict
        Dictionary containing the mapping between the original categorical values
        and the numbering obtained.
    '''
    if clean_name:
        # Clean the column's string values to have the same, standard format
        df = clean_naming(df, feature)
    # Get the unique values of the cateforical feature
    unique_values = df[feature].unique()
    if 'dask' in str(type(df)):
        # Make sure that the unique values are computed, in case we're using Dask
        unique_values = unique_values.compute()
    # Enumerate the unique values in the categorical feature and put them in a dictionary
    enum_dict = create_enum_dict(unique_values)
    if not apply_on_df:
        return enum_dict
    else:
        # Create a series from the enumerations of the original feature's categories
        if 'dask' in str(type(df)):
            enum_series = df[feature].map(lambda x: apply_dict_convertion(x, enum_dict, nan_value), meta=('x', int))
        else:
            enum_series = df[feature].map(lambda x: apply_dict_convertion(x, enum_dict, nan_value))
        return enum_series, enum_dict


def enum_category_conversion(df, enum_column, enum_dict, enum_to_category=None):
    '''Convert between enumerated encodings and their respective categories'
    names, in either direction.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe which the categorical feature belongs to.
    enum_column : string
        Name of the categorical feature which is encoded/enumerated. The
        feature's values must be single integer numbers, with a ';' separator if
        more than one category applies to a given row.
    enum_dict : dict
        Dictionary containing the category names that correspond to each
        enumeration number.
    enum_to_category : bool, default None
        Indicator on which the user can specify if the conversion is from
        numerical encodings to string categories names (True) or vice-versa
        (False). By default, it's not defined (None) and the method infers the
        direction of the conversion based on the input dictionary's key type.

    Returns
    -------
    categories : string
        String containing all the categories names of the current row. If more
        than one category is present, their names are separated by the ';'
        separator.
    '''
    # Separate the enumerations
    enums = str(df[enum_column]).split(';')
    # Check what direction the conversion is being done
    if not enum_to_category:
        # If all the keys are integers, then we're converting from enumerations to category names;
        # otherwise, it's the opposite direction
        enum_to_category = all([isinstance(item, int) for item in list(enum_dict.keys())])
    # Get the individual categories names
    if enum_to_category:
        categories = [enum_dict[int(n)] for n in enums]
    else:
        categories = [str(enum_dict[str(n)]) for n in enums]
    # Join the categories by a ';' separator
    categories = ';'.join(categories)
    return categories


def converge_enum(df1, df2, cat_feat_name, dict1=None, dict2=None, nan_value=0,
                  sort=True):
    '''Converge the categorical encoding (enumerations) on the same feature of
    two dataframes.

    Parameters
    ----------
    df1 : pandas.DataFrame or dask.DataFrame
        One of the dataframes that has the enumerated categorical feature, which
        encoding needs to be converged with the other.
    df2 : pandas.DataFrame or dask.DataFrame
        One of the dataframes that has the enumerated categorical feature, which
        encoding needs to be converged with the other.
    cat_feat_name : string
        Name of the categorical feature whose encodings need to be converged.
    dict1 : dict, default None
        Dictionary mapping between the category names and the first dataframe's
        encoding number. If not specified, the method will create the dictionary.
    dict2 : dict, default None
        Dictionary mapping between the category names and the second dataframe's
        encoding number. If not specified, the method will create the dictionary.
    nan_value : int, default 0
        Integer number that gets assigned to NaN and NaN-like values.
    sort : bool, default True
        If set to True, the final dictionary of mapping between categories names
        and enumeration numbers will be sorted alphabetically. In case sorting
        is used, the resulting dictionary and dataframes will always be the same.

    Returns
    -------
    data1_df : pandas.DataFrame or dask.DataFrame
        The first input dataframe after having its categorical feature converted
        to the new, converged enumeration encoding.
    data2_df : pandas.DataFrame or dask.DataFrame
        The second input dataframe after having its categorical feature
        converted to the new, converged enumeration encoding.
    all_data_dict : dict, default None
        New dictionary that maps both dataframes' unique categories to the
        converged enumeration encoding. Remember to save this dictionary, as
        this converged dictionary creation process is stochastic, if sorting is
        not performed.
    '''
    # Make copies to avoid potentially unwanted inplace changes
    data1_df = df1.copy()
    data2_df = df2.copy()
    if dict1 and dict2:
        data1_dict = dict1.copy()
        data2_dict = dict2.copy()
    else:
        # Determine each dataframe's dictionary of categories
        data1_df[cat_feat_name], data1_dict = enum_categorical_feature(data1_df, cat_feat_name, nan_value=nan_value)
        data2_df[cat_feat_name], data2_dict = enum_categorical_feature(data2_df, cat_feat_name, nan_value=nan_value)
    # Invert the dictionaries of categories
    data1_dict_inv = invert_dict(data1_dict)
    data2_dict_inv = invert_dict(data2_dict)
    data1_dict_inv[nan_value] = 'nan'
    data2_dict_inv[nan_value] = 'nan'
    # Revert back to the original dictionaries, now without multiple NaN-like categories
    data1_dict = invert_dict(data1_dict_inv)
    data2_dict = invert_dict(data2_dict_inv)
    # Get the unique categories of each dataframe
    data1_categories = list(data1_dict.keys())
    data2_categories = list(data2_dict.keys())
    # Combine all the unique categories into one single list
    all_categories = set(data1_categories + data2_categories)
    all_categories.remove('nan')
    if sort:
        all_categories = list(all_categories)
        all_categories.sort()
    # Create a new dictionary for the combined categories
    all_data_dict = create_enum_dict(all_categories)
    all_data_dict['nan'] = nan_value
    # Revert the feature of each dataframe to its original categories strings
    data1_df[cat_feat_name] = data1_df.apply(lambda df: enum_category_conversion(df, enum_column=cat_feat_name,
                                                                                       enum_dict=data1_dict_inv,
                                                                                       enum_to_category=True),
                                             axis=1, meta=('df', str))
    data2_df[cat_feat_name] = data2_df.apply(lambda df: enum_category_conversion(df, enum_column=cat_feat_name,
                                                                                       enum_dict=data2_dict_inv,
                                                                                       enum_to_category=True),
                                             axis=1, meta=('df', str))
    # Convert the features' values into the new enumeration
    data1_df[cat_feat_name] = data1_df.apply(lambda df: enum_category_conversion(df, enum_column=cat_feat_name,
                                                                                       enum_dict=all_data_dict,
                                                                                       enum_to_category=False),
                                             axis=1, meta=('df', str))
    data2_df[cat_feat_name] = data2_df.apply(lambda df: enum_category_conversion(df, enum_column=cat_feat_name,
                                                                                       enum_dict=all_data_dict,
                                                                                       enum_to_category=False),
                                             axis=1, meta=('df', str))
    return data1_df, data2_df, all_data_dict


def join_categorical_enum(df, cat_feat=[], id_columns=['patientunitstayid', 'ts'],
                          cont_join_method='mean', has_timestamp=None):
    '''Join rows that have the same identifier columns based on concatenating
    categorical encodings and on averaging continuous features.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe which will be processed.
    cat_feat : string, default []
        Name(s) of the categorical feature(s) which will have their values
        concatenated along the ID's.
    id_columns : list of strings, default ['patientunitstayid', 'ts']
        List of columns names which represent identifier columns. These are not
        supposed to be changed.
    cont_join_method : string, default 'mean'
        Defines which method to use when joining rows of continuous features.
        Can be either 'mean', 'min' or 'max'.
    has_timestamp : bool, default None
        If set to True, the resulting dataframe will be sorted and set as index
        by the timestamp column (`ts`). If not specified, the method will
        automatically look for a `ts` named column in the input dataframe.

    Returns
    -------
    data_df : pandas.DataFrame or dask.DataFrame
        Resulting dataframe from merging all the concatenated or averaged
        features.
    '''
    # Make a copy of the data to avoid potentially unwanted changes to the original dataframe
    data_df = df.copy()
    # Define a list of dataframes
    df_list = []
    # See if there is a timestamp column on the dataframe
    if has_timestamp is None:
        if 'ts' in id_columns:
            has_timestamp = True
        else:
            has_timestamp = False
    print('Concatenating categorical encodings...')
    for feature in tqdm(cat_feat):
        # Convert to string format
        data_df[feature] = data_df[feature].astype(str)
        # Join with other categorical enumerations on the same ID's
        data_to_add = data_df.groupby(id_columns)[feature].apply(lambda x: "%s" % ';'.join(x)).to_frame().reset_index()
        if has_timestamp:
            # Sort by time `ts` and set it as index
            data_to_add = data_to_add.set_index('ts')
        # Add to the list of dataframes that will be merged
        df_list.append(data_to_add)
    remaining_feat = list(set(data_df.columns) - set(cat_feat) - set(id_columns))
    print('Averaging continuous features...')
    for feature in tqdm(remaining_feat):
        # Join remaining features through their average, min or max value (just to be sure that there aren't missing or different values)
        if cont_join_method.lower() == 'mean':
            data_to_add = data_df.groupby(id_columns)[feature].mean().to_frame().reset_index()
        elif cont_join_method.lower() == 'min':
            data_to_add = data_df.groupby(id_columns)[feature].min().to_frame().reset_index()
        elif cont_join_method.lower() == 'max':
            data_to_add = data_df.groupby(id_columns)[feature].max().to_frame().reset_index()
        if has_timestamp:
            # Sort by time `ts` and set it as index
            data_to_add = data_to_add.set_index('ts')
        # Add to the list of dataframes that will be merged
        df_list.append(data_to_add)
    # Merge all dataframes
    print('Merging features\' dataframes...')
    if 'dask' in str(type(data_df)):
        data_df = reduce(lambda x, y: dd.merge(x, y, on=id_columns), df_list)
    else:
        data_df = reduce(lambda x, y: pd.merge(x, y, on=id_columns), df_list)
    print('Done!')
    return data_df


def prepare_embed_bag(df, feature):
    '''Prepare a categorical feature for embedding bag, i.e. split category
    enumerations into separate numbers, combine them into a single list and set
    the appropriate offsets as to when each row's group of categories end.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe that contains the categorical feature that will be embedded.
    feature : string
        Name of the categorical feature on which embedding bag will be applied.

    Returns
    -------
    embed_num : torch.Tensor
        List of all categorical enumerations, i.e. the numbers corresponding to
        each of the feature's categories, contained in the input series.
    offset : torch.Tensor
        List of when each row's categorical enumerations start, considering the
        embed_num list.
    '''
    embed_num = []
    count = 0
    offset = [count]
    for i in range(len(df)):
        # Separate digits in the same string
        if 'dask' in str(type(df)):
            # The line of code bellow corresponds to df[feature][i] in Pandas
            feature_val_i = df.head(i+1, npartitions=df.npartitions).tail(1)[feature].values[0]
        elif 'pandas' in str(type(df)):
            feature_val_i = df[feature][i]
        else:
            raise Exception(f'ERROR: `df` should either be a Pandas or Dask dataframe, not {type(df)}.')
        digits_list = feature_val_i.split(';')
        embed_num.append(digits_list)
        # Set the end of the current list
        count += len(digits_list)
        offset.append(count)
    # Flatten list
    embed_num = [int(item) for sublist in embed_num for item in sublist]
    # Convert to PyTorch tensor
    embed_num = torch.tensor(embed_num)
    offset = torch.tensor(offset)
    return embed_num, offset


# [TODO] Create a function that takes a set of embeddings (which will be used in
# an embedding bag) and reverts them back to the original text
# [TODO] Define an automatic method to discover which embedded category was more
# important by doing inference on individual embeddings of each category separately,
# seeing which one caused a bigger change in the output.


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


def remove_rows_unmatched_key(df, key, columns):
    '''Remove rows corresponding to the keys that weren't in the dataframe merged at the right.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe resulting from a asof merge which will be searched for missing values.
    key : string
        Name of the column which was used as the "by" key in the asof merge. Typically
        represents a temporal feature from a time series, such as days or timestamps.
    columns : list of strings
        Name of the column(s), originating from the dataframe which was merged at the
        right, which should not have any missing values. If it has, it means that
        the corresponding key wasn't present in the original dataframe. Even if there's
        just one column to analyze, it should be received in list format.

    Returns
    -------
    df : pandas.DataFrame or dask.DataFrame
        Returns the input dataframe but without the rows which didn't have any values
        in the right dataframe's features.
    '''
    for k in tqdm_notebook(df[key].unique()):
        # Variable that count the number of columns which don't have any value
        # (i.e. all rows are missing values) for a given identifier 'k'
        num_empty_columns = 0

        for col in columns:
            if df[df[key] == k][col].isnull().sum() == len(df[df[key] == k]):
                # Found one more column which is full of missing values for identifier 'k'
                num_empty_columns += 1

        if num_empty_columns == len(columns):
            # Eliminate all rows corresponding to the analysed key if all the columns
            # are empty for the identifier 'k'
            df = df[~(df[key] == k)]

    return df


def dataframe_to_padded_tensor(df, seq_len_dict, n_ids, n_inputs, id_column='subject_id', data_type='PyTorch', padding_value=999999):
    '''Converts a Pandas dataframe into a padded NumPy array or PyTorch Tensor.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Data in a Pandas dataframe format which will be padded and converted
        to the requested data type.
    seq_len_dict : dictionary
        Dictionary containing the original sequence lengths of the dataframe.
    n_ids : int
        Total number of subject identifiers in a dataframe.
        Example: Total number of patients in a health dataset.
    n_inputs : int
        Total number of input features present in the dataframe.
    id_column : string, default 'subject_id'
        Name of the column which corresponds to the subject identifier in the
        dataframe.
    data_type : string, default 'PyTorch'
        Indication of what kind of output data type is desired. In case it's
        set as 'NumPy', the function outputs a NumPy array. If it's 'PyTorch',
        the function outputs a PyTorch tensor.
    padding_value : numeric
        Value to use in the padding, to fill the sequences.

    Returns
    -------
    arr : torch.Tensor or numpy.ndarray
        PyTorch tensor or NumPy array version of the dataframe, after being
        padded with the specified padding value to have a fixed sequence
        length.
    '''
    # Max sequence length (e.g. patient with the most temporal events)
    max_seq_len = seq_len_dict[max(seq_len_dict, key=seq_len_dict.get)]

    # Making a padded numpy array version of the dataframe (all index has the same sequence length as the one with the max)
    arr = np.ones((n_ids, max_seq_len, n_inputs)) * padding_value

    # Iterator that outputs each unique identifier (e.g. each patient in the dataset)
    id_iter = iter(df[id_column].unique())

    # Count the iterations of ids
    count = 0

    # Assign each value from the dataframe to the numpy array
    for idt in id_iter:
        arr[count, :seq_len_dict[idt], :] = df[df[id_column] == idt].to_numpy()
        arr[count, seq_len_dict[idt]:, :] = padding_value
        count += 1

    # Make sure that the data type asked for is a string
    if not isinstance(data_type, str):
        raise Exception('ERROR: Please provide the desirable data type in a string format.')

    if data_type.lower() == 'numpy':
        return arr
    elif data_type.lower() == 'pytorch':
        return torch.from_numpy(arr)
    else:
        raise Exception('ERROR: Unavailable data type. Please choose either NumPy or PyTorch.')


def apply_zscore_norm(value, df=None, mean=None, std=None, categories_means=None,
                      categories_stds=None, groupby_columns=None):
    '''Performs z-score normalization when used inside a Pandas or Dask
    apply function.

    Parameters
    ----------
    value : int or float
        Original, unnormalized value.
    df : pandas.DataFrame or dask.DataFrame, default None
        Original pandas dataframe which is used to retrieve the
        necessary statistical values used in group normalization, i.e. when
        values are normalized according to their corresponding categories.
    mean : int or float, default None
        Average (mean) value to be used in the z-score normalization.
    std : int or float, default None
        Standard deviation value to be used in the z-score normalization.
    categories_means : dict, default None
        Dictionary containing the average values for each set of categories.
    categories_stds : dict, default None
        Dictionary containing the standard deviation values for each set of
        categories.
    groupby_columns : string or list of strings, default None
        Name(s) of the column(s) that contains the categories from which
        statistical values (mean and standard deviation) are retrieved.

    Returns
    -------
    value_norm : int or float
        Z-score normalized value.
    '''
    if not isinstance(value, numbers.Number):
        raise Exception(f'ERROR: Input value should be a number, not an object of type {type(value)}.')
    if mean and std:
        return (value - mean) / std
    elif df and categories_means and categories_stds and groupby_columns:
        try:
            if isinstance(groupby_columns, list):
                return (value - categories_means[tuple(df[groupby_columns])]) / \
                       categories_stds[tuple(df[groupby_columns])]
            else:
                return (value - categories_means[df[groupby_columns]]) / \
                       categories_stds[df[groupby_columns]]
        except:
            warnings.warn(f'Couldn\'t manage to find the mean and standard deviation values for the groupby columns {groupby_columns} with values {tuple(df[groupby_columns])}.')
            return np.nan
    else:
        raise Exception('ERROR: Invalid parameters. Either the `mean` and `std` or the `df`, `categories_means`, `categories_stds` and `groupby_columns` must be set.')


def apply_minmax_norm(value, df=None, min=None, max=None, categories_mins=None,
                      categories_maxs=None, groupby_columns=None):
    '''Performs minmax normalization when used inside a Pandas or Dask
    apply function.

    Parameters
    ----------
    value : int or float
        Original, unnormalized value.
    df : pandas.DataFrame or dask.DataFrame, default None
        Original pandas dataframe which is used to retrieve the
        necessary statistical values used in group normalization, i.e. when
        values are normalized according to their corresponding categories.
    min : int or float, default None
        Minimum value to be used in the minmax normalization.
    max : int or float, default None
        Maximum value to be used in the minmax normalization.
    categories_mins : dict, default None
        Dictionary containing the minimum values for each set of categories.
    categories_maxs : dict, default None
        Dictionary containing the maximum values for each set of categories.
    groupby_columns : string or list of strings, default None
        Name(s) of the column(s) that contains the categories from which
        statistical values (minimum and maximum) are retrieved.

    Returns
    -------
    value_norm : int or float
        Minmax normalized value.
    '''
    if not isinstance(value, numbers.Number):
        raise Exception(f'ERROR: Input value should be a number, not an object of type {type(value)}.')
    if mean and std:
        return (value - min) / (max - min)
    elif df and categories_means and categories_stds and groupby_columns:
        try:
            if isinstance(groupby_columns, list):
                return (value - categories_mins[tuple(df[groupby_columns])]) / \
                       (categories_maxs[tuple(df[groupby_columns])] - categories_mins[tuple(df[groupby_columns])])
            else:
                return (value - categories_mins[df[groupby_columns]]) / \
                       (categories_maxs[df[groupby_columns]] - categories_mins[df[groupby_columns]])
        except:
            warnings.warn(f'Couldn\'t manage to find the mean and standard deviation values for the groupby columns {groupby_columns} with values {tuple(df[groupby_columns])}.')
            return np.nan
    else:
        raise Exception('ERROR: Invalid parameters. Either the `min` and `max` or the `df`, `categories_mins`, `categories_maxs` and `groupby_columns` must be set.')


def apply_zscore_denorm(value, df=None, mean=None, std=None, categories_means=None,
                      categories_stds=None, groupby_columns=None):
    '''Performs z-score denormalization when used inside a Pandas or Dask
    apply function.

    Parameters
    ----------
    value : int or float
        Input normalized value.
    df : pandas.DataFrame or dask.DataFrame, default None
        Original pandas dataframe which is used to retrieve the
        necessary statistical values used in group denormalization, i.e. when
        values are denormalized according to their corresponding categories.
    mean : int or float, default None
        Average (mean) value to be used in the z-score denormalization.
    std : int or float, default None
        Standard deviation value to be used in the z-score denormalization.
    categories_means : dict, default None
        Dictionary containing the average values for each set of categories.
    categories_stds : dict, default None
        Dictionary containing the standard deviation values for each set of
        categories.
    groupby_columns : string or list of strings, default None
        Name(s) of the column(s) that contains the categories from which
        statistical values (mean and standard deviation) are retrieved.

    Returns
    -------
    value_denorm : int or float
        Z-score denormalized value.
    '''
    if not isinstance(value, numbers.Number):
        raise Exception(f'ERROR: Input value should be a number, not an object of type {type(value)}.')
    if mean and std:
        return value * std + mean
    elif df and categories_means and categories_stds and groupby_columns:
        try:
            if isinstance(groupby_columns, list):
                return value * categories_stds[tuple(df[groupby_columns])] \
                       + categories_means[tuple(df[groupby_columns])]

            else:
                return value * categories_stds[df[groupby_columns]] + \
                       categories_means[df[groupby_columns]]
        except:
            warnings.warn(f'Couldn\'t manage to find the mean and standard deviation values for the groupby columns {groupby_columns} with values {tuple(df[groupby_columns])}.')
            return np.nan
    else:
        raise Exception('ERROR: Invalid parameters. Either the `mean` and `std` or the `df`, `categories_means`, `categories_stds` and `groupby_columns` must be set.')


def apply_minmax_denorm(value, df=None, min=None, max=None, categories_mins=None,
                      categories_maxs=None, groupby_columns=None):
    '''Performs minmax denormalization when used inside a Pandas or Dask
    apply function.

    Parameters
    ----------
    value : int or float
        Input normalized value.
    df : pandas.DataFrame or dask.DataFrame, default None
        Original pandas dataframe which is used to retrieve the
        necessary statistical values used in group denormalization, i.e. when
        values are denormalized according to their corresponding categories.
    min : int or float, default None
        Minimum value to be used in the minmax denormalization.
    max : int or float, default None
        Maximum value to be used in the minmax denormalization.
    categories_mins : dict, default None
        Dictionary containing the minimum values for each set of categories.
    categories_maxs : dict, default None
        Dictionary containing the maximum values for each set of categories.
    groupby_columns : string or list of strings, default None
        Name(s) of the column(s) that contains the categories from which
        statistical values (minimum and maximum) are retrieved.

    Returns
    -------
    value_denorm : int or float
        Minmax denormalized value.
    '''
    if not isinstance(value, numbers.Number):
        raise Exception(f'ERROR: Input value should be a number, not an object of type {type(value)}.')
    if mean and std:
        return value * (max - min) + min
    elif df and categories_means and categories_stds and groupby_columns:
        try:
            if isinstance(groupby_columns, list):
                return value * (categories_maxs[tuple(df[groupby_columns])] - categories_mins[tuple(df[groupby_columns])]) + \
                       categories_mins[tuple(df[groupby_columns])]
            else:
                return value * (categories_maxs[df[groupby_columns]] - categories_mins[df[groupby_columns]]) + \
                       categories_mins[df[groupby_columns]]
        except:
            warnings.warn(f'Couldn\'t manage to find the mean and standard deviation values for the groupby columns {groupby_columns} with values {tuple(df[groupby_columns])}.')
            return np.nan
    else:
        raise Exception('ERROR: Invalid parameters. Either the `min` and `max` or the `df`, `categories_mins`, `categories_maxs` and `groupby_columns` must be set.')


def normalize_data(df, data=None, id_columns=['patientunitstayid', 'ts'],
                   normalization_method='z-score', columns_to_normalize=None,
                   columns_to_normalize_cat=None, embed_columns=None,
                   see_progress=True):
    '''Performs data normalization to a continuous valued tensor or dataframe,
       changing the scale of the data.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Original Pandas or Dask dataframe which is used to correctly calculate the
        necessary statistical values used in the normalization. These values
        can't be calculated from the tensor as it might have been padded. If
        the data tensor isn't specified, the normalization is applied directly
        on the dataframe.
    data : torch.Tensor, default None
        PyTorch tensor corresponding to the data which will be normalized
        by the specified normalization method. If the data tensor isn't
        specified, the normalization is applied directly on the dataframe.
    id_columns : list of strings, default ['subject_id', 'ts']
        List of columns names which represent identifier columns. These are not
        supposed to be normalized.
    normalization_method : string, default 'z-score'
        Specifies the normalization method used. It can be a z-score
        normalization, where the data is subtracted of it's mean and divided
        by the standard deviation, which makes it have zero average and unit
        variance, much like a standard normal distribution; it can be a
        min-max normalization, where the data is subtracted by its minimum
        value and then divided by the difference between the minimum and the
        maximum value, getting to a fixed range from 0 to 1.
    columns_to_normalize : list of strings, default None
        If specified, the columns provided in the list are the only ones that
        will be normalized. If set to False, no column will normalized directly,
        although columns can still be normalized in groups of categories, if
        specified in the `columns_to_normalize_cat` parameter. Otherwise, all
        continuous columns will be normalized.
    columns_to_normalize_cat : list of tuples of strings, default None
        If specified, the columns provided in the list are going to be
        normalized on their categories. That is, the values (column 2 in the
        tuple) are normalized with stats of their respective categories (column
        1 of the tuple). Otherwise, no column will be normalized on their
        categories.
    embed_columns : list of strings, default None
        If specified, the columns in the list, which represent features that
        will be embedded, aren't going to be normalized.
    see_progress : bool, default True
        If set to True, a progress bar will show up indicating the execution
        of the normalization calculations.

    Returns
    -------
    data : pandas.DataFrame or dask.DataFrame or torch.Tensor
        Normalized Pandas or Dask dataframe or PyTorch tensor.
    '''
    # Check if specific columns have been specified for normalization
    if columns_to_normalize is None:
        # List of all columns in the dataframe
        feature_columns = list(df.columns)
        # Normalize all non identifier continuous columns, ignore one hot encoded ones
        columns_to_normalize = feature_columns

        # List of all columns in the dataframe, except the ID columns
        [columns_to_normalize.remove(col) for col in id_columns]

        if embed_columns:
            # Prevent all features that will be embedded from being normalized
            [columns_to_normalize.remove(col) for col in embed_columns]

        # List of binary or one hot encoded columns
        binary_cols = list_one_hot_encoded_columns(df[columns_to_normalize])

        if binary_cols:
            # Prevent binary features from being normalized
            [columns_to_normalize.remove(col) for col in binary_cols]

        if not columns_to_normalize:
            print('No columns to normalize, returning the original dataframe.')
            return df

    if type(normalization_method) is not str:
        raise ValueError('Argument normalization_method should be a string. Available options \
                         are \'z-score\' and \'min-max\'.')

    if normalization_method.lower() == 'z-score':
        if columns_to_normalize is not False:
            # Calculate the means and standard deviations
            means = df[columns_to_normalize].mean()
            stds = df[columns_to_normalize].std()

            if 'dask' in str(type(df)):
                # Make sure that the values are computed, in case we're using Dask
                means = means.compute()
                stds = stds.compute()

            column_means = dict(means)
            column_stds = dict(stds)

        # Check if the data being normalized is directly the dataframe
        if data is None:
            # Treat the dataframe as the data being normalized
            data = df.copy()

            # Normalize the right columns
            if columns_to_normalize is not False:
                print(f'z-score normalizing columns {columns_to_normalize}...')
                for col in iterations_loop(columns_to_normalize, see_progress=see_progress):
                    data[col] = (data[col] - column_means[col]) / column_stds[col]

            if columns_to_normalize_cat:
                print(f'z-score normalizing columns {columns_to_normalize_cat} by their associated categories...')
                for col_tuple in iterations_loop(columns_to_normalize_cat, see_progress=see_progress):
                    # Calculate the means and standard deviations
                    means = df.groupby(col_tuple[0])[col_tuple[1]].mean()
                    stds = df.groupby(col_tuple[0])[col_tuple[1]].std()

                    if 'dask' in str(type(df)):
                        # Make sure that the values are computed, in case we're using Dask
                        means = means.compute()
                        stds = stds.compute()

                    categories_means = dict(means)
                    categories_stds = dict(stds)

                    # Normalize the right categories
                    if 'dask' in str(type(df)):
                        data[col_tuple[1]] = data.apply(lambda df: apply_zscore_norm(df, value=df[col_tuple[1]], categories_means=categories_means,
                                                                                     categories_stds=categories_stds, groupby_columns=col_tuple[0]),
                                                        axis=1, meta=('df', float))
                    else:
                        data[col_tuple[1]] = data.apply(lambda df: apply_zscore_norm(df, value=df[col_tuple[1]], categories_means=categories_means,
                                                                                     categories_stds=categories_stds, groupby_columns=col_tuple[0]),
                                                        axis=1)

        # Otherwise, the tensor is normalized
        else:
            if columns_to_normalize is not False:
                # Dictionary to convert the the tensor's column indeces into the dataframe's column names
                idx_to_name = dict(enumerate(df.columns))

                # Dictionary to convert the dataframe's column names into the tensor's column indeces
                name_to_idx = dict([(t[1], t[0]) for t in enumerate(df.columns)])

                # List of indeces of the tensor's columns which are needing normalization
                tensor_columns_to_normalize = [name_to_idx[name] for name in columns_to_normalize]

                # Normalize the right columns
                print(f'z-score normalizing columns {columns_to_normalize}...')
                for col in iterations_loop(tensor_columns_to_normalize, see_progress=see_progress):
                    data[:, :, col] = (data[:, :, col] - column_means[idx_to_name[col]]) / \
                                      column_stds[idx_to_name[col]]

    elif normalization_method.lower() == 'min-max':
        if columns_to_normalize is not False:
            mins = df[columns_to_normalize].min()
            maxs = df[columns_to_normalize].max()

            if 'dask' in str(type(df)):
                # Make sure that the values are computed, in case we're using Dask
                mins = means.compute()
                maxs = maxs.compute()

            column_mins = dict(mins)
            column_maxs = dict(maxs)

        # Check if the data being normalized is directly the dataframe
        if data is None:
            # Treat the dataframe as the data being normalized
            data = df.copy()

            if columns_to_normalize is not False:
                # Normalize the right columns
                print(f'min-max normalizing columns {columns_to_normalize}...')
                for col in iterations_loop(columns_to_normalize, see_progress=see_progress):
                    data[col] = (data[col] - column_mins[col]) / \
                                (column_maxs[col] - column_mins[col])

            if columns_to_normalize_cat:
                print(f'min-max normalizing columns {columns_to_normalize_cat} by their associated categories...')
                for col_tuple in columns_to_normalize_cat:
                    # Calculate the means and standard deviations
                    mins = df.groupby(col_tuple[0])[col_tuple[1]].min()
                    maxs = df.groupby(col_tuple[0])[col_tuple[1]].max()

                    if 'dask' in str(type(df)):
                        # Make sure that the values are computed, in case we're using Dask
                        mins = mins.compute()
                        maxs = maxs.compute()

                    categories_mins = dict(mins)
                    categories_maxs = dict(maxs)

                    # Normalize the right categories
                    if 'dask' in str(type(df)):
                        data[col_tuple[1]] = data.apply(lambda df: apply_minmax_norm(df, value=df[col_tuple[1]], categories_mins=categories_mins,
                                                                                     categories_maxs=categories_maxs, groupby_columns=col_tuple[0]),
                                                        axis=1, meta=('df', float))
                    else:
                        data[col_tuple[1]] = data.apply(lambda df: apply_minmax_norm(df, value=df[col_tuple[1]], categories_mins=categories_mins,
                                                                                     categories_maxs=categories_maxs, groupby_columns=col_tuple[0]),
                                                        axis=1)

        # Otherwise, the tensor is normalized
        else:
            if columns_to_normalize is not False:
                # Dictionary to convert the the tensor's column indeces into the dataframe's column names
                idx_to_name = dict(enumerate(df.columns))

                # Dictionary to convert the dataframe's column names into the tensor's column indeces
                name_to_idx = dict([(t[1], t[0]) for t in enumerate(df.columns)])

                # List of indeces of the tensor's columns which are needing normalization
                tensor_columns_to_normalize = [name_to_idx[name] for name in columns_to_normalize]

                # Normalize the right columns
                print(f'min-max normalizing columns {columns_to_normalize}...')
                for col in iterations_loop(tensor_columns_to_normalize, see_progress=see_progress):
                    data[:, :, col] = (data[:, :, col] - column_mins[idx_to_name[col]]) / \
                                      (column_maxs[idx_to_name[col]] - column_mins[idx_to_name[col]])

    else:
        raise ValueError(f'{normalization_method} isn\'t a valid normalization method. Available options \
                         are \'z-score\' and \'min-max\'.')

    return data


def denormalize_data(df, data=None, id_columns=['patientunitstayid', 'ts'],
                   normalization_method='z-score', columns_to_denormalize=None,
                   columns_to_denormalize_cat=None, embed_columns=None,
                   see_progress=True):
    '''Performs data denormalization to a continuous valued tensor or dataframe,
       changing the scale of the data.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Original Pandas or Dask dataframe which is used to correctly calculate the
        necessary statistical values used in the denormalization. These values
        can't be calculated from the tensor as it might have been padded. If
        the data tensor isn't specified, the denormalization is applied directly
        on the dataframe.
    data : torch.Tensor, default None
        PyTorch tensor corresponding to the data which will be denormalized
        by the specified denormalization method. If the data tensor isn't
        specified, the denormalization is applied directly on the dataframe.
    id_columns : list of strings, default ['subject_id', 'ts']
        List of columns names which represent identifier columns. These are not
        supposed to be denormalized.
    denormalization_method : string, default 'z-score'
        Specifies the denormalization method used. It can be a z-score
        denormalization, where the data is subtracted of it's mean and divided
        by the standard deviation, which makes it have zero average and unit
        variance, much like a standard normal distribution; it can be a
        min-max denormalization, where the data is subtracted by its minimum
        value and then divided by the difference between the minimum and the
        maximum value, getting to a fixed range from 0 to 1.
    columns_to_denormalize : list of strings, default None
        If specified, the columns provided in the list are the only ones that
        will be denormalized. If set to False, no column will denormalized directly,
        although columns can still be denormalized in groups of categories, if
        specified in the `columns_to_denormalize_cat` parameter. Otherwise, all
        continuous columns will be denormalized.
    columns_to_denormalize_cat : list of tuples of strings, default None
        If specified, the columns provided in the list are going to be
        denormalized on their categories. That is, the values (column 2 in the
        tuple) are denormalized with stats of their respective categories (column
        1 of the tuple). Otherwise, no column will be denormalized on their
        categories.
    embed_columns : list of strings, default None
        If specified, the columns in the list, which represent features that
        will be embedded, aren't going to be denormalized.
    see_progress : bool, default True
        If set to True, a progress bar will show up indicating the execution
        of the denormalization calculations.

    Returns
    -------
    data : pandas.DataFrame or dask.DataFrame or torch.Tensor
        Normalized Pandas or Dask dataframe or PyTorch tensor.
    '''
    # Check if specific columns have been specified for denormalization
    if columns_to_denormalize is None:
        # List of all columns in the dataframe
        feature_columns = list(df.columns)
        # Normalize all non identifier continuous columns, ignore one hot encoded ones
        columns_to_denormalize = feature_columns

        # List of all columns in the dataframe, except the ID columns
        [columns_to_denormalize.remove(col) for col in id_columns]

        if embed_columns:
            # Prevent all features that will be embedded from being denormalized
            [columns_to_denormalize.remove(col) for col in embed_columns]

        # List of binary or one hot encoded columns
        binary_cols = list_one_hot_encoded_columns(df[columns_to_denormalize])

        if binary_cols:
            # Prevent binary features from being denormalized
            [columns_to_denormalize.remove(col) for col in binary_cols]

        if not columns_to_denormalize:
            print('No columns to denormalize, returning the original dataframe.')
            return df

    if type(denormalization_method) is not str:
        raise ValueError('Argument denormalization_method should be a string. Available options \
                         are \'z-score\' and \'min-max\'.')

    if denormalization_method.lower() == 'z-score':
        if columns_to_denormalize is not False:
            # Calculate the means and standard deviations
            means = df[columns_to_denormalize].mean()
            stds = df[columns_to_denormalize].std()

            if 'dask' in str(type(df)):
                # Make sure that the values are computed, in case we're using Dask
                means = means.compute()
                stds = stds.compute()

            column_means = dict(means)
            column_stds = dict(stds)

        # Check if the data being denormalized is directly the dataframe
        if data is None:
            # Treat the dataframe as the data being denormalized
            data = df.copy()

            # Normalize the right columns
            if columns_to_denormalize is not False:
                print(f'z-score normalizing columns {columns_to_denormalize}...')
                for col in iterations_loop(columns_to_denormalize, see_progress=see_progress):
                    data[col] = data[col] * column_stds[col] + column_means[col]

            if columns_to_denormalize_cat:
                print(f'z-score normalizing columns {columns_to_denormalize_cat} by their associated categories...')
                for col_tuple in iterations_loop(columns_to_denormalize_cat, see_progress=see_progress):
                    # Calculate the means and standard deviations
                    means = df.groupby(col_tuple[0])[col_tuple[1]].mean()
                    stds = df.groupby(col_tuple[0])[col_tuple[1]].std()

                    if 'dask' in str(type(df)):
                        # Make sure that the values are computed, in case we're using Dask
                        means = means.compute()
                        stds = stds.compute()

                    categories_means = dict(means)
                    categories_stds = dict(stds)

                    # Normalize the right categories
                    if 'dask' in str(type(df)):
                        data[col_tuple[1]] = data.apply(lambda df: apply_zscore_denorm(df, value=df[col_tuple[1]], categories_means=categories_means,
                                                                                       categories_stds=categories_stds, groupby_columns=col_tuple[0]),
                                                        axis=1, meta=('df', float))
                    else:
                        data[col_tuple[1]] = data.apply(lambda df: apply_zscore_denorm(df, value=df[col_tuple[1]], categories_means=categories_means,
                                                                                       categories_stds=categories_stds, groupby_columns=col_tuple[0]),
                                                        axis=1)

        # Otherwise, the tensor is denormalized
        else:
            if columns_to_denormalize is not False:
                # Dictionary to convert the the tensor's column indeces into the dataframe's column names
                idx_to_name = dict(enumerate(df.columns))

                # Dictionary to convert the dataframe's column names into the tensor's column indeces
                name_to_idx = dict([(t[1], t[0]) for t in enumerate(df.columns)])

                # List of indeces of the tensor's columns which are needing denormalization
                tensor_columns_to_denormalize = [name_to_idx[name] for name in columns_to_denormalize]

                # Normalize the right columns
                print(f'z-score normalizing columns {columns_to_denormalize}...')
                for col in iterations_loop(tensor_columns_to_denormalize, see_progress=see_progress):
                    data[:, :, col] = data[:, :, col] * column_stds[idx_to_name[col]] + \
                                      column_means[idx_to_name[col]]

    elif denormalization_method.lower() == 'min-max':
        if columns_to_denormalize is not False:
            mins = df[columns_to_denormalize].min()
            maxs = df[columns_to_denormalize].max()

            if 'dask' in str(type(df)):
                # Make sure that the values are computed, in case we're using Dask
                mins = means.compute()
                maxs = maxs.compute()

            column_mins = dict(mins)
            column_maxs = dict(maxs)

        # Check if the data being denormalized is directly the dataframe
        if data is None:
            # Treat the dataframe as the data being denormalized
            data = df.copy()

            if columns_to_denormalize is not False:
                # Normalize the right columns
                print(f'min-max normalizing columns {columns_to_denormalize}...')
                for col in iterations_loop(columns_to_denormalize, see_progress=see_progress):
                    data[col] = data[col] * (column_maxs[col] - column_mins[col]) + \
                                column_mins[col]

            if columns_to_denormalize_cat:
                print(f'min-max normalizing columns {columns_to_denormalize_cat} by their associated categories...')
                for col_tuple in columns_to_denormalize_cat:
                    # Calculate the means and standard deviations
                    mins = df.groupby(col_tuple[0])[col_tuple[1]].min()
                    maxs = df.groupby(col_tuple[0])[col_tuple[1]].max()

                    if 'dask' in str(type(df)):
                        # Make sure that the values are computed, in case we're using Dask
                        mins = mins.compute()
                        maxs = maxs.compute()

                    categories_mins = dict(mins)
                    categories_maxs = dict(maxs)

                    # Normalize the right categories
                    if 'dask' in str(type(df)):
                        data[col_tuple[1]] = data.apply(lambda df: apply_minmax_denorm(df, value=df[col_tuple[1]], categories_mins=categories_mins,
                                                                                       categories_maxs=categories_maxs, groupby_columns=col_tuple[0]),
                                                        axis=1, meta=('df', float))
                    else:
                        data[col_tuple[1]] = data.apply(lambda df: apply_minmax_denorm(df, value=df[col_tuple[1]], categories_mins=categories_mins,
                                                                                       categories_maxs=categories_maxs, groupby_columns=col_tuple[0]),
                                                        axis=1)

        # Otherwise, the tensor is denormalized
        else:
            if columns_to_denormalize is not False:
                # Dictionary to convert the the tensor's column indeces into the dataframe's column names
                idx_to_name = dict(enumerate(df.columns))

                # Dictionary to convert the dataframe's column names into the tensor's column indeces
                name_to_idx = dict([(t[1], t[0]) for t in enumerate(df.columns)])

                # List of indeces of the tensor's columns which are needing denormalization
                tensor_columns_to_denormalize = [name_to_idx[name] for name in columns_to_denormalize]

                # Normalize the right columns
                print(f'min-max normalizing columns {columns_to_denormalize}...')
                for col in iterations_loop(tensor_columns_to_denormalize, see_progress=see_progress):
                    data[:, :, col] = data[:, :, col] * (column_maxs[idx_to_name[col]] - column_mins[idx_to_name[col]]) + \
                                      column_mins[idx_to_name[col]]

    else:
        raise ValueError(f'{denormalization_method} isn\'t a valid denormalization method. Available options \
                         are \'z-score\' and \'min-max\'.')

    return data


def missing_values_imputation(tensor):
    '''Performs missing values imputation to a tensor corresponding to a single column.

    Parameters
    ----------
    tensor : torch.Tensor
        PyTorch tensor corresponding to a single column which will be imputed.

    Returns
    -------
    tensor : torch.Tensor
        Imputed PyTorch tensor.
    '''
    # Replace NaN's with zeros
    tensor = torch.where(tensor != tensor, torch.zeros_like(tensor), tensor)

    return tensor


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


def sort_by_seq_len(data, seq_len_dict, labels=None, id_column=0):
    '''Sort the data by sequence length in order to correctly apply it to a
    PyTorch neural network.

    Parameters
    ----------
    data : torch.Tensor
        Data tensor on which sorting by sequence length will be applied.
    seq_len_dict : dict
        Dictionary containing the sequence lengths for each index of the
        original dataframe. This allows to ignore the padding done in
        the fixed sequence length tensor.
    labels : torch.Tensor, default None
        Labels corresponding to the data used, either specified in the input
        or all the data that the interpreter has.
    id_column : int, default 0
        Number of the column which corresponds to the subject identifier in
        the data tensor.

    Returns
    -------
    sorted_data : torch.Tensor, default None
        Data tensor already sorted by sequence length.
    sorted_labels : torch.Tensor, default None
        Labels tensor already sorted by sequence length. Only outputed if the
        labels data is specified in the input.
    x_lengths : list of int
        Sorted list of sequence lengths, relative to the input data.
    '''
    # Get the original lengths of the sequences, for the input data
    x_lengths = [seq_len_dict[id] for id in list(data[:, 0, id_column].numpy())]

    is_sorted = all(x_lengths[i] >= x_lengths[i+1] for i in range(len(x_lengths)-1))

    if is_sorted:
        # Do nothing if it's already sorted
        sorted_data = data
        sorted_labels = labels
    else:
        # Sorted indeces to get the data sorted by sequence length
        data_sorted_idx = list(np.argsort(x_lengths)[::-1])

        # Sort the x_lengths array by descending sequence length
        x_lengths = [x_lengths[idx] for idx in data_sorted_idx]

        # Sort the data by descending sequence length
        sorted_data = data[data_sorted_idx, :, :]

        if labels is not None:
            # Sort the labels by descending sequence length
            sorted_labels = labels[data_sorted_idx, :]

    if labels is None:
        return sorted_data, x_lengths
    else:
        return sorted_data, sorted_labels,  x_lengths


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


def pad_list(x_list, length, padding_value=999999):
    '''Pad a list with a specific padding value until the desired length is
    met.

    Parameters
    ----------
    x_list : list
        List which will be padded.
    length : int
        Desired length for the final padded list.
    padding_value : numeric
        Value to use in the padding, to fill the sequences.

    Returns
    -------
    x_list : list
        Resulting padded list'''
    return x_list + [padding_value] * (length - len(x_list))


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


def set_dosage_and_units(df, orig_column='dosage'):
    '''Separate medication dosage string column into numeric dosage and units
    features.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe containing the medication dosage information.
    orig_column : string, default 'dosage'
        Name of the original column, which will be split in two.

    Returns
    -------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe after adding the numeric dosage and units columns.
    '''
    # Start by assuming that dosage and unit are unknown
    dosage = np.nan
    unit = np.nan
    try:
        x = df[orig_column].split(' ')
        if len(x) == 2:
            try:
                x[0] = float(x[0])
            except:
                return dosage, unit
            if is_definitely_string(x[1]):
                # Add correctly formated dosage and unit values
                dosage = x[0]
                unit = x[1]
                return dosage, unit
            else:
                return dosage, unit
        else:
            return dosage, unit
    except:
        return dosage, unit


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
    return n_element


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
        features, labels, x_lengths = sort_by_seq_len(features, seq_len_dict, labels) # Sort the data by sequence length

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
            features, labels, x_lengths = sort_by_seq_len(features, seq_len_dict, labels) # Sort the data by sequence length

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

    optimizer = optim.Adam(model.parameters(), lr=lr)                       # Adam optimization algorithm
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
