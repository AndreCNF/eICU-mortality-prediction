from comet_ml import Experiment                         # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                                            # PyTorch to create and apply deep learning models
import pandas as pd                                     # Pandas to handle the data in dataframes
import dask.dataframe as dd                             # Dask to handle big data in dataframes
import numpy as np                                      # NumPy to handle numeric and NaN operations
import numbers                                          # numbers allows to check if data is numeric
from functools import reduce                            # Parallelize functions
import re                                               # Methods for string editing and searching, regular expression matching operations
import warnings                                         # Print warnings for bad practices
import utils                                            # Generic and useful methods
import data_processing                                  # Data processing and dataframe operations

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
    enum_dict = utils.invert_dict(enum_dict)
    # Move NaN to key 0
    enum_dict[np.nan] = nan_value
    # Search for NaN-like categories
    for key, val in enum_dict.items():
        if type(key) is str:
            if utils.is_string_nan(key):
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
        df = data_processing.clean_naming(df, feature)
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
            enum_series = df[feature].map(lambda x: utils.apply_dict_convertion(x, enum_dict, nan_value), meta=('x', int))
        else:
            enum_series = df[feature].map(lambda x: utils.apply_dict_convertion(x, enum_dict, nan_value))
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
    data1_dict_inv = utils.invert_dict(data1_dict)
    data2_dict_inv = utils.invert_dict(data2_dict)
    data1_dict_inv[nan_value] = 'nan'
    data2_dict_inv[nan_value] = 'nan'
    # Revert back to the original dictionaries, now without multiple NaN-like categories
    data1_dict = utils.invert_dict(data1_dict_inv)
    data2_dict = utils.invert_dict(data2_dict_inv)
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


def remove_nan_enum_from_string(x, nan_value=0):
    '''Removes missing values (NaN) from enumeration encoded strings.

    Parameters
    ----------
    x : string
        Original string, with possible NaNs included.
    nan_value : int, default 0
        Integer number that gets assigned to NaN and NaN-like values.

    Returns
    -------
    x : string
        NaN removed string.
    '''
    # Make sure that the NaN value is represented as a string
    nan_value = str(nan_value)
    # Only remove NaN values if the string isn't just a single NaN value
    if x != nan_value:
        # Remove NaN value that might have a following encoded value
        if f'{nan_value};' in x:
            x = re.sub(f'{nan_value};', '', x)
        # Remove NaN value that might be at the end of the string
        if nan_value in x:
            x = re.sub(f';{nan_value}', '', x)
        # If the string got completly emptied, place a single NaN value on it
        if x == '':
            x = nan_value
    return x


def join_categorical_enum(df, cat_feat=[], id_columns=['patientunitstayid', 'ts'],
                          cont_join_method='mean', has_timestamp=None,
                          nan_value=0, remove_listed_nan=True):
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
    nan_value : int, default 0
        Integer number that gets assigned to NaN and NaN-like values.
    remove_listed_nan : bool, default True
        If set to True, joined rows where non-NaN values exist have the NaN
        values removed.

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
    for feature in utils.iterations_loop(cat_feat):
        # Convert to string format
        data_df[feature] = data_df[feature].astype(str)
        # Join with other categorical enumerations on the same ID's
        data_to_add = data_df.groupby(id_columns)[feature].apply(lambda x: ';'.join(x)).to_frame().reset_index()
        if remove_listed_nan:
            # Remove NaN values from rows with non-NaN values
            data_to_add[feature] = data_to_add[feature].apply(lambda x: remove_nan_enum_from_string(x, nan_value))
        if has_timestamp:
            # Sort by time `ts` and set it as index
            data_to_add = data_to_add.set_index('ts')
        # Add to the list of dataframes that will be merged
        df_list.append(data_to_add)
    remaining_feat = list(set(data_df.columns) - set(cat_feat) - set(id_columns))
    print('Averaging continuous features...')
    for feature in utils.iterations_loop(remaining_feat):
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
