from comet_ml import Experiment                         # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                                            # PyTorch to create and apply deep learning models
import pandas as pd                                     # Pandas to handle the data in dataframes
import dask.dataframe as dd                             # Dask to handle big data in dataframes
import numpy as np                                      # NumPy to handle numeric and NaN operations
import numbers                                          # numbers allows to check if data is numeric
import warnings                                         # Print warnings for bad practices
import utils                                            # Generic and useful methods
import search_explore                                   # Methods to search and explore data

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
            raise Exception('ERROR: Column name not found in the dataframe.')

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
        ohe_columns = search_explore.list_one_hot_encoded_columns(ohe_df)

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
    for k in utils.iterations_loop(df[key].unique()):
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
    if min and max:
        return (value - min) / (max - min)
    elif df and categories_mins and categories_maxs and groupby_columns:
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
    if min and max:
        return value * (max - min) + min
    elif df and categories_mins and categories_maxs and groupby_columns:
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
        binary_cols = search_explore.list_one_hot_encoded_columns(df[columns_to_normalize])

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
                for col in utils.iterations_loop(columns_to_normalize, see_progress=see_progress):
                    data[col] = (data[col] - column_means[col]) / column_stds[col]

            if columns_to_normalize_cat:
                print(f'z-score normalizing columns {columns_to_normalize_cat} by their associated categories...')
                for col_tuple in utils.iterations_loop(columns_to_normalize_cat, see_progress=see_progress):
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
                        data[col_tuple[1]] = data.apply(lambda df: apply_zscore_norm(value=df[col_tuple[1]], df=df, categories_means=categories_means,
                                                                                     categories_stds=categories_stds, groupby_columns=col_tuple[0]),
                                                        axis=1, meta=('df', float))
                    else:
                        data[col_tuple[1]] = data.apply(lambda df: apply_zscore_norm(value=df[col_tuple[1]], df=df, categories_means=categories_means,
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
                for col in utils.iterations_loop(tensor_columns_to_normalize, see_progress=see_progress):
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
                for col in utils.iterations_loop(columns_to_normalize, see_progress=see_progress):
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
                        data[col_tuple[1]] = data.apply(lambda df: apply_minmax_norm(value=df[col_tuple[1]], df=df, categories_mins=categories_mins,
                                                                                     categories_maxs=categories_maxs, groupby_columns=col_tuple[0]),
                                                        axis=1, meta=('df', float))
                    else:
                        data[col_tuple[1]] = data.apply(lambda df: apply_minmax_norm(value=df[col_tuple[1]], df=df, categories_mins=categories_mins,
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
                for col in utils.iterations_loop(tensor_columns_to_normalize, see_progress=see_progress):
                    data[:, :, col] = (data[:, :, col] - column_mins[idx_to_name[col]]) / \
                                      (column_maxs[idx_to_name[col]] - column_mins[idx_to_name[col]])

    else:
        raise ValueError(f'{normalization_method} isn\'t a valid normalization method. Available options \
                         are \'z-score\' and \'min-max\'.')

    return data


def denormalize_data(df, data=None, id_columns=['patientunitstayid', 'ts'],
                   denormalization_method='z-score', columns_to_denormalize=None,
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
        binary_cols = search_explore.list_one_hot_encoded_columns(df[columns_to_denormalize])

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
                for col in utils.iterations_loop(columns_to_denormalize, see_progress=see_progress):
                    data[col] = data[col] * column_stds[col] + column_means[col]

            if columns_to_denormalize_cat:
                print(f'z-score normalizing columns {columns_to_denormalize_cat} by their associated categories...')
                for col_tuple in utils.iterations_loop(columns_to_denormalize_cat, see_progress=see_progress):
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
                        data[col_tuple[1]] = data.apply(lambda df: apply_zscore_denorm(value=df[col_tuple[1]], df=df, categories_means=categories_means,
                                                                                       categories_stds=categories_stds, groupby_columns=col_tuple[0]),
                                                        axis=1, meta=('df', float))
                    else:
                        data[col_tuple[1]] = data.apply(lambda df: apply_zscore_denorm(value=df[col_tuple[1]], df=df, categories_means=categories_means,
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
                for col in utils.iterations_loop(tensor_columns_to_denormalize, see_progress=see_progress):
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
                for col in utils.iterations_loop(columns_to_denormalize, see_progress=see_progress):
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
                        data[col_tuple[1]] = data.apply(lambda df: apply_minmax_denorm(value=df[col_tuple[1]], df=df, categories_mins=categories_mins,
                                                                                       categories_maxs=categories_maxs, groupby_columns=col_tuple[0]),
                                                        axis=1, meta=('df', float))
                    else:
                        data[col_tuple[1]] = data.apply(lambda df: apply_minmax_denorm(value=df[col_tuple[1]], df=df, categories_mins=categories_mins,
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
                for col in utils.iterations_loop(tensor_columns_to_denormalize, see_progress=see_progress):
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
