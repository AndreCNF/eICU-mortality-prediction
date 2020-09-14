# # Category to feature example
# ---
#
# Applying a method of category to feature conversion, where new features are created based on the categories of one categorical column and the values of another column. Working fine on Pandas, failing to use with multiple categories on Dask.

# ## Importing the necessary packages

import dask.dataframe as dd                # Dask to handle big data in dataframes
import pandas as pd                        # Pandas to load the data initially
from dask.distributed import Client        # Dask scheduler
import os                                  # os handles directory/workspace changes
import numpy as np                         # NumPy to handle numeric and NaN operations
from tqdm import tqdm_notebook             # tqdm allows to track code execution progress
from IPython.display import display        # Display multiple outputs on the same cell

# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# Set up local cluster
client = Client()
client

client.run(os.getcwd)


# ## Category to feature conversion method

def category_to_feature(df, categories_feature, values_feature):
    '''Convert a categorical column and its corresponding values column into
    new features, one for each category.

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
        # Convert category to feature
        data_df[category] = data_df.apply(lambda x: x[values_feature] if x[categories_feature] == category
                                                    else np.nan, axis=1)
    return data_df


# ## Creating data

# Encoded dataframes:

data_df = pd.DataFrame([[103, 0, 'cat_a', 'val_a1'], 
                        [103, 1, 'cat_a', 'val_a2'],
                        [103, 2, 'cat_b', 'val_b1'],
                        [104, 0, 'cat_c', 'val_c1'],
                        [105, 0, 'cat_a', 'val_a3'],
                        [106, 0, 'cat_c', 'val_c2'],
                        [107, 0, 'cat_b', 'val_b1'],
                        [108, 0, 'cat_b', 'val_b2'],
                        [108, 1, 'cat_d', 'val_d1'],
                        [108, 2, 'cat_a', 'val_a1'],
                        [108, 3, 'cat_a', 'val_a3'],], columns=['id', 'ts', 'categories', 'values'])
data_df

# ## Applying the method on Pandas
#
# Remember that we want each category (from `categories`) to turn into a feature, with values extracted from the column `values`.

category_to_feature(data_df, categories_feature='categories', values_feature='values')

# All is good, it worked as intended. Now let's try it on Dask.

# ## Applying the method on Dask
#
# Remember that we want each category (from `categories`) to turn into a feature, with values extracted from the column `values`.

data_ddf = dd.from_pandas(data_df, npartitions=1)
data_ddf.compute()

category_to_feature(data_ddf, categories_feature='categories', values_feature='values').compute()

# It failed! Notice how it just put all the new columns with the same values as the last added column: `cat_d`. We can confirm this if we print the dataframe step by step:

# Copy the dataframe to avoid potentially unwanted inplace changes
copied_df = data_ddf.copy()
copied_df.compute()

# Find the unique categories
categories = copied_df['categories'].unique()
if 'dask' in str(type(copied_df)):
    categories = categories.compute()
categories

# Create a feature for each category
for category in categories:
    # Convert category to feature
    copied_df[category] = copied_df.apply(lambda x: x['values'] if x['categories'] == category
                                                    else np.nan, axis=1)
    print(f'Dataframe after adding feature {category}:')
    display(copied_df.compute())


