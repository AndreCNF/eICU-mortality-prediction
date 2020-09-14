# # Imputation tests
# ---
#
# Playing around with imputation methods, to compare them and checking if their implementations in data-utils are working properly.

# ## Importing the necessary packages

import pandas as pd                        # Pandas to load the data initially
import numpy as np                         # NumPy to handle numeric and NaN operations
import data_utils as du                    # Data science and machine learning relevant methods

du.set_random_seed(42)

# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

du.set_pandas_library(lib='pandas')

# ## Creating dummy data

dmy_df = pd.DataFrame([[101, 0, np.nan],
                       [101, 1, 1],
                       [102, 0, np.nan],
                       [102, 1, np.nan],
                       [103, 0, 2],
                       [103, 1, np.nan],
                       [104, 0, np.nan],
                       [105, 0, np.nan],
                       [105, 1, np.nan],
                       [105, 2, 3],
                       [105, 3, np.nan],
                       [105, 4, 5],
                       [105, 5, np.nan],
                       [106, 0, 2],
                       [106, 0, np.nan],
                       [106, 0, np.nan],
                       [106, 0, np.nan],
                       [106, 0, 10]], columns=['id', 'ts', 'Var0'])
dmy_df

dmy_df.to_csv('imputation_test_dmy_df.csv')

# ## Testing imputation methods

# ### Zeros

zeros_df = du.data_processing.missing_values_imputation(dmy_df, method='zero', id_column='id')
zeros_df

zeros_df.to_csv('imputation_test_zeros_df.csv')

# ### Zig Zag

zigzag_df = du.data_processing.missing_values_imputation(dmy_df, method='zigzag', id_column='id')
zigzag_df

zigzag_df.to_csv('imputation_test_zigzag_df.csv')

# ### Interpolation

interpol_df = du.data_processing.missing_values_imputation(dmy_df, method='interpolation', id_column='id')
interpol_df

interpol_df.to_csv('imputation_test_interpol_df.csv')

# ## Creating large dummy data

id_col = np.concatenate([np.repeat(1, 25), 
                         np.repeat(2, 17), 
                         np.repeat(3, 56), 
                         np.repeat(4, 138), 
                         np.repeat(5, 2000), 
                         np.repeat(6, 100000)])
id_col

ts_col = np.concatenate([np.arange(25), 
                         np.arange(17), 
                         np.arange(56), 
                         np.arange(138), 
                         np.arange(2000),
                         np.arange(100000)])
ts_col

int_col = np.concatenate([np.random.randint(0, 50, size=(52236)), np.repeat(np.nan, 50000)])
np.random.shuffle(int_col)
int_col

float_col = np.concatenate([np.random.uniform(3, 15, size=(52236)), np.repeat(np.nan, 50000)])
np.random.shuffle(float_col)
float_col

bool_col_1 = np.concatenate([np.random.randint(0, 2, size=(42236)), np.repeat(np.nan, 60000)])
np.random.shuffle(bool_col_1)
bool_col_1

bool_col_2 = np.random.choice(a=[False, True, np.nan], size=(102236), p=[0.25, 0.25, 0.5])
np.random.shuffle(bool_col_2)
bool_col_2

bool_col_3 = np.random.choice(a=[False, True, pd.NaT], size=(102236), p=[0.25, 0.25, 0.5])
np.random.shuffle(bool_col_3)
bool_col_3

data = np.column_stack([id_col, ts_col, int_col, float_col, bool_col_1, bool_col_2, bool_col_3])
data

data_df = pd.DataFrame(data, columns=['id', 'ts', 'int_col', 'float_col', 'bool_col_1', 'bool_col_2', 'bool_col_3'])
data_df

data_df.dtypes

data_df.id = data_df.id.astype('uint')
data_df.ts = data_df.ts.astype('uint')
data_df.int_col = data_df.int_col.astype('Int32')
data_df.float_col = data_df.float_col.astype('float32')
data_df.bool_col_1 = data_df.bool_col_1.astype('boolean')
data_df.bool_col_2 = data_df.bool_col_2.astype('UInt8')
data_df.bool_col_3 = data_df.bool_col_3.astype('boolean')

data_df.dtypes

data_df

data_df.bool_col_1.unique()

data_df.bool_col_2.unique()

data_df.bool_col_3.unique()

du.search_explore.list_boolean_columns(data_df)

# ## Testing imputation methods

# ### Zeros

du.data_processing.missing_values_imputation(data_df, method='zero', id_column='id').tail(20)

# %%timeit
du.data_processing.missing_values_imputation(data_df, method='zero', id_column='id')

# ### Zig Zag

du.data_processing.missing_values_imputation(data_df, method='zigzag', id_column='id').tail(20)

# %%timeit
du.data_processing.missing_values_imputation(data_df, method='zigzag', id_column='id')

# ### Interpolation

du.data_processing.missing_values_imputation(data_df, method='interpolation', id_column='id').tail(20)

data_df.tail(20)

du.data_processing.missing_values_imputation(data_df, columns_to_imputate=['int_col', 'float_col'], method='interpolation', id_column='id')

# %%timeit
du.data_processing.missing_values_imputation(data_df, method='interpolation', id_column='id')

data_df.dtypes

tmp_df = du.data_processing.missing_values_imputation(data_df, method='interpolation', id_column='id')
tmp_df.dtypes

tmp_df


