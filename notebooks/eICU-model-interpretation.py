# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] {"Collapsed": "false"}
# # TCGA Model interpretation
# ---
#
# Interpreting the trained machine learning models, retrieving and analysing feature importance. The models were trained on the preprocessed the TCGA dataset from the Pancancer paper (https://www.ncbi.nlm.nih.gov/pubmed/29625048) into a single, clean dataset.
#
# The Cancer Genome Atlas (TCGA), a landmark cancer genomics program, molecularly characterized over 20,000 primary cancer and matched normal samples spanning 33 cancer types. This joint effort between the National Cancer Institute and the National Human Genome Research Institute began in 2006, bringing together researchers from diverse disciplines and multiple institutions.

# + [markdown] {"colab_type": "text", "id": "KOdmFzXqF7nq", "toc-hr-collapsed": true, "Collapsed": "false"}
# ## Importing the necessary packages

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl", "Collapsed": "false"}
import os                                  # os handles directory/workspace changes
import torch                               # PyTorch to create and apply deep learning models
import xgboost as xgb                      # Gradient boosting trees models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
import joblib                              # Save scikit-learn models in disk
from datetime import datetime              # datetime to use proper date and time formats
import time                                # Measure code execution time
import sys
import numpy as np                         # NumPy for simple mathematical operations
import shap                                # Interpretability package with intuitive plots
import pickle                              # Module to save python objects on disk

# + [markdown] {"Collapsed": "false"}
# Initialize the javascript visualization library to allow for shap plots:

# + {"Collapsed": "false"}
shap.initjs()

# + {"Collapsed": "false"}
# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# + {"Collapsed": "false"}
# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# Path to the dataset files
data_path = 'data/TCGA-Pancancer/cleaned/'
# Path to the trained models
models_path = 'code/tcga-cancer-classification/models/'
# Path where the model interpreter will be saved
interpreter_path = 'code/tcga-cancer-classification/interpreters/'
# Add path to the project scripts
sys.path.append('code/tcga-cancer-classification/scripts/')

# + {"Collapsed": "false"}
import modin.pandas as pd                  # Optimized distributed version of Pandas
import data_utils as du                    # Data science and machine learning relevant methods
from model_interpreter.model_interpreter import ModelInterpreter  # Model interpretability class
import Models                              # Machine learning models

# + [markdown] {"Collapsed": "false"}
# Allow pandas to show more columns:

# + {"Collapsed": "false"}
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

# + [markdown] {"Collapsed": "false"}
# Set the random seed for reproducibility:

# + {"Collapsed": "false"}
du.set_random_seed(42)

# + [markdown] {"Collapsed": "false"}
# ## Loading the data

# + {"Collapsed": "false"}
tcga_df = pd.read_csv(f'{data_path}normalized/tcga.csv')
tcga_df.head()

# + {"Collapsed": "false"}
tcga_df.participant_id.value_counts()

# + {"Collapsed": "false"}
(tcga_df.participant_id.value_counts() > 1).sum()

# + {"Collapsed": "false"}
tcga_df[tcga_df.participant_id == 'TCGA-SR-A6MX']

# + {"Collapsed": "false"}
tcga_df.dtypes

# + [markdown] {"Collapsed": "false"}
# Remove the original string ID column and use the numeric one instead:

# + {"Collapsed": "false"}
tcga_df = tcga_df.drop(columns=['participant_id'], axis=1)
tcga_df = tcga_df.rename(columns={'Unnamed: 0': 'sample_id'})
tcga_df.head()

# + [markdown] {"Collapsed": "false"}
# Convert the label to a numeric format:

# + {"Collapsed": "false"}
tcga_df.tumor_type_label.value_counts()

# + {"Collapsed": "false"}
tcga_df['tumor_type_label'], label_dict = du.embedding.enum_categorical_feature(tcga_df, 'tumor_type_label', nan_value=None, 
                                                                                forbidden_digit=None, clean_name=False)
tcga_df.tumor_type_label.value_counts()

# + [markdown] {"Collapsed": "false"}
# * Most common tumor type: BRCA (breast invasive carcinoma, number 16)
# * Less common tumor type: CHOL (cholangiocarcinoma, number 29)

# + {"Collapsed": "false"}
label_dict

# + {"Collapsed": "false"}
tcga_df.dtypes

# + [markdown] {"Collapsed": "false"}
# Convert to a PyTorch tensor:

# + {"Collapsed": "false"}
tcga_tsr = torch.from_numpy(tcga_df.to_numpy())
tcga_tsr

# + [markdown] {"Collapsed": "false"}
# Create a dataset:

# + {"Collapsed": "false"}
dataset = du.datasets.Tabular_Dataset(tcga_tsr, tcga_df)

# + {"Collapsed": "false"}
len(dataset)

# + {"Collapsed": "false"}
dataset.label_column

# + {"Collapsed": "false"}
dataset.y

# + [markdown] {"Collapsed": "false"}
# Get the train, validation and test sets data loaders, which will allow loading batches:

# + {"Collapsed": "false"}
train_dataloader, val_dataloader, test_dataloader = du.machine_learning.create_train_sets(dataset, test_train_ratio=0.2, validation_ratio=0.1,
                                                                                          batch_size=len(dataset), get_indeces=False)

# + [markdown] {"Collapsed": "false"}
# Get the full tensors with all the data from each set:

# + {"Collapsed": "false"}
train_features, train_labels = next(iter(train_dataloader))
val_features, val_labels = next(iter(val_dataloader))
test_features, test_labels = next(iter(test_dataloader))

# + {"Collapsed": "false"}
val_features

# + {"Collapsed": "false"}
len(train_features)

# + [markdown] {"Collapsed": "true"}
# ## Denormalizing the data
#
# [TODO]

# + {"Collapsed": "false"}
tcga_df.columns

# + [markdown] {"Collapsed": "false"}
# ## Adapting the data to XGBoost and Scikit-Learn

# + [markdown] {"Collapsed": "false"}
# Make a copy of the dataframe:

# + {"Collapsed": "false"}
sckt_tcga_df = tcga_df.copy()
sckt_tcga_df

# + [markdown] {"Collapsed": "false"}
# Convert categorical columns to string type:

# + {"Collapsed": "false"}
sckt_tcga_df.race = sckt_tcga_df.race.astype(str)
sckt_tcga_df.ajcc_pathologic_tumor_stage = sckt_tcga_df.ajcc_pathologic_tumor_stage.astype(str)

# + [markdown] {"Collapsed": "false"}
# One hot encode categorical features:

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false"}
sckt_tcga_df, new_cols= du.data_processing.one_hot_encoding_dataframe(sckt_tcga_df, columns=['race', 'ajcc_pathologic_tumor_stage'], 
                                                                      clean_name=False, clean_missing_values=False,
                                                                      has_nan=False, join_rows=False,
                                                                      get_new_column_names=True, inplace=True)
new_cols

# + {"Collapsed": "false"}
sckt_tcga_df.head()

# + [markdown] {"Collapsed": "false"}
# Remove the ID column:

# + {"Collapsed": "false"}
sckt_tcga_df = sckt_tcga_df.drop(columns='sample_id')
sckt_tcga_df.head()

# + [markdown] {"Collapsed": "false"}
# Convert to a PyTorch tensor:

# + {"Collapsed": "false"}
sckt_tcga_tsr = torch.from_numpy(sckt_tcga_df.to_numpy())
sckt_tcga_tsr

# + [markdown] {"Collapsed": "false"}
# Create a dataset:

# + {"Collapsed": "false"}
sckt_dataset = du.datasets.Tabular_Dataset(sckt_tcga_tsr, sckt_tcga_df)

# + {"Collapsed": "false"}
len(sckt_dataset)

# + {"Collapsed": "false"}
sckt_dataset.label_column

# + {"Collapsed": "false"}
sckt_dataset.y

# + [markdown] {"Collapsed": "false"}
# Get the train, validation and test sets data loaders, which will allow loading batches:

# + {"Collapsed": "false"}
sckt_train_dataloader, sckt_val_dataloader, sckt_test_dataloader = du.machine_learning.create_train_sets(sckt_dataset, test_train_ratio=0.2, validation_ratio=0.1,
                                                                                                         batch_size=len(sckt_dataset), get_indeces=False)

# + [markdown] {"Collapsed": "false"}
# Get the full tensors with all the data from each set:

# + {"Collapsed": "false"}
sckt_train_features, sckt_train_labels = next(iter(sckt_train_dataloader))
sckt_val_features, sckt_val_labels = next(iter(sckt_val_dataloader))
sckt_test_features, sckt_test_labels = next(iter(sckt_test_dataloader))

# + {"Collapsed": "false"}
sckt_val_features

# + {"Collapsed": "false"}
len(sckt_train_features)

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": true}
# ## Interpreting MLP

# + [markdown] {"Collapsed": "false"}
# ### Loading the model

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false"}
mlp_model = du.deep_learning.load_checkpoint(filepath=f'{models_path}mlp/checkpoint_26_01_2020_18_22.pth', ModelClass=Models.MLP)
mlp_model

# + [markdown] {"Collapsed": "false"}
# Check performance metrics:

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false"}
output, metrics = du.deep_learning.model_inference(mlp_model, dataloader=test_dataloader, metrics=['accuracy', 'F1', 'loss',
                                                                                                   'AUC', 'AUC_weighted'],
                                                   model_type='mlp', cols_to_remove=0)
metrics

# + [markdown] {"Collapsed": "false"}
# ### Interpreting the model

# + [markdown] {"Collapsed": "false"}
# Names of the features:

# + {"Collapsed": "false"}
feat_names = list(tcga_df.columns)
feat_names.remove('sample_id')
feat_names.remove('tumor_type_label')

# + {"Collapsed": "false"}
# feat_names

# + [markdown] {"Collapsed": "false"}
# Create a model interpreter object:

# + {"Collapsed": "false"}
# shap.DeepExplainer?

# + {"Collapsed": "false"}
# mlp_explainer = shap.DeepExplainer(mlp_model, test_features[:, 1:])

# + {"Collapsed": "false"}
mlp_interpreter = ModelInterpreter(mlp_model, data=dataset.X, labels=dataset.y, model_type='mlp',
                                   id_column=0, inst_column=None,
                                   fast_calc=True, SHAP_bkgnd_samples=500,
                                   random_seed=du.random_seed, feat_names=feat_names)
mlp_interpreter

# + [markdown] {"Collapsed": "false"}
# Calculate feature importance:

# + {"Collapsed": "false"}
# # mlp_explainer.shap_values?

# + {"Collapsed": "false"}
# mlp_explainer.interpret_model?

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false"}
start = time.time()
# mlp_shap_values = mlp_explainer.shap_values(test_features[:1000, 1:])
mlp_shap_values = mlp_interpreter.interpret_model(test_data=test_features[:1000], test_labels=test_labels[:1000],
                                                  instance_importance=False, feature_importance='shap')
display(mlp_shap_values)
end = time.time()
print(f'Cell execution time: {end - start} s')

# + [markdown] {"Collapsed": "false"}
# In a n1-high-memory-16 (16 vCPUs, 104 GB RAM) Google Cloud VM instance, the above calculation of SHAP values takes around 4 hours and 55 minutes.

# + {"Collapsed": "false"}
# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
shap_values_filename = f'{interpreter_path}mlp/shap_values_checkpoint_{current_datetime}.pickle'
# Save model interpreter object, with the instance and feature importance scores, in a pickle file
joblib.dump(mlp_shap_values, interpreter_filename)

# + {"Collapsed": "false"}
# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
interpreter_filename = f'{interpreter_path}mlp/model_interpreter_checkpoint_{current_datetime}.pickle'
joblib.dump(mlp_interpreter, interpreter_filename)

# + {"Collapsed": "false"}
# mlp_shap_values = joblib.load(shap_values_filename)
mlp_shap_values = joblib.load(f'{interpreter_path}mlp/shap_values_checkpoint_28_01_2020_17_33.pickle')
mlp_shap_values

# + {"Collapsed": "false"}
# mlp_explainer = joblib.load(interpreter_filename)
mlp_interpreter = joblib.load(f'{interpreter_path}mlp/model_interpreter_checkpoint_28_01_2020_17_34.pickle')
mlp_interpreter

# + [markdown] {"Collapsed": "false"}
# The interpreter gets different feature importance arrays for each output (in this case, each of the 33 possible tumor types), with one contribution value for each feature:

# + {"Collapsed": "false"}
mlp_shap_values = mlp_interpreter.feat_scores

# + {"Collapsed": "false"}
len(mlp_shap_values)

# + {"Collapsed": "false"}
mlp_shap_values[0].shape

# + {"Collapsed": "false"}
[scores.sum() for scores in mlp_shap_values]

# + {"Collapsed": "false"}
[scores[0].sum() for scores in mlp_shap_values]

# + [markdown] {"Collapsed": "false"}
# It appears that only the predicted class (the output with highest score) has the feature importance scores summing to one.

# + {"Collapsed": "false"}
np.argmax([scores.sum() for scores in mlp_shap_values])

# + {"Collapsed": "false"}
np.argmax([scores[0].sum() for scores in mlp_shap_values])

# + {"Collapsed": "false"}
pred = mlp_model(test_features[0, 1:].unsqueeze(0)).topk(1).indices.item()
pred

# + {"Collapsed": "false"}
int(test_labels[0])

# + [markdown] {"Collapsed": "false"}
# [TODO] Get a denormalized version of the tensors so as to see the original values in the plots.

# + [markdown] {"Collapsed": "false"}
# #### Summary plot

# + [markdown] {"Collapsed": "false"}
# Summary plot with all 33 tumor types (i.e. all output classes):

# + {"Collapsed": "false"}
test_features[:1000, 1:].unsqueeze(0).shape

# + {"Collapsed": "false"}
shap.summary_plot(mlp_shap_values,
                  features=test_features[:1000, 1:], 
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 15))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just breast cancer (i.e. just one output class; in this case, the most common tumor type):

# + {"Collapsed": "false"}
output_class = 16

# + {"Collapsed": "false"}
shap.summary_plot(mlp_shap_values[output_class], 
                  features=test_features[:1000, 1:],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just pancreatic cancer (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 8

# + {"Collapsed": "false"}
shap.summary_plot(mlp_shap_values[output_class], 
                  features=test_features[:1000, 1:],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just liver cancer (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 15

# + {"Collapsed": "false"}
shap.summary_plot(mlp_shap_values[output_class], 
                  features=test_features[:1000, 1:],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just lung adenocarcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 23

# + {"Collapsed": "false"}
shap.summary_plot(mlp_shap_values[output_class], 
                  features=test_features[:1000, 1:],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just lung squamous cell carcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 12

# + {"Collapsed": "false"}
shap.summary_plot(mlp_shap_values[output_class], 
                  features=test_features[:1000, 1:],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just thyroid cancer (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 21

# + {"Collapsed": "false"}
shap.summary_plot(mlp_shap_values[output_class], 
                  features=test_features[:1000, 1:],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just skin cutaneous melanoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 4

# + {"Collapsed": "false"}
shap.summary_plot(mlp_shap_values[output_class], 
                  features=test_features[:1000, 1:],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just prostate cancer (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 6

# + {"Collapsed": "false"}
shap.summary_plot(mlp_shap_values[output_class], 
                  features=test_features[:1000, 1:],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just rectum adenocarcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 13

# + {"Collapsed": "false"}
shap.summary_plot(mlp_shap_values[output_class], 
                  features=test_features[:1000, 1:],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just colon adenocarcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 11

# + {"Collapsed": "false"}
shap.summary_plot(mlp_shap_values[output_class], 
                  features=test_features[:1000, 1:],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just cervical squamous cell carcinoma and endocervical adenocarcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 10

# + {"Collapsed": "false"}
shap.summary_plot(mlp_shap_values[output_class], 
                  features=test_features[:1000, 1:],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just brain lower grade glioma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 22

# + {"Collapsed": "false"}
shap.summary_plot(mlp_shap_values[output_class], 
                  features=test_features[:1000, 1:],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just cholangiocarcinoma (i.e. just one output class; in this case, the least common tumor type):

# + {"Collapsed": "false"}
output_class = 29

# + {"Collapsed": "false"}
shap.summary_plot(mlp_shap_values[output_class], 
                  features=test_features[:1000, 1:],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# #### Sample force plot

# + {"Collapsed": "false"}
du.search_explore.find_val_idx(test_labels, 16)

# + {"Collapsed": "false"}
du.search_explore.find_val_idx(test_labels, 29)

# + {"Collapsed": "false"}
# Tumor type 16 (most common) sample
# sample = 27
# Tumor type 29 (less common) sample
sample = 225

# + {"Collapsed": "false"}
pred = mlp_model(test_features[sample, 1:].unsqueeze(0)).topk(1).indices.item()
pred

# + {"Collapsed": "false"}
mlp_interpreter.explainer.expected_value[pred]

# + {"Collapsed": "false"}
shap.force_plot(mlp_interpreter.explainer.expected_value[pred], 
                mlp_shap_values[pred][sample],
                features=test_features[sample, 1:].numpy(), 
                feature_names=feat_names)

# + [markdown] {"Collapsed": "false"}
# #### Sample decision plot

# + {"Collapsed": "false"}
shap.decision_plot(mlp_interpreter.explainer.expected_value[pred], 
                   mlp_shap_values[pred][sample],
                   features=test_features[sample, 1:].numpy(), 
                   feature_names=feat_names)

# + [markdown] {"Collapsed": "false"}
# #### Sample waterfall plot

# + {"Collapsed": "false"}
shap.waterfall_plot(mlp_interpreter.explainer.expected_value[pred], 
                    mlp_shap_values[pred][sample],
                    features=test_features[sample, 1:].numpy(), 
                    feature_names=feat_names)

# + {"Collapsed": "false"}
test_features[:, 1:].shape

# + {"Collapsed": "false"}
test_features[0].shape

# + {"Collapsed": "false"}
test_features[0].unsqueeze(0).shape

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": true}
# ## Interpreting XGBoost

# + [markdown] {"Collapsed": "false"}
# ### Loading the model

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false"}
xgb_model = joblib.load(f'{models_path}xgb/checkpoint_27_01_2020_04_47.model')
xgb_model

# + [markdown] {"Collapsed": "false"}
# Check performance metrics:

# + {"Collapsed": "false"}
pred = xgb_model.predict(sckt_test_features)

# + {"Collapsed": "false"}
acc = accuracy_score(sckt_test_labels, pred)
acc

# + {"Collapsed": "false"}
f1 = f1_score(sckt_test_labels, pred, average='weighted')
f1

# + {"Collapsed": "false"}
pred_proba = xgb_model.predict_proba(sckt_test_features)

# + {"Collapsed": "false"}
loss = log_loss(sckt_test_labels, pred_proba)
loss

# + {"Collapsed": "false"}
auc = roc_auc_score(sckt_test_labels, pred_proba, multi_class='ovr', average='macro')
auc

# + {"Collapsed": "false"}
auc_wgt = roc_auc_score(sckt_test_labels, pred_proba, multi_class='ovr', average='weighted')
auc_wgt

# + {"Collapsed": "false"}
pred

# + {"Collapsed": "false"}
pred.shape

# + {"Collapsed": "false"}
pred_proba

# + {"Collapsed": "false"}
pred_proba.shape

# + [markdown] {"Collapsed": "false"}
# ### Interpreting the model

# + [markdown] {"Collapsed": "false"}
# Names of the features:

# + {"Collapsed": "false"}
feat_names = list(sckt_tcga_df.columns)
feat_names.remove('tumor_type_label')

# + [markdown] {"Collapsed": "false"}
# Create a SHAP explainer object:

# + {"Collapsed": "false"}
# shap.TreeExplainer?

# + {"Collapsed": "false"}
xgb_explainer = shap.TreeExplainer(xgb_model)

# + [markdown] {"Collapsed": "false"}
# Calculate feature importance:

# + {"Collapsed": "false"}
# xgb_explainer.shap_values?

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false"}
start = time.time()
# shap_values = explainer.shap_values(sckt_test_features.numpy(), l1_reg='num_features(20)', nsamples=500)
xgb_shap_values = xgb_explainer.shap_values(sckt_test_features[:1000].numpy())
display(xgb_shap_values)
end = time.time()
print(f'Cell execution time: {end - start} s')

# + [markdown] {"Collapsed": "false"}
# In a n1-highmem-8 (8 vCPUs, 52 GB RAM) Google Cloud VM instance, the above calculation of SHAP values takes around 19 seconds.

# + {"Collapsed": "false"}
# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
shap_values_filename = f'{interpreter_path}xgb/shap_values_checkpoint_{current_datetime}.pickle'
# Save calculated SHAP values in a pickle file
joblib.dump(xgb_shap_values, shap_values_filename)

# + {"Collapsed": "false"}
# Filename and path where the model will be saved
explainer_filename = f'{interpreter_path}xgb/explainer_checkpoint_{current_datetime}.pickle'
# Save calculated SHAP values in a pickle file
joblib.dump(xgb_explainer, explainer_filename)

# + {"Collapsed": "false"}
# xgb_shap_values = joblib.load(shap_values_filename)
xgb_shap_values = joblib.load(f'{interpreter_path}xgb/shap_values_checkpoint_30_01_2020_18_07.pickle')
xgb_shap_values

# + {"Collapsed": "false"}
# xgb_shap_values = joblib.load(explainer_filename)
xgb_explainer = joblib.load(f'{interpreter_path}xgb/explainer_checkpoint_30_01_2020_18_07.pickle')
xgb_explainer

# + [markdown] {"Collapsed": "false"}
# The interpreter gets different feature importance arrays for each output (in this case, each of the 33 possible tumor types), with one contribution value for each feature:

# + {"Collapsed": "false"}
len(xgb_shap_values)

# + {"Collapsed": "false"}
xgb_shap_values[0].shape

# + {"Collapsed": "false"}
[scores.sum() for scores in xgb_shap_values]

# + [markdown] {"Collapsed": "false"}
# It appears that only the predicted class (the output with highest score) has the feature importance scores summing to one.

# + {"Collapsed": "false"}
np.argmax([scores.sum() for scores in xgb_shap_values])

# + {"Collapsed": "false"}
pred_0 = xgb_model.predict(sckt_test_features[0])[0]
pred_0

# + {"Collapsed": "false"}
int(sckt_test_labels[0])

# + [markdown] {"Collapsed": "false"}
# [TODO] Get a denormalized version of the tensors so as to see the original values in the plots.

# + [markdown] {"Collapsed": "false"}
# #### Summary plot

# + [markdown] {"Collapsed": "false"}
# Summary plot with all 33 tumor types (i.e. all output classes):

# + {"Collapsed": "false"}
shap.summary_plot(xgb_shap_values,
                  features=sckt_test_features[:1000], 
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 15))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just breast cancer (i.e. just one output class; in this case, the most common tumor type):

# + {"Collapsed": "false"}
output_class = 16

# + {"Collapsed": "false"}
shap.summary_plot(xgb_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just pancreatic cancer (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 8

# + {"Collapsed": "false"}
shap.summary_plot(xgb_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just liver cancer (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 15

# + {"Collapsed": "false"}
shap.summary_plot(xgb_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just lung adenocarcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 23

# + {"Collapsed": "false"}
shap.summary_plot(xgb_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just lung squamous cell carcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 12

# + {"Collapsed": "false"}
shap.summary_plot(xgb_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just thyroid cancer (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 21

# + {"Collapsed": "false"}
shap.summary_plot(xgb_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just skin cutaneous melanoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 4

# + {"Collapsed": "false"}
shap.summary_plot(xgb_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just prostate cancer (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 6

# + {"Collapsed": "false"}
shap.summary_plot(xgb_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just rectum adenocarcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 13

# + {"Collapsed": "false"}
shap.summary_plot(xgb_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just colon adenocarcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 11

# + {"Collapsed": "false"}
shap.summary_plot(xgb_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just cervical squamous cell carcinoma and endocervical adenocarcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 10

# + {"Collapsed": "false"}
shap.summary_plot(xgb_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just brain lower grade glioma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 22

# + {"Collapsed": "false"}
shap.summary_plot(xgb_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just cholangiocarcinoma (i.e. just one output class; in this case, the least common tumor type):

# + {"Collapsed": "false"}
output_class = 29

# + {"Collapsed": "false"}
shap.summary_plot(xgb_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# #### Sample force plot


# + {"Collapsed": "false"}
if 'xgb_explainer' in locals():
    xgb_expected_value = xgb_explainer.expected_value
else:
    # Calculate the expected value
    xgb_expected_value = np.mean(pred)


# + {"Collapsed": "false"}
xgb_expected_value

# + {"Collapsed": "false"}
(sckt_test_labels == 16).nonzero().squeeze()

# + {"Collapsed": "false"}
du.search_explore.find_val_idx(sckt_test_labels, 16)

# + {"Collapsed": "false"}
du.search_explore.find_val_idx(sckt_test_labels, 29)

# + {"Collapsed": "false"}
# Tumor type 16 (most common) sample
# For some reason, in this class' samples, the sum of SHAP values doesn't 
# ever add up to the exact margin value, although it still leads to relatively
# high values (> 0.7)
sample = 27
# Tumor type 29 (less common) sample
# sample = 225
# Other random sample
# sample = 1

# + {"Collapsed": "false"}
pred = int(xgb_model.predict(sckt_test_features[sample]))
pred

# + {"Collapsed": "false"}
xgb_shap_values[pred][sample]

# + {"Collapsed": "false"}
pred_margins = xgb_model.predict(sckt_test_features[sample], output_margin=True)
pred_margins

# + {"Collapsed": "false"}
pred_margins[0][pred]

# + {"Collapsed": "false"}
shap.force_plot(xgb_expected_value[pred],
                xgb_shap_values[pred][sample],
                features=sckt_test_features[sample].numpy(), 
                feature_names=feat_names)

# + [markdown] {"Collapsed": "false"}
# #### Sample decision plot

# + {"Collapsed": "false"}
shap.decision_plot(xgb_expected_value[pred],
                   xgb_shap_values[pred][sample],
                   features=sckt_test_features[sample].numpy(), 
                   feature_names=feat_names)

# + [markdown] {"Collapsed": "false"}
# #### Sample waterfall plot

# + {"Collapsed": "false"}
shap.waterfall_plot(xgb_expected_value[pred],
                    xgb_shap_values[pred][sample],
                    features=sckt_test_features[sample].numpy(), 
                    feature_names=feat_names)

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": true}
# ## Interpreting Logistic Regression

# + [markdown] {"Collapsed": "false"}
# ### Loading the model

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false"}
logreg_model = joblib.load(f'{models_path}logreg/checkpoint_27_01_2020_05_22.model')
logreg_model

# + [markdown] {"Collapsed": "false"}
# Check performance metrics:

# + {"Collapsed": "false"}
acc = logreg_model.score(sckt_test_features, sckt_test_labels)
acc

# + {"Collapsed": "false"}
pred = logreg_model.predict(sckt_test_features)

# + {"Collapsed": "false"}
f1 = f1_score(sckt_test_labels, pred, average='weighted')
f1

# + {"Collapsed": "false"}
pred_proba = logreg_model.predict_proba(sckt_test_features)

# + {"Collapsed": "false"}
loss = log_loss(sckt_test_labels, pred_proba)
loss

# + {"Collapsed": "false"}
auc = roc_auc_score(sckt_test_labels, pred_proba, multi_class='ovr', average='macro')
auc

# + {"Collapsed": "false"}
auc_wgt = roc_auc_score(sckt_test_labels, pred_proba, multi_class='ovr', average='weighted')
auc_wgt

# + [markdown] {"Collapsed": "false"}
# ### Interpreting the model

# + [markdown] {"Collapsed": "false"}
# Names of the features:

# + {"Collapsed": "false"}
feat_names = list(sckt_tcga_df.columns)
feat_names.remove('tumor_type_label')

# + {"Collapsed": "false"}
# feat_names

# + [markdown] {"Collapsed": "false"}
# Create a SHAP explainer object:

# + {"Collapsed": "false"}
# shap.LinearExplainer?

# + {"Collapsed": "false"}
# # shap.KernelExplainer?

# + {"Collapsed": "false"}
logreg_explainer = shap.LinearExplainer(logreg_model, sckt_train_features.numpy(), feature_perturbation='interventional')

# + [markdown] {"Collapsed": "false"}
# Calculate feature importance:

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false"}
start = time.time()
logreg_shap_values = logreg_explainer.shap_values(sckt_test_features[:1000].numpy())
display(logreg_shap_values)
end = time.time()
print(f'Cell execution time: {end - start} s')

# + [markdown] {"Collapsed": "false"}
# In a n1-standard-8 (8 vCPUs, 30 GB RAM) Google Cloud VM instance, the above calculation of SHAP values takes 12 seconds.
# In a n1-standard-16 (16 vCPUs, 60 GB RAM) Google Cloud VM instance, the above calculation of SHAP values takes less than 6 seconds.

# + {"Collapsed": "false"}
# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
shap_values_filename = f'{interpreter_path}logreg/shap_values_checkpoint_{current_datetime}.pickle'
# Save calculated SHAP values in a pickle file
joblib.dump(logreg_shap_values, shap_values_filename)

# + {"Collapsed": "false"}
# Filename and path where the model will be saved
explainer_filename = f'{interpreter_path}logreg/explainer_checkpoint_{current_datetime}.pickle'
# Save calculated SHAP values in a pickle file
joblib.dump(logreg_explainer, explainer_filename)

# + {"Collapsed": "false"}
# logreg_shap_values = joblib.load(shap_values_filename)
logreg_shap_values = joblib.load(f'{interpreter_path}logreg/shap_values_checkpoint_31_01_2020_01_49.pickle')
logreg_shap_values

# + {"Collapsed": "false"}
# logreg_shap_values = joblib.load(interpreter_filename)
logreg_explainer = joblib.load(f'{interpreter_path}logreg/explainer_checkpoint_31_01_2020_01_49.pickle')
logreg_explainer

# + [markdown] {"Collapsed": "false"}
# The interpreter gets different feature importance arrays for each output (in this case, each of the 33 possible tumor types), with one contribution value for each feature:

# + {"Collapsed": "false"}
len(logreg_shap_values)

# + {"Collapsed": "false"}
logreg_shap_values[0].shape

# + {"Collapsed": "false"}
[scores[0].sum() for scores in logreg_shap_values]

# + [markdown] {"Collapsed": "false"}
# Somehow the coefficients are not summing to 1... However, according to the LinearSHAP definition, perhaps it's not supposed to sum to 1, as it doesn't learn a separate linear model: "Assuming features are independent leads to interventional SHAP values which for a linear model are coef[i] * (x[i] - X.mean(0)[i]) for the ith feature."

# + {"Collapsed": "false"}
np.argmax([scores[0].sum() for scores in logreg_shap_values])

# + {"Collapsed": "false"}
logreg_shap_values[0][0]

# + {"Collapsed": "false"}
logreg_shap_values[0][0].sum()

# + {"Collapsed": "false"}
sckt_test_features[0].shape

# + {"Collapsed": "false"}
pred = logreg_model.predict(sckt_test_features[0].unsqueeze(0)).item()
pred

# + {"Collapsed": "false"}
int(sckt_test_labels[0])

# + [markdown] {"Collapsed": "false"}
# [TODO] Get a denormalized version of the tensors so as to see the original values in the plots.

# + [markdown] {"Collapsed": "false"}
# #### Summary plot

# + [markdown] {"Collapsed": "false"}
# Summary plot with all 33 tumor types (i.e. all output classes):

# + {"Collapsed": "false"}
shap.summary_plot(logreg_shap_values,
                  features=sckt_test_features[:1000], 
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 15))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just breast cancer (i.e. just one output class; in this case, the most common tumor type):

# + {"Collapsed": "false"}
output_class = 16

# + {"Collapsed": "false"}
shap.summary_plot(logreg_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just pancreatic cancer (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 8

# + {"Collapsed": "false"}
shap.summary_plot(logreg_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just liver cancer (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 15

# + {"Collapsed": "false"}
shap.summary_plot(logreg_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just lung adenocarcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 23

# + {"Collapsed": "false"}
shap.summary_plot(logreg_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just lung squamous cell carcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 12

# + {"Collapsed": "false"}
shap.summary_plot(logreg_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just thyroid cancer (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 21

# + {"Collapsed": "false"}
shap.summary_plot(logreg_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just skin cutaneous melanoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 4

# + {"Collapsed": "false"}
shap.summary_plot(logreg_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just prostate cancer (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 6

# + {"Collapsed": "false"}
shap.summary_plot(logreg_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just rectum adenocarcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 13

# + {"Collapsed": "false"}
shap.summary_plot(logreg_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just colon adenocarcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 11

# + {"Collapsed": "false"}
shap.summary_plot(logreg_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just cervical squamous cell carcinoma and endocervical adenocarcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 10

# + {"Collapsed": "false"}
shap.summary_plot(logreg_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just brain lower grade glioma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 22

# + {"Collapsed": "false"}
shap.summary_plot(logreg_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just cholangiocarcinoma (i.e. just one output class; in this case, the least common tumor type):

# + {"Collapsed": "false"}
output_class = 29

# + {"Collapsed": "false"}
shap.summary_plot(logreg_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# #### Sample force plot


# + {"Collapsed": "false"}
if 'logreg_explainer' in locals():
    logreg_expected_value = logreg_explainer.expected_value
else:
    # Calculate the expected value
    logreg_expected_value = np.mean(pred)


# + {"Collapsed": "false"}
logreg_expected_value

# + {"Collapsed": "false"}
(sckt_test_labels == 16).nonzero().squeeze()

# + {"Collapsed": "false"}
du.search_explore.find_val_idx(sckt_test_labels, 16)

# + {"Collapsed": "false"}
du.search_explore.find_val_idx(sckt_test_labels, 29)

# + {"Collapsed": "false"}
# Tumor type 16 (most common) sample
# sample = 27
# Tumor type 29 (less common) sample
sample = 225
# Other random sample
# sample = 1

# + {"Collapsed": "false"}
sckt_test_features[sample].shape

# + {"Collapsed": "false"}
pred = int(logreg_model.predict(sckt_test_features[sample].unsqueeze(0)))
pred

# + {"Collapsed": "false"}
logreg_shap_values[pred][sample]

# + {"Collapsed": "false"}
pred_logodds = logreg_model.predict_log_proba(sckt_test_features[sample].unsqueeze(0))
pred_logodds

# + {"Collapsed": "false"}
pred_logodds[0][pred]

# + {"Collapsed": "false"}
shap.force_plot(logreg_expected_value[pred],
                logreg_shap_values[pred][sample],
                features=sckt_test_features[sample].numpy(), 
                feature_names=feat_names)

# + [markdown] {"Collapsed": "false"}
# #### Sample decision plot

# + {"Collapsed": "false"}
shap.decision_plot(logreg_expected_value[pred],
                   logreg_shap_values[pred][sample],
                   features=sckt_test_features[sample].numpy(), 
                   feature_names=feat_names)

# + [markdown] {"Collapsed": "false"}
# #### Sample waterfall plot

# + {"Collapsed": "false"}
shap.waterfall_plot(logreg_expected_value[pred],
                    logreg_shap_values[pred][sample],
                    features=sckt_test_features[sample].numpy(), 
                    feature_names=feat_names)

# + [markdown] {"Collapsed": "false", "toc-hr-collapsed": true}
# ## Interpreting SVM

# + [markdown] {"Collapsed": "false"}
# ### Loading the model

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false"}
svm_model = joblib.load(f'{models_path}svm/checkpoint_27_01_2020_06_55.model')
svm_model

# + [markdown] {"Collapsed": "false"}
# Check performance metrics:

# + {"Collapsed": "false"}
acc = svm_model.score(sckt_test_features, sckt_test_labels)
acc

# + {"Collapsed": "false"}
pred = svm_model.predict(sckt_test_features)

# + {"Collapsed": "false"}
f1 = f1_score(sckt_test_labels, pred, average='weighted')
f1

# + {"Collapsed": "false"}
pred_proba = svm_model.predict_proba(sckt_test_features)

# + {"Collapsed": "false"}
loss = log_loss(sckt_test_labels, pred_proba)
loss

# + {"Collapsed": "false"}
auc = roc_auc_score(sckt_test_labels, pred_proba, multi_class='ovr', average='macro')
auc

# + {"Collapsed": "false"}
auc_wgt = roc_auc_score(sckt_test_labels, pred_proba, multi_class='ovr', average='weighted')
auc_wgt

# + [markdown] {"Collapsed": "false"}
# ### Interpreting the model

# + [markdown] {"Collapsed": "false"}
# Names of the features:

# + {"Collapsed": "false"}
feat_names = list(sckt_tcga_df.columns)
feat_names.remove('tumor_type_label')

# + {"Collapsed": "false"}
# feat_names

# + [markdown] {"Collapsed": "false"}
# Create a SHAP explainer object:

# + {"Collapsed": "false"}
# # shap.LinearExplainer?

# + {"Collapsed": "false"}
# shap.KernelExplainer?

# + {"Collapsed": "false"}
svm_explainer = shap.KernelExplainer(svm_model.predict_proba, np.zeros((1, len(feat_names))), max_bkgnd_samples=100)

# + [markdown] {"Collapsed": "false"}
# Calculate feature importance:

# + {"pixiedust": {"displayParams": {}}, "Collapsed": "false"}
start = time.time()
svm_shap_values = svm_explainer.shap_values(sckt_test_features[:500].numpy(), l1_reg='num_features(10)', nsamples=1000)
display(svm_shap_values)
end = time.time()
print(f'Cell execution time: {end - start} s')

# + {"Collapsed": "false"}
# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
# Filename and path where the model will be saved
interpreter_filename = f'{interpreter_path}svm/checkpoint_{current_datetime}.pickle'
# Save calculated SHAP values in a pickle file
joblib.dump(svm_shap_values, interpreter_filename)

# + {"Collapsed": "false"}
svm_shap_values = joblib.load(interpreter_filename)
svm_shap_values

# + [markdown] {"Collapsed": "false"}
# The interpreter gets different feature importance arrays for each output (in this case, each of the 33 possible tumor types), with one contribution value for each feature:

# + {"Collapsed": "false"}
len(svm_shap_values)

# + {"Collapsed": "false"}
[scores.sum() for scores in svm_shap_values]

# + [markdown] {"Collapsed": "false"}
# It appears that only the predicted class (the output with highest score) has the feature importance scores summing to one.

# + {"Collapsed": "false"}
np.argmax([scores.sum() for scores in svm_shap_values])

# + {"Collapsed": "false"}
pred = svm_model(sckt_test_features[0].unsqueeze(0)).topk(1).indices.item()
pred

# + {"Collapsed": "false"}
int(test_labels[0])

# + [markdown] {"Collapsed": "false"}
# [TODO] Get a denormalized version of the tensors so as to see the original values in the plots.

# + [markdown] {"Collapsed": "false"}
# #### Summary plot

# + {"Collapsed": "false"}
np.array(svm_shap_values).shape

# + {"Collapsed": "false"}
svm_shap_values.shape

# + {"Collapsed": "false"}
sckt_test_features[0].unsqueeze(0).numpy().shape

# + [markdown] {"Collapsed": "false"}
# Summary plot with all 33 tumor types (i.e. all output classes):

# + {"Collapsed": "false"}
sckt_test_features[:1000].unsqueeze(0).numpy().shape

# + {"Collapsed": "false"}
shap.summary_plot(svm_shap_values, 
                  features=sckt_test_features[:1000].numpy(), 
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just breast cancer (i.e. just one output class; in this case, the most common tumor type):

# + {"Collapsed": "false"}
output_class = 16

# + {"Collapsed": "false"}
shap.summary_plot(svm_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just pancreatic cancer (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 8

# + {"Collapsed": "false"}
shap.summary_plot(svm_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just liver cancer (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 15

# + {"Collapsed": "false"}
shap.summary_plot(svm_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just lung adenocarcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 23

# + {"Collapsed": "false"}
shap.summary_plot(svm_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just lung squamous cell carcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 12

# + {"Collapsed": "false"}
shap.summary_plot(svm_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just thyroid cancer (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 21

# + {"Collapsed": "false"}
shap.summary_plot(svm_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just skin cutaneous melanoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 4

# + {"Collapsed": "false"}
shap.summary_plot(svm_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just prostate cancer (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 6

# + {"Collapsed": "false"}
shap.summary_plot(svm_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just rectum adenocarcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 13

# + {"Collapsed": "false"}
shap.summary_plot(svm_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just colon adenocarcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 11

# + {"Collapsed": "false"}
shap.summary_plot(svm_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just cervical squamous cell carcinoma and endocervical adenocarcinoma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 10

# + {"Collapsed": "false"}
shap.summary_plot(svm_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just brain lower grade glioma (i.e. just one output class):

# + {"Collapsed": "false"}
output_class = 22

# + {"Collapsed": "false"}
shap.summary_plot(svm_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# Summary plot for just cholangiocarcinoma (i.e. just one output class; in this case, the least common tumor type):

# + {"Collapsed": "false"}
output_class = 29

# + {"Collapsed": "false"}
shap.summary_plot(svm_shap_values[output_class], 
                  features=sckt_test_features[:1000],
                  feature_names=feat_names, plot_type='bar',
                  plot_size=(15, 10))

# + [markdown] {"Collapsed": "false"}
# #### Sample force plot

# + {"Collapsed": "false"}
sample = 0

# + {"Collapsed": "false"}
# [TODO] Not being able to do force plots on multiple outputs
# shap.force_plot(interpreter.explainer.expected_value, 
#                 interpreter.feat_scores, 
#                 features=test_features[sample, 1:].unsqueeze(0).numpy(), 
#                 feature_names=feat_names)

# + {"Collapsed": "false"}
shap.force_plot(svm_explainer.expected_value[pred], 
                svm_shap_values[pred], 
                features=sckt_test_features[sample].unsqueeze(0).numpy(), 
                feature_names=feat_names)

# + [markdown] {"Collapsed": "false"}
# #### Sample decision plot

# + {"Collapsed": "false"}
sample = 0

# + {"Collapsed": "false"}
# [TODO] Not being able to do decision plots on multiple outputs
# shap.multioutput_decision_plot(interpreter.explainer.expected_value, 
#                                interpreter.feat_scores, row_index=0,
#                                features=test_features[sample, 1:].unsqueeze(0).numpy(), 
#                                feature_names=feat_names)

# + {"Collapsed": "false"}
shap.decision_plot(svm_explainer.expected_value[pred], 
                   svm_shap_values[pred], 
                   features=sckt_test_features[sample].unsqueeze(0).numpy(), 
                   feature_names=feat_names)

# + {"Collapsed": "false"}
sckt_test_features[:].shape

# + {"Collapsed": "false"}
sckt_test_features[0].shape

# + {"Collapsed": "false"}
sckt_test_features[0].unsqueeze(0).shape

# + {"Collapsed": "false"}

