# import sys
# print("Python version")
# print (sys.version)
# # python -c 'help("modules")'
# from subprocess import Popen               # Run shell commands
# print('Shell which commands:')
# Popen('which pip', shell=True).wait()
# Popen('which pip3', shell=True).wait()
# Popen('which python', shell=True).wait()
# import pkg_resources
# print('pip packages:')
# print([p.project_name for p in pkg_resources.working_set])
print('DEBUG: Running distribute_training.py')
from comet_ml import Experiment            # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import os                                  # os handles directory/workspace changes
print(f'DEBUG: Comet ML environment variable:\n{os.environ["COMET_DISABLE_AUTO_LOGGING"]}')
import ray                                 # Distributed computing
from ray.util.sgd import TorchTrainer      # Distributed training package
import argparse                            # Read terminal arguments
import sys
sys.path.append('/home/ubuntu/')           # Add the folder on which local files are uploaded
import utils                               # RaySGD custom constructors and data pipelines
import data_utils as du                    # Data science and machine learning relevant methods
import yaml                                # Save and load YAML files
import sys

# Parse the terminal arguments
parser = argparse.ArgumentParser()
parser.add_argument('--comet_ml_api_key', type=str, help='Comet.ml API key')
parser.add_argument('--comet_ml_workspace', type=str, help='Comet.ml workspace where data the experiment will be uploaded')
parser.add_argument('--comet_ml_project_name', type=str, help='Comet.ml project name where data the experiment will be uploaded')
parser.add_argument('--comet_ml_save_model', type=bool, help='Boolean to decide whether the trained models are uploaded to Comet.ml')
parser.add_argument('--config_file_name', type=str, help='Name of the configuration file.')
args = parser.parse_args()
# Load the configuration dictionary
stream_config = open(args.config_file_name, 'r')
config = yaml.load(stream_config, Loader=yaml.FullLoader)
print(f'DEBUG: config={config}')
# Set the random seed for reproducibility
du.set_random_seed(config.get('random_seed', 42))
# Add the Comet ML credentials to the configuration dictionary
config['comet_ml_api_key'] = args.comet_ml_api_key
config['comet_ml_workspace'] = args.comet_ml_workspace
config['comet_ml_project_name'] = args.comet_ml_project_name
config['comet_ml_save_model'] = args.comet_ml_save_model
# Make sure that all None configuration are correctly formated as None, not a string
for key, val in config.items():
    if str(val).lower() == 'none':
        config[key] = None
# Start ray
# ray.init(address='auto', resources=dict(CPU=120, GPU=120))
ray.init(address='auto')
print('DEBUG: Started Ray.')
# NOTE: These could actually just be the current VM's resources. If it's the head node,
# we might need some extra resources just to add new nodes.
print(f'DEBUG: The cluster\'s total resources: \n{ray.cluster_resources()}')
print(f'DEBUG: The cluster\'s currently available resources: \n{ray.available_resources()}')
# Create the trainer
trainer = TorchTrainer(
        model_creator=utils.eICU_model_creator,
        data_creator=utils.eICU_data_creator,
        optimizer_creator=utils.eICU_optimizer_creator,
        training_operator_cls=utils.eICU_Operator,
        num_workers=config.get('num_workers', 1),
        config=config,
        use_gpu=True,
        use_fp16=config.get('use_fp16', False),
        use_tqdm=True)
print(f'DEBUG: Created the TorchTrainer object.')
# Train the model
for epoch in du.utils.iterations_loop(range(config.get('n_epochs', 1)), see_progress=config.get('see_progress', True), desc='Epochs'):
    stats = trainer.train(info=dict(epoch_idx=epoch))
