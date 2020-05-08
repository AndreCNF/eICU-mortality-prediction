from comet_ml import Experiment            # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import ray                                 # Distributed computing
from ray.util.sgd import TorchTrainer      # Distributed training package
import argparse                            # Read terminal arguments
import utils                               # RaySGD custom constructors and data pipelines
import data_utils as du                    # Data science and machine learning relevant methods
import yaml                                # Save and load YAML files

# Set the random seed for reproducibility
du.set_random_seed(42)
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
ray.init(address='auto')
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
# Train the model
for epoch in du.utils.iterations_loop(range(config.get('n_epochs', 1)), see_progress=config.get('see_progress', True), desc='Epochs'):
    stats = trainer.train(info=dict(epoch_idx=epoch))
