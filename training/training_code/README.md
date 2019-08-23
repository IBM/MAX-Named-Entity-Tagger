## Overview

The content of this directory is uploaded to a Watson Machine Learning service instance in the IBM Cloud.

Directory content:

- `train-max-model.sh`: Main entry point. WML executes this script to train the model.
- `training_requirements.txt`: Defines that packages that will be installed before training is started.
- `data_prep.py`: Prepares dataset for training
- `data_utils.py`: Utilities for processing input data and handling vocabulary and word vector files
- `train_ner.py`: Model training script
