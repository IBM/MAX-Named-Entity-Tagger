## Overview

The content of this directory is uploaded to a Watson Machine Learning service instance in the IBM Cloud.

Directory content:

- `train-max-model.sh`: Main entry point. WML executes this script to train the model.
- `training_requirements.txt`: Defines that packages that will be installed before training is started.
- `data_prep.py`: Generate vocabularies for model training
- `process_iob.py`: Utility for converting IOB format data
- `metrics.py`: Utilities for computing multi-class evaluation metrics
- `params.py`: Training configuration
- `train_ner.py`: Model training script
