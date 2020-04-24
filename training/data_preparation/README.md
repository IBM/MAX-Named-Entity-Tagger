## How to prepare your data for training

Follow the instructions in this document to prepare your data for model training.

- [Prerequisites](#prerequisites)
- [Organize data directory](#organize-data-directory)
- [Preparing your data](#preparing-your-data)

## Prerequisites

Training the Named Entity Tagger model requires:

- three input datasets - for training, validation and testing. The data input format is (example for training dataset shown):
  - `train.words.txt`: a file containing the input sentences
  - `train.tags.txt`: a file containing the input labels (entity tags) for each sentence
  
  Each of these files contains one sentence per line. Look at the [sample data](../sample_training_data/data/train.txt) for an example.
  
  The data can also be provided in [IOB format](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)). 
  In this case, use the `process_iob.py` script to convert IOB data to the format above.
  
  ```
  python training_code/process_iob.py --data_path data_directory/path_to_iob_data --output_path data_directory/data
  ```

  
- [GloVe word embedding vectors](https://nlp.stanford.edu/projects/glove/)

### Download GloVe vectors

You can download the GloVe 6B (Wikipedia 2014 + Gigaword 5) pre-trained word vectors from [this link](http://nlp.stanford.edu/data/glove.6B.zip). Unzip the file and take note of the location - referred to as `GLOVE_DIR` in these instructions.

## Organize data directory

Trainable models should adhere to the standard directory structure below:

```
|-- data_directory
    |-- assets
    |-- data
    |-- initial_model
```

1. `assets` holds ancillary files required for training (typically these are generated during the data preparation phase).
2. `data` folder holds the data required for training.
3. `initial_model` folder holds the initial checkpoint files to initiate training.

If a particular directory is not required, it can be omitted.

## Preparing your data

The `data_prep.py` script in the `training_code` directory will read in the training dataset, together with the GloVe vectors, and create vocabulary files for words, characters and entity tags for model training. It will also write out the required processed GloVe vectors for the `words` vocabulary file (trimming the vectors down to only the vocabulary present in your training data).

From the `training` base folder in the model repository, run the following command. This assumes the directory structure above.

```
python training_code/data_prep.py --data_path data_directory/data/ --output_path data_directory/assets/ --glove_path GLOVE_DIR
```

This requires two data files present in `data_directory/data/`, named `train.words.txt` and `train.tags.txt`.

Once completed, this will generate the following files in the `assets` subfolder, ready for executing training:

- `vocab.chars.txt` - vocabulary of characters in the training dataset
- `vocab.tags.txt` - vocabulary of entity tags in the training dataset
- `vocab.words.txt` - vocabulary of words in the training dataset
- `glove.npz` - pre-trained GloVe weight vectors corresponding to `vocab.words.txt` vocabulary

**Note** here we generate the vocabularies only on the _training_ dataset. This gives a better representation of model performance on new, unseen data (when evaluating on the validation and test datasets), since in real deployments the model would not be able to access future data (that was not in the training data) for building the vocabulary.


Proceed to [training your model](../README.md#train-the-model), noting to set the training data directory to the same `data_directory` used here to prepare the data for training.
