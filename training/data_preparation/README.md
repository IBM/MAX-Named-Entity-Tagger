## How to prepare your data for training

Follow the instructions in this document to prepare your data for model training.

- [Prerequisites](#prerequisites)
- [Organize data directory](#organize-data-directory)
- [Preparing your data](#preparing-your-data)

## Prerequisites

Training the Named Entity Tagger model requires:

- two input datasets - `train.txt` for training and `valid.txt` for validation, in [IOB format](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)). Look at the [sample data](../sample_training_data/data/train.txt) for an example.
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

The `data_prep.py` script in the `training_code` directory will read in the training and validation datasets, together with the GloVe vectors, and create vocabulary files for words, characters and entity tags for model training. It will also write out the required processed GloVe vectors for the `words` vocabulary file (trimming the vectors down to only the vocabulary present in your training data).

From the `training` base folder in the model repository, run the following command. This assumes the directory structure above.

```
python training_code/data_prep.py --data_path data_directory/data/ --output_path data_directory/assets/ --glove_path GLOVE_DIR
```

This requires two data files present in `data_directory/data/`, named `train.txt` and `valid.txt`.

Once completed, this will generate the following files in the `assets` subfolder, ready for executing training:

- `chars.txt` - vocabulary of characters in your data files
- `tags.txt` - vocabulary of entity tags in your data files
- `words.txt` - vocabulary of words in your data files
- `glove.6B.300d.trimmed.npz` - pre-trained GloVe weight vectors corresponding to `words.txt` vocabulary

Proceed to [training your model](../README.md#train-the-model), noting to set the training data directory to the same `data_directory` used here to prepare the data for training.
