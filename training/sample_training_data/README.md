# Sample training data

This directory contains two small sample training files to quickly test out model training: `train.txt` and `valid.txt`. For the purposes of testing, the contents of both files are identical.

The sample file is taken from the [TensorFlow sequence tagging repo](https://github.com/guillaumegenthial/sequence_tagging/blob/master/data/test.txt) under the [LICENSE](https://github.com/guillaumegenthial/sequence_tagging/blob/master/LICENSE.txt) of the repo - [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt).

Also provided are the ancillary assets required for training:

* `chars.txt` - vocabulary of characters in sample data files
* `tags.txt` - vocabulary of entity tags present in sample data files
* `words.txt` - vocabulary of words in sample data files
* `glove.6B.300d.trimmed.npz` - pre-trained GloVe weight vectors corresponding to `words.txt` vocabulary

These ancillary assets are generated from your training data files during [data preparation](../data_preparation/README.md).
