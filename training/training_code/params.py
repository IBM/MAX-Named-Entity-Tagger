#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

_params = {                 
    'dim_words': 100,                 # word embedding dim
    'dim_chars': 32,                 # char embedding dim
    'dropout': 0.15,                   # dropout rate
    'num_oov_buckets': 1,             # out-of-vocab buckets for lookup tables
    'epochs': 10,                     # number of epochs to train
    'checkpoint_interval': 5,         # checkpoint interval in epochs
    'batch_size': 32,                 # batch size for training
    'buffer': 15000,                  # buffer for training dataset shuffle 
    'char_lstm_size': 64,             # hidden size for char bi-lstm layer
    'lstm_size': 128,                 # hidden size for concatenated bi-lstm layer
    'words': 'vocab.words.txt',       # filename for words vocab
    'chars': 'vocab.chars.txt',       # filename for chars vocab
    'tags': 'vocab.tags.txt',         # filename for tags vocab
    'glove': 'glove.npz',             # filename for glove embeddings
    'glove_cased': False,             # whether glove embeddings are cased. If False then lowercase word pre-processing is applied
    'export_dir': 'saved_model',      # result subfolder for exporting SavedModel after training
    'scores_dir': 'scores'          # result subfolder for writing evaluation results
}

def get_params():
    return _params.copy()



'''

_params = {                 
    'dim_words': 100,                 # word embedding dim
    'dim_chars': 100,                 # char embedding dim
    'dropout': 0.5,                   # dropout rate
    'num_oov_buckets': 1,             # out-of-vocab buckets for lookup tables
    'epochs': 20,                     # number of epochs to train
    'checkpoint_interval': 5,         # checkpoint interval in epochs
    'batch_size': 32,                 # batch size for training
    'buffer': 15000,                  # buffer for training dataset shuffle 
    'char_lstm_size': 25,             # hidden size for char bi-lstm layer
    'lstm_size': 100,                 # hidden size for concatenated bi-lstm layer
    'words': 'vocab.words.txt',       # filename for words vocab
    'chars': 'vocab.chars.txt',       # filename for chars vocab
    'tags': 'vocab.tags.txt',         # filename for tags vocab
    'glove': 'glove.npz',             # filename for glove embeddings
    'glove_cased': False,             # whether glove embeddings are cased. If False then lowercase word pre-processing is applied
    'export_dir': 'saved_model',      # subfolder of output folder, for exporting SavedModel after training
    'scores_dir': 'scores'          # result subfolder for writing evaluation results
}
'''