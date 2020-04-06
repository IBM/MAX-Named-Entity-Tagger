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

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Embedding, Dropout, \
    Bidirectional, LSTM, Lambda, Input, Activation, Masking
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint
from f1 import F1Score
import sys
from data_utils import load_vocab, write_vocab, get_processing_word, \
    CoNLLDataset, minibatches, pad_sequences
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Train a Named Entity Recognition model in Keras')
parser.add_argument('--data_path', required=True, help='path to training data')
parser.add_argument('--model_path', required=True, help='path where model will be saved')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs to train')

args = parser.parse_args()

data_dir = args.data_path.rstrip('/')
result_dir = args.model_path.rstrip('/')

train_filename = "{}/data/train.txt".format(data_dir)
valid_filename = "{}/data/valid.txt".format(data_dir)

use_chars = True
max_iter = None

print('Loading vocab files and word vectors from {}'.format(data_dir))
vocab_tags = load_vocab("{}/assets/tags.txt".format(data_dir))
vocab_chars = load_vocab("{}/assets/chars.txt".format(data_dir))
vocab_words = load_vocab("{}/assets/words.txt".format(data_dir))

n_words = len(vocab_words)
n_char = len(vocab_chars)
n_tags = len(vocab_tags)
pad_tag = n_tags
n_labels = n_tags + 1

# coNLL data for train
train = CoNLLDataset(train_filename, get_processing_word(vocab_words, vocab_chars, lowercase=True, chars=use_chars),
                     get_processing_word(vocab_tags, lowercase=False, allow_unk=False), max_iter)
# coNLL data for validation#coNLL
valid = CoNLLDataset(valid_filename, get_processing_word(vocab_words, vocab_chars, lowercase=True, chars=use_chars),
                     get_processing_word(vocab_tags, lowercase=False, allow_unk=False), max_iter)

emb_data = np.load("{}/assets/glove.6B.300d.trimmed.npz".format(data_dir))
embeddings = emb_data["embeddings"]

# Hyperparameters
dim_word = 300
dim_char = 100
hidden_size_char = 100  # lstm on chars
hidden_size_lstm = 300  # lstm on word embeddings
nepochs = args.epochs
lr = 0.0105
lr_decay = 0.0005
batch_size = 10
dropout = 0.5

# Process training dataset
print('Creating training dataset...')
words, labels = list(minibatches(train, len(train)))[0]  # NOTE: len(train) will return entire dataset!
char_ids, word_ids = zip(*words)

word_ids, sequence_lengths = pad_sequences(word_ids, pad_tok=pad_tag)
char_ids, word_lengths = pad_sequences(char_ids, pad_tok=pad_tag, nlevels=2)
labels, _ = pad_sequences(labels, pad_tok=pad_tag)


# Convert word and char ids to np arrays; one-hot encode labels
char_ids_arr = np.array(char_ids)
word_ids_arr = np.array(word_ids)
labels_arr = np.array(labels)
labels_arr_one_hot = np.eye(n_labels)[labels_arr]

# Process validation dataset
print('Creating validation dataset...')
words_valid, labels_valid = list(minibatches(valid, len(valid)))[0]
char_ids_valid, word_ids_valid = zip(*words_valid)
word_ids_valid, sequence_lengths_valid = pad_sequences(word_ids_valid, pad_tok=pad_tag)
char_ids_valid, word_lengths_valid = pad_sequences(char_ids_valid, pad_tok=pad_tag, nlevels=2)
labels_valid, _ = pad_sequences(labels_valid, pad_tok=pad_tag)

# Convert word and char ids to np arrays; one-hot encode labels
char_ids_arr_valid = np.array(char_ids_valid)
word_ids_arr_valid = np.array(word_ids_valid)
labels_arr_valid = np.array(labels_valid)
labels_arr_one_hot_valid = np.eye(n_labels)[labels_arr_valid]



# === Model code ===
def _build_embeddings(weights, use_bidirectional=False):
    # The first 'branch' of the model embeds words.
    word_emb_input = Input((None, ), name='word_input')
    mask_word = Masking(mask_value=pad_tag)(word_emb_input)
    word_emb_output = Embedding(n_words, dim_word, weights=[weights], trainable=False)(mask_word)

    # The second 'branch' of the model embeds characters.
    # Note: end to end paper claims to have applied dropout layer on character embeddings before inputting
    # to a CNN in addition to before both layers of BLSTM
    char_emb_input = Input((None, None), name='char_input')
    # Reshape: Input is sentences, words, characters. For characters, we want to just operate it over the character sentence by
    # number of words and seq of characters so we reshape so that we have words by characters
    char_emb_output = Lambda(lambda x: tf.keras.backend.reshape(x, (-1, tf.keras.backend.shape(x)[-1])))(char_emb_input)
    mask_char = Masking(mask_value=pad_tag)(char_emb_output)
    char_emb_output = Embedding(n_char, dim_char)(mask_char)
    char_emb_output = Dropout(dropout)(char_emb_output)

    # construct LSTM layers. Option to use 1 Bidirectonal layer, or one forward and one backward LSTM layer.
    # Empirical results appear better with 2 LSTM layers hence it is the default.
    if use_bidirectional:
        char_emb_output = Bidirectional(LSTM(hidden_size_char, return_sequences=False))(char_emb_output)
    else:
        fw_LSTM = LSTM(hidden_size_char, return_sequences=False)(char_emb_output)
        bw_LSTM = LSTM(hidden_size_char, return_sequences=False, go_backwards=True)(char_emb_output)
        char_emb_output = concatenate([fw_LSTM, bw_LSTM])
    # Use dropout to prevent overfitting (as a regularizer)
    char_emb_output = Dropout(dropout)(char_emb_output)
    # Reshape back
    char_emb_output = Lambda(lambda x, z: tf.keras.backend.reshape(x, (-1, tf.shape(z)[1], 2 * hidden_size_char)),
                             arguments={"z": word_emb_input})(char_emb_output)
    return word_emb_input, word_emb_output, char_emb_input, char_emb_output


def _build_model(embedding_weights, char_bidirectional=False, concat_bidirectional=True):
    word_emb_input, word_emb_output, char_emb_input, char_emb_output = _build_embeddings(embedding_weights, char_bidirectional)
    # concatenate word embedding and character embedding
    x = concatenate([word_emb_output, char_emb_output])
    x = Dropout(dropout)(x)
    # construct LSTM layers. Option to use 1 Bidirectonal layer, or one forward and one backward LSTM layer.
    # Empirical results appear better with bidirectional LSTM here, hence it is the default.
    if concat_bidirectional:
        x = Bidirectional(LSTM(hidden_size_lstm, return_sequences=True))(x)
    else:
        fw_LSTM_2 = LSTM(hidden_size_lstm, return_sequences=True)(x)
        bw_LSTM_2 = LSTM(hidden_size_lstm, return_sequences=True, go_backwards=True)(fw_LSTM_2)
        x = concatenate([fw_LSTM_2, bw_LSTM_2])

    x = Dropout(dropout)(x)
    scores = Dense(n_labels)(x)
    # Activation Function
    x = Activation("softmax", name='predict_output')(scores)
    # create model
    model = Model([word_emb_input, char_emb_input], x)
    return model


# === Build model ===

num_classes = len(labels_arr_one_hot)
f1 = F1Score(num_classes,average='micro')

model = _build_model(embeddings)
# Optimizer: Adam shows best results
adam_op = Adam(lr=lr, decay=lr_decay)
model.compile(optimizer=adam_op, loss='categorical_crossentropy', metrics=['accuracy', f1])


# Remove training output folder before saving the model files of the current run
shutil.rmtree('training_output', ignore_errors=True)

model_filename = "Epoch-{epoch:02d}"
checkpoint_path = os.path.join('training_output/checkpoints/', model_filename)


callbacks = ModelCheckpoint(filepath=checkpoint_path,
                            period=1,
                            verbose=1,
                            save_weights_only=True,
                           )

'''
# Load latest checkpoints and restart training
latest = tf.train.latest_checkpoint('training_output/checkpoints/')
print('The latest checkpoint is:', latest)
model.load_weights(latest)
'''

# train model
print('Beginning model fitting...')
model.fit([word_ids_arr, char_ids_arr], labels_arr_one_hot, batch_size=batch_size, callbacks=[callbacks], epochs=nepochs,
          validation_data=([word_ids_arr_valid, char_ids_arr_valid], labels_arr_one_hot_valid))

# Export keras model to TF SavedModel format
print('Exporting SavedModel to {}'.format(result_dir))
model.trainable = False
with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        result_dir,
        inputs={t.name: t for t in model.inputs},
        outputs={t.name: t for t in model.outputs})

# export vocabs
print('Writing vocab files to {}'.format(result_dir))
write_vocab(vocab_words, '{}/words.txt'.format(result_dir))
write_vocab(vocab_chars, '{}/chars.txt'.format(result_dir))
write_vocab(vocab_tags, '{}/tags.txt'.format(result_dir))

print('Completed training!')
