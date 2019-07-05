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

from maxfw.model import MAXModelWrapper

import numpy as np
import re
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Embedding, Dropout, \
    Bidirectional, LSTM, Lambda, Input, Activation, Masking
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.models import Model
import logging
from core.utils import get_processing_word, load_vocab, pad_sequences
from config import DEFAULT_MODEL_PATH, MODEL_ID, MODEL_META_DATA as model_meta

logger = logging.getLogger()


class ModelWrapper(MAXModelWrapper):

    MODEL_META_DATA = model_meta

    pat = re.compile(r'(\W+)')
    dim_word = 300              # word embeddings dim
    dim_char = 100              # char embeddings dim
    hidden_size_lstm = 300      # lstm on combined word & char embeddings
    hidden_size_char = 100      # lstm on chars
    dropout = 0.5               # not used for inference but required to build network layers

    """Model wrapper for TensorFlow models in SavedModel format"""
    def __init__(self, path=DEFAULT_MODEL_PATH, model=MODEL_ID):
        logger.info('Loading model from: {}...'.format(path))

        # load assets first to enable model definition
        self._load_assets(path)

        # create model and load weights
        weights_path = "{}/{}.h5".format(path, model)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model = self._create_model()
            self.model.load_weights(weights_path)
            logger.info('Loaded model')

    def _load_assets(self, path):
        vocab_tags = load_vocab(path + "/tags.txt")
        vocab_chars = load_vocab(path + "/chars.txt")
        vocab_words = load_vocab(path + "/words.txt")

        self.proc_fn = get_processing_word(vocab_words, vocab_chars, lowercase=True, chars=True)

        self.id_to_tag = {idx: v for v, idx in vocab_tags.items()}
        self.n_words = len(vocab_words)
        self.n_char = len(vocab_chars)
        n_tags = len(vocab_tags)
        self.pad_tag = n_tags
        self.n_labels = n_tags + 1

    def _create_model(self):
        # create word-level layers
        word_emb_input = Input((None,))
        mask_word = Masking(mask_value=self.pad_tag)(word_emb_input)
        word_emb_output = Embedding(self.n_words, self.dim_word, trainable=False)(mask_word)

        # create character-level layers
        char_emb_input = Input((None, None))
        char_emb_output = Lambda(lambda x: tf.keras.backend.reshape(x, (-1, tf.keras.backend.shape(x)[-1])))(char_emb_input)
        mask_char = Masking(mask_value=self.pad_tag)(char_emb_output)
        char_emb_output = Embedding(self.n_char, self.dim_char)(mask_char)
        char_emb_output = Dropout(self.dropout)(char_emb_output)
        fw_LSTM = LSTM(self.hidden_size_char, return_sequences=False)(char_emb_output)
        bw_LSTM = LSTM(self.hidden_size_char, return_sequences=False, go_backwards=True)(char_emb_output)
        char_emb_output = concatenate([fw_LSTM, bw_LSTM])
        char_emb_output = Dropout(self.dropout)(char_emb_output)
        char_emb_output = Lambda(lambda x, z: tf.keras.backend.reshape(x, (-1, tf.shape(z)[1], 2 * self.hidden_size_char)),
                                 arguments={"z": word_emb_input})(char_emb_output)

        # concatenates word layers and character layers
        x = concatenate([word_emb_output, char_emb_output])
        x = Dropout(self.dropout)(x)
        x = Bidirectional(LSTM(self.hidden_size_lstm, return_sequences=True))(x)
        x = Dropout(self.dropout)(x)
        scores = Dense(self.n_labels)(x)
        softmax = Activation("softmax")(scores)
        model = Model([word_emb_input, char_emb_input], softmax)
        return model

    def _pre_process(self, x):
        words_raw = re.split(self.pat, x)
        words_raw = [w.strip() for w in words_raw]      # strip whitespace
        words_raw = [w for w in words_raw if w]         # keep only non-empty terms, keeping raw punctuation
        words = [self.proc_fn(w) for w in words_raw]
        char_ids, word_ids = zip(*words)
        word_ids, _ = pad_sequences([word_ids], pad_tok=self.pad_tag)
        char_ids, _ = pad_sequences([char_ids], pad_tok=self.pad_tag, nlevels=2)
        word_ids_arr = np.array(word_ids)
        char_ids_arr = np.array(char_ids)
        return words_raw, word_ids_arr, char_ids_arr

    def _post_process(self, x):
        return [self.id_to_tag[i] for i in x.ravel()]

    def _predict(self, word_ids_arr, char_ids_arr):
        with self.graph.as_default():
            pred = self.model.predict([word_ids_arr, char_ids_arr], 1)
        return np.argmax(pred, -1)

    def predict(self, x):
        words, word_ids_arr, char_ids_arr = self._pre_process(x)
        labels_pred_arr = self._predict(word_ids_arr, char_ids_arr)
        labels_pred = self._post_process(labels_pred_arr)
        return labels_pred, words
