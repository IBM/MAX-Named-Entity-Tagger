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
from tensorflow.python.saved_model import tag_constants

import logging
from core.utils import get_processing_word, load_vocab, pad_sequences
from config import DEFAULT_MODEL_PATH, MODEL_META_DATA as model_meta

logger = logging.getLogger()


class ModelWrapper(MAXModelWrapper):

    MODEL_META_DATA = model_meta

    """Model wrapper for TensorFlow models in SavedModel format"""
    def __init__(self, path=DEFAULT_MODEL_PATH):
        logger.info('Loading model from: {}...'.format(path))

        # load assets first to enable model definition
        self._load_assets(path)

        # Loading the tf SavedModel
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        tf.saved_model.loader.load(self.sess, [tag_constants.SERVING], DEFAULT_MODEL_PATH)

        self.word_ids_tensor = self.sess.graph.get_tensor_by_name('word_input:0')
        self.char_ids_tensor = self.sess.graph.get_tensor_by_name('char_input:0')
        self.output_tensor = self.sess.graph.get_tensor_by_name('predict_output/truediv:0')
        self.pat = re.compile(r'(\W+)')

    def _load_assets(self, path):
        vocab_tags = load_vocab(path + "/tags.txt")
        vocab_chars = load_vocab(path + "/chars.txt")
        vocab_words = load_vocab(path + "/words.txt")

        self.proc_fn = get_processing_word(vocab_words, vocab_chars, lowercase=True, chars=True)

        self.id_to_tag = {idx: v for v, idx in vocab_tags.items()}
        self.dict_vocal_tags_len = len(vocab_tags)
        self.id_to_tag.update({self.dict_vocal_tags_len: 'O'})
        print('id to tag', self.id_to_tag)
        self.n_words = len(vocab_words)
        self.n_char = len(vocab_chars)
        n_tags = len(vocab_tags)
        self.pad_tag = n_tags
        self.n_labels = n_tags + 1

    def batch_inference_fn(self, x):

        char_ids_list = []
        word_ids_list = []
        words_raw_list = []
        for inp in x:
            words_raw = re.split(self.pat, inp)
            words_raw = [w.strip() for w in words_raw]  # strip whitespace
            words_raw = [w for w in words_raw if w]  # keep only non-empty terms, keeping raw punctuation
            words = [self.proc_fn(w) for w in words_raw]
            char_ids, word_ids = zip(*words)
            char_ids_list.append(char_ids)
            word_ids_list.append(word_ids)
            words_raw_list.append(words_raw)
        return words_raw_list, char_ids_list, word_ids_list

    def _pre_process(self, x):

        # Perform batch inference
        words_raw_list, char_ids_list, word_ids_list = self.batch_inference_fn(x)
        word_ids, _ = pad_sequences(word_ids_list, pad_tok=self.pad_tag)
        char_ids, _ = pad_sequences(char_ids_list, pad_tok=self.pad_tag, nlevels=2)
        word_ids_arr = np.array(word_ids)
        char_ids_arr = np.array(char_ids)
        return words_raw_list, word_ids_arr, char_ids_arr

    def _post_process(self, x):
        final_result = []
        for i in x:
            final_result.append([self.id_to_tag[j] for j in
                                 i.ravel() if j != self.dict_vocal_tags_len])
        return final_result

    def _predict(self, word_ids_arr, char_ids_arr):
        pred = self.sess.run(self.output_tensor, feed_dict={
            self.word_ids_tensor: word_ids_arr,
            self.char_ids_tensor: char_ids_arr
        })
        return np.argmax(pred, -1)

    def predict(self, x):
        words, word_ids_arr, char_ids_arr = self._pre_process(x)
        labels_pred_arr = self._predict(word_ids_arr, char_ids_arr)
        labels_pred = self._post_process(labels_pred_arr)
        return labels_pred, words
