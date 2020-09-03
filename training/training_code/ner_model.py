#
# Copyright 2018-2020 IBM Corp. All Rights Reserved.
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

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_text as text


def get_vocab(vocab, num_oov_buckets):
    if isinstance(vocab, str):
        return get_vocab_from_path(vocab, num_oov_buckets)
    elif isinstance(vocab, list):
        return get_vocab_from_list(vocab, num_oov_buckets)
    elif isinstance(vocab, dict):
        return get_vocab_from_dict(vocab, num_oov_buckets)


def get_vocab_from_dict(mapping, num_oov_buckets):
    return _get_vocab_from_kv(list(mapping.keys()), list(mapping.values()), num_oov_buckets)


def get_vocab_from_list(keys, num_oov_buckets):
    values = [i for i, _ in enumerate(keys)]
    return _get_vocab_from_kv(keys, values, num_oov_buckets)


def get_vocab_from_path(path, num_oov_buckets):
    initializer = tf.lookup.TextFileInitializer(path,
        tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
        tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER)
    return _get_vocab(initializer, num_oov_buckets)


def reverse_vocab_from_vocab(vocab, default_value='O'):
    exported_tensors = vocab.export()
    keys = exported_tensors[1]
    values = exported_tensors[0]
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys, values, key_dtype=keys.dtype, value_dtype=values.dtype)
    return tf.lookup.StaticHashTable(initializer, default_value)


def _get_vocab_from_kv(keys, values, num_oov_buckets):
    keys = tf.convert_to_tensor(keys, dtype=tf.string)
    values = tf.convert_to_tensor(values, dtype=tf.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys, values, key_dtype=tf.string, value_dtype=tf.int64)
    return _get_vocab(initializer, num_oov_buckets)


def _get_vocab(initializer, num_oov_buckets):
    if num_oov_buckets:
        return tf.lookup.StaticVocabularyTable(initializer, num_oov_buckets)
    else:
        return tf.lookup.StaticHashTable(initializer, -1)


def _get_reverse_vocab(path, default_value='O'):
    tfi = tf.lookup.TextFileInitializer(path,
        tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER,
        tf.string, tf.lookup.TextFileIndex.WHOLE_LINE)
    return tf.lookup.StaticHashTable(tfi, default_value)


def _build_model(num_words, num_chars, num_tags, params):
    glove_file = params['glove']
    # model params
    dropout = params['dropout']
    char_embedding_dim = params['dim_chars']
    word_embedding_dim = params['dim_words']
    lstm_size = params['lstm_size']
    char_lstm_size = params['char_lstm_size']

    # === Build character embedding branch ===
    char_ids = tf.keras.layers.Input((None, None), name='char_ids', dtype=tf.int64)
    word_lengths = tf.keras.layers.Input((None, ), name='word_lengths', dtype=tf.int64)

    char_embeddings = tf.keras.layers.Embedding(num_chars, char_embedding_dim, name='char_embeddings')(char_ids)
    char_embeddings = tf.keras.layers.Dropout(dropout)(char_embeddings)

    # reshape using dim of embeddings
    dim_words = tf.shape(input=char_embeddings)[1]
    dim_chars = tf.shape(input=char_embeddings)[2]
    flat_chars = tf.reshape(char_embeddings, [-1, dim_chars, char_embedding_dim])

    # mask the char embedding input
    chars_mask = tf.sequence_mask(word_lengths, dim_chars)
    flat_chars_mask = tf.reshape(chars_mask, [-1, dim_chars])

    # === Bi-LSTM over character embeddings ===
    char_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(char_lstm_size, return_sequences=False), name='bilstm_chars')(flat_chars, mask=flat_chars_mask)
    char_lstm_output = tf.reshape(char_lstm, [-1, dim_words, char_lstm_size * 2])

    # === Build word model branch ===
    word_ids = tf.keras.layers.Input((None, ), name='word_ids', dtype=tf.int64)
    sent_lengths = tf.keras.layers.Input((), name='sent_lengths', dtype=tf.int64)

    # mask word input
    words_mask = tf.sequence_mask(sent_lengths, dim_words)

    word_embedding_weights = np.load(glove_file)['embeddings']
    # add an extra vector for the out-of-vocab bucket
    word_embedding_weights = np.vstack([word_embedding_weights, [[0.] * word_embedding_dim]])
    # weight_init = tf.keras.initializers.Constant(word_embedding_weights)
    word_emb_layer = tf.keras.layers.Embedding(num_words, word_embedding_dim,
                                            #    embeddings_initializer=weight_init,
                                               trainable=False,
                                               name='word_embeddings')
    word_embeddings = word_emb_layer(word_ids)                                           
    word_emb_layer.set_weights([word_embedding_weights])

    embeddings = tf.keras.layers.Concatenate(axis=-1, name='concat_embeddings')([word_embeddings, char_lstm_output])
    embeddings = tf.keras.layers.Dropout(dropout)(embeddings)

    # Bi-LSTM on concatenated embeddings
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, return_sequences=True), name='bilstm')(embeddings, mask=words_mask)
    output = tf.keras.layers.Dropout(dropout)(lstm)
    # bi-lstm logits for CRF layer
    logits = tf.keras.layers.Dense(num_tags, name='logits')(output)
    # confidence prediction branch logits
    conf_logits = tf.keras.layers.Dense(num_tags, name='confidence_logits')(output)

    model = tf.keras.Model([word_ids, sent_lengths, char_ids, word_lengths], [logits, conf_logits])
    return model


class CRFLayer(tf.keras.layers.Layer):
    def __init__(self, dim, name='crf_layer'):
        super(CRFLayer, self).__init__(name=name)
        self.dim = dim
        initializer = tf.initializers.GlorotNormal()
        self.crf_params = self.add_weight(
            name='crf_params',
            shape=(self.dim, self.dim),
            initializer=initializer, 
            trainable=True)
            
    def call(self, x):
        logits = x[0]
        sent_lengths = x[1]
        pred_ids, _ = tfa.text.crf.crf_decode(logits, self.crf_params, sent_lengths)
        return pred_ids


class NERModel(tf.keras.Model):
    def __init__(self, params, vocab_words, vocab_chars, vocab_tags, name='ner_model'):
        super(NERModel, self).__init__(name=name)

        self.params = params.copy()
        # load vocab assets
        self.vocab_words = vocab_words
        self.vocab_chars = vocab_chars
        self.vocab_tags = vocab_tags
        self.reverse_vocab_tags = reverse_vocab_from_vocab(vocab_tags)

        # self.vocab_words, self.vocab_chars, self.vocab_tags, self.reverse_vocab_tags =\
            # self._load_vocabs(self.params)
        self.tokenizer = text.WhitespaceTokenizer()
        # config values
        self.num_words = self.vocab_words.size().numpy()
        self.num_chars = self.vocab_chars.size().numpy()
        self.num_tags = self.vocab_tags.size().numpy()
        self.batch_size = params.get('batch_size', 32)
        self.pad_value = params.get('pad_value', '<pad>')
        self.pad_tag = params.get('pad_tag', 'O')
        self.pad_len = params.get('pad_len', 0)
        self.lower_case = params['lower_case']

        # build bi-lstm base model
        self.base_model = _build_model(self.num_words, self.num_chars, self.num_tags, self.params)
        self.crf_layer = CRFLayer(self.num_tags)

        # initializer = tf.initializers.GlorotNormal()
        # self.crf_params = tf.Variable(initializer((self.num_tags, self.num_tags)), name='crf_params')

    @tf.function(input_signature=[tf.TensorSpec((None,), dtype=tf.string)])
    def serve_text_input(self, words):
        word_tokens = self.tokenizer.tokenize(words)
        return self.serve_token_input(word_tokens)

    @tf.function(input_signature=[tf.RaggedTensorSpec((None, None), dtype=tf.string)])
    def serve_token_input(self, tokens):
        input_tokens = tokens # copy for returning later
        char_tokens = tf.strings.unicode_split(tokens, 'UTF-8')

        if self.lower_case:
            tokens = tf.ragged.map_flat_values(tf.strings.lower, tokens)
        sent_lengths = tokens.row_lengths()
        word_ids = self.vocab_words.lookup(tokens.to_tensor(self.pad_value))
        word_lengths = char_tokens.row_lengths(axis=-1).to_tensor(self.pad_len)
        char_ids = self.vocab_chars.lookup(char_tokens.to_tensor(self.pad_value))

        features = {
            'word_ids': word_ids,
            'sent_lengths': sent_lengths,
            'char_ids': char_ids,
            'word_lengths': word_lengths
        }
        # predict
        _, pred_ids, pred_probs = self.predict_crf_conf_probs(features)
        
        mask = tf.sequence_mask(sent_lengths)
        pred_tags = self.reverse_vocab_tags.lookup(tf.cast(pred_ids, dtype=tf.int64))
        # return ragged tensors to take into account the input sentence lengths
        pred_ids = tf.ragged.boolean_mask(pred_ids, mask)
        pred_tags = tf.ragged.boolean_mask(pred_tags, mask)
        pred_probs = tf.ragged.boolean_mask(pred_probs, mask)
        return [pred_tags, pred_ids, pred_probs, input_tokens]

    @tf.function
    def call(self, x, training=None):
        return self.base_model(x, training)

    @tf.function
    def predict_conf(self, x, training):
        _, conf_logits = self.base_model(x, training)
        return conf_logits

    @tf.function
    def predict_crf(self, x, training):
        sent_lengths = x['sent_lengths']
        logits, _ = self.base_model(x, training)
        # pred_ids, _ = tfa.text.crf.crf_decode(logits, self.crf_params, sent_lengths)
        pred_ids = self.crf_layer([logits, sent_lengths])
        return logits, pred_ids

    @tf.function
    def predict_crf_conf_probs(self, x):
        sent_lengths = x['sent_lengths']
        logits, conf_logits = self(x)
        # pred_ids, _ = tfa.text.crf.crf_decode(logits, self.crf_params, sent_lengths)
        pred_ids = self.crf_layer([logits, sent_lengths])
        pred_probs = tf.keras.activations.softmax(conf_logits)
        return logits, pred_ids, pred_probs

    @tf.function
    def loss(self, y, logits, sent_lengths):
        # log_likelihood, _ = tfa.text.crf.crf_log_likelihood(logits, y, sent_lengths, self.crf_params)
        log_likelihood, _ = tfa.text.crf.crf_log_likelihood(logits, y, sent_lengths, self.crf_layer.crf_params)
        return tf.reduce_mean(input_tensor=-log_likelihood)

    @tf.function
    def conf_loss(self, y, logits, mask):
        ll = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
        return tf.reduce_mean(tf.boolean_mask(ll, mask))