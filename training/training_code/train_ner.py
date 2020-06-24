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

import json
import logging
from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_text as text

from metrics import compute_cm, metrics_from_confusion_matrix


def _map_tag_dataset(raw_tags, tokenizer, vocab_tags, pad_tag):
    tag_tokens = raw_tags.map(tokenizer.tokenize)
    return tag_tokens.map(lambda x: vocab_tags.lookup(x.to_tensor(pad_tag)))


def _map_word_dataset(raw_words, tokenizer, vocab_words, vocab_chars, pad_value, pad_len, lower_case):
    word_tokens = raw_words.map(tokenizer.tokenize)
    char_tokens = word_tokens.map(lambda w: tf.strings.unicode_split(w, 'UTF-8'))
    if lower_case:
        word_tokens = word_tokens.map(lambda x: tf.ragged.map_flat_values(tf.strings.lower, x))
    word_ids = word_tokens.map(lambda x: (vocab_words.lookup(x.to_tensor(pad_value)), x.row_lengths()))
    char_ids = char_tokens.map(lambda x: (vocab_chars.lookup(x.to_tensor(pad_value)), x.row_lengths(axis=-1).to_tensor(pad_len)))
    zipped = tf.data.Dataset.zip((word_ids, char_ids))
    dataset = zipped.map(lambda x, y:
        {'word_ids': x[0], 'sent_lengths': x[1], 'char_ids': y[0], 'word_lengths': y[1]})
    return dataset


def create_dataset(fwords, ftags, params, vocab_words, vocab_chars, vocab_tags, shuffle=False):
    tokenizer = text.WhitespaceTokenizer()

    batch_size = params.get('batch_size', 32)
    prefetch = params.get('prefetch', 2)
    buffer = params.get('buffer', 10000)
    pad_value = params.get('pad_value', '<pad>')
    pad_tag = params.get('pad_tag', 'O')
    pad_len = params.get('pad_len', 0)
    lower_case = params['lower_case']

    raw_words = tf.data.TextLineDataset(fwords).batch(batch_size)
    raw_tags = tf.data.TextLineDataset(ftags).batch(batch_size)

    # tag dataset
    tag_ids = _map_tag_dataset(raw_tags, tokenizer, vocab_tags, pad_tag)
    # word and char datasets
    word_char_data = _map_word_dataset(
        raw_words, tokenizer, vocab_words, vocab_chars, pad_value, pad_len, lower_case)

    dataset = tf.data.Dataset.zip((word_char_data, tag_ids))
    if shuffle:
        dataset = dataset.shuffle(buffer)
    return dataset.prefetch(prefetch)


def _get_vocab(path, num_oov_buckets):
    tfi = tf.lookup.TextFileInitializer(path,
        tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
        tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER)
    if num_oov_buckets:
        return tf.lookup.StaticVocabularyTable(tfi, num_oov_buckets)
    else:
        return tf.lookup.StaticHashTable(tfi, -1)

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
    weight_init = tf.keras.initializers.Constant(word_embedding_weights)
    word_embeddings = tf.keras.layers.Embedding(num_words, word_embedding_dim,
                                               embeddings_initializer=weight_init,
                                               trainable=False,
                                               name='word_embeddings')(word_ids)

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


def _get_ckpt_manager(params, ner_model, ckpt_dir):
    if params['learning_rate']:
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    else:
        # default learning rate
        optimizer = tf.keras.optimizers.Adam()
    # assume defaults for confidence branch
    conf_optimizer = tf.keras.optimizers.Adam()
    ckpt = (tf.train.Checkpoint(
        epochs_trained=tf.Variable(0),
        steps_per_epoch=tf.Variable(0),
        conf_epochs_trained=tf.Variable(0),
        # params=params,
        optimizer=optimizer,
        conf_optimizer=conf_optimizer,
        ner_model=ner_model))
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
    return ckpt, manager


def train_confidence_branch(ner_model, params, train_data, valid_data, ckpt_dir):
    x_sig = train_data.element_spec[0]
    y_sig = train_data.element_spec[1]

    # restore from checkpoint if exists
    ckpt, manager = _get_ckpt_manager(params, ner_model, ckpt_dir)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print('Restored from checkpoint: {}. Epochs trained for confidence branch: {}'.format(
            manager.latest_checkpoint, ckpt.conf_epochs_trained.numpy()))
    else:
        print('No checkpoint found in: {}. Initializing training from scratch.'.format(ckpt_dir))

    # === set up optimizer and metrics ===
    optimizer = ckpt.conf_optimizer
    softmax_metric = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)
    # only train confidence dense layer weights
    conf_weights = [w for w in ner_model.trainable_variables if 'confidence' in w.name]

    @tf.function(input_signature=[x_sig, y_sig])
    def conf_train_step(x, y):
        sent_lengths = x['sent_lengths']
        mask = tf.sequence_mask(sent_lengths)
        with tf.GradientTape() as tape:
            logits = ner_model.predict_conf(x, training=True)
            loss = ner_model.conf_loss(y, logits, mask)
        # confidence weights
        grads = tape.gradient(loss, conf_weights)
        optimizer.apply_gradients(zip(grads, conf_weights))
        return loss
    
    @tf.function(input_signature=[x_sig, y_sig])
    def conf_valid_step(x, y):
        sent_lengths = x['sent_lengths']
        mask = tf.sequence_mask(sent_lengths)
        logits = ner_model.predict_conf(x, training=False)
        # update metric
        softmax_metric(y, logits, sample_weight=mask)

    # === confidence branch training ===
    conf_epochs = params['conf_epochs']
    if conf_epochs:
        print('Beginning training confidence branch for {} epochs'.format(conf_epochs))
    else:
        print('Skipping confidence training (conf_epochs: {})'.format(conf_epochs))
    for epoch in range(1, conf_epochs + 1):
        print('\nepoch {}/{}'.format(epoch, conf_epochs))
        # init keras progress bar
        total_steps = ckpt.steps_per_epoch.numpy()
        total_steps = total_steps if total_steps > 0 else None
        progbar = tf.keras.utils.Progbar(total_steps, interval=0.5, stateful_metrics=['val_loss'])

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_data):
            conf_loss = conf_train_step(x_batch_train, y_batch_train)
            progbar.update(step, values=[('loss', conf_loss.numpy())])
        
        # evaluate on validation set
        for x_batch_val, y_batch_val in valid_data:
            conf_valid_step(x_batch_val, y_batch_val)
        progbar.update(step + 1, values=[('val_loss', softmax_metric.result())])
        softmax_metric.reset_states()
        ckpt.conf_epochs_trained.assign_add(1)

    # # save final checkpoint 
    save_path = manager.save()
    print('Saved final checkpoint (epochs trained: {}, conf epochs trained: {}): {}'.format(
        ckpt.epochs_trained.numpy(), ckpt.conf_epochs_trained.numpy(), save_path))
    return ner_model


def train(ner_model, params, train_data, valid_data, ckpt_dir):
    # extract signatures for training and validation functions
    x_sig = train_data.element_spec[0]
    y_sig = train_data.element_spec[1]

    # restore from checkpoint if exists
    ckpt, manager = _get_ckpt_manager(params, ner_model, ckpt_dir)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print('Restored from checkpoint: {}. Epochs trained: {}'.format(manager.latest_checkpoint, ckpt.epochs_trained.numpy()))
    else:
        print('No checkpoint found in: {}. Initializing training from scratch.'.format(ckpt_dir))

    # === set up optimizer and metrics ===
    optimizer = ckpt.optimizer
    acc = tf.keras.metrics.Accuracy()
    val_acc = tf.keras.metrics.Accuracy()
    # for f1 
    all_tags, all_tags_idx = ner_model.vocab_tags.export()
    indices = [i for i, t in zip(all_tags_idx.numpy(), all_tags.numpy()) if t != b'O']

    # === define functions for training and evaluation ===
    # only update main network weights
    network_weights = [w for w in ner_model.trainable_variables if 'confidence' not in w.name]

    @tf.function(input_signature=[x_sig, y_sig])
    def train_step(x, y):
        sent_lengths = x['sent_lengths']
        mask = tf.sequence_mask(sent_lengths)
        with tf.GradientTape() as tape:
            logits, pred_ids = ner_model.predict_crf(x, training=True)
            loss = ner_model.loss(y, logits, sent_lengths)

        # compute and apply grads to network weights
        grads = tape.gradient(loss, network_weights)
        optimizer.apply_gradients(zip(grads, network_weights))

        # Update training metrics
        acc(y, pred_ids, sample_weight=mask)
        return pred_ids, loss

    @tf.function(input_signature=[x_sig, y_sig])
    def valid_step(x, y):
        sent_lengths = x['sent_lengths']
        mask = tf.sequence_mask(sent_lengths)
        _, pred_ids = ner_model.predict_crf(x, training=False)
        # Update validation metrics
        val_acc(y, pred_ids, sample_weight=mask)
        return pred_ids, mask

    epochs = params['epochs']
    checkpoint_interval = params['checkpoint_interval']
    if epochs:
        print('Beginning Bi-LSTM training for {} epochs. Checkpointing interval {} epochs'.format(epochs, checkpoint_interval))
    else:
        print('Skipping Bi-LSTM training (epochs: {})'.format(epochs))
    for epoch in range(1, epochs + 1):
        print("\nepoch {}/{}".format(epoch, epochs))
        # init keras progress bar
        total_steps = ckpt.steps_per_epoch.numpy()
        total_steps = total_steps if total_steps > 0 else None
        progbar = tf.keras.utils.Progbar(total_steps, interval=0.5, stateful_metrics=['train_acc', 'val_acc', 'val_f1'])

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_data):
            _, loss = train_step(x_batch_train, y_batch_train)
            progbar.update(step, values=[('loss', loss.numpy())])
        
        # first epoch training, set the total steps (batches) per epoch in checkpoint and progbar
        if ckpt.steps_per_epoch.numpy() == 0:
            ckpt.steps_per_epoch.assign(step + 1)
        if progbar.target is None:
            progbar.target = step + 1

        # f1 computation 
        zeros = tf.initializers.Zeros()
        total_cm = tf.Variable(zeros((ner_model.num_tags, ner_model.num_tags), dtype=tf.float64))

        # evaluate on validation set
        for x_batch_val, y_batch_val in valid_data:
            pred_ids, mask = valid_step(x_batch_val, y_batch_val)
            cm = compute_cm(y_batch_val, pred_ids, ner_model.num_tags, mask)
            total_cm.assign_add(cm)

        pr, rec, f1 = metrics_from_confusion_matrix(total_cm, indices)
        metric_values = [('train_acc', acc.result()), ('val_acc', val_acc.result()), ('val_f1', f1)]
        progbar.update(step + 1, values=metric_values)
        # Reset metrics at the end of each epoch
        acc.reset_states()
        val_acc.reset_states()

        ckpt.epochs_trained.assign_add(1)
        if ckpt.epochs_trained.numpy() % checkpoint_interval == 0 or epoch == epochs:
            save_path = manager.save()
            print('Saved checkpoint for epoch {} (total epochs trained: {}): {}'.format(epoch, ckpt.epochs_trained.numpy(), save_path))

    return ner_model, manager


def generate_mc_metrics(params, ner_model, dataset):
    @tf.function(input_signature=[dataset.element_spec[0]])
    def test_predict_step(x):
        _, pred_ids = ner_model.predict_crf(x, training=False)
        return pred_ids

    def split_span(s):
        for match in re.finditer(r"\S+", s):
            span = match.span()
            yield match.group(0), span[0], span[1] - 1

    def drop_iob_tags(df):
        new = df['entity'].str.split("-", n = 1, expand = True)
        df.drop(columns =['entity'], inplace = True)
        df['entity']= new[1]
        return df

    sent_id = 1
    golds = []
    preds = []

    reverse_words = _get_reverse_vocab(params['words'], default_value='UNK')
    reverse_tags = _get_reverse_vocab(params['tags'], default_value='O')
    
    for x, y in dataset:
        sent_lengths = x['sent_lengths']
        pred_ids = test_predict_step(x)
        word_batch = reverse_words.lookup(x['word_ids'])
        pred_tag_batch = reverse_tags.lookup(tf.cast(pred_ids, tf.int64))
        gold_tag_batch = reverse_tags.lookup(tf.cast(y, tf.int64))
        for words, sent_len, pred_tags, gold_tags in zip(word_batch, sent_lengths, pred_tag_batch, gold_tag_batch):
            words = words[:sent_len]
            pred_tags = pred_tags[:sent_len]
            gold_tags = gold_tags[:sent_len]
            sent = ' '.join([w.decode() for w in words.numpy()])
            word_spans = [s for s in split_span(sent)]
            pred_tags = [t.decode() for t in pred_tags.numpy()]
            gold_tags = [t.decode() for t in gold_tags.numpy()]
            for w, pt, gt in zip(word_spans, pred_tags, gold_tags):
                if pt != 'O':
                    preds.append({'sentence_id': sent_id,
                        'start_char_offset': w[-2],
                        'end_char_offset': w[-1],
                        'entity': pt})
                if gt != 'O':
                    golds.append({'sentence_id': sent_id,
                        'start_char_offset': w[-2],
                        'end_char_offset': w[-1],
                        'entity': gt if gt != 'UNK' else 'O'})
            sent_id += 1
        
    golds_df = pd.DataFrame(golds, columns=['sentence_id', 'start_char_offset', 'end_char_offset', 'entity'])
    preds_df = pd.DataFrame(preds, columns=['sentence_id', 'start_char_offset', 'end_char_offset', 'entity'])
        
    # Drop IOB Tags and consider only the entity
    golds_df = drop_iob_tags(golds_df)
    preds_df = drop_iob_tags(preds_df)
    ents = golds_df.entity.unique()
    scores = []

    merged = pd.merge(golds_df, preds_df, how='inner')
    support = len(merged)
    p = support / len(preds_df)
    r = support / len(golds_df)
    f1 = (2 * p * r) / (p + r)
    scores.append({'entity': '_OVERALL_', 'f1': f1, 'precision': p, 'recall': r, 'support': support})

    for e in ents:
        pf = preds_df.loc[preds_df['entity'] == e]
        gf = golds_df.loc[golds_df['entity'] == e]
        merged = pd.merge(gf, pf, how='inner')
        support = len(merged)
        p = support / len(pf) if len(pf) > 0 else 0
        r = support / len(gf) if len(gf) > 0 else 0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
        scores.append({'entity': e, 'f1': f1, 'precision': p, 'recall': r, 'support': support})

    return pd.DataFrame(scores)


class NERModel(tf.keras.Model):
    def __init__(self, params, name='ner_model'):
        super(NERModel, self).__init__(name=name)

        self.params = params.copy()
        # load vocab assets
        self.vocab_words, self.vocab_chars, self.vocab_tags, self.reverse_vocab_tags =\
            self._load_vocabs(self.params)
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

        # crf model param
        initializer = tf.initializers.GlorotNormal()
        self.crf_params = tf.Variable(initializer((self.num_tags, self.num_tags)), name='crf_params')

    @staticmethod
    def _load_vocabs(params):
        vocab_words = _get_vocab(params['words'], params['num_oov_buckets'])
        vocab_chars = _get_vocab(params['chars'], params['num_oov_buckets'])
        vocab_tags = _get_vocab(params['tags'], 0)
        reverse_vocab_tags = _get_reverse_vocab(params['tags'])
        return vocab_words, vocab_chars, vocab_tags, reverse_vocab_tags

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
        pred_ids, _ = tfa.text.crf.crf_decode(logits, self.crf_params, sent_lengths)
        return logits, pred_ids

    @tf.function
    def predict_crf_conf_probs(self, x):
        sent_lengths = x['sent_lengths']
        logits, conf_logits = self(x)
        pred_ids, _ = tfa.text.crf.crf_decode(logits, self.crf_params, sent_lengths)
        pred_probs = tf.keras.activations.softmax(conf_logits)
        return logits, pred_ids, pred_probs

    @tf.function
    def loss(self, y, logits, sent_lengths):
        log_likelihood, _ = tfa.text.crf.crf_log_likelihood(logits, y, sent_lengths, self.crf_params)
        return tf.reduce_mean(input_tensor=-log_likelihood)

    @tf.function
    def conf_loss(self, y, logits, mask):
        ll = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
        return tf.reduce_mean(tf.boolean_mask(ll, mask))


if __name__ == '__main__':
    # set up config params
    import argparse

    parser = argparse.ArgumentParser(description='Train the Char Embedding Bi-LSTM Named Entity Recognition model')
    # options for file paths/names
    path_group = parser.add_argument_group('File name and path options')
    path_group.add_argument('--data_path', required=True,
        help='path to training data. Assumes assets in "assets/" folder and data in "data/" folder')
    path_group.add_argument('--result_path', required=True, help='path where results and model output will be saved')
    path_group.add_argument('--words', default='vocab.words.txt', help='filename for words vocab')
    path_group.add_argument('--chars', default='vocab.chars.txt', help='filename for character vocab')
    path_group.add_argument('--tags', default='vocab.tags.txt', help='filename for tags vocab')
    path_group.add_argument('--glove', default='glove.npz', help='filename for glove embeddings npz file')
    path_group.add_argument('--export_dir', default='saved_model', help='filename for glove embeddings npz file')
    # options for if data needs to be prepared
    prep_group = parser.add_argument_group('Data preparation options')
    prep_group.add_argument('--raw_glove_path', required=False,
        help='''path to raw glove vectors. If this option is specified, the data preparation script will be run first 
        in order to generate the trimmed glove embeddings and vocabularies. 
        This assumes that the input data is in the "data/" folder and the vocabs and embeddings will be 
        written to "assets/" folder''')
    prep_group.add_argument('--min_count', type=int, default=1, help='drop words < min_count occurence in the vocabulary (default: 1)')
    prep_group.add_argument('--glove_type', default='6B', help='Corpus for GloVe embeddings: 6B or 840B (default 6B)')
    # model options
    model_group = parser.add_argument_group('Model definition and training options')
    model_group.add_argument('--dim_words', type=int, default=100, help='word embedding dimension')
    model_group.add_argument('--dim_chars', type=int, default=32, help='character embedding dimension')
    model_group.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    model_group.add_argument('--num_oov_buckets', type=int, default=1, help='out-of-vocabulary buckets for lookup tables')
    model_group.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    model_group.add_argument('--conf_epochs', type=int, default=0,
        help='number of epochs to train confidence branch after training main network (default: 0 i.e. no confidence branch training)')
    model_group.add_argument('--checkpoint_interval', type=int, default=5, help='checkpoint interval in epochs')
    model_group.add_argument('--batch_size', type=int, default=32, help='batch size')
    model_group.add_argument('--learning_rate', type=float, required=False, help='learning rate for optimizer')
    model_group.add_argument('--buffer', type=int, default=2000, help='buffer for training dataset shuffle')
    model_group.add_argument('--char_lstm_size', type=int, default=64, help='hidden size for char bi-lstm layer')
    model_group.add_argument('--lstm_size', type=int, default=128, help='hidden size for concatenated bi-lstm layer')
    model_group.add_argument('--glove_cased', dest='lower_case', action='store_false',
        help='set this flag if glove word embeddings should be cased (default: uncased)')
    model_group.set_defaults(lower_case=True)

    args = parser.parse_args()
    
    data_path = args.data_path
    ASSET_DIR = Path(data_path, 'assets')
    DATA_DIR = Path(data_path, 'data')
    RESULT_DIR = Path(args.result_path)
    params = vars(args).copy()

    if params['raw_glove_path']:
        print('Option --raw_glove_path specified. Running data preparation script.')
        # first we prepare the dataset 
        from data_prep import data_prep      
        data_prep(DATA_DIR, Path(params['raw_glove_path']), ASSET_DIR, params['dim_words'],
            params['glove_type'], params['lower_case'], params['min_count'])

    # Logging
    Path(RESULT_DIR).mkdir(exist_ok=True)
    tf.compat.v1.logging.set_verbosity(logging.INFO)
    handlers = [
        logging.FileHandler('{}/main.log'.format(RESULT_DIR)),
        logging.StreamHandler(sys.stdout)
    ]
    logging.getLogger('tensorflow').handlers = handlers

    params['words'] = str(Path(ASSET_DIR, params['words']))
    params['chars'] = str(Path(ASSET_DIR, params['chars']))
    params['tags'] = str(Path(ASSET_DIR, params['tags']))
    params['glove'] = str(Path(ASSET_DIR, params['glove']))
    params['export_dir'] = str(Path(RESULT_DIR, params['export_dir'])) if 'export_dir' in params else None

    print('Training run params: {}'.format(params))
    print('Training data location: {}'.format(DATA_DIR))
    print('Training asset location: {}'.format(ASSET_DIR))
    print('Training result location: {}'.format(RESULT_DIR))

    # write params used for result dir
    with Path(RESULT_DIR, 'params.json').open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    def fwords(name):
        return str(Path(DATA_DIR, '{}.words.txt'.format(name)))

    def ftags(name):
        return str(Path(DATA_DIR, '{}.tags.txt'.format(name)))

    print('Building model.')
    ner_model = NERModel(params)
    # create datasets
    print('Creating datasets.')
    train_data = create_dataset(
        fwords('train'), ftags('train'), params, ner_model.vocab_words, ner_model.vocab_chars, ner_model.vocab_tags, shuffle=True)
    valid_data = create_dataset(
        fwords('valid'), ftags('valid'), params, ner_model.vocab_words, ner_model.vocab_chars, ner_model.vocab_tags)
    test_data = create_dataset(
        fwords('test'), ftags('test'), params, ner_model.vocab_words, ner_model.vocab_chars, ner_model.vocab_tags)

    # train model
    ckpt_dir = '{}/ckpts'.format(RESULT_DIR)
    ner_model, _ = train(ner_model, params, train_data, valid_data, ckpt_dir)
    # train confidence branch
    ner_model = train_confidence_branch(ner_model, params, train_data, valid_data, ckpt_dir)

    # generate entity-level metrics for all three sets of data
    for name, data in (('train', train_data),
                       ('validation', valid_data),
                       ('test', test_data)):
        print(f'Computing multi-class metrics for {name} data')
        scores = generate_mc_metrics(params, ner_model, train_data)
        print(f'{name} scores:')
        print(scores)
        csv_out = str(Path(RESULT_DIR, f'{name}_scores.csv'))
        scores.to_csv(csv_out, index=False)
        print('Wrote scores to: {}'.format(csv_out))

    if 'export_dir' in params:
    # export to SavedModel
        print('Exporting SavedModel to: {}'.format(params['export_dir']))
        tf.saved_model.save(ner_model, params['export_dir'],
            signatures={tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: ner_model.serve_text_input,
            'serve_token_input': ner_model.serve_token_input})
