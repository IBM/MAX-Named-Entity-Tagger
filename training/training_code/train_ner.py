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


def _map_word_dataset(raw_words, tokenizer, vocab_words, vocab_chars, pad_value, pad_len, lower):
    word_tokens = raw_words.map(tokenizer.tokenize)
    char_tokens = word_tokens.map(lambda w: tf.strings.unicode_split(w, 'UTF-8'))
    if lower:
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
    lower = not params['glove_cased']

    raw_words = tf.data.TextLineDataset(fwords).batch(batch_size)
    raw_tags = tf.data.TextLineDataset(ftags).batch(batch_size)

    # tag dataset
    tag_ids = _map_tag_dataset(raw_tags, tokenizer, vocab_tags, pad_tag)
    # word and char datasets
    word_char_data = _map_word_dataset(
        raw_words, tokenizer, vocab_words, vocab_chars, pad_value, pad_len, lower)

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
    logits = tf.keras.layers.Dense(num_tags, name='logits')(output)

    # TODO add confidence predictions
    # softmax = tf.keras.layers.Activation('softmax', name='predict_output')(logits)
    # pred_ids = tf.argmax(softmax, axis=-1)

    model = tf.keras.Model([word_ids, sent_lengths, char_ids, word_lengths], logits)

    return model


def train(ner_model, params, train_data, valid_data):

    x_sig = train_data.element_spec[0]
    y_sig = train_data.element_spec[1]

    # === set up optimizer and metrics ===
    optimizer = tf.keras.optimizers.Adam()
    acc = tf.keras.metrics.Accuracy()
    val_acc = tf.keras.metrics.Accuracy()
    # for f1 
    all_tags, all_tags_idx = ner_model.vocab_tags.export()
    indices = [i for i, t in zip(all_tags_idx.numpy(), all_tags.numpy()) if t != b'O']

    ckpt_dir = '{}/ckpts'.format(RESULT_DIR)
    ckpt = (tf.train.Checkpoint(
        epochs_trained=tf.Variable(0),
        steps_per_epoch=tf.Variable(0),
        # params=params,
        optimizer=optimizer,
        # metrics=metrics,
        ner_model=ner_model))
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print('Restored from checkpoint: {}. Epochs trained: {}'.format(manager.latest_checkpoint, ckpt.epochs_trained.numpy()))
    else:
        print('No checkpoint found in: {}. Initializing training from scratch.'.format(ckpt_dir))

    # === define functions for training and evaluation ===
    @tf.function(input_signature=[x_sig, y_sig])
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits, pred_ids, sent_lengths = ner_model.training_predict(x, training=True)
            loss = ner_model.loss(logits, y, sent_lengths)

        # compute and apply grads
        grads = tape.gradient(loss, ner_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, ner_model.trainable_variables))

        # Update training metric.
        mask = tf.sequence_mask(sent_lengths)
        acc(y, pred_ids, sample_weight=mask)
        return pred_ids, loss

    @tf.function(input_signature=[x_sig, y_sig])
    def valid_step(x, y):
        _, pred_ids, sent_lengths = ner_model.training_predict(x, training=False)
        mask = tf.sequence_mask(sent_lengths)
        val_acc(y, pred_ids, sample_weight=mask)
        # f1
        return pred_ids, mask

    epochs = params['epochs']
    checkpoint_interval = params.get('checkpoint_interval', 5)
    print('Beginning training for {} epochs. Checkpointing interval {} epochs'.format(epochs, checkpoint_interval))
    for epoch in range(1, epochs + 1):
        print("\nepoch {}/{}".format(epoch, epochs))
        # init keras progress bar
        total_steps = ckpt.steps_per_epoch.numpy()
        total_steps = total_steps if total_steps > 0 else None
        progbar = tf.keras.utils.Progbar(total_steps, interval=0.5, stateful_metrics=['acc', 'val_acc', 'val_f1'])

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

        for x_batch_val, y_batch_val in valid_data:
            pred_ids, mask = valid_step(x_batch_val, y_batch_val)
            cm = compute_cm(y_batch_val, pred_ids, ner_model.num_tags, mask)
            total_cm.assign_add(cm)

        pr, rec, f1 = metrics_from_confusion_matrix(total_cm, indices)
        metric_values = [('acc', acc.result()), ('val_acc', val_acc.result()), ('val_f1', f1)]
        progbar.update(step + 1, values=metric_values)
        # Reset metrics at the end of each epoch
        acc.reset_states()
        val_acc.reset_states()
        ckpt.epochs_trained.assign_add(1)

        if ckpt.epochs_trained.numpy() % checkpoint_interval == 0:
            save_path = manager.save()
            print("Saved checkpoint for epoch {} (total epochs trained: {}): {}".format(epoch, ckpt.epochs_trained.numpy(), save_path))
    
    # save final checkpoint 
    save_path = manager.save()
    print("Saved final checkpoint for epoch {} (total epochs trained: {}): {}".format(epoch, ckpt.epochs_trained.numpy(), save_path))
    return ner_model, manager


def generate_mc_metrics(params, ner_model, dataset):
    @tf.function(input_signature=[dataset.element_spec[0]])
    def test_predict_step(x):
        _, pred_ids, sent_lengths = ner_model.training_predict(x, training=False)
        return pred_ids, sent_lengths

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
        pred_ids, sent_lengths = test_predict_step(x)
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
        self.lower = not params['glove_cased']

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
    def serve(self, words):
        word_tokens = self.tokenizer.tokenize(words)
        input_tokens = word_tokens # copy for returning later
        char_tokens = tf.strings.unicode_split(word_tokens, 'UTF-8')

        if self.lower:
            word_tokens = tf.ragged.map_flat_values(tf.strings.lower, word_tokens)
        sent_lengths = word_tokens.row_lengths()
        word_ids = self.vocab_words.lookup(word_tokens.to_tensor(self.pad_value))
        word_lengths = char_tokens.row_lengths(axis=-1).to_tensor(0)
        char_ids = self.vocab_chars.lookup(char_tokens.to_tensor(self.pad_value))

        features = {
            'word_ids': word_ids,
            'sent_lengths': sent_lengths,
            'char_ids': char_ids,
            'word_lengths': word_lengths
        }
        # predict
        logits = self.base_model(features, training=False)
        pred_ids, _ = tfa.text.crf.crf_decode(logits, self.crf_params, sent_lengths)
        
        pred_tags = self.reverse_vocab_tags.lookup(tf.cast(pred_ids, dtype=tf.int64))
        # return ragged tensors to take into account the input sentence lengths
        pred_ids = tf.ragged.boolean_mask(pred_ids, tf.sequence_mask(sent_lengths))
        pred_tags = tf.ragged.boolean_mask(pred_tags, tf.sequence_mask(sent_lengths))
        return [pred_tags, pred_ids, input_tokens]
        # return {
        #     'pred_tags': pred_tags,
        #     'pred_ids': pred_ids,
        #     'sent_lengths': sent_lengths,
        #     'word_tokens': input_tokens if self.lower else word_tokens
        # }

    @tf.function
    def call(self, x, training=None):
        return self.base_model(x, training)

    @tf.function
    def training_predict(self, x, training=None):
        sent_lengths = x['sent_lengths']
        logits = self(x, training)
        pred_ids, _ = tfa.text.crf.crf_decode(logits, self.crf_params, sent_lengths)
        return logits, pred_ids, sent_lengths

    @tf.function
    def loss(self, logits, y, word_lengths):
        log_likelihood, _ = tfa.text.crf.crf_log_likelihood(logits, y, word_lengths, self.crf_params)
        return tf.reduce_mean(input_tensor=-log_likelihood)


if __name__ == '__main__':
    # Params
    from params import get_params
    import argparse

    parser = argparse.ArgumentParser(description='Train the Char Embedding Bi-LSTM Named Entity Recognition model')
    parser.add_argument('--data_path', required=True,
        help='path to training data. Assumes assets in "assets/" folder and data in "data/" folder')
    parser.add_argument('--result_path', required=True, help='path where results and model output will be saved')

    args = parser.parse_args()
    ASSET_DIR = Path(args.data_path, 'assets')
    DATA_DIR = Path(args.data_path, 'data')
    RESULT_DIR = Path(args.result_path)

    # Logging
    Path(RESULT_DIR).mkdir(exist_ok=True)
    tf.compat.v1.logging.set_verbosity(logging.INFO)
    handlers = [
        logging.FileHandler('{}/main.log'.format(RESULT_DIR)),
        logging.StreamHandler(sys.stdout)
    ]
    logging.getLogger('tensorflow').handlers = handlers

    params = get_params()
    params['words'] = str(Path(ASSET_DIR, params['words']))
    params['chars'] = str(Path(ASSET_DIR, params['chars']))
    params['tags'] = str(Path(ASSET_DIR, params['tags']))
    params['glove'] = str(Path(ASSET_DIR, params['glove']))
    params['export_dir'] = str(Path(RESULT_DIR, params['export_dir'])) if 'export_dir' in params else None
    params['scores_dir'] = str(Path(RESULT_DIR, params['scores_dir'])) if 'scores_dir' in params else None

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
    ner_model, _ = train(ner_model, params, train_data, valid_data)

    # export to SavedModel
    if 'export_dir' in params:
        print('Exporting SavedModel to: {}'.format(params['export_dir']))
        tf.saved_model.save(ner_model, params['export_dir'], signatures=ner_model.serve)

    # generate entity-level metrics for test data
    if 'scores_dir' in params:
        Path(params['scores_dir']).mkdir(exist_ok=True)
        print('Computing multi-class metrics for test data')
        test_scores = generate_mc_metrics(params, ner_model, test_data)
        print('Test scores:')
        print(test_scores)
        csv_out = str(Path(params['scores_dir'], 'test_scores.csv'))
        test_scores.to_csv(csv_out, index=False)
        print('Wrote scores to: {}'.format(csv_out))