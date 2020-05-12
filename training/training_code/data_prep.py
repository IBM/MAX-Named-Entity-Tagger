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
# Adapted from https://github.com/guillaumegenthial/tf_ner/blob/master/data/example/build_vocab.py
# and https://github.com/guillaumegenthial/tf_ner/blob/master/data/example/build_glove.py

import argparse
from collections import Counter
from pathlib import Path
import numpy as np


def data_prep(data_dir, glove_dir, output_dir, dim_word, glove_type, lower_case, min_count):
    # 1. Words
    # Get Counter of words on all the data, filter by min count, save

    Path(output_dir).mkdir(exist_ok=True)
    # we build the vocab only on training data
    print('Building word vocab (may take a while)')
    train_file = '{}/train.words.txt'.format(data_dir)
    word_vocab_file = '{}/vocab.words.txt'.format(output_dir)
    counter_words = Counter()
    with Path(train_file).open() as f:
        print('Reading training words file: {}'.format(train_file))
        for line in f:
            counter_words.update(line.strip().split())

    if lower_case:
        print('Generating lower-case word vocab')
        vocab_words = {w.lower() for w, c in counter_words.items() if c >= min_count}
    else:
        print('Generating cased word vocab')
        vocab_words = {w for w, c in counter_words.items() if c >= min_count}

    vocab_words_list = sorted(list(vocab_words))

    word_to_idx = {w: idx for idx, w in enumerate(vocab_words_list)}
    size_vocab = len(word_to_idx)

    with Path(word_vocab_file).open('w') as f:
        for w in vocab_words_list:
            f.write('{}\n'.format(w))
            
    print('- done. Kept {} out of {}'.format(
        size_vocab, len(counter_words)))
    print('Wrote word vocab to: {}'.format(word_vocab_file))

    # 2. Chars
    # Get all the characters from the vocab words
    char_vocab_file = '{}/vocab.chars.txt'.format(output_dir)
    print('Building character vocab from word vocab')
    vocab_chars = set()
    for w in vocab_words:
        vocab_chars.update(w)

    with Path(char_vocab_file).open('w') as f:
        for c in sorted(list(vocab_chars)):
            f.write('{}\n'.format(c))
    print('- done. Found {} chars'.format(len(vocab_chars)))
    print('Wrote char vocab to: {}'.format(char_vocab_file))

    # 3. Tags
    # Get all tags from the training set

    train_tags_file = '{}/train.tags.txt'.format(data_dir)
    tag_vocab_file = '{}/vocab.tags.txt'.format(output_dir)
    print('Building tag vocab tags (may take a while)')
    vocab_tags = set()
    with Path(train_tags_file).open() as f:
        print('Reading training tags file: {}'.format(train_tags_file))
        for line in f:
            vocab_tags.update(line.strip().split())

    with Path(tag_vocab_file).open('w') as f:
        for t in sorted(list(vocab_tags)):
            f.write('{}\n'.format(t))
    print('- done. Found {} tags.'.format(len(vocab_tags)))
    print('Wrote tag vocab to: {}'.format(tag_vocab_file))

    # === process Glove vectors ===

    glove_filename = '{}/glove.{}.{}d.txt'.format(glove_dir, glove_type, dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    glove_out = '{}/glove.npz'.format(output_dir)
    # Array of zeros
    embeddings = np.zeros((size_vocab, dim_word))

    # Get relevant glove vectors
    found = 0
    print('Reading GloVe file: {} (may take a while)'.format(glove_filename))
    with Path(glove_filename).open() as f:
        for line_idx, line in enumerate(f):
            if line_idx % 100000 == 0:
                print('- At line {}'.format(line_idx))
            line = line.strip().split()
            if len(line) != dim_word + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
    print('- done. Found {} vectors for {} words'.format(found, size_vocab))

    # Save np.array to file
    np.savez_compressed(glove_out, embeddings=embeddings)
    print('Wrote glove embeddings to: {}'.format(glove_out))
    print('Completed data preparation')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare a dataset for training and extract vocabulary files')
    parser.add_argument('--data_path', required=True,
        help='path to training data: one file with a sentence per line; one file with tags per line')
    parser.add_argument('--glove_path', required=True, help='path to glove vectors')
    parser.add_argument('--output_path', required=True, help='path to write outputs')
    parser.add_argument('--dim', type=int, default=100, help='dimension of GloVe word embedding vectors (default: 100)')
    parser.add_argument('--min_count', type=int, default=1, help='drop words < min_count occurence in the vocabulary (default: 1)')
    parser.add_argument('--glove_type', default='6B', help='Corpus for GloVe embeddings: 6B or 840B (default 6B)')
    parser.add_argument('--lower_case', type=bool, default=True,
        help='If True, word vocab is generated for lowercase words only (chars are still cased). ' + 
        'This MUST be matched with same setting in training script (default: True)')

    args = parser.parse_args()
    data_dir = args.data_path.rstrip('/')
    glove_dir = args.glove_path.rstrip('/')
    output_dir = args.output_path.rstrip('/')
    dim_word = args.dim
    glove_type = args.glove_type
    lower_case = args.lower_case
    min_count = args.min_count
    # execute prepare data script
    data_prep(data_dir, glove_dir, output_dir, dim_word, glove_type, lower_case, min_count)
