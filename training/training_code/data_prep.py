# Adapted from https://github.com/guillaumegenthial/sequence_tagging/blob/master/build_data.py

import argparse
from data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word


parser = argparse.ArgumentParser(description='Prepare a IOB-formatted dataset for training and extract vocabulary files')
parser.add_argument('--data_path', required=True, help='path to training data')
parser.add_argument('--glove_path', required=True, help='path to glove vectors')
parser.add_argument('--output_path', required=True, help='path to write outputs')
parser.add_argument('--dim', type=int, default=300, help='dimension of GloVe word embedding vectors')

args = parser.parse_args()

data_dir = args.data_path.rstrip('/')
glove_dir = args.glove_path.rstrip('/')
output_dir = args.output_path.rstrip('/')
dim_word = args.dim

# training data
train_filename = "{}/train.txt".format(data_dir)
valid_filename = "{}/valid.txt".format(data_dir)
# glove files
glove_filename = "{}/glove.6B.{}d.txt".format(glove_dir, dim_word)
# trimmed embeddings (created from glove_filename with build_data.py)
filename_trimmed = "{}/glove.6B.{}d.trimmed.npz".format(output_dir, dim_word)

words_filename = "{}/words.txt".format(output_dir)
tags_filename = "{}/tags.txt".format(output_dir)
chars_filename = "{}/chars.txt".format(output_dir)

processing_word = get_processing_word(lowercase=True)

train = CoNLLDataset(train_filename, processing_word)
valid  = CoNLLDataset(valid_filename, processing_word)

# Build word and tag vocabs
vocab_words, vocab_tags = get_vocabs([train, valid])
vocab_glove = get_glove_vocab(glove_filename)

vocab = vocab_words & vocab_glove
vocab.add(UNK)
vocab.add(NUM)

# Save vocab
write_vocab(vocab, words_filename)
write_vocab(vocab_tags, tags_filename)

# Trim GloVe Vectors
vocab = load_vocab(words_filename)
export_trimmed_glove_vectors(vocab, glove_filename,
                            filename_trimmed, dim_word)

# Build and save char vocab
train = CoNLLDataset(train_filename)
vocab_chars = get_char_vocab(train)
write_vocab(vocab_chars, chars_filename)
