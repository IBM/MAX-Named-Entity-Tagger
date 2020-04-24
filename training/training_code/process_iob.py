import unicodedata
import csv  
import argparse
from pathlib import Path


def out_file(path, name, wt):
    return '{}/{}.{}.txt'.format(path, name, wt)


def process_iob(filename):
    with open(filename) as f:
        sentences = []
        tags = []
        words, word_tags = [], []
        for line in f:
                line = line.strip()
                # line = unicodedata.normalize('NFKD', line)
                if line.startswith("-DOCSTART-"):
                    continue
                elif len(line) == 0:
                    if len(words) > 0 and len(word_tags) > 0:
                        # if 'Wilma' in words:
                            # print(words)
                        sentences += [words]
                        tags += [word_tags]
                        words, word_tags = [], []
                    continue
                else:
                    ls = line.split(' ')
                    word, word_tag = ls[0], ls[-1]
                    words += [word]
                    word_tags += [word_tag]
        return sentences, tags


def write_output(data, out_file):
    with open(out_file, 'w') as f:
        writer = csv.writer(f, delimiter=' ', quoting=csv.QUOTE_NONE, escapechar='\\')
        writer.writerows(data)
    print('Wrote {}'.format(out_file))

# === Main script ===
parser = argparse.ArgumentParser(description=
    'Convert IOB format data to a word file and tag file for model training')
parser.add_argument('--data_path', required=True,
    help='path to training data in IOB format')
parser.add_argument('--output_path', required=True, help='path to write output data')
parser.add_argument('--files', required=False, type=str, default='train,valid,test',
    help='comma-delimited list of filenames to process - .txt extensions are assumed (default: train, valid, test)')

args = parser.parse_args()

data_dir = args.data_path.rstrip('/')
output_dir = args.output_path.rstrip('/')
Path(output_dir).mkdir(exist_ok=True)
files = args.files
files = files.split(',')

for f in files:
    fname = '{}/{}.txt'.format(data_dir, f)
    print('Processing file: {}'.format(fname))
    sentences, tags = process_iob(fname)
    write_output(sentences, out_file(output_dir, f, 'words'))
    write_output(tags, out_file(output_dir, f, 'tags'))
