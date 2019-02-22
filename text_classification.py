#!/usr/bin/python3
# encoding=utf-8
import sys
import os
import json
from pprint import pprint
from collections import defaultdict
import re
import fastText as fasttext
from magpie import Magpie


label = {'yfc':'/data/CXR8/NTUH/YFC_sentences_label.txt'}
output = {'yfc':'/data/CXR8/NTUH/YFC.label.json'}

def label2json(label_file: str, json_file: str):
    labeled_sentences = list()

    with open(label_file, 'rt', encoding='utf-8-sig') as fp:
        pattern = re.compile(r'(?P<label>__label__\w+)')
        for line in fp.readlines():
            line = line.strip()
            sentence = defaultdict(list)
            for item in [s.strip() for s in pattern.split(line) if len(s.strip()) > 0]:
                if item.startswith('__label__'):
                    sentence['label'].append(item)
                else:
                    sentence['sentence'] = item
            labeled_sentences.append(sentence)
    with open(json_file, 'wt', encoding='utf-8-sig') as fp:
        json.dump(labeled_sentences, fp, indent=4)
    return labeled_sentences


def json2magpie():
    save_root = '/data/CXR8/NTUH/YFC_magpie'
    with open('/data/CXR8/NTUH/YFC.label.json', 'rt', encoding='utf-8-sig') as fp:
        labeled_sentences = json.load(fp)
    all_label = set()
    for idx, d in enumerate(labeled_sentences):
        identity = '{:>05}'.format(idx)
        txt_fp = os.path.join(save_root, '{}.txt'.format(identity))
        lab_fp = os.path.join(save_root, '{}.lab'.format(identity))
        sentence = d['sentence']
        all_label.update(d['label'])
        if len(d['label']) > 1:
            lab = "\n".join(d['label'])
        else:
            lab = d['label'][0] + '\n'
        with open(txt_fp, 'wt', encoding='utf-8-sig') as fp:
            fp.write(sentence)
        with open(lab_fp, 'wt', encoding='utf-8-sig') as fp:
            fp.write(lab)
    all_label = '\n'.join(list(all_label))
    with open('/data/CXR8/NTUH/YFC_labels', 'wt', encoding='utf-8-sig') as fp:
        fp.write(all_label)



def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


def train_model():
    train_data = '/home/hsinming/fastText/data/YFC.train'
    valid_data = '/home/hsinming/fastText/data/YFC.test'

    # train_supervised uses the same arguments and defaults as the fastText cli
    model = fasttext.train_supervised(input=train_data, epoch=5000, lr=0.1, wordNgrams=4, verbose=2, minCount=10)
    print_results(*model.test(valid_data))
    model.save_model("/data/CXR8/NTUH/YFC.bin")


def main():
    all_labels = []
    with open('/data/CXR8/NTUH/YFC_labels', 'rt', encoding='utf-8-sig') as fp:
        for line in fp.readlines():
            line = line.strip()
            all_labels.append(line)

    magpie = Magpie()
    magpie.train_word2vec('/data/CXR8/NTUH/YFC_magpie', vec_dim=100)
    magpie.fit_scaler('/data/CXR8/NTUH/YFC_magpie')
    magpie.train('/data/CXR8/NTUH/YFC_magpie', all_labels, test_ratio=0.1, epochs=300)
    magpie.save_word2vec_model('/data/CXR8/NTUH/YFC_word2vec_model')
    magpie.save_scaler('/data/CXR8/NTUH/YFC_scalar', overwrite=True)
    magpie.save_model('/data/CXR8/NTUH/YFC_model.h5')

if __name__ == '__main__':
    status = main()
    sys.exit(status)
