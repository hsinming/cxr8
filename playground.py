#!/usr/bin/python3
# encoding=utf8

import sys
import torch
from torchvision import models
import torch.nn as nn
from pprint import pprint
import json
from report import match_sentence_semantic



se_resnet50_weight = './se_resnet50_weight.pkl'

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        net = models.densenet121(pretrained=True)
        for mod in net.named_modules():
            print(mod)
        print()
    def forward(self, x):
        return x

def main():
    name = 'YFC'
    th = 0.97
    number = 5
    start = 0

    for i in range(start, start + number):
        try:
            with open('/data/CXR8/NTUH/{}_CommonSent_{}.json'.format(name, i - 1), 'rt', encoding='utf-8-sig') as fp:
                sentences = json.load(fp)
        except:
            with open('/data/CXR8/NTUH/{}_sentences_unlabel.txt'.format(name, i - 1), 'rt', encoding='utf-8-sig') as fp:
                sentences = []
                for line in fp.readlines():
                    line = line.strip()
                    sentences.append(line)

        sentences = list(set(sentences))
        # matchList = match_sentence(sentences, th)
        matchList = match_sentence_semantic(sentences, th)
        count_matchList = sorted([(d['sent_a'], len(d['sent_b'])) for d in matchList],
                                 key=lambda x: x[1], reverse=True)
        # exclude the sentence of only single match
        common_sent = sorted([s for (s, c) in count_matchList if c > 1])

        with open('/data/CXR8/NTUH/{}_MatchList_{}.json'.format(name, i), 'wt') as fp:
            json.dump(matchList, fp, indent=4, sort_keys=True)
        with open('/data/CXR8/NTUH/{}_MatchCount_{}.json'.format(name, i), 'wt') as fp:
            json.dump(count_matchList, fp, indent=4, sort_keys=True)
        with open('/data/CXR8/NTUH/{}_CommonSent_{}.json'.format(name, i), 'wt') as fp:
            json.dump(common_sent, fp, indent=4, sort_keys=True)

    pprint(count_matchList[:50])
    print('There are {} common sentence.'.format(len(common_sent)))




if __name__=='__main__':
    status = main()
    sys.exit(status)
