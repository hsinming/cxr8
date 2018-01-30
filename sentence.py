#!/usr/bin/python3
# encoding=utf-8
# from http://nlpforhackers.io/splitting-text-into-sentences/

import sys
from nltk import sent_tokenize
from nltk.corpus import gutenberg
from pprint import pprint
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer


def sentence_tokenizer(training_text=None):
    if training_text is None:
        training_text = ""
        for file_id in gutenberg.fileids():
            training_text += gutenberg.raw(file_id)

    trainer = PunktTrainer()
    trainer.INCLUDE_ALL_COLLOCS = True

    if isinstance(training_text, str):
        trainer.train(training_text)
    tokenizer = PunktSentenceTokenizer(trainer.get_params())

    return tokenizer


def main():
    sentences = "Mr. Smith bought cheapsite.com for 1.5 million dollars, i.e. he paid a lot for it. Did he mind? Adam Jones Jr. thinks he didn't. In any case, this isn't true... Well, with a probability of .9 it isn't."
    tokenizer = sentence_tokenizer()
    tokenizer._params.abbrev_types.add('dr')
    tokenizer._params.abbrev_types.add('mr')
    tokenizer._params.abbrev_types.add('jr')
    pprint(tokenizer.tokenize(sentences))
    print()
    pprint(sent_tokenize(sentences))

if __name__ == '__main__':
    status = main()
    sys.exit(status)