#!/usr/bin/python3
# encoding=utf-8
# Inspired by http://nlpforhackers.io/wordnet-sentence-similarity/

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import sys
import itertools


def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'

    if tag.startswith('V'):
        return 'v'

    if tag.startswith('J'):
        return 'a'

    if tag.startswith('R'):
        return 'r'

    return None


def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None

    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None


def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))

    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        synset_similarity = [synset.path_similarity(ss) for ss in synsets2 if synset.path_similarity(ss)]
        if synset_similarity:
            best_score = max(synset_similarity)
        else:
            best_score = None

        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1

    # Average the values
    try:
        score /= count
    except ZeroDivisionError:
        score = 0.0
    return score


def symmetric_sentence_similarity(sentence1, sentence2):
    """ compute the symmetric sentence similarity using Wordnet """
    return (sentence_similarity(sentence1, sentence2) + sentence_similarity(sentence2, sentence1)) / 2


def main():
    sentences = ['A large mass in the right upper lung.',
                 'A big tumor in the right upper lung.',
                 'A small nodule at right pulmonary hilum.',
                 'Multiple nodules in bilateral lungs.',
                 'Multiple ill defined opacities in bilateral lungs with bilateral pleural effusion.',
                 'Patchy opacity at left lower lung with mild left pleural effusion.',
                 'Many small tumors in bilateral lungs.',
                 "Bilateral ill defined opacities and mild pleural effusion."]



    for a, b in itertools.combinations(sentences, 2):
        sim = symmetric_sentence_similarity(a, b)
        print('Sentence A: {}'.format(a))
        print('Sentence B: {}'.format(b))
        print('==> Similarity: {:.4f}'.format(sim))
        print()



if __name__ == '__main__':
    status = main()
    sys.exit(status)