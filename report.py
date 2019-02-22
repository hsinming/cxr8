#!/usr/bin/python3
# encoding=utf-8
from pandas import DataFrame, ExcelWriter
import os
import sys
from glob import glob
import json
import re
import hunspell
import copy
from pprint import pprint
from difflib import SequenceMatcher
from itertools import combinations
from nltk import sent_tokenize
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from operator import itemgetter
import spacy


matcher = SequenceMatcher(lambda x: x in ' ,.;?()<>0123456789+-*/=!@#$%^&', ' ', ' ')
spellchecker = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
spellchecker.add_dic('/usr/share/hunspell/en_med_glut.dic')
medical_wordlist = '/data/CXR8/NTUH/wordlist.txt'
radiology_wordlist = '/data/CXR8/NTUH/radiology_word.txt'
json_repo = {'yfc': '/data/CXR8/NTUH/YFC_Reports',
             'ycc': '/data/CXR8/NTUH/YCC_Reports',
             'jyc': '/data/CXR8/NTUH/JYC_Reports',
             'wjl': '/data/CXR8/NTUH/WJL_Reports',
             'many': '/data/CXR8/NTUH/MANY_Reports'}


class Report(object):
    json_repo = {'yfc': '/data/CXR8/NTUH/YFC_Reports',
                 'ycc': '/data/CXR8/NTUH/YCC_Reports',
                 'jyc': '/data/CXR8/NTUH/JYC_Reports',
                 'wjl': '/data/CXR8/NTUH/WJL_Reports',
                 'many': '/data/CXR8/NTUH/MANY_Reports'}
    def __init__(self, name: str, root_path='/data/CXR8/NTUH'):
        self.name = name.upper()
        self.root_path = root_path
        self.excel_path = os.path.join(root_path, '{}_report.xls'.format(name.upper()))
        self.report_path = os.path.join(root_path, '{}_report.txt'.format(name.upper()))
        self.raw_sent_path = os.path.join(root_path, '{}_raw_sentence.txt'.format(name.upper()))
        self.corrected_sent_path = os.path.join(root_path, '{}_corrected_sentence.txt'.format(name.upper()))
        self.json_repo = self.__class__.json_repo.get(name.lower(), '')
        self.json_files = self._get_json_list()
        self.reports = self._get_report_body()
        self._save_list(self.reports, self.report_path)
        self.raw_sentences = self._split_into_sentence()
        self._save_list(self.raw_sentences, self.raw_sent_path)
        corrected_sentences = []
        for s in self.raw_sentences:
            corrected_sentences.append(self._correct_typo(s))
        self.corrected_sentences = corrected_sentences
        self._save_list(self.corrected_sentences, self.corrected_sent_path)

    def __str__(self):
        return "{name}'s reports".format(self.name)

    def __repr__(self):
        self.__str__()

    def _get_json_list(self) -> list:
        json_list = []
        json_list += glob(os.path.join(self.json_repo, '*.json'))
        print('There are {} reports loaded.'.format(len(json_list)))
        return json_list

    def _get_report_body(self) -> list:
        reportBody_list = []
        for jsf in self.json_files:
            with open(jsf, 'r', encoding='utf-8-sig') as fp:
                try:
                    whole_report = json.load(fp)
                except:
                    continue

                for key, value in whole_report.items():
                    if isinstance(value, dict):
                        try:
                            raw_report = value.get('ReportBody', '')
                        except:
                            continue
                        report = re.split(r'\r\n', raw_report)
                        for r in report:
                            reportBody_list.append(r.strip())
                    else:
                        continue

        return reportBody_list

    def _split_into_sentence(self) -> list:
        """
        A sentence tokenizer by regular expression.
        Return a list of sentences, each of which is a string.
        """
        sentence_list = []
        filter = re.compile(
            r'(?:(?:Chest)?(?:PA|AP|PA/AP|AP/PA)?(?:CXR)?.*?(?:shows|show|showed)?\s*:?\s*)?([a-zA-Z]+.*?(?<!\d)(?<!Dr)(?<!DR)(?<!Bil)(?<!bil)(?<!esp)(?<!Esp)(?<!susp)(?<!Susp))[.,;:]')

        for report in self.reports:
            for line in report.splitlines():
                line = line.strip()
                match = filter.findall(line)
                if match:
                    sentence_list += match
        sentence_list = list(set(sentence_list))
        sentence_list.sort()
        print('These reports contain {} different sentences.'.format(len(sentence_list)))
        return sentence_list

    def _correct_typo(self, input_str: str) -> str:
        """
        Hunspell spell checker for string of words or a sentence.
        :param input: a string
        :return: a string
        """
        _add_word_to_dictionary(spellchecker, radiology_wordlist)
        enc = spellchecker.get_dic_encoding()

        is_single_word = bool(len(re.split(r'\s+', input_str)) == 1)
        is_sentence = not is_single_word

        to_check = copy.deepcopy(input_str)

        if is_single_word:
            if to_check.isalpha() and (not to_check.isupper()):
                try:
                    ok = spellchecker.spell(to_check)  # False if it is a typo.
                except UnicodeEncodeError:  # Not in a supported encoding.
                    return to_check

                if not ok:
                    suggestions = spellchecker.suggest(to_check)
                    if suggestions:
                        try:
                            best = suggestions[0].decode(enc)
                        except:
                            best = suggestions[0]
                        corrected = best
                    else:  # Hunspell has no suggestion, so send back the original word.
                        corrected = to_check
                else:  # Already correct.
                    corrected = to_check
            else:  # Abbreviation or numbers
                corrected = to_check
            return corrected
        elif is_sentence:
            corrected_sentence = []
            words = re.split(r'\s+', to_check)
            for word in words:
                corrected_word = correct_typo(word)
                corrected_sentence.append(corrected_word)
            corrected_sentence = " ".join(corrected_sentence)
            return corrected_sentence

    def _save_list(self, list_to_save: list, output_fn: str):
        to_write = '\n'.join(list_to_save)
        with open(output_fn, 'wt') as fp:
            fp.write(to_write)

    def _load_list(self, input_fn: str):
        sentences = []
        with open(input_fn, 'rt') as fp:
            for line in fp.readlines():
                sentences.append(line.strip())
        return sentences


def _add_word_to_dictionary(spellchecker, wordlist_path):
    with open(wordlist_path, 'r') as fp:
        for line in fp.readlines():
            for word in line.split():
                spellchecker.add(word)


def _get_json_list(repos: (str, list, dict)) -> list:
    json_list = []
    if isinstance(repos, list):
        for repo in repos:
            json_list += glob(os.path.join(repo, '*.json'))
    elif isinstance(repos, str):
        json_list += glob(os.path.join(repos, '*.json'))
    elif isinstance(repos, dict):
        for name, path in repos.items():
            json_list += glob(os.path.join(path, '*.json'))
    else:
        raise NotImplementedError('Input type: {}'.format(type(repos)))
    return json_list


def _show_json(json_fn):
    with open(json_fn, 'r', encoding='utf-8-sig') as fp:
        report = json.load(fp)
        pprint(report)


def _report_pool(json_list):
    reports = []
    for idx, fn in enumerate(json_list):
        patient_id = os.path.basename(fn).split('.json')[0]
        with open(fn, 'r', encoding='utf-8-sig') as fp:
            case = json.load(fp)
            reports.append(dict())
            for accNO, item_dict in case.items():
                reports[idx]['PatientID'] = patient_id
                reports[idx]['AccessNo'] = item_dict['AccessNo']
                reports[idx]['ReportDr'] = item_dict['ReportDr']
                reports[idx]['ReportBody'] = item_dict['ReportBody']
                reports[idx]['OrderDescription'] = item_dict['OrderDescription']
    return reports


def _get_report_body(json_list: list, show=False) -> list:
    reportBody_list = []
    for jsf in json_list:
        with open(jsf, 'r', encoding='utf-8-sig') as fp:
            try:
                js_dict = json.load(fp)
            except:
                continue
            else:
                if show:
                    _show_json(jsf)
                    print()

            for key, value in js_dict.items():
                if isinstance(value, dict):
                    try:
                        raw_report = value.get('ReportBody', '')
                    except:
                        continue
                    report = re.split(r'\r\n', raw_report)
                    for r in report:
                        reportBody_list.append(r.strip())
                else:
                    if show:
                        print('{} contains no dictionary.'.format(key))
                        _show_json(jsf)
                    continue


    return reportBody_list


def correct_typo(input_str: str) -> str:
    """
    Hunspell spell checker for string of words or a sentence.
    :param input: a string
    :return: a string
    """
    _add_word_to_dictionary(spellchecker, radiology_wordlist)
    enc = spellchecker.get_dic_encoding()

    is_single_word = bool(len(re.split(r'\s+', input_str))==1)
    is_sentence = not is_single_word

    to_check = copy.deepcopy(input_str)

    if is_single_word:
        if to_check.isalpha() and (not to_check.isupper()):
            try:
                ok = spellchecker.spell(to_check)    # False if it is a typo.
            except UnicodeEncodeError:    # Not in a supported encoding.
                return to_check

            if not ok:
                suggestions = spellchecker.suggest(to_check)
                if suggestions:
                    try:
                        best = suggestions[0].decode(enc)
                    except:
                        best = suggestions[0]
                    corrected = best
                else:   # Hunspell has no suggestion, so send back the original word.
                    corrected = to_check
            else:  # Already correct.
                corrected = to_check
        else:  # Abbreviation or numbers
            corrected = to_check
        return corrected
    elif is_sentence:
        corrected_sentence = []
        words = re.split(r'\s+', to_check)
        for word in words:
            corrected_word = correct_typo(word)
            corrected_sentence.append(corrected_word)
        corrected_sentence = " ".join(corrected_sentence)
        return corrected_sentence


def sentence_splitter_regex(report_list: list, show=False) -> list:
    """
    A sentence tokenizer by regular expression.
    Return a list of sentences (string).
    """
    sentence_list = []
    filter = re.compile(r'(?:[Cc]hest.*?(?:shows|show|showed)\s*:?\s+)?([a-zA-Z]+.*?(?<!\d)(?<!Dr)(?<!DR)(?<!Bil)(?<!bil)(?<!esp)(?<!Esp)(?<!susp)(?<!Susp))[.,;:]')

    for report in report_list:
        if show:
            print('Original report: {}'.format(repr(report)))
        for line in report.splitlines():
            line = line.strip()
            match = filter.findall(line)
            if match:
                sentence_list += [m.strip() for m in match]

    sentence_list = list(dict.fromkeys(sentence_list))
    sentence_list.sort()

    return sentence_list


def sentence_splitter_nltk(reports: list) -> list:
    """
    A sentence tokenizer from NLTK
    """
    output = []
    for report in reports:
        sentences = sent_tokenize(report)
        output += [s.strip() for s in sentences]
    output = list(set(output))
    output.sort()
    return output


def get_raw_sentences(json_repo: (str, list, dict)) -> list:
    json_list = _get_json_list(json_repo)
    report_list = _get_report_body(json_list)
    raw_sentences = sentence_splitter_regex(report_list)
    return raw_sentences


def save_list(sentences_list, output):
    sentence_to_write = '\n'.join(sentences_list)
    with open(output, 'w') as fp:
        fp.write(sentence_to_write)


def load_list(input_fn):
    sentences = []
    with open(input_fn, 'r') as fp:
        for line in fp.readlines():
            sentences.append(line.strip())
    return sentences


def save_to_excel(input, output, overwrite=True):
    if overwrite and os.path.exists(output):
        os.remove(output)
    writer = ExcelWriter(output)
    input_df = DataFrame(input)
    input_df.to_excel(writer, 'sheet1')
    try:
        writer.save()
    except:
        print('Saving {} Failed.'.format(output))
    else:
        print('Saving {} succeeded.'.format(output))


def remove_similar_sentence(sentence_list: list, th: float) -> tuple:
    matcher = SequenceMatcher(lambda x: x in ' ,.;?()<>0123456789+-*/=!@#$%^&', ' ', ' ')
    similar = set()
    for a, b in filter(similar.isdisjoint, combinations(sentence_list, 2)):
        matcher.set_seqs(a, b)
        if th < matcher.ratio() < 1:
            similar.update([a, b])
            print('"{}"\n"{}"\nSimilarity: {:.3f}'.format(a, b, matcher.ratio()))
            print()
    dissimilar = list(set(sentence_list) - similar)
    dissimilar.sort()
    similar = list(set(similar))
    similar.sort()
    return dissimilar, similar


def match_sentence(sentence_list: list, th: float) -> list:
    matcher = SequenceMatcher(lambda x: x in ' ,.;?()<>0123456789+-*/=!@#$%^&', ' ', ' ')
    matchList = []
    for a in sentence_list:
        matchDict = defaultdict(list)   # record similarity mapping for each sentence
        for b in filter(lambda x: x != a, sentence_list):  # exclude a == b
            matcher.set_seqs(a.lower(), b.lower())
            similarity_ratio = matcher.ratio()

            if similarity_ratio > th:
                matchDict['sent_a'] = a
                matchDict['sent_b'].append({b:similarity_ratio})

                print('"{}"\n"{}"\nSequence similarity: {:.3f}'.format(a, b, similarity_ratio))
                print()

        if len(matchDict['sent_b']) > 0:
            matchList.append(matchDict)
    return matchList


def match_sentence_semantic(sentence_list: list, th: float) -> list:
    nlp = spacy.load('en')
    matchList = []
    for a in sentence_list:
        matchDict = defaultdict(list)   # record similarity mapping for each sentence
        for b in filter(lambda x: x != a, sentence_list):  # exclude a == b
            similarity_ratio = nlp(a.lower()).similarity(nlp(b.lower()))
            if similarity_ratio > th:
                matchDict['sent_a'] = a
                matchDict['sent_b'].append({b:similarity_ratio})

                print('"{}"\n"{}"\nSemantic similarity: {:.3f}'.format(a, b, similarity_ratio))
                print()

        if len(matchDict['sent_b']) > 0:
            matchList.append(matchDict)
    return matchList


def _mean_similarity(a: str, sentences: list):
    number = len(sentences)
    sentences_set = set(sentences)
    sent_a_set = set()
    sent_a_set.add(a)
    sent_b_set = sentences_set - sent_a_set
    sum = 0.0
    matcher.set_seq1(a)
    for b in sent_b_set:
        matcher.set_seq2(b)
        sum += matcher.ratio()
    avg = sum / (number - 1)
    return (a, avg)


def find_common_sentence(sentences: list, k: int) -> dict:
    pool = Pool(cpu_count())
    arg = [(s, sentences) for s in sentences]
    mean = pool.starmap_async(_mean_similarity, arg)
    pool.close()
    pool.join()
    mean = sorted(mean.get(), key=itemgetter(1), reverse=True)
    result = dict(mean[i] for i in range(k))
    return result



def main():
    name = 'YFC'
    th = 0.96
    number = 3
    start = 0

    for i in range(start, start + number):
        try:
            with open('/data/CXR8/NTUH/{}_CommonSent_{}.json'.format(name, i-1), 'rt', encoding='utf-8-sig') as fp:
                sentences = json.load(fp)
        except:
            sentences = Report(name.lower()).corrected_sentences

        sentences = list(set(sentences))
        #matchList = match_sentence(sentences, th)
        matchList = match_sentence_semantic(sentences, th)
        count_matchList = sorted([(d['sent_a'], len(d['sent_b'])) for d in matchList],
                                 key=lambda x: x[1], reverse=True)
        # exclude the sentence of only single match
        common_sent = sorted([s for (s ,c) in count_matchList if c > 1])

        with open('/data/CXR8/NTUH/{}_MatchList_{}.json'.format(name, i), 'wt') as fp:
            json.dump(matchList, fp, indent=4, sort_keys=True)
        with open('/data/CXR8/NTUH/{}_MatchCount_{}.json'.format(name, i), 'wt') as fp:
            json.dump(count_matchList, fp, indent=4, sort_keys=True)
        with open('/data/CXR8/NTUH/{}_CommonSent_{}.json'.format(name, i), 'wt') as fp:
            json.dump(common_sent, fp, indent=4, sort_keys=True)


    pprint(count_matchList[:50])
    print('There are {} common sentence.'.format(len(common_sent)))


    return





if __name__ == '__main__':
    status = main()
    sys.exit(status)
