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
from difflib import SequenceMatcher
from itertools import combinations
from wordnet_sentence_similarity import symmetric_sentence_similarity
from short_sentence_similarity import similarity


spellchecker = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
matcher = SequenceMatcher(lambda x: x in ' ,.;?()<>0123456789+-*/=!@#$%^&', ' ', ' ')
medical_wordlist = '/data/CXR8/NTUH/wordlist.txt'
radiology_wordlist = '/data/CXR8/NTUH/radiology_word.txt'
json_repo = {'yfc':'/data/CXR8/NTUH/YFC_Reports', 'ycc':'/data/CXR8/NTUH/YCC_Reports'}
root = '/data/CXR8/NTUH/'
report_xls = '/data/CXR8/NTUH/YFC_reports.xls'
raw_output = '/data/CXR8/NTUH/YFC_raw.txt'
corrected_output = '/data/CXR8/NTUH/YFC_corrected.txt'
final_output = '/data/CXR8/NTUH/YFC_80_wordnet.txt'


def _add_word_to_dictionary(spellchecker, wordlist):
    with open(wordlist, 'r') as fp:
        for line in fp.readlines():
            for word in line.split():
                spellchecker.add(word)


def _correct_word(spellchecker, words, add_to_dict = None):
    enc = spellchecker.get_dic_encoding()
    wrong = []
    corrected = []

    if add_to_dict is not None:
        if isinstance(add_to_dict, (list, set, tuple)):
            for w in add_to_dict:
                spellchecker.add(w)
        elif isinstance(add_to_dict, str):
            for w in add_to_dict.split():
                spellchecker.add(w)

    if isinstance(words, str):
        words_to_check = words.split()
        words_are_strings = True
    elif isinstance(words, (list, set, tuple)):
        words_to_check = copy.deepcopy(words)
        words_are_strings = False

    for word in words_to_check:
        if word.isalpha() and not word.isupper():
            try:
                ok = spellchecker.spell(word)
            except UnicodeEncodeError:
                corrected.append(word)
                continue

            if not ok:
                wrong.append(word)
                suggestions = spellchecker.suggest(word)

                if suggestions:
                    try:
                        best = suggestions[0].decord(enc)
                    except:
                        best = suggestions[0]

                    corrected.append(best)
                else:
                    corrected.append(word)
            else:
                corrected.append(word)
        else:
            corrected.append(word)

    if words_are_strings:
        corrected = " ".join(corrected)
    else:
        corrected = type(words)(corrected)

    wrong = list(dict.fromkeys(wrong))

    return corrected, wrong


def _get_json_list(repos):
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
        raise 'Wrong input type: {}'.format(type(repos))
    return json_list


def _show_json(json_fn):
    with open(json_fn, 'r', encoding='utf-8-sig') as fp:
        report_dict = json.load(fp)
        report_dict = json.dumps(report_dict, indent=4, ensure_ascii=False)
        print(report_dict)


def report_pool(json_list):
    json_pool = []
    for idx, jsf in enumerate(json_list):
        patient_id = os.path.basename(jsf).split('.json')[0]
        with open(jsf, 'r', encoding='utf-8-sig') as fp:
            js_dict = json.load(fp)
            json_pool.append(dict())
            for accNO, item_dict in js_dict.items():
                json_pool[idx]['PatientID'] = patient_id
                json_pool[idx]['AccessNo'] = item_dict['AccessNo']
                json_pool[idx]['ReportDr'] = item_dict['ReportDr']
                json_pool[idx]['ReportBody'] = item_dict['ReportBody']
                json_pool[idx]['OrderDescription'] = item_dict['OrderDescription']
    return json_pool


def _get_report_body(json_list, show=True):
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
                        reportBody_list.append(value.get('ReportBody', ''))
                    except:
                        continue
                else:
                    print('{} contains no dictionary.'.format(key))
                    continue


    return reportBody_list


def save_to_excel(input, output, overwrite=True):
    if overwrite and os.path.exists(output):
        os.remove(output)
    writer = ExcelWriter(output)
    pool_df = DataFrame(input)
    pool_df.to_excel(writer, 'sheet1')
    writer.save()


def _extract_sentences(report_list, show=True):
    sentence_list = []
    filter1 = re.compile(r'.*(?:EXAMINATION:\s*)(?P<exam>.*)(?:\r\n)(?:FINDINGS:\r\n)(?P<finding>.*$)')
    filter2 = re.compile(r'(\b[^\s].*?\.)')
    filter3 = re.compile(r'(\b[^\s\d,.;\(\)<>\-\*\?]+?.+?[.;)\?]+?)')

    for report in report_list:
        if show:
            print('Original report: {}'.format(repr(report)))
        for line in report.splitlines():
            line = line.strip()

            if show:
                print('After splitline and strip: {}'.format(line))

            matchlist = filter3.findall(line)

            if show:
                print('Candidate sentences: {}'.format(matchlist))
                print()

            for m in matchlist:
                m = m.strip()
                if len(m) > 5 and len(m.split()) > 2:
                    sentence_list.append(m)

    sentence_list = list(dict.fromkeys(sentence_list))
    sentence_list.sort()

    return sentence_list


def get_raw_sentences(json_repo):
    json_list = _get_json_list(json_repo)
    report_list = _get_report_body(json_list, show=False)
    raw_sentences = _extract_sentences(report_list, show=False)
    return raw_sentences


def get_corrected_sentences(raw_sentences):
    _add_word_to_dictionary(spellchecker, medical_wordlist)
    _add_word_to_dictionary(spellchecker, radiology_wordlist)
    corrected_sentences = []
    for sentence in raw_sentences:
        corrected, _ = _correct_word(spellchecker, sentence)
        corrected_sentences.append(corrected)
    return corrected_sentences


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


def grouping_sentence_method1(matcher, sentence_list, th):
    similar = set()
    for a, b in filter(similar.isdisjoint, combinations(sentence_list, 2)):
        matcher.set_seqs(a, b)

        if th < matcher.ratio() < 1:
            shorter = a if len(b) > len(a) else b
            similar.add(shorter)
            print('"{}"\n"{}"\nSimilarity: {:.3f}'.format(a, b, matcher.ratio()))
            print('\n==> Remove "{}"'.format(shorter))
            print()
    dissimilar = list(set(sentence_list) - similar)
    dissimilar.sort()
    return dissimilar, similar


def grouping_sentence_method2(sentence_list, th):
    similar = set()
    for a, b in filter(similar.isdisjoint, combinations(sentence_list, 2)):
        similarity = symmetric_sentence_similarity(a, b)

        if th < similarity < 1:
            shorter = a if len(b) > len(a) else b
            similar.add(shorter)
            print('"{}"\n"{}"\nSimilarity: {:.3f}'.format(a, b, similarity))
            print('\n==> Remove "{}"'.format(shorter))
            print()
    dissimilar = list(set(sentence_list) - similar)
    dissimilar.sort()
    return dissimilar, similar


def grouping_sentence_method3(sentence_list, th, info_content_norm):
    similar = set()
    for a, b in filter(similar.isdisjoint, combinations(sentence_list, 2)):
        similar_ratio = similarity(a, b, info_content_norm)

        if th < similar_ratio < 1:
            shorter = a if len(b) > len(a) else b
            similar.add(shorter)
            print('"{}"\n"{}"\nSimilarity: {:.3f}'.format(a, b, similar_ratio))
            print('\n==> Remove "{}"'.format(shorter))
            print()
    dissimilar = list(set(sentence_list) - similar)
    dissimilar.sort()
    return dissimilar, similar


def main():
    raw = get_raw_sentences(json_repo['yfc'])
    corrected = get_corrected_sentences(raw)
    #dissimilar, similar = remove_similar_sentence(matcher, corrected, 0.6)
    dissimilar, similar = grouping_sentence_method3(corrected[:100], 0.6, True)

    print('{} sentences are preserved.'.format(len(dissimilar)))

    save_list(corrected, corrected_output)
    save_list(dissimilar, final_output)






if __name__ == '__main__':
    status = main()
    sys.exit(status)
