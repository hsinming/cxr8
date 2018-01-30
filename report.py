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
from nltk import sent_tokenize
from wordnet_sentence_similarity import symmetric_sentence_similarity
from short_sentence_similarity import similarity
from sentence import sentence_tokenizer


spellchecker = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
spellchecker.add_dic('/usr/share/hunspell/en_med_glut.dic')
matcher = SequenceMatcher(lambda x: x in ' ,.;?()<>0123456789+-*/=!@#$%^&', ' ', ' ')
medical_wordlist = '/data/CXR8/NTUH/wordlist.txt'
radiology_wordlist = '/data/CXR8/NTUH/radiology_word.txt'
json_repo = {'yfc': '/data/CXR8/NTUH/YFC_Reports',
             'ycc': '/data/CXR8/NTUH/YCC_Reports',
             'jyc': '/data/CXR8/NTUH/JYC_Reports',
             'wjl': '/data/CXR8/NTUH/WJL_Reports'}
root = '/data/CXR8/NTUH/'
report_xls = '/data/CXR8/NTUH/YCC_reports.xls'
raw_output = '/data/CXR8/NTUH/WJL_raw.txt'
corrected_output = '/data/CXR8/NTUH/WJL_corrected.txt'
final_output = '/data/CXR8/NTUH/WJL_80.txt'


def _split_into_sentences(text):
    caps = "([A-Z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt|Jr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|me|edu)"

    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    if "..." in text: text = text.replace("...", "<prd><prd><prd>")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def _add_word_to_dictionary(spellchecker, wordlist_path):
    with open(wordlist_path, 'r') as fp:
        for line in fp.readlines():
            for word in line.split():
                spellchecker.add(word)


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
        raise NotImplementedError('Input type: {}'.format(type(repos)))
    return json_list


def _show_json(json_fn):
    with open(json_fn, 'r', encoding='utf-8-sig') as fp:
        report_dict = json.load(fp)
        report_dict = json.dumps(report_dict, indent=4, ensure_ascii=False)
        print(report_dict)


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


def _get_report_body(json_list, show=False):
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


def _extract_sentences(report_list, show=False):
    sentence_list = []
    filter = re.compile(r'[a-zA-Z]+.+?[^\d][.;]')

    for report in report_list:
        if show:
            print('Original report: {}'.format(repr(report)))
        for line in report.splitlines():
            line = line.strip()

            if show:
                print('After splitline and strip: {}'.format(repr(line)))

            match = filter.findall(line)
            if match:
                for m in match:
                    if len(re.split(r'\s+', m.strip())) > 1:
                        sentence_list.append(m.strip())

    sentence_list = list(dict.fromkeys(sentence_list))
    sentence_list.sort()

    return sentence_list


def correct_typo(spellchecker, input):
    assert isinstance(input, (str, list))
    _add_word_to_dictionary(spellchecker, radiology_wordlist)
    enc = spellchecker.get_dic_encoding()
    output = []
    is_list = isinstance(input, list)
    is_string = isinstance(input, str)
    if is_string:
        is_single_word = not bool(re.search(r'\s+', input))   # not split by space
    if is_list:
        is_single_word = False
    is_sentence = bool(is_string and not is_single_word)

    to_check = copy.deepcopy(input)

    if is_single_word:
        if to_check.isalpha() and not to_check.isupper():
            try:
                ok = spellchecker.spell(to_check)    # False if it is a typo.
            except UnicodeEncodeError:    # Not in a supported encoding.
                output.append(to_check)
                return output

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
        output.append(corrected)
    elif is_sentence:  # a sentence
        corrected_sentence = []
        words = re.split(r'\s+', to_check)
        for word in words:
            corrected_word = correct_typo(spellchecker, word)[0]
            corrected_sentence.append(corrected_word)
        corrected_sentence_string = " ".join(corrected_sentence)
        output.append(corrected_sentence_string)
    elif is_list:   # a list of sentences
        for sentence in to_check:
            corrected_sentence = correct_typo(spellchecker, sentence)  # a list has a single sentence
            output += corrected_sentence

    return output


def sentence_splitter(report, tokenizer):
    is_list = isinstance(report, list)
    is_str = isinstance(report, str)
    sentence_list = []

    if is_list:
        for each in report:
            sentence_list += sentence_splitter(each, tokenizer)
    if is_str :
        sentences = tokenizer.tokenize(report)
        for s in sentences:
            if len(re.split(r'\s+', s.strip())) > 1:
                sentence_list.append(s.strip())
    sentence_list = list(dict.fromkeys(sentence_list))
    sentence_list.sort()

    return sentence_list


def get_raw_sentences(json_repo):
    json_list = _get_json_list(json_repo)
    report_list = _get_report_body(json_list)
    #raw_sentences = _extract_sentences(report_list)
    raw_sentences = sentence_splitter(report_list, sentence_tokenizer())
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


def grouping_sentence_method1(matcher, sentence_list, th, keep):
    assert keep in ['shorter', 'longer']
    similar = set()
    for a, b in filter(similar.isdisjoint, combinations(sentence_list, 2)):
        matcher.set_seqs(a, b)

        if th < matcher.ratio() < 1:
            shorter = a if len(b) > len(a) else b
            longer = a if len(a) > len(b) else b
            if keep == 'shorter': remove = longer
            if keep == 'longer': remove = shorter
            similar.add(remove)
            print('"{}"\n"{}"\nSimilarity: {:.3f}'.format(a, b, matcher.ratio()))
            print('==> Remove "{}"'.format(remove))
            print()
    dissimilar = list(set(sentence_list) - similar)
    dissimilar.sort()
    return dissimilar


def grouping_sentence_method2(sentence_list, th, keep):
    assert keep in ['shorter', 'longer']
    similar = set()
    for a, b in filter(similar.isdisjoint, combinations(sentence_list, 2)):
        similarity = symmetric_sentence_similarity(a, b)

        if th < similarity < 1:
            shorter = a if len(b) > len(a) else b
            longer = a if len(a) > len(b) else b
            if keep == 'shorter': remove = longer
            if keep == 'longer': remove = shorter
            similar.add(remove)
            print('"{}"\n"{}"\nSimilarity: {:.3f}'.format(a, b, similarity))
            print('==> Remove "{}"'.format(remove))
            print()
    dissimilar = list(set(sentence_list) - similar)
    dissimilar.sort()
    return dissimilar


def grouping_sentence_method3(sentence_list, th, keep, info_content_norm):
    assert keep in ['shorter', 'longer']
    similar = set()
    for a, b in filter(similar.isdisjoint, combinations(sentence_list, 2)):
        similar_ratio = similarity(a, b, info_content_norm)

        if th < similar_ratio < 1:
            shorter = a if len(re.split(r'\W+',b)) > len(re.split(r'\W+',a)) else b
            longer = a if len(re.split(r'\W+',b)) < len(re.split(r'\W+',a)) else b
            if keep == 'shorter': remove = longer
            if keep == 'longer': remove = shorter
            similar.add(remove)
            print('"{}"\n"{}"\nSimilarity: {:.3f}'.format(a, b, similar_ratio))
            print('==> Remove "{}"'.format(remove))
            print()
    dissimilar = list(set(sentence_list) - similar)
    dissimilar.sort()
    return dissimilar


def main():
    raw_sentences = get_raw_sentences(json_repo['wjl'])
    corrected_sentences = correct_typo(spellchecker, raw_sentences)

    #dissimilar = grouping_sentence_method1(matcher, corrected_sentences, 0.6, 'shorter')
    dissimilar = grouping_sentence_method3(corrected_sentences, 0.8, 'shorter', True)

    print('{} sentences are preserved.'.format(len(dissimilar)))

    save_list(corrected_sentences, corrected_output)
    save_list(dissimilar, final_output)
    return





if __name__ == '__main__':
    status = main()
    sys.exit(status)
