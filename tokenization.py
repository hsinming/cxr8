#!/usr/bin/python3
#encoding=utf-8

import sys
from pprint import pprint
from segtok.segmenter import split_multi
from report import json_repo
from report import _get_json_list, _get_report_body



def main():
    json_files = _get_json_list(json_repo['wjl'])
    reports = _get_report_body(json_files, show=False)
    sentences = split_multi(reports[6])
    for s in sentences:
        print(s)



if __name__ == '__main__':
    status = main()
    sys.exit(status)