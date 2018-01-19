# -*- coding: utf-8 -*-
import csv
import random
from collections import OrderedDict

source_path = "/data/CXR8/Data_Entry_2017.csv"
dist_path_train = "./Train_Label_simple.csv"
dist_path_val = "./Val_Label_simple.csv"
dist_path_test = "./Test_Label_simple.csv"

INCLUDE_NO_FINDING = True


disease_categories = {
        'Atelectasis': 0,
        'Cardiomegaly': 1,
        'Effusion': 2,
        'Infiltration': 3,
        'Mass': 4,
        'Nodule': 5,
        'Pneumonia': 6,
        'Pneumothorax': 7,
        'Consolidation': 8,
        'Edema': 9,
        'Emphysema': 10,
        'Fibrosis': 11,
        'Pleural_Thickening': 12,
        'Hernia': 13,
        }

# Re-classify the disease into common imaging findings.
finding_categories = OrderedDict()
finding_categories['Opacity'] = ['Atelectasis', 'Pneumonia', 'Consolidation']
finding_categories['Infiltration'] = ['Infiltration']
finding_categories['Edema'] = ['Edema']
finding_categories['Cardiomegaly'] = ['Cardiomegaly']
finding_categories['Effusion'] = ['Effusion']
finding_categories['Tumor'] = ['Mass', 'Nodule']
finding_categories['Pneumothorax'] = ['Pneumothorax']
finding_categories['Emphysema'] = ['Emphysema']
finding_categories['Fibrosis'] = ['Fibrosis']
finding_categories['Pleural_Thickening'] = ['Pleural_Thickening']
finding_categories['Hernia'] = ['Hernia']

# Another mapping of imaging finding
disease_to_finding = {disease:k for k, v in finding_categories.items() for disease in v}

if __name__ == '__main__':
    with open(source_path) as f:
        with open(dist_path_train, "w+", newline='') as wf_train:
            with open(dist_path_val, "w+", newline='') as wf_val:
                with open(dist_path_test, "w+", newline='') as wf_test:
                    writer_train = csv.writer(wf_train)
                    writer_val = csv.writer(wf_val)
                    writer_test = csv.writer(wf_test)
                    lines = f.read().splitlines()
                    #lines = lines[0:100]

                    # remove the title
                    del lines[0]

                    col = ['FileName'] + list(finding_categories.keys())

                    writer_train.writerow(col)
                    writer_val.writerow(col)
                    writer_test.writerow(col)
                    random.shuffle(lines)

                    # number of films
                    line_number = len(lines)

                    for i in range(line_number):
                        split = lines[i].split(',')
                        file_name = split[0]
                        label_string = split[1]
                        if not INCLUDE_NO_FINDING and label_string == 'NO Finding':
                            continue

                        labels = label_string.split('|')
                        vector = [0 for _ in range(len(finding_categories))]
                        for label in labels:
                            # Exclude "No Finding" because it is not accurate.
                            if label != "No Finding":
                                vector[list(finding_categories.keys()).index(disease_to_finding[label])] = 1
                        vector.insert(0, file_name)
                        if i <= line_number*0.7:
                            writer_train.writerow(vector)#70%
                        elif i > line_number*0.7 and i <= line_number*0.9:
                            writer_val.writerow(vector)#20%
                        else :
                            writer_test.writerow(vector)#10%
    print("Label data generated")
                    
