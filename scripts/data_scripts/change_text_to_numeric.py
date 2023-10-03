#!/usr/bin/env python3
import sys
import os
import io
from sklearn.preprocessing import LabelEncoder

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 change_text_to_numeric.py data-file\n")
    sys.exit(1)

f_input = sys.argv[1]
f_output = os.path.join("data", "stage3", "train.csv")
os.makedirs(os.path.join("data", "stage3"), exist_ok=True)

def process_data(fd_in, fd_out):
    arr_experience_level = []
    arr_job_title = []
    arr_salary_in_usd = []
    arr_company_size = []

    for line in fd_in:
        line = line.rstrip('\n').split(',')
        arr_experience_level.append(line[0])
        arr_job_title.append(line[1])
        arr_salary_in_usd.append(line[2])
        arr_company_size.append(line[3])

    for i in range(len(arr_company_size)):
        if arr_company_size[i] == 'L':
            arr_company_size[i] = 2
        elif arr_company_size[i] == 'M':
            arr_company_size[i] = 1            
        else:
            arr_company_size[i] = 0
            
    for i in range(len(arr_experience_level)):
        if arr_experience_level[i] == 'SE':
            arr_experience_level[i] = 3
        elif arr_experience_level[i] == 'MI':
            arr_experience_level[i] = 2       
        elif arr_experience_level[i] == 'EN':
            arr_experience_level[i] = 1        
        else:
            arr_experience_level[i] = 0 
        
    label_encoder = LabelEncoder()
    arr_job_title = label_encoder.fit_transform(arr_job_title)          

    for p_experience_level, p_job_title, p_salary_in_usd, p_company_size in zip(arr_experience_level, arr_job_title, arr_salary_in_usd, arr_company_size):
        fd_out.write("{},{},{},{}\n".format(p_experience_level, p_job_title, p_salary_in_usd, p_company_size))

with io.open(f_input, encoding="utf8") as fd_in:
    with io.open(f_output, "w", encoding="utf8") as fd_out:
        process_data(fd_in, fd_out)