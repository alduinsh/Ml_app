#!/usr/bin/env python3
import sys
import os
import io

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 fill_na.py data-file\n")
    sys.exit(1)

f_input = sys.argv[1]
f_output = os.path.join("data", "stage2", "train.csv")
os.makedirs(os.path.join("data", "stage2"), exist_ok=True)

def process_data(fd_in, fd_out):
    arr_experience_level = []
    arr_job_title = []
    arr_salary_in_usd = []
    arr_company_size = []

    for line in fd_in:
        line = line.rstrip('\n').split(',')
        arr_experience_level.append(line[0])
        arr_job_title.append(line[1])
        arr_company_size.append(line[3])

        if line[2].isdigit() and int(line[2]) < 300000:
            arr_salary_in_usd.append(int(line[2]))
        else:
            arr_salary_in_usd.append(0)

    s = sum(arr_salary_in_usd)
    salary_count = len(arr_salary_in_usd)
    
    for i in range(salary_count):
        if arr_salary_in_usd[i] == 0:
            arr_salary_in_usd[i] = round(s / salary_count, 3)

    for p_experience_level, p_job_title, p_salary_in_usd, p_company_size in zip(arr_experience_level, arr_job_title, arr_salary_in_usd, arr_company_size):
        fd_out.write("{},{},{},{}\n".format(p_experience_level, p_job_title, p_salary_in_usd, p_company_size))

with io.open(f_input, encoding="utf8") as fd_in:
    with io.open(f_output, "w", encoding="utf8") as fd_out:
        process_data(fd_in, fd_out)
