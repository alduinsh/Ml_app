import os
import sys
import pickle
import json

import pandas as pd
from sklearn.linear_model import LinearRegression

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython evaluate.py data-file model\n")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])
X = df[['p_experience_level', 'p_job_title', 'p_company_size']]
y = df['p_salary_in_usd']

with open(sys.argv[2], "rb") as fd:
    clf = pickle.load(fd)

score = clf.score(X, y)

prc_file = os.path.join("evaluate", "score.json")
os.makedirs(os.path.join("evaluate"), exist_ok=True)

with open(prc_file, "w") as fd:
    json.dump({"score": score}, fd)
