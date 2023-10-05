#!/usr/bin/env python3
import os
import sys
import pickle

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


if len(sys.argv) != 4:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train.py data-file model algo\n")
    sys.exit(1)

data_file = sys.argv[1]
model_file = sys.argv[2]
algorithm = sys.argv[3]


df = pd.read_csv(data_file)
X = df.iloc[:, [0, 1, 3]]
y = df.iloc[:, 2]

if algorithm == "linear":
    clf = LinearRegression()
elif algorithm == "tree":
    clf = DecisionTreeRegressor()
else:
    sys.stderr.write("Invalid algorithm. Supported algorithms: linear, tree\n")
    sys.exit(1)

clf.fit(X, y)

with open(model_file, "wb") as fd:
    pickle.dump(clf, fd)

