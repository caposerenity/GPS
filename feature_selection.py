import numpy as np
import torch
import tqdm
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def select_id_relevant(memory):
    X = np.array(memory.cpu())
    y = np.arange(len(memory))
    #lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    lsvc = LogisticRegression(C=1).fit(X,y)
    model = SelectFromModel(lsvc, prefit=True)
    id_relevant = model.get_support()
    print(id_relevant)
    return id_relevant
