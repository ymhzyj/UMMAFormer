import json
import csv
import numpy as np
import os
import numpy as np
from sklearn import metrics

with open('paper_results/val_results.json', 'r') as fid:
    test_results = json.load(fid)

test_dict={}
for k,res in test_results['results'].items():
    test_dict[k]=res[0]['score']


with open('data/lavdf/annotations/metadata.json', 'r') as fid:
    gt = json.load(fid)
dict_data = {}
ious=[]
for item in gt:
    if item['split']=='test':
        k = os.path.basename(item['file'])[:-4]
        if(item['n_fakes']>0):
            dict_data[k]=1
        else:
            dict_data[k]=0

gts=[]
preds=[]
for k,v in dict_data.items():
    gts.append(v)
    preds.append(test_dict[k])
    

y = np.array(gts)
pred = np.array(preds)
fpr, tpr, thresholds = metrics.roc_curve(y, pred)
print(metrics.auc(fpr, tpr))




