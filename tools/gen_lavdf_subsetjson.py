import json


with open('data/lavdf/annotations/metadata.json', 'r') as fid:
    gt = json.load(fid)
dict_data = []
for value in gt:
    if (not value['modify_video']) and (value['modify_audio']):
        continue
    dict_data.append(value)
    
with open("data/lavdf/annotations/metadata_vmsub.json","w") as f:
    json.dump(dict_data,f)
    print("加载入文件完成...")