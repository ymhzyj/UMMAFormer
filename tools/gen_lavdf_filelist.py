import json
from tqdm import tqdm
import os

json_file='data/lavdf/annotations/metadata.json'
with open(json_file, 'r') as fid:
    json_data = json.load(fid)
json_db = json_data

# if label_dict is not available
# if self.label_dict is None:
#     # label_dict = {}
#     label_dict = {'real':0, 'fake': 1}


# fill in the db (immutable afterwards)
train_list=[]
dev_list=[]
test_list=[]
for value in tqdm(json_db):
    key = os.path.splitext(os.path.basename(value['file']))[0]
    # skip the video if not in the split
    if value['split'].lower() in 'train':
        train_list.append(os.path.join('data/lavdf/video',value['file']+'\n'))
    elif value['split'].lower() in 'dev':
        dev_list.append(os.path.join('data/lavdf/video',value['file']+'\n'))
    else:
        test_list.append(os.path.join('data/lavdf/video',value['file']+'\n'))

# end=len(dev_list)//2000
# for i in range(0,end):
#     f=open("data/lavdf/filelist/dev_{}.txt".format(i),"w")
#     f.writelines(dev_list[2000*i:2000*(i+1)])
#     f.close()

# f=open("data/lavdf/filelist/dev_{}.txt".format(end),"w")
# f.writelines(dev_list[2000*end:])
# f.close()

# end=len(test_list)//2000
# for i in range(0,end):
#     f=open("data/lavdf/filelist/test_{}.txt".format(i),"w")
#     f.writelines(test_list[2000*i:2000*(i+1)])
#     f.close()

# f=open("data/lavdf/filelist/test_{}.txt".format(end),"w")
# f.writelines(test_list[2000*end:])
# f.close()

f=open("data/lavdf/filelist/train.txt","w")
 
f.writelines(train_list)
f.close()

f=open("data/lavdf/filelist/dev.txt","w")
 
f.writelines(dev_list)
f.close()
f=open("data/lavdf/filelist/test.txt","w")
 
f.writelines(test_list)
f.close()