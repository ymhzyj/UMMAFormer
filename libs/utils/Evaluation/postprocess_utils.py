import sys
import numpy as np
import pandas as pd
import json
import os
from joblib import Parallel, delayed

import yaml


# with open("./config/anet.yaml", 'r', encoding='utf-8') as f:
#         tmp = f.read()
#         config = yaml.load(tmp, Loader=yaml.FullLoader)


# vid_path = config['dataset']['test']['visual_feature_path']
# nms_thresh = config['testing']['nms_thresh'] 



def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def get_infer_dict(vid_anno,subset='test'):
    # df = pd.read_csv(vid_info)
    database = load_json(vid_anno)
    video_dict = {}
    for i in range(len(database)):
        video_name = video_name = os.path.splitext(os.path.basename(database[i]['file']))[0]
        # if os.path.exists(os.path.join(vid_path+"/test",video_name+".npy")):
        video_info = database[i]
        video_new_info = {}
        video_new_info['duration_frame'] = video_info['video_frames']
        video_new_info['duration_second'] = video_info['duration']
        video_new_info["feature_frame"] = video_info['video_frames']
        video_subset = video_info["split"]
        video_anno = video_info['fake_periods']
        video_new_info['fake_periods'] = video_info['fake_periods']
        if len(video_anno) > 0:
            video_label = 'Fake'
            if video_subset == subset:
                    video_dict[video_name] = video_new_info
    return video_dict



def Soft_NMS(df, nms_threshold=1e-5, num_prop=100):
 
    df = df.sort_values(by="score", ascending=False)

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])
    tlabel = list(df.label.values[:])

    rstart = []
    rend = []
    rscore = []
    rlabel = []


    while len(tscore) > 1 and len(rscore) < num_prop and max(tscore)>0:
        max_index = tscore.index(max(tscore))
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx], tend[idx])
                if tmp_iou > 0:
                    tscore[idx] = tscore[idx] * (np.exp(-np.square(tmp_iou)*10) / nms_threshold)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        rlabel.append(tlabel[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)
        tlabel.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    newDf['label'] = rlabel

    return newDf



def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - min(s1, s2)
    return float(Aand) / (Aor - Aand + (e2-s2))



def multithread_detection(video_name, video_cls, pred_prop):
    
    old_df = pred_prop[pred_prop.video_name == video_name]
    # print(df)
    
    df = pd.DataFrame()
    df['score'] = old_df.score.values[:]
    df['label'] = old_df.label.values[:]
    df['xmin'] = old_df.xmin.values[:]
    df['xmax'] = old_df.xmax.values[:]

    best_score = np.max(video_cls)

    # if len(df) > 1:
    #     df = Soft_NMS(df, nms_thresh)
    df = df.sort_values(by="score", ascending=False)
    proposal_list = []

    for j in range(min(100, len(df))):
            tmp_proposal = {}
            tmp_proposal["label"] = 'Fake'
            tmp_proposal["score"] = float(df.score.values[j])*best_score
            tmp_proposal["segment"] = [max(0, df.xmin.values[j]),
                                    min(1, df.xmax.values[j])]
            proposal_list.append(tmp_proposal)

    return {video_name: proposal_list}






