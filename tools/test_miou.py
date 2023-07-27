import json
import csv
import numpy as np


def calculate_iou(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return max(0.0, iou)
# thds = [x for x in np.linspace(0.05, 0.95, 20)]

# thds = [np.around(
#     np.linspace(0.05, 0.95, 20),  # numpy数组或列表
#     decimals=2  # 保留几位小数
# )]
thds = [0.05+0.05*i for i in range(0,19)]
thds = np.around(
    thds,  # numpy数组或列表
    decimals=2  # 保留几位小数
)
# thds = [x for x in np.linspace(0.05,0.95,19)]
split = 'test_cellular'
with open('paper_results/psynd_tsn_fpn_recoveryselfattentionnonormhrlrfullbackbone_2023_04_10_09_23_18/test_results.json'.format(split), 'r') as fid:
    test_results = json.load(fid)
mIoUs=[]
for thd in thds:
    test_dict={}
    for k,res in test_results['results'].items():
        segment = []
        for r_ in res:
            if r_['score']>thd:
                segment.append(r_['segment'])
        if len(segment)==0:
            continue
        # k = k.split('-')[0]
        test_dict[k]=segment

        
    # with open('data/SynDetect_gutenburg/test_annotation.csv', newline='') as csvfile:
    #     # 读取CSV文件内容
    #     reader = csv.reader(csvfile)
    #     # 将CSV文件内容转换为字典
    #     dict_data = {rows[0]: [[float(rows[1]),float(rows[2])]] for rows in reader}

    with open('data/Psynd/annotations/metadatabyola.json', 'r') as fid:
        gt = json.load(fid)
    dict_data = {}
    ious=[]
    for k,res in gt.items():
        if res['split']==split:
            segment = res['fake_periods']
            durations = res['duration']
            audio_frames = res['audio_frames']
            gt_frames = np.zeros(audio_frames)
            for seg in segment:
                start_index = int((seg[0]/durations)*audio_frames)
                end_index = int((seg[1]/durations)*audio_frames)
                gt_frames[start_index:end_index] = 1
            pred_frames = np.zeros(audio_frames)
            if k in test_dict.keys():
                for seg in test_dict[k]:
                    start_index = int((seg[0]/durations)*audio_frames)
                    end_index = int((seg[1]/durations)*audio_frames)
                    pred_frames[start_index:end_index] = 1
            tp = (gt_frames == pred_frames).sum()
            fp = (gt_frames != pred_frames).sum()
            iou = tp / (tp + 2*fp)
            ious.append(iou)
    # ious = list()
    # not right



    mi = np.mean(ious) * 100.0
    mIoUs.append(mi)
print("{} iou:{}".format(split,np.mean(mIoUs)))
print("{} iou best:{}".format(split,np.max(mIoUs)))

