import numpy as np
import matplotlib.pyplot as plt
from .eval_proposal import ANETproposal
from .eval_detection import ANETdetection
# from .postprocess_utils import multithread_detection , get_infer_dict, load_json
import os
from joblib import Parallel, delayed
import json
import pandas as pd
# def run_evaluation(ground_truth_file, proposal_file, dataset_name='',
#                    max_avg_nr_proposal=100,
#                    tiou_thre=np.linspace(0.5, 1.0, 11), subset='test'):
#     anet_proposal = ANETproposal(ground_truth_file, proposal_file,
#                                  dataset_name=dataset_name,
#                                  tiou_thresholds=tiou_thre, max_avg_nr_proposals=max_avg_nr_proposal,
#                                  subset=subset, verbose=True, check_status=False)

#     anet_proposal.evaluate()

#     recall = anet_proposal.recall
#     average_recall = anet_proposal.avg_recall
#     average_nr_proposal = anet_proposal.proposals_per_video

#     return (average_nr_proposal, average_recall, recall)

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def plot_metric(args, average_nr_proposals, average_recall, recall, tiou_thresholds=np.linspace(0.5, 1.0, 11)):
    fn_size = 14
    plt.figure(num=None, figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    colors = ['k', 'r', 'yellow', 'b', 'c', 'm', 'b', 'pink', 'lawngreen', 'indigo']
    area_under_curve = np.zeros_like(tiou_thresholds)
    for i in range(recall.shape[0]):
        area_under_curve[i] = np.trapz(recall[i], average_nr_proposals)

    for idx, tiou in enumerate(tiou_thresholds[::2]):
        ax.plot(average_nr_proposals, recall[2 * idx, :], color=colors[idx + 1],
                label="tiou=[" + str(tiou) + "],area=" + str(int(area_under_curve[2 * idx] * 100) / 100.),
                linewidth=4, linestyle='-', marker=None)

    ax.plot(average_nr_proposals, average_recall, color=colors[0],
            label="tiou=0.5:0.1:1.0," + "area=" + str(int(np.trapz(average_recall, average_nr_proposals) * 100) / 100.),
            linewidth=4, linestyle='-', marker=None)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1]] + handles[:-1], [labels[-1]] + labels[:-1], loc='best')

    plt.ylabel('Average Recall', fontsize=fn_size)
    plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
    plt.grid(b=True, which="both")
    plt.ylim([0, 1.0])
    plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
    plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
    plt.savefig(os.path.join(args.output["work_dir"],args.model_name,args.output["output_path"], args.eval["save_fig_path"]))


def evaluation_proposal(gt_filename,pred_filename,tious,subset,max_avg_nr_proposal=100):
    anet_proposal = ANETproposal(gt_filename,pred_filename,
                                tiou_thresholds=tious, max_avg_nr_proposals=max_avg_nr_proposal,
                                subset=subset, verbose=True, check_status=False)

    anet_proposal.evaluate()

    recall = anet_proposal.recall
    average_recall = anet_proposal.avg_recall
    average_nr_proposal = anet_proposal.proposals_per_video

    # print("AR@10 is \t", np.mean(recall[:, 9]))
    # print("AR@20 is \t", np.mean(recall[:, 19]))
    # print("AR@50 is \t", np.mean(recall[:, 49]))
    # print("AR@100 is \t", np.mean(recall[:, 99]))
    result = f'Proposal: AR@10 {np.mean(recall[:, 9])*100:.3f} \t'
    result+=f'AR@20 {np.mean(recall[:, 19])*100:.3f} \t'
    result+=f'AR@50 {np.mean(recall[:, 49])*100:.3f} \t'
    result+=f'AR@100 {np.mean(recall[:, 99])*100:.3f} \t'
    with open(pred_filename.replace('.json','.txt'), 'a') as fobj:
        fobj.write(f'{result}\n')
    return (np.mean(recall[:, 9])+np.mean(recall[:, 19])+np.mean(recall[:, 49])+np.mean(recall[:, 99]))/4*100

def evaluation_detection(gt_filename,pred_filename,tious,subset):
    anet_detection = ANETdetection(
    ground_truth_filename=gt_filename,
    prediction_filename=pred_filename,
    tiou_thresholds=tious,
    subset=subset, verbose=True, check_status=False)
    anet_detection.evaluate()

    mAP_at_tIoU = [f'mAP@{t:.2f} {mAP*100:.3f}' for t, mAP in zip(anet_detection.tiou_thresholds, anet_detection.mAP)]
    results = f'Detection: average-mAP {anet_detection.average_mAP*100:.3f} {" ".join(mAP_at_tIoU)}'
    print(results)
    with open(pred_filename.replace('.json','.txt'), 'a') as fobj:
        fobj.write(f'{results}\n')
    return np.mean(anet_detection.mAP)*100


def detection_thread(vid,pred_data,cls_data_cls):
    proposal_list = []
    old_df = pred_data[pred_data.video_name == vid]
    # print(df)
    df = pd.DataFrame()
    df['score'] = old_df.score.values[:]
    df['label'] = old_df.label.values[:]
    df['xmin'] = old_df.xmin.values[:]
    df['xmax'] = old_df.xmax.values[:]
    best_score=np.max(cls_data_cls[vid])
    for j in range(min(100, len(df))):
            tmp_proposal = {}
            tmp_proposal["label"] = 'Fake'
            tmp_proposal["score"] = float(df.score.values[j])*best_score
            tmp_proposal["segment"] = [max(0, df.xmin.values[j]),
                                    df.xmax.values[j]]
            proposal_list.append(tmp_proposal)
    return {vid: proposal_list}

def post_process_multi(pred_data,output_file,cls_score_file=None):
    
    pred_videos = list(pred_data.video_name.values[:])
    pred_videos = set(pred_videos)
    cls_data_cls = {}
    if cls_score_file is not None:
        best_cls = load_json(cls_score_file)
    
        for idx, vid in enumerate(pred_videos):
            if vid in pred_videos:
                cls_data_cls[vid] = best_cls[vid]
    else:
        for idx, vid in enumerate(pred_videos):
            if vid in pred_videos:
                cls_data_cls[vid] = [1,1]        

    parallel = Parallel(n_jobs=16, prefer="processes")
    detection = parallel(delayed(detection_thread)(vid, pred_data,cls_data_cls)
                        for vid in pred_videos)
    detection_dict = {}
    [detection_dict.update(d) for d in detection]

    output_dict = {"version": "ANET v1.3, Lavdf", "results": detection_dict, "external_data": {}}

    with open(output_file, "w") as out:
        json.dump(output_dict, out)
        
        
def run_evaluation(preds, ground_truth_file, proposal_file, dataset_name='',
                   max_avg_nr_proposal=100,
                   tiou_thre=np.linspace(0.5, 1.0, 11), subset='test',cls_score_file=None):
    preds = pd.DataFrame({
                'video_name' : preds['video-id'],
                'xmin' : preds['t-start'].tolist(),
                'xmax': preds['t-end'].tolist(),
                'label': preds['label'].tolist(),
                'score': preds['score'].tolist()
            })
    print("saving detection results...")
    post_process_multi(preds,proposal_file,cls_score_file)
    print("evaluion detection results...")
    mAP=evaluation_detection(ground_truth_file,proposal_file,tiou_thre,subset)
    print("evaluion proposal results...")
    mAR=evaluation_proposal(ground_truth_file,proposal_file,tiou_thre,subset,max_avg_nr_proposal)
    return mAP,mAR
    
