import os
import json
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
from ..utils import remove_duplicate_annotations

@register_dataset("tvil")
class TVILDataset(Dataset):
    def __init__(
        self,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        feat_folder,      # folder for features
        audio_feat_folder, # folder for audio features
        json_file,        # json file for annotations
        feat_stride,      # temporal stride of the feats
        num_frames,       # number of frames for each feat
        default_fps,      # default fps
        downsample_rate,  # downsample rate for feats
        max_seq_len,      # maximum sequence length during training
        trunc_thresh,     # threshold for truncate an action segment
        crop_ratio,       # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,        # input feat dim
        audio_input_dim, # input audio feat dim
        num_classes,      # number of action categories
        file_prefix,      # feature file prefix if any
        file_ext,         # feature file extension if any
        audio_file_ext,  # audio feature file extension if any
        force_upsampling  # force to upsample to max_seq_len
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        self.audio_feat_folder= audio_feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.audio_file_ext=audio_file_ext
        self.json_file = json_file

        # anet uses fixed length features, make sure there is no downsampling
        self.force_upsampling = force_upsampling

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.audio_input_dim=audio_input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = {'Fake':0}
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db = self._load_json_db(self.json_file)
        # proposal vs action categories
        assert (num_classes == 1)
        self.data_list = dict_db
        
        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'VIL',
            'tiou_thresholds': np.linspace(0.5, 0.95, 10),
            'empty_label_ids': []
        }
        print("{} subset has {} videos".format(self.split,len(self.data_list)))
    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_db = json.load(fid)

        # if label_dict is not available
        # if self.label_dict is None:
        #     label_dict = {}
        #     for key, value in json_db.items():
        #         for act in value['annotations']:
        #             label_dict[act['label']] = act['label_id']
        dict_db = tuple()
        # fill in the db (immutable afterwards)
        for key,value in json_db.items():
            # key =  os.path.splitext(os.path.basename(value['file']))[0]
            # skip the video if not in the split
            if value['split'].lower() not in self.split:
                continue
            if isinstance(self.file_prefix, list):
                assert len(self.file_prefix)==2
                feat_file = os.path.join(self.feat_folder, self.file_prefix[0], value['split'].lower(),
                                        value['file'][:-4] + self.file_ext)
            else:
                feat_file = os.path.join(self.feat_folder, self.file_prefix, value['split'].lower(),
                                        value['file'][:-4] + self.file_ext)
            if not os.path.exists(feat_file):
                continue
            
            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            elif 'video_frames' in value:
                fps = value['video_frames'] / value['duration']
            else:
                assert False, "Unknown video FPS."
            duration = value['duration']
            
            # video_labels=int(value['modify_video'])
            # audio_labels=int(value['modify_audio'])
            # av_labels = np.array([video_labels])
            # get annotations if available
            if ('fake_periods' in value) and (len(value['fake_periods']) > 0):
                valid_acts = value['fake_periods']
                num_acts = len(valid_acts)
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                for idx, act in enumerate(valid_acts):
                    segments[idx][0] = act[0]
                    segments[idx][1] = act[1]
                    labels[idx] = 0
            else:
                    segments = None
                    labels = None
            dict_db += ({'id': value['file'][:-4],
                         'fps' : fps,
                         'duration' : duration,
                         'split': value['split'].lower(),
                         'segments' : segments,
                         'labels' : labels,
            }, )

        return dict_db

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        
        if isinstance(self.file_prefix,list):
            filename1 = os.path.join(self.feat_folder,self.file_prefix[0], video_item['split'],
                                    video_item['id'] + self.file_ext)
            feats1 = np.load(filename1).astype(np.float32)      
            filename2 = os.path.join(self.feat_folder,self.file_prefix[1], video_item['split'],
                                    video_item['id'] + self.file_ext)
            feats2 = np.load(filename2).astype(np.float32)
            if feats1.shape[0] != feats2.shape[0]:
                feature_length=max(feats1.shape[0],feats1.shape[0])
                feats1 = np.resize(feats1,(feature_length,feats1.shape[1]))  
                feats2 = np.resize(feats2,(feature_length,feats2.shape[1]))
            feats= np.concatenate((feats1,feats2),axis=1) 
        else:
            filename = os.path.join(self.feat_folder,self.file_prefix , video_item['split'],
                                    video_item['id'] + self.file_ext)
            feats = np.load(filename).astype(np.float32)
        audio_feats= None
        if self.audio_feat_folder is not None:
            audio_filename = os.path.join(self.audio_feat_folder,video_item['split'],
                        video_item['id'] + self.file_ext)
            audio_feats = np.load(audio_filename)
        

        # we support both fixed length features / variable length features
        # case 1: variable length features for training
        if self.feat_stride > 0 and (not self.force_upsampling):
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
            if self.downsample_rate > 1:
                feats = feats[::self.downsample_rate, :]
                feat_stride = self.feat_stride * self.downsample_rate
        # case 2: variable length features for input, yet resized for training
        elif self.feat_stride > 0 and self.force_upsampling:
            feat_stride = float(
                (feats.shape[0] - 1) * self.feat_stride + self.num_frames
            ) / self.max_seq_len
            # center the features
            num_frames = feat_stride
        # case 3: fixed length features for input
        else:
            # deal with fixed length feature, recompute feat_stride, num_frames
            seq_len = feats.shape[0]
            assert seq_len <= self.max_seq_len
            if self.force_upsampling:
                # reset to max_seq_len
                seq_len = self.max_seq_len
            feat_stride = video_item['duration'] * video_item['fps'] / seq_len
            # center the features
            num_frames = feat_stride
        feat_offset = 0.5 * num_frames / feat_stride

        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # resize the features if needed
        if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
            resize_feats = F.interpolate(
                feats.unsqueeze(0),
                size=self.max_seq_len,
                mode='linear',
                align_corners=False
            )
            feats = resize_feats.squeeze(0)
            
        if (self.audio_feat_folder is not None):
            audio_feats = torch.from_numpy(np.ascontiguousarray(audio_feats.transpose()))
            resize_audio_feats = F.interpolate(
                audio_feats.unsqueeze(0),
                size=feats.shape[1],
                mode='linear',
                align_corners=False
            )
            audio_feats = resize_audio_feats.squeeze(0)
            feats=torch.cat([feats,audio_feats],dim=0)
        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])
            # for activity net, we have a few videos with a bunch of missing frames
            # here is a quick fix for training
            if self.is_training:
                vid_len = feats.shape[1] + feat_offset
                valid_seg_list, valid_label_list = [], []
                for seg, label in zip(segments, labels):
                    if seg[0] >= vid_len:
                        # skip an action outside of the feature map
                        continue
                    # skip an action that is mostly outside of the feature map
                    ratio = (
                        (min(seg[1].item(), vid_len) - seg[0].item())
                        / (seg[1].item() - seg[0].item())
                    )
                    if ratio >= self.trunc_thresh:
                        valid_seg_list.append(seg.clamp(max=vid_len))
                        # some weird bug here if not converting to size 1 tensor
                        valid_label_list.append(label.view(1))
                segments = torch.stack(valid_seg_list, dim=0)
                labels = torch.cat(valid_label_list)
        else:
            segments, labels = None, None
            
        # av_labels = torch.from_numpy(video_item['av_labels'])
        
        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : num_frames}

        # no truncation is needed
        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )

        return data_dict
