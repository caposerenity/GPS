import os.path as osp
import os
import numpy as np
from scipy.io import loadmat
import pickle
import copy
from .base import BaseDataset
from PIL import Image
import torch
from collections import OrderedDict
class LUPDG(BaseDataset):
    def __init__(self, root, transforms):
        self.split = 'train'
        self.transforms = transforms
        self.annotations = []
        self.videos = {}
        self.pkls = []
        self.all_ids = OrderedDict()
        self.all_videos = []
        self.root = root
        hw_pkl = open('/path/path/dets-NL/Imagesize.pkl','rb')
        hw_pkl = pickle.load(hw_pkl)
        cnt = 0
        for (troot, _, files) in os.walk(self.root, followlinks=True):
            for f in sorted(files):
                path = os.path.join(troot, f)
                words = path.split('/')
                if not words[4] in ['germany+berlin', 'china+beijing','us+new_york']:
                    continue
                #if not words[5] in ['VO_1AdYRGW8']:
                #    continue
                #if cnt >=272:
                #    break
                if path in hw_pkl.keys():
                    w, h = hw_pkl[path]
                else:
                    h, w = 720, 1280
                info = dict(
                    pklname = '/path/path/dets-NL/merge/'+words[4]+'+'+words[5]+'.pkl',
                    filename = path,
                    height = h,
                    width = w
                )
                self.annotations.append(info)
                self.pkls.append('/path/path/dets-NL/merge/'+
                words[4]+'+'+words[5]+'.pkl')
                frame_id = int(f.split('.')[0])
                self.all_videos.append(self.pkls[-1])

                if self.pkls[-1] not in self.videos:
                    self.videos[self.pkls[-1]] = {}
                    cnt+=1
                self.videos[self.pkls[-1]][frame_id] = len(self.annotations)-1

        self.all_videos = list(OrderedDict.fromkeys(self.all_videos).keys())
        print(f'num of files:{len(self.annotations)}')
        self.ids = [[-1]]*len(self.annotations)
        self.boxes = [[-1]]*len(self.annotations)
        self.conf = [[-1]]*len(self.annotations)
        self.instance_nums = 0
        for pkl in set(self.pkls):
            file = open(pkl,'rb')
            pkl_anno = pickle.load(file)
            file.close()
            for frame_id in pkl_anno.keys():
                if frame_id not in self.videos[pkl].keys():
                    continue
                idx = self.videos[pkl][frame_id]
                self.ids[idx] = []
                self.boxes[idx] = []
                self.conf[idx] = []
                for i in range(len(pkl_anno[frame_id])):
                    if int(pkl_anno[frame_id][i][5])!= -1:
                        self.all_ids[int(pkl_anno[frame_id][i][5])] = self.all_videos.index(pkl)
                        #self.all_ids.add( int(pkl_anno[frame_id][i][5]) )
                        self.ids[idx].append( int(pkl_anno[frame_id][i][5]) )
                    else:
                        self.ids[idx].append( -1 )
                    self.boxes[idx].append( pkl_anno[frame_id][i][:4] )
                    self.conf[idx].append( pkl_anno[frame_id][i][4] )
                    self.instance_nums += 1

        self.all_ids = OrderedDict(sorted(self.all_ids.items(), key=lambda t: t[0]))
        self.ordered_ids = list(self.all_ids.keys())
        for i in range(len(self.ids)):
            ids_in_frame = self.ids[i]
            for j in range(len((ids_in_frame))):
                if self.ids[i][j]!=-1:
                    self.ids[i][j] = self.ordered_ids.index(self.ids[i][j])+1
                else:
                    self.ids[i][j] = 20001
        self.videos = torch.tensor(list(self.all_ids.values()))

        #self.all_ids = list(OrderedDict.fromkeys(self.all_ids).keys())
        

        print(f'num of videos:{len(self.all_videos)}')
        print(f'num of ids:{len(self.all_ids)}')
        print(f'max num of id:{max(self.all_ids)}')
        print(f'num of instances:{self.instance_nums}')
        self._set_group_flag()

    '''def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.annotations[i]
            if img_info['height'] > img_info['width']:
                self.flag[i] = 1'''

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.annotations[i]
            if img_info['filename'].split('/')[4] == 'china+beijing':
                self.flag[i] = 1
            elif img_info['filename'].split('/')[4] == 'us+new_york':
                self.flag[i] = 2

    def __getitem__(self, idx):
        img_info = self.annotations[idx]

        gt_masks_ann = []
        gt_ids = copy.deepcopy(self.ids[idx])
        for i in range(len(gt_ids)):
            # if gt_ids[i]!=-1:
            #     gt_ids[i] = self.all_ids.index(gt_ids[i])
            # else:
            #     gt_ids[i] = 20001
            gt_masks_ann.append(None)
        labels = torch.as_tensor(gt_ids, dtype=torch.int64) 
        target = dict(
            img_name = img_info['filename'],
            video_id = self.videos[labels[labels!=20001][0] -1 ],
            idx = idx,
            boxes = torch.as_tensor(copy.deepcopy(self.boxes[idx]), dtype=torch.float32),
            labels = labels,
            flag = torch.as_tensor([self.flag[idx]], dtype=torch.int64)
            )
        
        img = Image.open(img_info['filename']).convert("RGB")
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
