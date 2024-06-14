from collections import OrderedDict
import numpy as np
import torch
import tqdm
from scipy import optimize
import os
from collections import defaultdict

def to_device(images, targets, device):
    images = [image.to(device) for image in images]
    for t in targets:
        t["boxes"] = t["boxes"].to(device)
        t["labels"] = t["labels"].to(device)
    return images, targets

def generate_binary_part(dataset):
    with torch.no_grad():
        annotationn = dataset.annotations
        all_ids = dataset.ordered_ids
        ids = dataset.ids
        middle_of_ids = []
        frames_in_id = defaultdict(list)
        for idx, anno in enumerate(annotationn):
            ids_i = ids[idx]
            for id in ids_i:
                if id!=20001:
                    name = os.path.basename(anno['filename'])
                    frames_in_id[id].append(int(name[:-4]))
        for id in sorted(frames_in_id.keys()):
            frame_idxes = sorted(frames_in_id[id])
            mid_idx = frame_idxes[len(frame_idxes)//2]
            middle_of_ids.append(mid_idx)
        return middle_of_ids

def generate_pseudo_label(model, data_loader, device, dataset):
    model.eval()
    memory = model.module.roi_heads.reid_loss.lut
    #print(memory.sum())
    videos = dataset.videos
    assigned_num = 0
    merged_id = 0
    e_list = {}
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm.tqdm(data_loader)):
            images, targets = to_device(images, targets, device)
             
            gt_boxes = [t["boxes"] for t in targets]
            
            embeddings = model(images, targets)
            '''if 14 in targets[0]['labels']:
                embeddings = model(images, targets)
                e_list[int(targets[0]['img_name'].split('/')[-1][:7])] = embeddings[torch.nonzero(targets[0]['labels']==14).squeeze()]'''
            num = 0
            #continue
            for j in range(len(gt_boxes)):
                idx = targets[j]["idx"]
                video_id = targets[j]["video_id"]
                
                #取j这一帧内instance的feat
                embeddings_in_frame = torch.stack(embeddings[num:num+len(gt_boxes[j])])
                num+= len(gt_boxes[j])

                #只有一帧内尚未有id标签的会被赋pseudo label
                gt_labels = targets[j]["labels"]
                #unlabeled_idxs = torch.nonzero(gt_labels==20001).squeeze(1)
                labeled_idxs = torch.nonzero(gt_labels!=20001).squeeze(1)

                #取这一帧所在video中出现过的所有id的feat
                same_video_indexes = (videos==video_id)
                feat_of_same_video = memory[0:10141][same_video_indexes]
                feat_idxs = torch.nonzero(same_video_indexes).squeeze(1).tolist()

                '''id_similarity = torch.cosine_similarity(feat_of_same_video.unsqueeze(1), feat_of_same_video, dim=-1).cpu()
                id_similarity[id_similarity<0.9] = 0
                id_similarity = id_similarity - torch.diag_embed(torch.diag(id_similarity))
                new_feat_idxs = feat_idxs.copy()
                for k in range(len(id_similarity)):
                    most_simialr = torch.argmax(id_similarity[k])
                    if id_similarity[k][most_simialr]>0 and most_simialr < k:
                        new_feat_idxs[k] = new_feat_idxs[most_simialr]
                        merged_id += 1'''
                
                similarity = torch.cosine_similarity(embeddings_in_frame, feat_of_same_video, dim=-1).cpu()
                for k in range(len(labeled_idxs)):
                    #把原本有标签的对应标签相似度改为1，不影响其他匹配
                    label = gt_labels[labeled_idxs[k].item()]
                    similarity[labeled_idxs[k].item()][feat_idxs.index(label-1)] = 1.0
                similarity[similarity<0.5] = 0
                
                #匈牙利算法做匹配,需先拓展为方阵,左右上下
                pad = torch.nn.ZeroPad2d(padding=(0,max(len(similarity),len(similarity[0]))-len(similarity[0])
                ,0,max(len(similarity),len(similarity[0]))-len(similarity)))
                similarity = pad(similarity)
                row_ind,col_ind=optimize.linear_sum_assignment(similarity, maximize=True)

                for k in range(len(gt_labels)):
                    if col_ind[k]>=len(feat_idxs): #如果拓展了video中的id，对不存在的id不分配
                        continue
                    new_id = feat_idxs[col_ind[k]] + 1
                    if max(similarity[k])!=0:
                        if max(similarity[k])!=1:
                            assigned_num += 1
                        dataset.ids[idx][k] = new_id
                    '''print(new_id)
                    print(dataset.ordered_ids[new_id])
                    print(targets[j]['img_name'])
                    print(targets[j]['boxes'][k])
                    print('___________________')
                    exit()'''
                #print(dataset.ids[idx])
    print('instances assigned pseudo labels')
    print(assigned_num)
    print('merged_ids')
    print(merged_id)
    '''li = []
    for e  in sorted(e_list):
        li.append(e_list[e])
    print(len(li))
    torch.save(torch.stack(li), '4.pth')'''
    return dataset
