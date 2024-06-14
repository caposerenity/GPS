import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch.cuda.amp import autocast,custom_fwd,custom_bwd
import numpy as np
from .triplet import TripletLossFilter

def iou_match_grid(box1, box2):
    x11, y11, x12, y12 = np.split(box1, 4, axis=1)
    x21, y21, x22, y22 = np.split(box2, 4, axis=1)
    xa = np.maximum(x11, np.transpose(x21))
    xb = np.minimum(x12, np.transpose(x22))
    ya = np.maximum(y11, np.transpose(y21))
    yb = np.minimum(y12, np.transpose(y22))

    area_inter = np.maximum(0, (xb - xa + 1)) * np.maximum(0, (yb - ya + 1))

    area_1 = (x12 - x11 + 1) * (y12 - y11 + 1)
    area_2 = (x22 - x21 + 1) * (y22 - y21 + 1)
    area_union = area_1 + np.transpose(area_2) - area_inter
    iou = area_inter / area_union
    return iou

def random_fourier_features(x, w=None, b=None, num_f=None, sum=True, sigma=None, seed=None):
    if num_f is None:
        num_f = 1
    n = x.size(0)
    r = x.size(1)
    x = x.view(n, r, 1)
    c = x.size(2)
    if sigma is None or sigma == 0:
        sigma = 1
    if w is None:
        w = 1 / sigma * (torch.randn(size=(num_f, c))).to(x.device)
        b = 2 * np.pi * torch.rand(size=(r, num_f))
        b = b.repeat((n, 1, 1))

    Z = torch.sqrt(torch.tensor(2.0 / num_f).to(x.device))

    mid = torch.matmul(x, w.t())

    mid = mid + b.to(x.device)
    mid -= mid.min(dim=1, keepdim=True)[0]
    mid /= mid.max(dim=1, keepdim=True)[0]
    mid *= np.pi / 2.0

    if sum:
        Z = Z * (torch.cos(mid) + torch.sin(mid))
    else:
        Z = Z * torch.cat((torch.cos(mid), torch.sin(mid)), dim=-1)
    return Z

def cov(x, w=None):
    if w is None:
        n = x.shape[0]
        cov = torch.matmul(x.t(), x) / n
        e = torch.mean(x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    else:
        w = w.view(-1, 1)
        cov = torch.matmul((w * x).t(), x)
        e = torch.sum(w * x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())

    return res

class OIM(autograd.Function):
    @staticmethod
    #@custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, targets, belong_to_first_half, non_id_feat, lut, cq, header, first_pos_sample, second_pos_sample, momentum):
        ctx.save_for_backward(inputs, targets, belong_to_first_half)#, lut, cq, header, momentum)
        ctx.lut = lut
        ctx.cq = cq
        ctx.header = header
        ctx.first_pos_sample = first_pos_sample
        ctx.second_pos_sample = second_pos_sample
        ctx.momentum = momentum
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(torch.cat((cq, non_id_feat)).t())
        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    @staticmethod
    #@custom_bwd(cast_inputs=torch.float32)
    def backward(ctx, grad_outputs):
        inputs, targets, belong_to_first_half = ctx.saved_tensors #, lut, cq, header, momentum = ctx.saved_tensors

        # inputs, targets = tensor_gather((inputs, targets))

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat([ctx.lut, ctx.cq], dim=0))
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        for x, y, is_first_half in zip(inputs, targets, belong_to_first_half):
            if y < len(ctx.lut):
                ctx.lut[y] = ctx.momentum * ctx.lut[y] + (1.0 - ctx.momentum) * x
                ctx.lut[y] /= ctx.lut[y].norm()
                if is_first_half:
                    ctx.first_pos_sample[y] = ctx.momentum * ctx.first_pos_sample[y] + (1.0 - ctx.momentum) * x
                    ctx.first_pos_sample[y] /= ctx.first_pos_sample[y].norm()
                else:
                    ctx.second_pos_sample[y] = ctx.momentum * ctx.second_pos_sample[y] + (1.0 - ctx.momentum) * x
                    ctx.second_pos_sample[y] /= ctx.second_pos_sample[y].norm()
            else:
                ctx.cq[ctx.header] = x
                ctx.header = (ctx.header + 1) % ctx.cq.size(0)
        return grad_inputs, None, None, None, None, None, None, None, None, None


def oim(inputs, targets, belong_to_first_half, non_id_feat, lut, cq, header, first_pos_sample, second_pos_sample, momentum=0.5):
    return OIM.apply(inputs, targets,belong_to_first_half, non_id_feat, lut, cq, torch.tensor(header), first_pos_sample, second_pos_sample, torch.tensor(momentum))


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.tri_loss = TripletLossFilter()
        self.middle_of_frame_idx = None
        #self.norm = torch.nn.InstanceNorm1d(1)

        self.register_buffer("lut", torch.nn.Parameter(torch.zeros(self.num_pids, self.num_features), requires_grad=False))
        self.register_buffer("cq", torch.nn.Parameter(torch.zeros(self.num_unlabeled, self.num_features), requires_grad=False))
        self.register_buffer("id_relevant_dims", torch.nn.Parameter(torch.ones(self.num_features)>0, requires_grad=False))
        
        #前半部分prototype
        self.register_buffer("first_pos_sample", torch.nn.Parameter(torch.zeros(self.num_pids, self.num_features), requires_grad=False))
        self.register_buffer("second_pos_sample", torch.nn.Parameter(torch.zeros(self.num_pids, self.num_features), requires_grad=False))

        self.header_cq = 0

    def forward(self, inputs, roi_label, frame_idxes):
        #inputs = self.norm(inputs.unsqueeze(1)).squeeze(1)
        # iou作mask，同一帧内相邻的instance距离要远
        '''in_box_regs = torch.cat(in_box_regs)
        
        loss_same_frame = 0
        start = 0
        box_regs = in_box_regs.detach().cpu()
        for i in range(len(instance_nums)):
            end = start+instance_nums[i]
            #mask = ((roi_label[i] >= 1) & (roi_label[i] !=20001))
            iou_mask = torch.tensor(iou_match_grid(box_regs[start:end], box_regs[start:end]))
            iou_mask = ((iou_mask>0.2) & (iou_mask<0.5))
            similarities = torch.cosine_similarity(inputs[start:end].unsqueeze(1), inputs[start:end], dim=-1)
            iou_mask = iou_mask.to(similarities.device)
            
            #similarities = similarities * iou_mask
            #trace = torch.sum(torch.diag(similarities)[mask])
            similarities = similarities[iou_mask]
            if len(similarities) == 0:
                continue
            loss_same_frame += torch.sum(similarities) /( len(similarities)*len(similarities) ) 
            start = end'''

        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1

        belong_to_first_half = []
        for idx,l in enumerate(list(label)):
            frame_idx =frame_idxes[idx//128]
            if l>=0 and l!=20000:
                belong_to_first_half.append(frame_idx<=self.middle_of_frame_idx[l])
            else:
                belong_to_first_half.append(False)
        belong_to_first_half = torch.tensor(belong_to_first_half).to(inputs.device)

        inds = label >= 0
        label = label[inds]
        belong_to_first_half = belong_to_first_half[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)

        # loss_triplet = 0
        # instance_nums = [torch.sum(inds[start:start+128]) for start in [0,128,256,384,512]]
        # start = 0
        # for i in range(len(instance_nums)):
            #end = instance_nums[i]
        first_half = belong_to_first_half[label!=20000]
        with_id_label = label[label!=20000]
        prototype_feat = self.second_pos_sample[with_id_label]
        prototype_feat[first_half] = self.first_pos_sample[with_id_label][first_half]
        tmp_feat = torch.cat( (inputs, prototype_feat))

        #tmp_feat = torch.cat( (inputs, self.lut[with_id_label])).detach()
        tmp_label = torch.cat((label, with_id_label))
        loss_triplet = self.tri_loss( tmp_feat, tmp_label)
        #thre = torch.cosine_similarity(self.pos_sample[with_id_label], self.lut[with_id_label], -1)<0.5
        #self.pos_sample[with_id_label][thre] = self.pos_sample[with_id_label][thre]*0.5+inputs[label!=20000][thre]*0.5
        #start = end

        cfeaturecs = random_fourier_features(inputs)
        loss_cov = 0
        for i in range(cfeaturecs.size()[-1]):
            cfeaturec = cfeaturecs[:, :, i]

            cov1 = cov(cfeaturec)
            cov_matrix = cov1 * cov1

            if len(torch.nonzero(self.id_relevant_dims))==len(self.id_relevant_dims):
                loss_cov += torch.sum(cov_matrix) - torch.trace(cov_matrix)
            else:
                id_relevant = cov_matrix[self.id_relevant_dims]
                cov_re_irr = torch.transpose(id_relevant, 0, 1)[~self.id_relevant_dims]

                loss_cov += torch.sum(cov_re_irr)

        non_id_feat = inputs[((label < 0)&(label == 20000))]
        
        projected = oim(inputs, label, belong_to_first_half, non_id_feat, self.lut, self.cq, self.header_cq, self.first_pos_sample, self.second_pos_sample, self.momentum)
        projected *= self.oim_scalar

        self.header_cq = (
            self.header_cq + (label >= self.num_pids).long().sum().item()
        ) % self.num_unlabeled
        loss_oim = F.cross_entropy(projected, label, ignore_index=20000)
        #print('label:')
        #print(label)
        #print('pred:')
        #print(torch.argmax(projected, dim=1))
        if len(label[label!=20000])==0:
            loss_oim = torch.tensor(0.0).to(label.device)
        
        return loss_oim, loss_cov, loss_triplet
