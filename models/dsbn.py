import torch
import torch.nn as nn
import collections

class DSBN2d(nn.Module):
    def __init__(self, planes, domains = 4):
        super(DSBN2d, self).__init__()
        self.num_features = planes
        self.BNs = nn.ModuleList()
        #额外define一个bn用于inference
        for i in range(domains+1):
            #self.BNs.append(nn.SyncBatchNorm(planes))
            self.BNs.append(nn.BatchNorm2d(planes))

    def forward(self, x, domain = 0, similarity=None):
        if similarity is None:
            return self.BNs[domain](x)
        else:
            #print(similarity)
            weights = []
            for i in range(len(self.BNs)-1):
                weights.append(self.BNs[i].state_dict())
            w_dict = collections.OrderedDict()
            w_dict['weight'] = (weights[0]['weight']*similarity[0] + weights[1]['weight']*similarity[1] + weights[2]['weight']*similarity[2] + weights[3]['weight']*similarity[3])
            w_dict['bias'] = (weights[0]['bias']*similarity[0] + weights[1]['bias']*similarity[1] + weights[2]['bias']*similarity[2] + weights[3]['bias']*similarity[3])
            w_dict['running_mean'] = (weights[0]['running_mean']*similarity[0] + weights[1]['running_mean']*similarity[1] + weights[2]['running_mean']*similarity[2] + weights[3]['running_mean']*similarity[3])
            w_dict['running_var'] = (weights[0]['running_var']*similarity[0] + weights[1]['running_var']*similarity[1] + weights[2]['running_var']*similarity[2] + weights[3]['running_var']*similarity[3])
            w_dict['num_batches_tracked'] = weights[0]['num_batches_tracked']
            self.BNs[-1].load_state_dict(w_dict)
            return self.BNs[-1](x)


class DSBN1d(nn.Module):
    def __init__(self, planes, domains = 4):
        super(DSBN1d, self).__init__()
        self.num_features = planes
        self.BNs = nn.ModuleList()
        for i in range(domains):
            self.BNs.append(nn.BatchNorm1d(planes))

    def forward(self, x, domain = 0):
        return self.BNs[domain](x)

def convert_dsbn(model):
    for _, (child_name, child) in enumerate(model.named_children()):
        #if next(model.parameters()).is_cuda:
        #print(child_name)
        #print(child)
        #continue
        assert(not next(model.parameters()).is_cuda)
        if isinstance(child, nn.BatchNorm2d):
            m = DSBN2d(child.num_features)
            for i in range(4):
                m.BNs[i].load_state_dict(child.state_dict())
            setattr(model, child_name, m)
            print("convert_dsbn 2D")
            print(child_name)
        elif isinstance(child, nn.BatchNorm1d):
            m = DSBN1d(child.num_features)
            for i in range(4):
                m.BNs[i].load_state_dict(child.state_dict())
            setattr(model, child_name, m)
            print("convert_dsbn 1D")
            print(child_name)
        else:
            convert_dsbn(child)

def convert_bn(model):
    for _, (child_name, child) in enumerate(model.named_children()):
        assert(not next(model.parameters()).is_cuda)
        weights = []
        if isinstance(child, DSBN2d):
            m = nn.BatchNorm2d(child.num_features)
            for i in range(len(child.BNs)):
                weights.append(child.BNs[i].state_dict())
            w_dict = collections.OrderedDict()
            
            w_dict['weight'] = (weights[0]['weight'] + weights[1]['weight'] + weights[2]['weight']+ weights[3]['weight'])/4
            w_dict['bias'] = (weights[0]['bias'] + weights[1]['bias'] + weights[2]['bias'] + weights[3]['bias'])/4
            w_dict['running_mean'] = (weights[0]['running_mean'] + weights[1]['running_mean'] + weights[2]['running_mean'] + weights[3]['running_mean'])/4
            w_dict['running_var'] = (weights[0]['running_var'] + weights[1]['running_var'] + weights[2]['running_var'] + weights[3]['running_var'])/4
            w_dict['num_batches_tracked'] = weights[0]['num_batches_tracked']
            m.load_state_dict(w_dict)
            setattr(model, child_name, m)
        elif isinstance(child, DSBN1d):
            m = nn.BatchNorm1d(child.num_features)
            for i in range(len(child.BNs)):
                weights.append(child.BNs[i].state_dict())
            m['weight'] = (weights[0]['weight'] + weights[1]['weight'] + weights[2]['weight'] + weights[3]['weight'])/4
            m['bias'] = (weights[0]['bias'] + weights[1]['bias'] + weights[2]['bias'] + weights[3]['bias'])/4
            m['running_mean'] = (weights[0]['running_mean'] + weights[1]['running_mean'] + weights[2]['running_mean'] + weights[3]['running_mean'])/4
            m['running_var'] = (weights[0]['running_var'] + weights[1]['running_var'] + weights[2]['running_var'] + weights[3]['running_var'])/4
            m['num_batches_tracked'] = weights[0]['num_batches_tracked']
            #m.load_state_dict(child.BN_S.state_dict())
            setattr(model, child_name, m)
        else:
            convert_bn(child)
