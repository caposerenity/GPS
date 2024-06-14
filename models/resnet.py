from collections import OrderedDict
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from .dsbn import DSBN1d, DSBN2d
from torchvision.models.detection.backbone_utils import BackboneWithFPN,resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import TwoMLPHead

class Backbone(nn.Sequential):
    def __init__(self, resnet):
        super(Backbone, self).__init__(
            OrderedDict(
                [
                    ["conv1", resnet.conv1],
                    ["bn1", resnet.bn1],
                    ["relu", resnet.relu],
                    ["maxpool", resnet.maxpool],
                    #["in1", torch.nn.InstanceNorm2d(64)],
                    ["layer1", resnet.layer1],  # res2
                    ["in2", torch.nn.InstanceNorm2d(256)],
                    ["layer2", resnet.layer2],  # res3
                    ["in3", torch.nn.InstanceNorm2d(512)],
                    ["layer3", resnet.layer3],  # res4
                    #["in4", torch.nn.InstanceNorm2d(1024)],
                ]
            )
        )
        self.out_channels = 1024

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = super(Backbone, self).forward(x)
        return OrderedDict([["feat_res4", feat]])


class Res5Head(nn.Sequential):
    def __init__(self, resnet):
        super(Res5Head, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
        self.out_channels = [1024, 2048]
        self.prototype = torch.nn.Parameter(torch.zeros(4, 1024), requires_grad=False)
        self.prototype_init = False
        #使用FPN后接两个 1x1 卷积变回1024维
        #另一种方案是直接用512维的reid feat
        #self.representation_size = 1024
        #self.linear1 = torch.nn.Conv2d(256, out_channels=self.representation_size, kernel_size=1, stride=1, padding=0)
        #self.linear2 = torch.nn.Conv2d(self.representation_size, out_channels=self.representation_size, kernel_size=1, stride=1, padding=0)


    # def forward(self, x):
    #     feat = super(Res5Head, self).forward(x)
    #     x = F.adaptive_max_pool2d(x, 1)
    #     feat = F.adaptive_max_pool2d(feat, 1)
    #     return OrderedDict([["feat_res4", x], ["feat_res5", feat]])

    def bottleneck_forward(self, bottleneck, x, domain, similarity=None):
        identity = x

        out = bottleneck.conv1(x)
        out = bottleneck.bn1(out, domain, similarity)
        out = bottleneck.relu(out)
        out = bottleneck.conv2(out)
        out = bottleneck.bn2(out, domain, similarity)
        out = bottleneck.relu(out)
        out = bottleneck.conv3(out)
        out = bottleneck.bn3(out, domain, similarity)
        if bottleneck.downsample is not None:
            for module in bottleneck.downsample:
                if not isinstance(module, DSBN2d):
                    identity = module(x)
                else:
                    identity = module(identity, domain, similarity)
        out += identity
        out = bottleneck.relu(out)
        return out

    def forward(self, x, domain=0, training=True):
        #对于reid head的dsbn特殊处理
        #需要取出没有child的module组成list一次执行，可以避免递归中重新实现所有带is_source的forward
        #Bottleneck的forward步骤有缺失
        

        module_seq=[]
        is_reid_head = False
        for _, (child_name, child) in enumerate(self.named_modules()):
            if isinstance(child, DSBN2d) or isinstance(child, DSBN1d):
                is_reid_head = True
            if isinstance(child, torchvision.models.resnet.Bottleneck):
                module_seq.append(child)

        #x = F.relu(self.linear1(x))
        #x = F.relu(self.linear2(x))
        
        
        feat = x.clone()
        x = F.adaptive_max_pool2d(x, 1)
        #if is_reid_head:
        if True:
            if training:
                if self.prototype.device != x.device:
                    self.prototype = self.prototype.to(x.device)
                output = torch.mean(x, dim=0).view(-1,1024).detach()
                #print(domain) 
                self.prototype[domain] = self.prototype[domain]*0.9 + output*0.1
                for module in module_seq:
                    feat = self.bottleneck_forward(module, feat, domain)
            else:
                similarity = []
                output = torch.mean(x, dim=0).view(-1,1024)
                similarity = torch.cosine_similarity(output, self.prototype, dim=-1)
                similarity = torch.nn.functional.softmax(similarity,dim=0)
                #print(similarity)
                #exit()
                for module in module_seq:
                    feat = self.bottleneck_forward(module, feat, domain, similarity)
            
        else:
            feat = super(Res5Head, self).forward(feat)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])


def build_resnet(name="resnet50", pretrained=True):
    resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)    
 
    #resnet = resnet_fpn_backbone(name,  trainable_layers=3, pretrained=pretrained, norm_layer=nn.BatchNorm2d)
    
    # freeze layers
    resnet.conv1.weight.requires_grad_(False)
    resnet.bn1.weight.requires_grad_(False)
    resnet.bn1.bias.requires_grad_(False)


    # return_layers = {'layer1': 'feat_res2', 'layer2': 'feat_res3', 'layer3': "feat_res4"}
    # in_channels_list = [256,512,1024]
    # out_channels = 256
    
    # resnet_with_fpn = BackboneWithFPN(Backbone(resnet),
    #     return_layers,in_channels_list,out_channels)
    # resnet_with_fpn.body.conv1.weight.requires_grad_(False)
    # resnet_with_fpn.body.bn1.weight.requires_grad_(False)
    # resnet_with_fpn.body.bn1.bias.requires_grad_(False)
    
    return Backbone(resnet), Res5Head(resnet)
