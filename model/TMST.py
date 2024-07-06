import sys
sys.path.insert(0, '')

import torch
import torch.nn as nn

from model.activation import activation_factory
import torch.nn.functional as F
#from .Attention import *
#from .Attention import *

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

class Sequential_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Sequential_layer, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)        
        self.relu = nn.ReLU()
        self.Avp = nn.AdaptiveAvgPool2d(1)
        self.Max = nn.AdaptiveMaxPool2d(1)
        self.combine_conv = nn.Conv2d(2, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):    
        x = self.conv(x)
        N,C,T,V = x.size()
        x = x.permute(0,2,1,3).contiguous()
        Q = self.Avp(x)
        K = self.Max(x)
        Combine = torch.cat([Q,K],dim=2)
        Combine = self.combine_conv(Combine.permute(0,2,1,3).contiguous()).permute(0,2,1,3).contiguous()
        out = (x * self.sigmoid(Combine).expand_as(x)).permute(0,2,1,3).contiguous()      
        return out
                             
class Temporal_tran(nn.Module):
    def __init__(self, in_channels, out_channels, Frames, kernel_size, stride=1):
        super(Temporal_tran, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))     
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.AvpTemTrans = nn.AdaptiveAvgPool2d(1)
        self.MaxTemTrans = nn.AdaptiveMaxPool2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()        
        self.soft = nn.Softmax(-1)
        self.linear = nn.Linear(Frames,Frames)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        conv_init(self.conv)
        
    def forward(self, x):    
        x = self.conv(x)
        N,C,T,V=x.size()
        x1 = x[:,:C//2,:,:]
        x2 = x[:,C//2:C,:,:]                
        Q = self.AvpTemTrans(x1.permute(0,2,1,3).contiguous())
        K = self.MaxTemTrans(x2.permute(0,2,1,3).contiguous())
        Q = self.relu(self.linear(Q.squeeze(-1).squeeze(-1)))
        K = self.relu(self.linear(K.squeeze(-1).squeeze(-1)))       
        atten = self.sigmoid(torch.einsum('nt,nm->ntm', (Q, K)))                   
        out = self.bn(torch.einsum('nctv,ntm->ncmv', (x, atten)))      
        return out


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation1=1, dilation2=1):
        super(TemporalConv, self).__init__()
        pad1 = (kernel_size + (kernel_size-1) * (2**dilation1-1) - 1) // 2
        pad2 = (kernel_size + (kernel_size-1) * (2**dilation2-1) - 1) // 2
        
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad1, 0),
            stride=(stride, 1),
            dilation=(2**dilation1, 1))
        
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad2, 0),
            stride=(stride, 1),
            dilation=(2**dilation2, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        #self.temporal_att = AttentionBlock(out_channels, out_channels, out_channels)

    def forward(self, x): # 128 64 64 25
        x1 = self.conv1(x) # 128 8 64 25
        x1 = self.bn(x1)
        tanh_x1 = torch.tanh(x1)
        sigmoid_x1 = torch.sigmoid(x1)
        x1 = tanh_x1 * sigmoid_x1 # 128 8 64 25
        
        x2 = self.conv2(x) # 128 8 64 25
        x2 = self.bn(x2)
        tanh_x2 = torch.tanh(x2)
        sigmoid_x2 = torch.sigmoid(x2)
        x2 = tanh_x2 * sigmoid_x2 # 128 8 64 25
        
        x = x1+x2 # 128 8 64 25
        # N_, C_, T_, V_ = x.shape
        # x = self.temporal_att(x)
        # x = x.view(N_, C_, T_, V_)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 frames,
                 kernel_size=3,
                 stride=1,
                 dilations1=[1,2],
                 dilations2=[2,1],
                 residual=True,
                 residual_kernel_size=1,
                 activation='relu'):

        super().__init__()
        assert out_channels % (len(dilations1) + 3) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.frames = frames
        self.num_branches = len(dilations1) + 3
        branch_channels = out_channels // self.num_branches

        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                activation_factory(activation),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation1=dilation1,
                    dilation2=dilation2),
                
            )
            for dilation1, dilation2 in zip(dilations1, dilations2)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            activation_factory(activation),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))
        
        self.branches.append(nn.Sequential(
            Temporal_tran(in_channels, branch_channels, self.frames, kernel_size=1, stride=stride),
            nn.BatchNorm2d(branch_channels)
        ))        
        
        self.branches.append(nn.Sequential(
            Sequential_layer(in_channels, branch_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(branch_channels)
        ))  

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        self.act = activation_factory(activation)
        

    def forward(self, x):
        # Input dim: (N,C,T,V) 128 64 64 25
        res = self.residual(x) #32, 288, 100, 20
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        #out = self.temporal_att(out)
        out = self.act(out)
        return out


if __name__ == "__main__":
    mstcn = MultiScale_TemporalConv(288, 288)
    x = torch.randn(32, 288, 100, 20)
    mstcn.forward(x)
    for name, param in mstcn.named_parameters():
        print(f'{name}: {param.numel()}')
    print(sum(p.numel() for p in mstcn.parameters() if p.requires_grad))
