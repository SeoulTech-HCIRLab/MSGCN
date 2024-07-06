import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from einops import rearrange, repeat

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)
    

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(EdgeConv, self).__init__()
        
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels*2, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
    
    def forward(self, x, dim=4): # N, C, T, V 128, 80, 64, 25]
        
        if dim == 3:
            N, C, L = x.size()
            pass
        else:
            N, C, T, V = x.size()
            x = x.mean(dim=-2, keepdim=False) # N, C, V
        
        x = self.get_graph_feature(x, self.k)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        
        if dim == 3:
            pass
        else:
            x = repeat(x, 'n c v -> n c t v', t=T)
        
        return x
        
    def knn(self, x, k):

        inner = -2 * torch.matmul(x.transpose(2, 1), x) # N, V, V
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = - xx - inner - xx.transpose(2, 1)
        
        idx = pairwise_distance.topk(k=k, dim=-1)[1] # N, V, k
        return idx
    
    def get_graph_feature(self, x, k, idx=None):
        N, C, V = x.size()
        if idx is None:
            idx = self.knn(x, k=k)
        device = x.get_device()
        
        idx_base = torch.arange(0, N, device=device).view(-1, 1, 1) * V
        
        idx = idx + idx_base
        idx = idx.view(-1)
        
        x = rearrange(x, 'n c v -> n v c')
        feature = rearrange(x, 'n v c -> (n v) c')[idx, :]
        feature = feature.view(N, V, k, C)
        x = repeat(x, 'n v c -> n v k c', k=k)
        
        feature = torch.cat((feature - x, x), dim=3)
        feature = rearrange(feature, 'n v k c -> n c v k')
        
        return feature
    
class ChannelSpecific(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=4, mid_reduction=1, loop_times=4, fuse_alpha=0.15):
        super(ChannelSpecific, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rel_reduction = rel_reduction
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        # self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        # self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        self.loop_times = loop_times
        self.fuse_alpha = fuse_alpha
        self.alpha = nn.Parameter(torch.ones(self.rel_reduction))
            
    def forward(self, x):
        n, c, t, v = x.shape
        x1, x2 = self.conv3(x.mean(-2, keepdim=True)), self.conv3(x.mean(-2, keepdim=True))
        N, C, T, V = x1.shape
        x1 = x1.reshape(n, self.rel_reduction, int(C//self.rel_reduction), -1, v)
        x2 = x2.reshape(n, self.rel_reduction, int(C//self.rel_reduction), -1, v)
        graph = self.sigmoid(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        graph = torch.einsum('nkctuv,k->nkctuv', graph, self.alpha) # 64, 8, 10, 1, 25  N, Grp, Ch, T, V, V
        graph = graph.mean(0,  keepdim=True).squeeze()
        Grp, Ch, V_, _ = graph.shape
        graph = graph.view(Grp*Ch, V, _)
        graph = graph.mean(0,  keepdim=True)
        return graph
        
class Atten(nn.Module):
    def __init__(self, out_channels):
        super(Atten, self).__init__()
        self.out_channels=out_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(-1) 
        self.bn = nn.BatchNorm2d(out_channels)
        self.linear = nn.Linear(25,25)
#        self.linear2 = nn.Linear(25,25)
        bn_init(self.bn, 1)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x): 
        N, C, T, V = x.size()
        x1 = x[:,:C//2,:,:]
        x2 = x[:,C//2:C,:,:]
        Q_o = Q = self.avg_pool(x1.permute(0,3,1,2).contiguous())
        K_o = K = self.avg_pool(x2.permute(0,3,1,2).contiguous())
        Q = self.relu(self.linear(Q.squeeze(-1).squeeze(-1)))
        K = self.relu(self.linear(K.squeeze(-1).squeeze(-1)))
        atten = self.soft(torch.einsum('nv,nw->nvw', (Q, K))).unsqueeze(1).repeat(1,self.out_channels,1,1)  
        return atten, Q_o, K_o

class MSCAM(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups=8, coff_embedding=4, num_subset=3,t_stride=1,t_padding=0,t_dilation=1,bias=True,first=False,residual=True):
        super(MSCAM, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.groups=groups
        self.out_channels=out_channels
        self.in_channels=in_channels
        self.num_subset = 3
        self.alpha = nn.Parameter(torch.ones(1))
        self.Pre_def = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32),[3,1,25,25]), dtype=torch.float32, requires_grad=True).repeat(1,groups,1,1), requires_grad=True)
        self.A = Variable(torch.from_numpy(np.reshape(A.astype(np.float32),[3,1,25,25]).repeat(groups,axis=1)), requires_grad=False) 
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(25,25)
        self.Att = Atten(out_channels//4)
        self.AvG = nn.AdaptiveAvgPool2d(1) 
        self.Ref_conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.Channelcorrelation = ChannelSpecific(in_channels, out_channels)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * num_subset,
            kernel_size=(1, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        self.edgeConv = EdgeConv(in_channels, out_channels, k=5)
        self.conv_down_edge = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
            
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-1)
        self.relu = nn.ReLU()         
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        
    def pearson_correlation(self, X, threshold=0.5):
        Q_c_list = []
        num_joints = X.shape[1]
        R_c = np.zeros((num_joints, num_joints))
        for i in range(num_joints):
            for j in range(num_joints):
                if i == j:
                    R_c[i,j] = 1.0
                else:
                    X_i = X[:,i]
                    X_j = X[:,j]
                    mean_ = np.mean((X_i - np.mean(X_i, axis=0)) * (X_j - np.mean(X_j, axis=0)), axis=0)
                    std_ = np.std(X_i, axis=0) * np.std(X_j, axis=0)
                    pearson = 0.
                    if std_ == 0:
                        pearson = 0.
                    elif std_ > 0:
                        pearson = np.nan_to_num((mean_/std_), copy=False, nan=0.0, posinf=None, neginf=None)
                    r_ij = pearson
                    R_c[i,j] = np.mean(r_ij)

        # Apply a threshold to the correlation matrix and set the remaining correlations to 1
        threshold = 0.5
        mask = (R_c >= threshold).astype(np.float16)
        Q_c = self.softmax(R_c * mask)
        Q_c_list.append(Q_c)
        Q_c = np.squeeze(np.stack(Q_c_list, axis=-1), axis=2)
        return Q_c
            
            
    def softmax(self, x):
        # Apply the softmax function along the last dimension of the input
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def forward(self, x0):
        
        N, C, T, V = x0.size()
        A_correlation = self.Channelcorrelation(x0)
        A = self.A.cuda(x0.get_device()) + self.Pre_def
        norm_learn_A = A.repeat(1,self.out_channels//self.groups,1,1)
        A_final=torch.zeros([N,self.num_subset,self.out_channels,25,25],dtype=torch.float32).detach().to(x0.get_device())
        x = x0         #[128, 80, 64, 25]
        x = self.conv(x) # [128, 240, 64, 25] C*3
        n, kc, t, v = x.size()
        x = x.view(n, self.num_subset, kc// self.num_subset, t, v)
        for i in range(self.num_subset):
            x1,Q1,K1 = self.Att(x[:,i,:(kc// self.num_subset)//4,:,:])
            x2,Q2,K2 = self.Att(x[:,i,(kc// self.num_subset)//4:((kc// self.num_subset)//4)*2,:,:])
            x3,Q3,K3 = self.Att(x[:,i,((kc// self.num_subset)//4)*2:((kc// self.num_subset)//4)*3,:,:])
            x4,Q4,K4 = self.Att(x[:,i,((kc// self.num_subset)//4)*3:((kc// self.num_subset)//4)*4,:,:])
            
            x1_2 = self.soft(torch.einsum('nv,nw->nvw', (self.relu(self.linear(K1.squeeze(-1).squeeze(-1))), self.relu(self.linear(Q2.squeeze(-1).squeeze(-1)))))).unsqueeze(1).repeat(1,self.out_channels//4,1,1)
            x2_3 = self.soft(torch.einsum('nv,nw->nvw', (self.relu(self.linear(K2.squeeze(-1).squeeze(-1))), self.relu(self.linear(Q3.squeeze(-1).squeeze(-1)))))).unsqueeze(1).repeat(1,self.out_channels//4,1,1)
            x3_4 = self.soft(torch.einsum('nv,nw->nvw', (self.relu(self.linear(K3.squeeze(-1).squeeze(-1))), self.relu(self.linear(Q4.squeeze(-1).squeeze(-1)))))).unsqueeze(1).repeat(1,self.out_channels//4,1,1)

            x1 = x1/2 + x1_2/4
            x2 = x2/2 + x1_2/4 + x2_3/4
            x3 = x3/2 + x2_3/4 + x3_4/4
            x4 = x4/2 + x3_4/4
            
            atten = torch.cat([x1,x2,x3,x4],dim=1)
            A_final[:,i,:,:,:] = atten * 0.5 + norm_learn_A[i]
    
    
        A_final = A_final + A_correlation + norm_learn_A
        x = torch.einsum('nkctv,nkcvw->nctw', (x, A_final))   
        
        C_in = self.AvG(x)
        C_in = self.Ref_conv(C_in.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) 
        C_out = x + x * self.sigmoid(C_in).expand_as(x)               
        # edge_conv_ here
        edge_conv = self.edgeConv(x0)
        edge_conv = self.conv_down_edge(edge_conv)
        # addition of the edge conv and the CR_OUT
        C_out = C_out + edge_conv
        
        out = self.bn(C_out)
        out += self.down(x0)
        out = self.relu(out)
        return out

