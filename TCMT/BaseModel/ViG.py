import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# GCN 模型定义
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCN, self).__init__()
        self.layer1 = dglnn.GraphConv(in_feats, hidden_size)
        self.layer2 = dglnn.GraphConv(hidden_size, out_feats)

    def forward(self, g, features):
        x = self.layer1(g, features)
        x = F.relu(x)
        x = self.layer2(g, x)
        return x

# MLP 模型定义
class MLP(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(in_feats, hidden_size)
        self.layer2 = nn.Linear(hidden_size, out_feats)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 多头注意力模型定义
class ViGmodel(nn.Module):
    def __init__(self, gcn_hidden_size, mlp_hidden_size):
        super(ViGmodel, self).__init__()
        self.gcn = GCN(in_feats=512, hidden_size=gcn_hidden_size, out_feats=256)
        self.mlp = MLP(in_feats=256, hidden_size=mlp_hidden_size, out_feats=196)

    def forward(self, g, features):
        # GCN
        gcn_output = self.gcn(g, features)

        # 残差连接
        gcn_output += features

        # MLP
        mlp_output = self.mlp(gcn_output)

        # 残差连接
        final_output = mlp_output + gcn_output

        return final_output

# # 初始化模型
# gcn_hidden_size = 128
# mlp_hidden_size = 256
# num_classes = 10  # 根据你的任务设定
#
# model = ViGmodel(gcn_hidden_size, mlp_hidden_size)
# output = model(g, features)


