import nltk
import numpy as np
import torch
import torch.nn as nn
from torchcrf import CRF
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import Linear, CrossEntropyLoss
from sklearn.metrics import log_loss

from TCMT.DataProcess.ImagePatch import image_patches
from TCMT.DataProcess.ImageFeature import patches_features, Neighbors, create_Graph
from TCMT.BaseModel.ViG import ViGmodel
from TCMT.BaseModel.GCNModel import GCNModel
from TCMT.BaseModel.CrossTransformer import Cross_Transformer

#Image Process
def Image_Graph(image_path):
    # 使用函数获取 patches
    patches = image_patches(image_path, (14, 14))

    # 提取特征向量
    image_feature = patches_features(patches)
    # print(image_feature.shape[1])
    # 查看特征向量的形状
    # print("Image:", image_feature.shape)  # (196,512)
    # print(image_feature)
    # KNN算法寻找当前特征向量的邻居，8个
    k_neighbors = 8
    indices = Neighbors(k_neighbors, image_feature)
    # 根据邻居节点构建图
    image_feature = torch.tensor(image_feature)
    image_graph = create_Graph(indices)
    return image_feature, image_graph, patches

#ViG model construct
def image_GCN(image_feature, patches):
    # Image GCN parameter
    input_size1 = image_feature.shape[1]
    hidden_size1 = image_feature.shape[1]  # 隐藏层的大小
    num_layers1 = 2  # GCN layer
    num_node1 = len(patches)

    # image GCN模型和 text GCN模型
    image_gcn = GCNModel(input_size1, hidden_size1, num_layers1, num_node1)
    return image_gcn

#Transformer Train
def Transformers(image_output):
    # 转换image的维度
    desired_size = 768
    pad_size = desired_size - image_output.size(1)
    image_reshaped = torch.cat([image_output, torch.zeros(image_output.size(0), pad_size)], dim=1)
    print(image_reshaped.shape)  # (196,768)

    featureinteraction = Cross_Transformer(768)
    final_output = featureinteraction(image_reshaped, image_reshaped, image_reshaped)
    return final_output

#DeepSentiBank



# #训练
# image_path = "E:/PythonProject2/TCMT/Datasets/sandwich.jpg"
# image_feature, image_graph, patches = Image_Graph(image_path)
# image_gcn = image_GCN(image_feature, patches)  # construct model
# image_output = image_gcn(image_feature, image_graph)   #torch.Size([196, 512])
# print(image_output.shape)
# final_output = Transformers(image_output)
# print(final_output.shape)








