import dgl
import torch
import numpy as np
from PIL import Image
import networkx as nx
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms
from sklearn.neighbors import NearestNeighbors
from TCMT.DataProcess.ImagePatch import image_patches



# 加载预训练的ResNet模型
def ResNet():
    model = models.resnet18(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # 移除最后的全连接层
    # 设置模型为评估模式
    model.eval()
    return model


#将图片转化为特征向量表示
def patches_features(patches):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    for patch in patches:
        img = Image.fromarray(patch)
        img = transform(img).unsqueeze(0)

        # 使用ResNet模型提取特征
        with torch.no_grad():
            model = ResNet()
            feature = model(img)

        features.append(feature.squeeze().numpy())

    # 将特征堆叠成数组
    features = np.vstack(features)
    return features



#使用KNN算法寻找当前特征向量的邻居，8个
def Neighbors(k_neighbors,features):
    # 初始化NearestNeighbors
    knn = NearestNeighbors(n_neighbors=k_neighbors+1, metric='euclidean')

    # 训练模型
    knn.fit(features)

    # 对每个数据点找到最近的邻居
    distances, indices = knn.kneighbors(features)

    for i, neighbors in enumerate(indices):
        # 去掉当前向量自身
        neighbors = neighbors[1:]
        # print(f"Nearest neighbors for point {i}: {neighbors}")

    return indices

def create_Graph(indices):
    nodes = []  # 节点
    arcs = []  # 边
    for i, neighbors in enumerate(indices):
        neighbors = neighbors[1:]
        # print(f"Nearest neighbors for point {i}: {neighbors}")
        for arc in neighbors:
            nodes.append(i)
            arcs.append(arc)
    # print(len(nodes))
    # print(len(arcs))

    # Create a DGL graph
    graph = (arcs, nodes)
    image_graph = torch.tensor(graph)
    g = dgl.graph(graph)  # 构建图

    return image_graph

    # # 获取 NetworkX 图对象
    # nxg = g.to_networkx()
    # # 绘制图
    # pos = nx.spring_layout(nxg)  # 使用 spring_layout 算法进行布局
    # nx.draw(nxg, pos)
    # plt.show()


# #展示patch图片
# def show_patches(patches):
#     num_patches = len(patches)
#     rows = int(num_patches**0.5)
#     cols = (num_patches + rows - 1) // rows
#
#     plt.figure(figsize=(10, 10))
#
#     for i, patch in enumerate(patches):
#         plt.subplot(rows, cols, i + 1)
#         plt.imshow(patch)
#         plt.axis('off')  # 关闭坐标轴
#         plt.title('')    # 设置标题为空字符串
#
#     plt.show()


# # 图片路径
# image_path = "E:/PythonProject2/VisionLanguageMABSA/Datasets/sandwich.jpg"
#
# # 使用函数获取 patches
# patches = image_patches(image_path, (14, 14))
# # show_patches(patches)
#
# # 提取特征向量
# Image_features = patches_features(patches)
#
# # 查看特征向量的形状
# print("Shape of features:", Image_features.shape)
#
# # KNN算法寻找当前特征向量的邻居，8个
# k_neighbors = 8
# indices = Neighbors(k_neighbors,Image_features)
#
# #根据邻居节点构建图
# graph = create_Graph(indices)
# nx.draw(graph.to_networkx(), with_labels=True)
# plt.show()
