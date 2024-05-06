import nltk
import numpy as np
import torch
import torch.nn as nn
from torchcrf import CRF
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import Linear, CrossEntropyLoss
from sklearn.metrics import log_loss

from TCMT.DataProcess.DataProcess import load_dataset, annotate_sentence
from TCMT.TextualExtraction.TextualModule import text_feature_func, Image_to_text_caption, Cross_Transformers
from TCMT.VisualPrediction.VisualModule import Image_Graph, image_GCN, Transformers
from TCMT.BaseModel.FasterRCNN import FasterRCNN_to_Tensor
from TCMT.TextualVisual.Textual_Visual_Module import Cross_Transformers2
from TCMT.BaseModel.FCLayer import FCLayer


data_path = "E:/PythonProject2/VisionLanguageMABSA/Datasets/twitter2015/train.txt"
image_path = "E:/PythonProject2/VisionLanguageMABSA/Datasets/twitter2015_images"
samples = load_dataset(data_path,image_path)

#获取数据集
def get_dataset(sample):
    image_path = sample['image_path']
    sentence = sample['sentence']
    aspect_terms = sample['aspect_term']
    sentiments = sample['sentiment']
    return image_path, sentence, aspect_terms, sentiments

#将数据集处理为标签类型，实现对每个单词的7分类任务
annotation_list = []
for sample in samples:
     annotations = annotate_sentence(sample['sentence'], sample['aspect_term'], sample['sentiment'])
     annotation_list.append(annotations)


#FCLayer
hidden_size1 = 256
hidden_size2 = 256
num_label = 7
def FC_func(embedding_feature):
    input_size = embedding_feature.shape[1]
    fc_layer = FCLayer(input_size, hidden_size1, hidden_size2, num_label)
    return  fc_layer


if __name__ == "__main__":
    num_epochs = 50
    batch_size = 32

    # model parameter
    num_heads = 8
    # 定义各个组件的学习率
    learning_rate = 0.00002

    for epoch in range(num_epochs):
        # Iterate through batches
        for start_idx in range(0, len(samples), batch_size):
            end_idx = start_idx + batch_size
            batch_samples = samples[start_idx:end_idx]

            # datasets Train
            for sample in batch_samples:
                image_path, sentence, aspect_terms, sentiments = get_dataset(sample)  # get dataset,image and text

                #label
                annotations = annotate_sentence(sample['sentence'], sample['aspect_term'], sample['sentiment'])

                #Textual Module
                text_feature_embedding = text_feature_func(sentence)
                preds_embedding = Image_to_text_caption(image_path)
                text_hidden = Cross_Transformers(text_feature_embedding, preds_embedding)
                # 创建FC_layer
                fc_layer = FC_func(text_hidden)
                # 前向传播
                pred_text = fc_layer(text_hidden)

                #Visual Module
                image_feature, image_graph, patches = Image_Graph(image_path)
                image_gcn = image_GCN(image_feature, patches)  # construct model
                image_output = image_gcn(image_feature, image_graph)   #torch.Size([196, 512])
                image_hidden = Transformers(image_output)
                # print(image_hidden.shape)
                #创建FC_layer
                fc_layer = FC_func(image_hidden)
                # 前向传播
                pred_visual = fc_layer(image_hidden)

                #Textual-Visual Module
                target_embedding = FasterRCNN_to_Tensor(image_path)
                cross_modal_hidden = Cross_Transformers2(target_embedding, text_hidden, image_hidden)
                # 创建FC_layer
                fc_layer= FC_func(cross_modal_hidden)
                # 前向传播
                pred_textual_visual = fc_layer(cross_modal_hidden)

                # 将标签转换为索引，每个标签对应一个索引
                label_to_index = {'O': 0, 'B-NEU': 1, 'I-NEU': 2, 'B-POS': 3, 'I-POS': 4, 'B-NEG': 5, 'I-NEG': 6}
                labels_indices = [label_to_index[label] for label in annotations]
                # 将标签索引转化为张量
                label_tensor = torch.tensor(labels_indices)


                #交叉熵损失函数
                criterion = CrossEntropyLoss()
                Text_loss = criterion(pred_text, label_tensor)
                Visual_loss = criterion(pred_visual, label_tensor)
                Textual_Visual_loss = criterion(pred_textual_visual, label_tensor)

                # 计算总的损失 a,b,c
                # 默认权重参数
                a = 1/3
                b = 1/4
                c = 1/3
                total_loss = a * Visual_loss + b * Text_loss + c * Textual_Visual_loss

                print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item()}, Text_loss:{Text_loss}, Image_loss:{Visual_loss}, Text_image_loss:{Textual_Visual_loss},')

                # Set up the optimizer
                # 合并所有参数和优化器
                all_parameters = list(image_gcn.parameters()) + list(fc_layer.parameters())
                optimizer = Adam(all_parameters, lr=learning_rate)
                scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
                # Backward pass and optimization step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()








