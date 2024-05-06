import torch
import nltk
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import torchvision.models as models
from transformers import BertTokenizer, BertModel
from pycocotools.coco import COCO

from TCMT.BaseModel.CrossTransformer import Cross_Transformer

# 构建 COCO 标签词典
coco_labels = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
    9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
    53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant',
    65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'TV', 73: 'laptop', 74: 'mouse', 75: 'remote',
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
}


def FasterRCNN_to_Tensor(image_path):
    # 加载图像
    image = Image.open(image_path)

    # 加载预训练模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # 加载 ResNet 模型，选择合适的预训练权重
    resnet_model = models.resnet50(pretrained=True)
    resnet_model.eval()

    # 图像预处理
    transform = T.Compose([T.ToTensor()])

    # 预处理图像并将其转换为模型输入格式
    input_image = transform(image)
    input_image = input_image.unsqueeze(0)  # 添加批处理维度

    # 将图像输入模型并进行预测
    with torch.no_grad():
        predictions = model(input_image)

    # 初始化一个空的张量，用于存储所有拼接后的张量
    concatenated_tensors = []
    # 可视化检测结果
    draw = ImageDraw.Draw(image)
    for score, box, label_index in zip(predictions[0]['scores'], predictions[0]['boxes'], predictions[0]['labels']):
        if score > 0.75:  # 设置置信度阈值
            label_text = coco_labels[label_index.item()]  # 将索引映射到文本标签
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
            draw.text((box[0], box[1]), f"{label_text}: {score:.2f}", fill="red")  # 输出类别和置信度分数

            # 提取目标区域
            x1, y1, x2, y2 = box.int().tolist()
            cropped_image = image.crop((x1, y1, x2, y2))
            # 图像预处理
            cropped_image = transform(cropped_image)
            cropped_image = cropped_image.unsqueeze(0)  # 添加批处理维度
            # 使用 ResNet 模型提取特征
            with torch.no_grad():
                features = resnet_model(cropped_image)

            # 定义线性变换层
            linear_layer = nn.Linear(1000, 768)
            # 将输入张量通过线性变换层
            features_tensor = linear_layer(features)
            # print(features_tensor.shape)

            label_text = nltk.word_tokenize(label_text)
            # 获取每个标签的标签向量
            model_name = 'E:/bert-base-cased'  # 您可以选择其他预训练模型
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)

            marked_text = ["[CLS]"] + label_text + ["[SEP]"]

            # tokenized_text = tokenizer.tokenize(marked_text)
            input_ids = torch.tensor(tokenizer.encode(marked_text, add_special_tokens=True)).unsqueeze(0)  # 添加批次维度
            outputs = model(input_ids)
            labels_embedding = outputs.last_hidden_state
            # 使用 squeeze 函数去除维度为 1 的维度
            labels_tensor = labels_embedding.squeeze(0)
            # print(labels_tensor.shape)

            # Concated
            concatenated_tensor = torch.cat((features_tensor, labels_tensor), dim=0)
            concatenated_tensors.append(concatenated_tensor)

    # 将所有拼接后的张量沿着批处理维度拼接成一个张量
    final_tensor = torch.cat(concatenated_tensors, dim=0)
    # print(final_tensor.shape)

    # # 显示结果图像
    # image.show()
    return final_tensor


# image_path = 'E:/PythonProject2/TCMT/Datasets/46211.jpg'
# final_tensor = FasterRCNN_to_Tensor(image_path)
# print(final_tensor.shape)



