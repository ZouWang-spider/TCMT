import torch
import nltk
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
from models.experimental import attempt_load
from transformers import BertTokenizer, BertModel
import torchvision.models as models



def YOLOv5_to_Tensor(image_path):
    # 加载图像
    image = Image.open(image_path)

    # 加载 YOLOv5 模型
    Yoolov5model = attempt_load('yolov5s.pt', map_location=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')).fuse().autoshape()

    # 加载 ResNet 模型
    resnet_model = models.resnet50(pretrained=True)
    resnet_model.eval()

    # 图像预处理
    transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    # 图像预处理
    input_image = transform(image)

    # 将图像输入 YOLOv5 模型进行目标检测
    results = Yoolov5model(input_image)

    # 提取目标区域，并进行预处理
    concatenated_tensors = []
    for result in results.pred:
        for det in result:
            if det[4] > 0.75:  # 设置置信度阈值
                x1, y1, x2, y2 = det[:4].int().tolist()
                cropped_image = image.crop((x1, y1, x2, y2))
                cropped_image = transform(cropped_image)
                cropped_image = cropped_image.unsqueeze(0)  # 添加批处理维度

                # 使用 ResNet 模型提取特征
                with torch.no_grad():
                    features = resnet_model(cropped_image)
                features_tensor = features.mean(dim=[2, 3])  # 池化或平均特征
                # print(features_tensor.shape)

                # 获取标签的嵌入
                label_text = coco_labels[det[5].item()]
                label_text = nltk.word_tokenize(label_text)
                model_name = 'bert-base-uncased'
                tokenizer = BertTokenizer.from_pretrained(model_name)
                model = BertModel.from_pretrained(model_name)
                marked_text = ["[CLS]"] + label_text + ["[SEP]"]
                input_ids = torch.tensor(tokenizer.encode(marked_text, add_special_tokens=True)).unsqueeze(0)
                outputs = model(input_ids)
                labels_embedding = outputs.last_hidden_state.squeeze(0)
                # print(labels_embedding.shape)

                # 拼接特征张量和标签嵌入
                concatenated_tensor = torch.cat((features_tensor, labels_embedding), dim=1)
                concatenated_tensors.append(concatenated_tensor)

    # 将所有拼接后的张量沿着批处理维度拼接成一个张量
    final_tensor = torch.cat(concatenated_tensors, dim=0)
    # print(final_tensor.shape)

    return final_tensor
