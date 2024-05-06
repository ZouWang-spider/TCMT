import nltk
import numpy as np
import torch
import json
from PIL import Image
import pytesseract
import torch.nn as nn
from torchcrf import CRF
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import Linear, CrossEntropyLoss
from sklearn.metrics import log_loss
from transformers import BertTokenizer, BertModel

from TCMT.BaseModel.CrossTransformer import Cross_Transformer
from TCMT.BaseModel.BiAffine import BiAffine, BERT_Embedding
from TCMT.BaseModel.Image_to_text import Vit_to_text
from TCMT.BaseModel.FCLayer import FCLayer



#syn_feature Process
def text_feature_func(sentence):
    # part of Language
    text = nltk.word_tokenize(sentence)
    text_graph, rels, pos = BiAffine(sentence)
    word_feature, dependency_feature, pos_feature = BERT_Embedding(sentence, rels, pos)
    syn_feature = torch.cat((word_feature, dependency_feature, pos_feature), dim=1)
    #Pooling k=3
    max_pooling = nn.MaxPool1d(kernel_size=3)
    text_feature_embedding = max_pooling(syn_feature)
    return text_feature_embedding


#Generate Text Caption
def Image_to_text_caption(image_patch):
    preds = Vit_to_text([image_patch])
    preds = preds[0]

    gener_text = nltk.word_tokenize(preds)

    model_name = 'bert-base-uncased'  # 您可以选择其他预训练模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    marked_text1 = ["[CLS]"] + gener_text + ["[SEP]"]

    # tokenized_text = tokenizer.tokenize(marked_text)
    input_ids1 = torch.tensor(tokenizer.encode(marked_text1, add_special_tokens=True)).unsqueeze(0)  # 添加批次维度
    outputs1 = model(input_ids1)
    preds_embedding = outputs1.last_hidden_state

    return preds_embedding

#OCR decter text
def OCR(image_path):
   image = Image.open(image_path)
   # 使用 pytesseract 进行 OCR
   OCR_text = pytesseract.image_to_string(image)
   return OCR_text


#Face description text
def face_description(image_filename):
    with open("E:/PythonProject2/TCMT/BaseModel/FITE/face_descriptions/twitter15_face_discription.json", 'r') as f:
        data = json.load(f)
    if image_filename in data:
        return data[image_filename]
    else:
        return "Text not found for image: " + image_filename

# # 输入图片文件名，并获取面部描述
# image_filename = '74960.jpg'
# text = face_description(image_filename)
# print(text)


#Cross_Transformer Train
def Cross_Transformers(text_embedding, caption_embedding):
    featureinteraction = Cross_Transformer(768)
    text_hidden = featureinteraction(text_embedding, caption_embedding, caption_embedding)
    return text_hidden


# #Aspect term Extraction
# hidden_size1 = 256
# num_aspect_tags = 2
# num_sentiment_classes = 3
# def FC_func(text_hidden):
#     hidden_size2 = text_hidden.shape[0]
#     input_size = text_hidden.shape[1]
#     fcmodel = FCLayer(input_size, hidden_size1, hidden_size2, num_aspect_tags, num_sentiment_classes)
#     return fcmodel


# image_path, sentence, aspect_terms, sentiments = get_dataset(sample)
# text_feature_embedding = text_feature_func(sentence)
# preds_embedding = Image_to_text_caption(image_path)
#
# text_hidden = Cross_Transformers(text_feature_embedding, preds_embedding)
# fcmodel = FC_func(text_hidden)
#
# label = fcmodel(text_hidden)
