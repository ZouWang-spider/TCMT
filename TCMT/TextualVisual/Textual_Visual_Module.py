import nltk
import numpy as np
import torch
import torch.nn as nn
from torchcrf import CRF
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import Linear, CrossEntropyLoss
from sklearn.metrics import log_loss
from transformers import BertTokenizer, BertModel

from TCMT.BaseModel.CrossTransformer import Cross_Transformer2
from TCMT.BaseModel.FCLayer import FCLayer
from TCMT.BaseModel.FasterRCNN import FasterRCNN_to_Tensor



#faster R-CNN to tensor
# image_path = 'E:/PythonProject2/TCMT/Datasets/46211.jpg'
# final_tensor = FasterRCNN_to_Tensor(image_path)


#Cross_Transformer Train
def Cross_Transformers2(target_embedding, textual_tensor, visual_tensor):
    featureinteraction = Cross_Transformer2(768)
    cross_modal_hidden = featureinteraction(target_embedding, visual_tensor, visual_tensor,textual_tensor)

    return cross_modal_hidden




