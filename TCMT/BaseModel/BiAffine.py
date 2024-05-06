from supar import Parser
import nltk
import dgl
import networkx as nx
import torch
import matplotlib.pyplot as plt
from stanfordcorenlp import StanfordCoreNLP
from transformers import BertTokenizer, BertModel


#使用BiAffine对句子进行处理得到arcs、rels、probs
def BiAffine(sentence):
      text = nltk.word_tokenize(sentence)
      # print(text)

      #POS feature
      nlp = StanfordCoreNLP(r'D:\StanfordCoreNLP\stanford-corenlp-4.5.4', lang='en')
      ann = nlp.pos_tag(sentence)
      pos_tags = [pair[1] for pair in ann]
      # print(pos_tags)


      parser = Parser.load('biaffine-dep-en')  # 'biaffine-dep-roberta-en'解析结果更准确
      dataset = parser.predict([text], prob=True, verbose=True)
      # print(dataset.sentences[0])

      #dependency feature
      rels = dataset.rels[0]
      # print(f"arcs:  {dataset.arcs[0]}\n"
      #       f"rels:  {dataset.rels[0]}\n"
      #       f"probs: {dataset.probs[0].gather(1, torch.tensor(dataset.arcs[0]).unsqueeze(1)).squeeze(-1)}")

      # 构建句子的图，由弧-->节点
      arcs = dataset.arcs[0]  # 边的信息
      edges = [i + 1 for i in range(len(arcs))]
      for i in range(len(arcs)):
            if arcs[i] == 0:
                  arcs[i] = edges[i]

      # 将节点的序号减一，以便适应DGL graph从0序号开始
      arcs = [arc - 1 for arc in arcs]
      edges = [edge - 1 for edge in edges]
      graph = (arcs, edges)
      graph_line = '({}, {})\n'.format(graph[0], graph[1])  # 将图信息转为字符串
      # print("graph:", graph)
      # print(graph_line)

      # Create a DGL graph
      text_graph = torch.tensor(graph)
      g = dgl.graph((arcs, edges))
      # nx.draw(g.to_networkx(), with_labels=True)
      # plt.show()

      return text_graph, rels, pos_tags





#节点特征
def BERT_Embedding(sentence, rels, pos):
      text = nltk.word_tokenize(sentence)


      # 初始化BERT标记器和模型
      # 加载BERT模型和分词器
      model_name = 'bert-base-uncased'  # 您可以选择其他预训练模型
      tokenizer = BertTokenizer.from_pretrained(model_name)
      model = BertModel.from_pretrained(model_name)

      # 获取单词节点特征
      # 标记化句子
      marked_text1 = ["[CLS]"] + text + ["[SEP]"]
      marked_text2 = ["[CLS]"] + rels + ["[SEP]"]
      marked_text3 = ["[CLS]"] + pos + ["[SEP]"]

      # 将分词转化为词向量
      # tokenized_text = tokenizer.tokenize(marked_text)
      input_ids1 = torch.tensor(tokenizer.encode(marked_text1, add_special_tokens=True)).unsqueeze(0)  # 添加批次维度
      outputs1 = model(input_ids1)

      input_ids2 = torch.tensor(tokenizer.encode(marked_text2, add_special_tokens=True)).unsqueeze(0)  # 添加批次维度
      outputs2 = model(input_ids2)

      input_ids3 = torch.tensor(tokenizer.encode(marked_text3, add_special_tokens=True)).unsqueeze(0)  # 添加批次维度
      outputs3 = model(input_ids3)

      # 获取词向量
      word_embeddings = outputs1.last_hidden_state
      dependency_embeddings = outputs2.last_hidden_state
      pos_embeddings = outputs3.last_hidden_state

      # 提取单词对应的词向量（去掉特殊标记的部分）
      word_embeddings = word_embeddings[:, 1:-1, :]  # 去掉[CLS]和[SEP]标记
      dependency_embeddings = dependency_embeddings[:, 1:-1, :]  # 去掉[CLS]和[SEP]标记
      pos_embeddings = pos_embeddings[:, 1:-1, :]  # 去掉[CLS]和[SEP]标记

      # 使用切片操作去除第一个和最后一个元素
      word_feature = word_embeddings[0][1:-1, :]  # 单词特征
      dependency_feature = dependency_embeddings[0][1:-1, :]  # 单词特征
      pos_feature = pos_embeddings[0][1:-1, :]  # 单词特征

      return word_feature, dependency_feature, pos_feature


# sentence ='Johor Laksa , some say the best in Malaysia .'
# text_graph, rels, pos_tags = BiAffine(sentence)
# word_feature, dependency_feature, pos_feature = BERT_Embedding(sentence, rels, pos_tags)
# print(word_feature.shape)  #(6,768)
# print(word_feature)

