import os
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import defaultdict


def get_image_tensor(image_path, transform=None):
    image = Image.open(image_path).convert("RGB")
    if transform:
        image = transform(image)   #对图像进行预处理转化为torch类型
    return image



def load_dataset(data_file,image_path):
    with open(data_file, "r") as f:
        lines = f.readlines()
    # # 处理句子中包含多词方面词, 使用 defaultdict 创建一个以 sentence 为键，值为包含该句子信息的列表的字典
    # sentence_dict = defaultdict(lambda: {"sentence": "", "aspect_terms": [], "sentiments": [], "image_path": ""})

    num_samples = len(lines) // 4    #数据集数量
    samples = []

    for i in range(num_samples):
        sentence = lines[i * 4].strip()
        aspect_term = lines[i * 4 + 1].strip()
        sentiment = int(lines[i * 4 + 2].strip())
        image_filename = lines[i * 4 + 3].strip()

        # Replace $T$ with aspect_term in the sentence
        sentence = sentence.replace('$T$', aspect_term)

        # 将信息添加到字典中
        # sentence_dict[sentence]["sentence"] = sentence
        # sentence_dict[sentence]["aspect_terms"].append(aspect_term)
        # sentence_dict[sentence]["sentiments"].append(sentiment)
        # # sentence_dict[sentence]["image_filename"] = image_filename

        # 图像路径拼接
        full_image_path = os.path.join(image_path, image_filename)

        # 读取并预处理图像
        image_tensor = get_image_tensor(full_image_path, transform=None)
        # sentence_dict[sentence]["image_path"] = full_image_path

        sample = {
            "sentence": sentence,
            "aspect_term": aspect_term,
            "sentiment": sentiment,
            "image_filename": image_filename,
            "image_path": full_image_path

        }
        # # 将字典的值转换为列表
        # samples = list(sentence_dict.values())
        samples.append(sample)
    return samples

#构建标签
def annotate_sentence(sentence, aspect_term, sentiment):
    words = sentence.split()  # 将句子拆分为单词
    print(words)
    annotations = ['O'] * len(words)  # 初始化所有单词的标注为 O

    aspect_term_tokens = aspect_term.split()


    # 将方面词拆分为单词，并查找其在句子中的位置
    strat_index = None
    for i in range(len(words)):
        if words[i:i + len(aspect_term_tokens)] == aspect_term_tokens:
            strat_index = i
            break

    long = len(aspect_term_tokens)

    # 如果方面词是多词类型，则标注为 B-POS 和 I-POS
    if sentiment == 1:
        if len(aspect_term_tokens) > 1:
            annotations[strat_index] = 'B-POS'
            for i in range(strat_index + 1, strat_index + long):
                annotations[i] = 'I-POS'
        # 如果方面词是单词类型，则直接标注为 B-POS
        else:
            annotations[strat_index] = 'B-POS'
    elif sentiment == 0:
        if len(aspect_term_tokens) > 1:
            annotations[strat_index] = 'B-NEU'
            for i in range(strat_index + 1, strat_index + long):
                annotations[i] = 'I-NEU'
        # 如果方面词是单词类型，则直接标注为 B-POS
        else:
            annotations[strat_index] = 'B-NEU'
    elif sentiment == -1:
            if len(aspect_term_tokens) > 1:
                annotations[strat_index] = 'B-NEG'
                for i in range(strat_index + 1, strat_index + long):
                    annotations[i] = 'I-NEG'
            # 如果方面词是单词类型，则直接标注为 B-POS
            else:
                annotations[strat_index] = 'B-NEG'
    return annotations



data_path = "E:/PythonProject2/VisionLanguageMABSA/Datasets/twitter2015/train.txt"
image_path = "E:/PythonProject2/VisionLanguageMABSA/Datasets/twitter2015_images"
samples = load_dataset(data_path,image_path)

for sample in samples:
    print(sample)
    print(sample['sentence'])
    print(sample['image_path'])
    print(sample['aspect_term'])
    print(sample['sentiment'])
    annotations = annotate_sentence(sample['sentence'], sample['aspect_term'], sample['sentiment'])
    print(annotations)


