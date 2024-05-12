from collections import defaultdict

# 存储多词方面词的情感极性数量
pos = defaultdict(int)
neu = defaultdict(int)
neg = defaultdict(int)


path = 'E:/PythonProject2/VisionLanguageMABSA/Datasets/twitter2017/test.txt'
# 从文件中逐行读取数据
with open(path, 'r', encoding='utf-8') as file:
    while True:
        # 逐行读取数据
        sentence = file.readline().strip()
        if not sentence:
            break  # 如果读到文件末尾，则退出循环

        # 读取下一行数据
        aspect_term = file.readline().strip()
        sentiment = int(file.readline().strip())
        image_file_name = file.readline().strip()

        # 判断方面词是否为多词方面词
        if " " in aspect_term:
            if sentiment == 1:
                pos[aspect_term] += 1
            elif sentiment == 0:
                neu[aspect_term] += 1
            elif sentiment == -1:
                neg[aspect_term] += 1

# 输出多词方面词对应的情感极性数量
# 统计多词方面词对应的情感极性数量
positive_count = sum(pos.values())
neutral_count = sum(neu.values())
negative_count = sum(neg.values())

# 输出三种类别分别的数量
print(f"Positive count: {positive_count}")
print(f"Neutral count: {neutral_count}")
print(f"Negative count: {negative_count}")
