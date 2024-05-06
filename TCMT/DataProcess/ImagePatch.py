import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms


def image_patches(image_path, patch_size):
    # 打开图像
    img = Image.open(image_path)

    # 将图像转换为 NumPy 数组
    img_array = np.array(img)

    # 获取图像的形状
    height, width, _ = img_array.shape

    # 计算每个 patch 的大小
    patch_height = height // patch_size[0]
    patch_width = width // patch_size[1]

    # 存储所有的 patches
    patches = []

    # 分割图像
    for i in range(patch_size[0]):
        for j in range(patch_size[1]):
            patch = img_array[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :]
            patches.append(patch)

    return patches


# # 图片路径
# image_path = "E:/PythonProject2/VisionLanguageMABSA/Datasets/twitter2015_images/414619.jpg"
#
# # 定义 patch 的行列数
# patch_size = (14, 14)
#
# # 获取所有 patches
# patches = split_image_into_patches(image_path, patch_size)
#
# # 可以通过遍历 patches 来处理每个小块
# for i, patch in enumerate(patches):
#     # 处理 patch，比如保存、显示等
#     # 这里只是打印 patch 的形状
#     print(f"Patch {i + 1}: {patch.shape}")


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

# image_path = "E:/PythonProject2/TCMT/Datasets/sandwich.jpg"
# # 使用前面示例中的函数获取 patches
# patches = image_patches(image_path, (14, 14))  #196
#
#
# # 显示 patches
# show_patches(patches)







