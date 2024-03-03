import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
import cv2

from scipy.ndimage import label
from skimage.filters import gabor
from skimage.filters import threshold_otsu
import pandas as pd
import os

def detect_repeating_patterns_21_30(image_path, size=(128, 128), threshold=1.0):
    # 以灰度模式加载图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 将图像缩小到较小的尺寸以加快计算速度
    img = cv2.resize(img, size)

    # 将图像归一化到0-1范围
    img = img / 255.0

    # 减去均值并除以标准差以归一化图像
    img = (img - np.mean(img)) / np.std(img)

    # 计算图像的自相关函数
    correlation = correlate2d(img, img, mode='same')

    # 显示自相关图
    # plt.imshow(correlation, cmap='hot')
    # plt.show()

    # 应用阈值
    thresholded = correlation > threshold

    # 找出所有超过阈值的区域
    labeled, num_labels = label(thresholded)

    # 判断图像是否具有重复的纹理或模式
    has_repeating_patterns = num_labels > 1
    # print(has_repeating_patterns)

    return correlation, has_repeating_patterns, labeled


def process_images_in_folder(folder_path, output_file):
    # 创建一个空的DataFrame，用于存储结果
    results = pd.DataFrame(columns=['image_path', 'has_repeating_patterns'])

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 只处理jpg和png文件
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            try:
                # 对每个图像执行函数
                _, has_repeating_patterns, _ = detect_repeating_patterns_21_30(image_path)
                print(has_repeating_patterns)

                # 将结果添加到DataFrame
                results = results.append({
                    'image_path': image_path,
                    'has_repeating_patterns': has_repeating_patterns
                }, ignore_index=True)

            except Exception as e:
                print(f'Error processing file {image_path}: {e}')

    # 将结果保存到csv文件
    results.to_csv(output_file, index=False)
