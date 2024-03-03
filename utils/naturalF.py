import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd

def is_natural_color_histogram_advanced_14_00(
    image_path
    # threshold=0.3, 
    # brightness_threshold=(45, 195), 
    # contrast_threshold=(45, 195), 
    # saturation_threshold=(55, 175)
    ):

    # 加载图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 将图像从BGR转化为HSV

    # 计算亮度和对比度
    brightness = img[...,2].mean()
    contrast = img[...,2].std()
    # brightness_check = brightness_threshold[0] <= brightness <= brightness_threshold[1]
    # contrast_check = contrast_threshold[0] <= contrast <= contrast_threshold[1]

    # 计算饱和度，并与阈值进行比较
    saturation = img[...,1].mean()
    # saturation_check = saturation_threshold[0] <= saturation <= saturation_threshold[1]

    # 计算颜色直方图
    color = ('h', 's', 'v')
    entropy_values = []
    # entropy_check = []
    for i, col in enumerate(color):
        histogram = cv2.calcHist([img], [i], None, [256], [0, 256])

        # 计算直方图的熵
        hist_normalized = histogram.ravel() / histogram.sum()
        hist_normalized = hist_normalized[hist_normalized > 0]
        entropy = -1 * (hist_normalized * np.log2(hist_normalized)).sum()
        entropy_values.append(entropy)
        # entropy_check.append(entropy >= threshold)

    # 综合判断
    checks = {
        'brightness': brightness,
        'contrast': contrast,
        'saturation': saturation,
        'h_entropy': entropy_values[0],
        's_entropy': entropy_values[1],
        'v_entropy': entropy_values[2]
        # 'is_natural': all([brightness_check, contrast_check, saturation_check, all(entropy_check)])
    }

    return checks


# 将指定文件夹中的所有图片的结果保存到一个CSV文件中
def process_images_in_folder(folder_path, csv_output_path):
    # 获取文件夹中的所有图片
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # 创建一个空的DataFrame来保存结果
    df = pd.DataFrame()

    # 对每一张图片执行函数并保存结果
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(folder_path, image_file)
        result = is_natural_color_histogram_advanced_14_00(image_path)
        
        # 创建一个新的字典，首先添加图片路径，然后添加其他结果
        result_with_image_path = {"image_path": image_path}
        result_with_image_path.update(result)
        
        df = df.append(result_with_image_path, ignore_index=True)

    # 将结果写入CSV文件
    df.to_csv(csv_output_path, index=False)

