import cv2
import numpy as np
import pandas as pd
import tqdm
from tqdm import tqdm
import os

def is_symmetric_sift_16_55(image_path, size=(256, 256), min_match_count=35):
    # 读取图像
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, size)

    # 创建 SIFT 特征提取器
    sift = cv2.SIFT_create()

    # 计算图像的 SIFT 特征
    keypoints1, descriptors1 = sift.detectAndCompute(img, None)

    # 翻转图像
    flipped = cv2.flip(img, 1)

    # 计算翻转图像的 SIFT 特征
    keypoints2, descriptors2 = sift.detectAndCompute(flipped, None)

    # 检查描述符是否为 None 或者为空
    if descriptors1 is None or len(descriptors1) < 1 or descriptors2 is None or len(descriptors2) < 1:
        print(f"No descriptors found in one or both of the images. Skipping {image_path}")
        return False, 0

    # 检查描述符的数量
    if len(descriptors1) < 2 or len(descriptors2) < 2:
        print(f"Not enough descriptors found in one or both of the images. Skipping {image_path}")
        return False, 0

    # 创建 FLANN 匹配器
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 使用 FLANN 匹配器匹配两个图像的 SIFT 特征
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 存储良好的匹配
    good = []

    # 使用 Lowe's ratio 测试挑选出良好的匹配
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # 如果良好的匹配数量大于最小匹配数量，则认为图像是对称的
    if len(good) > min_match_count:
        return True, len(good)
    else:
        return False, len(good)
    
def process_images(directory, output_file):
    # 创建一个空的 DataFrame 用于存储结果
    # results = pd.DataFrame(columns=['image_path', 'is_symmetric', 'good_matches_count'])
    results = pd.DataFrame(columns=['image_path', 'is_symmetric'])

    # 遍历指定目录中的所有文件
    image_files = [f for f in os.listdir(directory) if f.endswith(".jpg") or f.endswith(".png")]
    for filename in tqdm(image_files, desc="Processing images"):
        # 构造图像路径
        image_path = os.path.join(directory, filename)
            
        # 执行 is_symmetric_sift_16_55 函数并获取结果
        # is_symmetric, good_matches_count = is_symmetric_sift_16_55(image_path)
        is_symmetric, _ = is_symmetric_sift_16_55(image_path)

        # 将结果添加到 DataFrame 中
        # results = results.append({'image_path': image_path, 'is_symmetric': is_symmetric, 'good_matches_count': good_matches_count}, ignore_index=True)
        results = results.append({'image_path': image_path, 'is_symmetric': is_symmetric}, ignore_index=True)

    # 将结果写入到 CSV 文件中
    results.to_csv(output_file, index=False)

