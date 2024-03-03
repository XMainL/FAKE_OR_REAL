import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import csv

def noise_detection_17_40(image_path, h=12.5, noise_ratio_threshold=0.1825):
    # 读取图像
    img = cv2.imread(image_path)

    # 转换为Lab色彩空间
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # 分离L, a, b通道
    l_channel, a_channel, b_channel = cv2.split(img_lab)

    # 对L通道进行CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_channel = clahe.apply(l_channel)

    # 重新合并L, a, b通道
    img_clahe = cv2.merge([l_channel, a_channel, b_channel])
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_Lab2BGR)

    # 使用非局部均值算法去噪
    dst = cv2.fastNlMeansDenoisingColored(img_clahe, h=h)

    # 计算噪声图像
    noise_img = cv2.absdiff(img_clahe, dst)

    # 转换噪声图像为灰度图，然后进行二值化
    noise_img_gray = cv2.cvtColor(noise_img, cv2.COLOR_BGR2GRAY)
    _, noise_img_bin = cv2.threshold(noise_img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 计算噪声比例
    noise_ratio = np.sum(noise_img_bin == 255) / (img_clahe.shape[0] * img_clahe.shape[1])
    # print(noise_ratio)

    # 判断是否存在大量噪声
    noise_level = ""
    is_noisy = noise_ratio > noise_ratio_threshold
    if noise_ratio > noise_ratio_threshold * 1.10:
        noise_level = "3"
    elif noise_ratio > noise_ratio_threshold * 1.05:
        noise_level = "2"
    elif noise_ratio > noise_ratio_threshold:
        noise_level = "1"
    else:
        noise_level = "0"

    # 显示原始图像、去噪后图像和噪声图像
    # fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # axs[0].imshow(cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB))
    # axs[0].set_title('CLAHE Image')
    # axs[1].imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    # axs[1].set_title('Denoised Image')
    # axs[2].imshow(noise_img_bin, cmap='gray')
    # axs[2].set_title('Noise Image')
    # plt.show()

    return img_clahe, dst, noise_img_bin, noise_ratio, is_noisy, noise_level

def process_directory(image_directory, output_csv_path):
    # 创建CSV文件，并写入列名
    with open(output_csv_path, 'w', newline='') as csvfile:
        # fieldnames = ['image_path', 'is_noisy', 'noise_ratio', 'noise_level'] # 去除 is_noisy 变量
        fieldnames = ['image_path', 'noise_ratio', 'noise_level']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 遍历文件夹中的所有文件
        for filename in os.listdir(image_directory):
            # 检查文件是否为图片
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                image_path = os.path.join(image_directory, filename)

                # 进行噪声检测
                _, _, _, noise_ratio, is_noisy, noise_level = noise_detection_17_40(image_path)

                # 将结果写入CSV文件
                # writer.writerow({'image_path': image_path, 'noise_ratio': noise_ratio, 'is_noisy': is_noisy, 'noise_level': noise_level})
                writer.writerow({'image_path': image_path, 'noise_ratio': noise_ratio, 'noise_level': noise_level})
