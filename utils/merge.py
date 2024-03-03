import pandas as pd
from utils import noise
from utils import repeat
from utils import symmetric_sift
from utils import naturalF
from skimage.filters import gabor
from skimage.filters import threshold_otsu

def merge_tables_and_save(table1_csv, table2_csv, table3_csv, table4_csv, output_csv):
    # 读取 CSV 文件
    table1 = pd.read_csv(table1_csv)
    table2 = pd.read_csv(table2_csv)
    table3 = pd.read_csv(table3_csv)
    table4 = pd.read_csv(table4_csv)

    # 按照 image_path 列合并所有表
    merged = table1.merge(table2, on='image_path').merge(table3, on='image_path').merge(table4, on='image_path')

    # 将合并后的数据保存到新的 CSV 文件中
    merged.to_csv(output_csv, index=False)

    return merged

if __name__=='__main__':
    image_folder = './dataset/train/fake/'
    noise_csv = noise.process_directory(image_folder, './dataset/train/fake/__train_fake_noise_new.csv')
    selfRrelevance_csv = repeat.process_images_in_folder(image_folder, './dataset/train/fake/__train_fake_selfRrelevance_new.csv')
    symmetry_csv = symmetric_sift.process_images(image_folder, './dataset/train/fake/__train_fake_symmetry_new.csv')
    natural_csv = naturalF.process_images_in_folder(image_folder, './dataset/train/fake/__train_fake_natural_new.csv')

    tabel1_path = './dataset/train/fake/__train_fake_noise_new.csv'
    tabel2_path = './dataset/train/fake/__train_fake_selfRrelevance_new.csv'
    tabel3_path = './dataset/train/fake/__train_fake_symmetry_new.csv'
    tabel4_path = './dataset/train/fake/__train_fake_natural_new.csv'
    merge_tabel = merge_tables_and_save(tabel1_path, tabel2_path, tabel3_path, tabel4_path, './model/input_files/__train_fake_new.csv')
