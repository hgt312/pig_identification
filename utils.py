import glob
import os
import argparse
import random
import shutil


origin_path = '/media/hgt/share2/jdd/Pig_Identification_Qualification_Train'
train_dir = '/media/hgt/share2/jdd/train'  # 训练集数据
dev_dir = '/media/hgt/share2/jdd/dev'  # 验证集数据
root_path = '/media/hgt/share2/jdd'


def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def separate():
    if os.path.exists(os.path.join(root_path, 'dev')):
        pass
    else:
        os.mkdir(os.path.join(root_path, 'dev'))
    if os.path.exists(os.path.join(root_path, 'train')):
        pass
    else:
        os.mkdir(os.path.join(root_path, 'train'))
    for i in range(1, 31):
        if os.path.exists(os.path.join(root_path, 'dev', str(i))):
            pass
        else:
            os.mkdir(os.path.join(root_path, 'dev', str(i)))
        if os.path.exists(os.path.join(root_path, 'train', str(i))):
            pass
        else:
            os.mkdir(os.path.join(root_path, 'train', str(i)))
    for i in range(1, 31):
        now_path = os.path.join(origin_path, str(i))
        name_list = list(os.path.join(now_path, name) for name in os.listdir(now_path))
        random.shuffle(name_list)
        print(name_list[:3])
        for num in range(len(name_list)):
            if num < int(len(name_list) * 0.7):
                shutil.copy(name_list[num], os.path.join(train_dir, str(i)))
            else:
                shutil.copy(name_list[num], os.path.join(dev_dir, str(i)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--separate", action="store_true")
    args = parser.parse_args()
    if args.separate:
        separate()


if __name__ == '__main__':
    main()
