import os
from random import shuffle, seed

from PIL import Image


def get_img_names(data_dir):
    return list(filter(lambda x: x.endswith('.jpg') or x.endswith('.jpeg') or
                                 x.endswith('.png'), os.listdir(data_dir)))


def is_a_ratio(ratio):
    return (isinstance(ratio, int) or isinstance(ratio, float)) and 0 <= ratio <= 1


def split_data(src_dir, tar_dir, train_percent, valid_percent, test_percent, _seed=0):
    if not os.path.exists(src_dir):
        raise NotADirectoryError(f'{src_dir} is not a directory')
    if not is_a_ratio(train_percent) or not is_a_ratio(valid_percent) or \
            not is_a_ratio(test_percent) or train_percent + valid_percent + test_percent != 1:
        raise ValueError("ratio not valid")

    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)
    print("target directory:", tar_dir)  # test info

    dtypes_ratios = {"train": train_percent, "valid": valid_percent, "test": test_percent}
    img_names = get_img_names(src_dir)
    print("img numbers:", len(img_names))  # test info
    seed(_seed)
    shuffle(img_names)
    dtype_sizes = []
    for dir_type in dtypes_ratios:
        dtype_sizes.append([dir_type, int(len(img_names) * dtypes_ratios[dir_type])])
    if sum(list(sizes[1] for sizes in dtype_sizes)) != len(img_names):
        dtype_sizes[-1][1] = abs(len(img_names) - sum(list(sizes[1] for sizes in dtype_sizes[:-1])))
    dtype_sizes = dict(dtype_sizes)
    start = 0
    for dir_type in dtype_sizes:
        os.makedirs(os.path.join(tar_dir, f"{dir_type}"), exist_ok=True)
        print("current directory:", dir_type)  # test info
        for i in range(int(dtype_sizes[dir_type])):
            (Image.open(os.path.join(src_dir, f"{img_names[start + i]}"))
             .save(os.path.join(tar_dir, f"{dir_type}", f"{img_names[start + i]}")))
        start += dtype_sizes[dir_type]
