# -*- coding: utf-8 -*-#
"""
@author:Galen
@file: pre_classify.py
@time: 2019/06/27
@description:
"""
import random
import os

from collections import defaultdict


def read_data(path):
    with open(path, "r") as f:
        while True:
            line = f.readline().strip()
            if not line:
                break
            yield line


def save_txt(filename, data_list, model="a"):
    with open(filename, model, encoding='UTF-8', errors='ignore') as f:
        for data in data_list:
            f.write(data + '\n')


def sort_dict(dict_words, reverse=True, site=1):
    """
    字典排序
    reverse: False 升序，True降序
    site: 0 第一个元素，1是第二个元素
    :param dict_words:
    :return:
    """
    keys = dict_words.keys()
    values = dict_words.values()
    list_one = [(key, val) for key, val in zip(keys, values)]
    list_sort = sorted(list_one, key=lambda x: x[site], reverse=reverse)
    return list_sort


def clean_data():
    """

    :return:
    """
    data_path = 'data/xfyun/apptype_train.dat'
    save_path = 'data/xfyun/apptype_train.txt'
    cate_path = 'data/xfyun/apptype_count.txt'
    cate_count = defaultdict(int)
    data_set = []
    for line in read_data(data_path):
        line_list = line.strip().split("\t")
        line_list = [x.replace("\t", "").replace("\n", "").strip() for x in line_list]
        if len(line_list) == 3:
            if len(line_list[0]) == 32:
                data_set.append(line)
            else:
                print('line_list = 3', len(line_list), line_list)
                data_set[-1] = data_set[-1] + line.replace("\t", "").replace("\n", "").strip()
        else:
            print('line_list != 3', len(line_list), line_list)
            data_set[-1] = data_set[-1] + line.replace("\t", "").replace("\n", "").strip()
    with open(save_path, 'w')as f:
        for line in data_set:
            # 001357CD179A515D6C0B91C7462D6C32\t<种类>|<种类>\t内容
            line_list = line.split("\t")
            cate_list = line_list[1].split("|")
            if len(line_list[2]) <= 0:
                print(line_list)
            for c in cate_list:
                cate_count[c] += 1
                f.write("{0}\t{1}\n".format(c, line_list[2]))
    cate_count_list = sort_dict(cate_count)
    with open(cate_path, 'w')as f:
        for k, v in cate_count_list:
            # print(k, v)
            f.write("{0}\t{1}\n".format(k, v))


def cate_save(context_list, data_type):
    save_dir = "./data/xfyun"
    data_save = os.path.join(save_dir, "{0}.{1}.txt".format("xunfei", data_type))
    print("save path:", data_save)
    save_txt(data_save, context_list, model='w')


def generate_data():
    data_path = 'data/xfyun/apptype_train.txt'
    with open(data_path, 'r', encoding='utf-8') as f:
        data_list = f.read().splitlines()
        print(len(data_list))
        random.shuffle(data_list)
        val_list = data_list[0:1000]
        cate_save(val_list, "val")
        train_list = data_list[1000:]
        cate_save(train_list, "train")


if __name__ == "__main__":
    # python xunfei/pre_classify.py
    clean_data()
    generate_data()
