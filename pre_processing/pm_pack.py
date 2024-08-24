# -*- coding: utf-8 -*-
import h5py
import joblib
import os
from PIL import Image
import numpy as np


def pack(dir_stft):
    feature_ori_list = []
    feature_ck_list = []
    feature_wh_list = []
    feature_res_list = []
    label_list=[]
    for file in os.listdir(dir_stft):
        I_stft = Image.open(dir_stft+file).convert('L')
        I_stft = np.array(I_stft)
        feature_ori_list.append(I_stft)
        I_stft = Image.open(dir_stft+'../ck/'+file).convert('L')
        feature_ck_list.append(np.array(I_stft))
        I_stft = Image.open(dir_stft+'../wh/'+file).convert('L')
        feature_wh_list.append(np.array(I_stft))
        I_stft = Image.open(dir_stft + '../res/' + file).convert('L')
        feature_res_list.append(np.array(I_stft))
        txt_dir = '../data/ICBHI/'

        num = int(file.split('.')[0][22:])
        txt_name = file[:22] + '.txt'
        array = np.loadtxt(txt_dir + txt_name)
        crackles = array[:, 2:4][int(num), 0]
        wheezes = array[:, 2:4][int(num), 1]
        if crackles == 0 and wheezes == 0:
            label = 0
        elif crackles == 1 and wheezes == 0:
            label = 1
        elif crackles == 0 and wheezes == 1:
            label = 2
        else:
            label = 3
        label_list.append(label)
    return feature_ori_list, feature_ck_list, feature_wh_list, feature_res_list, label_list

def one_hot(x, K):
    # x is a array from np
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

if __name__ == '__main__':
    com_path = '../analysis/tqwt/train/'

    ori, ck, wh, res, label = pack(com_path+'ori/')
    joblib.dump((ori, ck, wh, res, label), open('../pack/official/tqwt1_4_train.p', 'wb'))
    print('Done!')

