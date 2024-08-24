#!/usr/bin/python                                                       


import os
import random
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

import librosa
from tqdm import tqdm
from config.utils import  *
from  config.gtg import  *

import torch
import torchaudio
from  torchaudio import  functional as F



import matplotlib.pyplot as plt
import numpy as np

#from IPython.display import Audio, display

from tqwt_tools import DualQDecomposition





# https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_challenge_train_test.txt
# official 6-4 split dataset;
# code phrase
class image_loader_pitch_1channel(Dataset):
    def __init__(self, data_dir, folds_file, test_fold, train_flag, params_json, input_transform=None, stetho_id=-1,
                 aug_scale=None):

        "part1: 取出所有的文件名称中，设备名称，病人标号；　然后统计每个设备下，　所包含的病人编号　"
        # getting device-wise information
        # self.file_to_device: 是一个字典，　
        self.file_to_device = {} # 该字典中包含920项内容，key: 文件名称，　value: 该文件名所对应的设备编号；　
        device_to_id = {}  #　字典中包含四项内容，给四个设备编号从0-3,　key: 设备名称，　value: 设备编号；
        device_id = 0  # 用于给设备编号，　遇到的第一个设备编号为０；
        files = os.listdir(data_dir) #data_dir 该路径下所包含的所有文件,　1840= 920份音频文件 + 920份标注文件；
        device_patient_list = [] #二维列表，维度１代表设备编号，维度2代表该设备编号下所对应的病人编号；代表每个设备编号号下，　所包含的病人的编号；　
        pats = []   # 用于存放126个的病人编号，　编号从100-226;

        for f in files:
            device = f.strip().split('_')[-1].split('.')[0]
            # device: 文件名中的设备名；　通过对文件名按下划线划分，取出最后一个下划线后的位置；
            if device not in device_to_id:
                device_to_id[device] = device_id   # 第一个设备的编号从0开始；
                #  device_to_id[] 字典包含４个内容，分别包括４个设备， key: 设备名称AKGC，　value: 设备的编号0~3； 　
                device_id += 1 # 设备名称的编号+１，　代表下一个设备编号；
                device_patient_list.append([])  # 增加设备与病人的编号之间的关系列表；

            self.file_to_device[f.strip().split('.')[0]] = device_to_id[device]
            # file_to_device:　920项内容，表示文件名称以及所对应的设备编号;  key: 文件名称，　value:　设备编号；　

            pat = f.strip().split('_')[0] #　取出当前文件名中的病人编号；　 通过将文件名称按照下划线分割，第一个字符串代表:　病人编号
            if pat not in device_patient_list[device_to_id[device]]:
                # 若病人编号不在当前设备编号所对应的列表中,  则将该病人编号加入　当前设备编号下(0-3)的列表中
                # device_to_id[设备名]：设备编号；　device_patient_list[设备编号]：当前设备编号所对应的列表中，　所包含的已有的病人编号　
                device_patient_list[device_to_id[device]].append(pat)
            if pat not in pats:
                pats.append(pat)

        print("DEVICE DICT", device_to_id)
        for idx in range(device_id):  # 从而现在打印出四个设备，每个设备编号下，　所包含的病人的个数；　
            print("Each　Device cover the numbers of patients", idx, len(device_patient_list[idx]))   # 显示每个设备所对应病人的个数



        "part2: 　获取病人的字典在当前的折数中；　"
        # get patients dict in current fold based on train flag
        # folds_file: 每个病人编号所在对应的折数中；　　5个folds ，每个folds 中的对应的病人编号；
        all_patients = open(folds_file).read().splitlines()
        # 列表中126项，　每个病人编号所对应的折数；


        patient_dict = {}  # key: patient编号--idx， value:　fold--第几折0-4;
        # 当test_fold =4 时，
        # 在训练阶段时， patient_dict[] 　只会有101 项病人， 不会包含第四折中的病人；
        # 在非训练阶段时， patient_dict[] 只会有25 项病人， 只包含第四折中的病人；


        for line in all_patients:
            idx, fold = line.strip().split(' ')  # 取出病人病号，　第几折；
            if train_flag and int(fold) != test_fold:
                patient_dict[idx] = fold  #　如果是训练阶段，　则将除了test_fold折数中，其余的四折病人加入到其中；
            elif train_flag == False and int(fold) == test_fold:
                patient_dict[idx] = fold  #　如果非训练阶段，　则将test_fold折数中的病人加入到其中；



        "part3: 　获取病人的字典在当前的折数中；　"
        # extracting the audiofilenames and the data for breathing cycle and it's label
        print(" Getting filenames  ...")
        # filenames: 920 份文件名称，
        # rec_annotation_dict: 字典中包含了920项内容，key: 文件名，　value: dataFrame
        # 每个内容包含了该音频的分段标注信息， 即起始 ，终止时间， crackle, wheeze；
        filenames, rec_annotations_dict = get_annotations(data_dir)

        # 提取出文件名称，　rec_annotations该文件中多个呼吸分段的起始，　终止时间以及对应的label;
        if stetho_id >= 0: # 是否按照设备型号，进行划分；
            self.filenames = [s for s in filenames if
                              s.split('_')[0] in patient_dict and self.file_to_device[s] == stetho_id]
        else:
            # patient_dict　代表其中所包含的病人编号，
            # slef.filenames: 代表了　从filenames中的920份文件中取出　patient_dict病人编号所对应的文件；
            self.filenames = [s for s in filenames if s.split('_')[0] in patient_dict]
            # 当test_fold =4 时，
            # 在训练阶段时， patient_dict[] 会有101 项病人，
            # 此时， self.filenames = 722　个文件名称；:  从filenames(920)份文件 中取出，patient_dict的病人不会包含第四折中的病人；

            # 在非训练阶段时， patient_dict[] 会有25 项病人， 只包含第四折中的病人；
            # 此时， self.filenames = 198　个文件；

        # each sample is a tuple with
        # id_0: audio_data, id_1: label, id_2: file_name,
        # id_3: cycle id, id_4: aug id, id_5: split id
        self.audio_data = []

        self.labels = []
        self.train_flag = train_flag
        self.data_dir = data_dir
        self.input_transform = input_transform

        # parameters for spectrograms
        self.sample_rate = 10000
        self.desired_length = 8
        self.n_mels = 128
        self.nfft = 1024
        self.hop =  100   #  self.nfft // 2
        self.f_max = 2000

        self.dump_images = False  # You can set False if already exits;
        self.filenames_with_labels = []


        "part４: 从每一份完整音频文件中获取，　分割出每个小的音频　"
        # get individual breathing cycles from each audio file
        print("Exracting Individual Cycles")
        self.cycle_list = []
        self.classwise_cycle_list = [[], [], [], []]
        # classwise_cycle_list:  是一个列表，总共４个，按照标签将四个类的呼吸音周期，
        # 列表张每一项，都包含了该类别下，所有的子音频；
        # 每个子音频的数据都是一个五元组，　每个元组的组成如下  = [原始数据，标签，文件名，当前文件名中的第几个子音频，自带的占位０]

        self.classes_with_duration_list = [ [], [], [], [] ]
        # 按照类别将，　　每个类别下各个子音频的持续时间添加到其中；





        # for idx, file_name in tqdm(enumerate(self.filenames)):
        # 取出self.filenames 中对应的文件名，　在测试和训练阶段不同；
        for idx, file_name in enumerate(tqdm(self.filenames, desc='get individual cycles from each audio file:')):

            # note : d[i][3]:　类别标签　０-normal, 1-crackle, 2-wheeze
            data = get_sound_samples(rec_annotations_dict[file_name], file_name, data_dir, self.sample_rate)
            # data　是一个列表，其中个数依据当前的record中划分了多少个子音频片段出来；
            # data[0]: str文件名称，　data[1:n]: 　代表了划分出n个子音频片段出来；

            #  data[i]: 　i=0, 表示当前record的文件名称；  当i从1开始时，每个子音频使用元组表示；
            #　data[i][j]每个子音频片段用元组表示，　每个元组中所包含的内容有四项：　
            # 第一项：该子音频所对应的原始音频数据，
            # 第二项，三项代表该子音频所对应的起始，终止时间；　
            #　四项：　该子音频所对应的类别标签；

            # 1. 统计出四个类别下， 每个类别下, 各自样本所持续的时间； dur =  end - start;
            for cycle_in_curr_record, cur_data in  enumerate( data[1:]  ):
                cycle_dur =  cur_data[2] - cur_data[1]
                #　由于此时的 cur_data[3] 代表的是子音频的标签，　所以范围0-3　符合四个列表的范围；
                self.classes_with_duration_list[cur_data[3]].append(cycle_dur)




            #  cycle_idx： 提取出的是第几个子音频；
            # ｄ: 该子音频下，所对应的元组，　每个元组包含五个成分；
            cycles_with_labels = [(d[0], d[3], file_name, cycle_idx, 0) for cycle_idx, d in enumerate(data[1:])]
            # cycles_with_labels：将当前record中所有子音频的信息添加到其中，从而构成一个列表，　列表中的个数＝　当前子音频的个数，
            # 列表是中每一项都是tuple元组，
            # 每个元组的组成如下 = [当前子音频数据，标签， 文件名，当前文件名中的第几个子音频，自带的占位０]


            self.cycle_list.extend(cycles_with_labels)
            # cycles_with_labels:　是将当前record的子音频添加到其中；
            # cycle_list: 则是将所有record　所对应的子音频全部放在一起，从而构成一个列表；
            # 列表中每一项同上，组成如下= [子音频数据，标签，文件名，当前文件名中的第几个子音频，自带的占位０]　　

            # cycle_idx： 当前record中的第几个子音频，即cycles_with_labels中第几项；
            # ｄ: 这里的d代表即上述列表中的每一项具体的子音频，而列表中每一项都是元组，
            for cycle_idx, d in enumerate(cycles_with_labels):
                # 元组中的数据，即ｄ[3]　第几个呼吸周期，ｄ[1]:类别标签；　
                # self.filenames_with_labels() ; 文件名称 +　当前文件名第几个呼吸周期　+　标签
                self.filenames_with_labels.append(file_name + '_' + str(d[3]) + '_' + str(d[1]))
                self.classwise_cycle_list[d[1]].append(d)
                # classwise_cycle_list: 是一个列表总共４项，
                # 按照标签将四个类的呼吸音周期，将每个子音频划分到所对应的类别列表当中去；

        # concatenation based augmentation scheme
        if train_flag and aug_scale:
            self.new_augment(scale=aug_scale)
            #　将新生成的数据加入其中，　增加到 cycle_list(),  filenames_with_labels　这两项中的内容；

        #     '''　以下为新生成的数据格式，不匹配出错的地方；
        #     self.cycle_list.append((new_sample, 0, normal_i[2] + '-' + normal_j[2], idx, 0))
        #     # cycle_list:　
        #     原始的数据格式为：则是将所有record　所对应的子音频全部放在一起，从而构成一个列表；
        #     # 列表中每一项组成如下= [子音频数据，标签，文件名，当前文件名中的第几个子音频，自带的占位０]　
        #
        #     但是，新添加的后，合成音频的数据格式却不同：
        #     # ( 拼接的子音频数据，　类别标签，　子音频i所对应的文件名称-子音频j所对应的文件名称, 在当前类别下是第几个新增加的样本, 占位0　)
        #     '''
        #
        #
        #     '''
        #     self.filenames_with_labels.append(normal_i[2] + '-' + normal_j[2] + '_' + str(idx) + '_0')
        #     # filenames_with_labels:
        #      原始的数据格式：  str = ' 文件名称_当前文件名第几个呼吸周期_类别标签　'
        #
        #     但是，新添加的后，合成音频的数据格式却不同：
        # 　　 str = '子音频i所对应的文件名称-子音频j所对应的文件名称_在当前类别下是第几个新增加的样本_类别标签'
        #     '''




        "part5,　　拼接到统一的固定长度　"
        # split and pad each cycle to the desired length
        #     # cycle_list:　列表，则是将所有record　所对应的子音频全部放在一起，从而构成一个列表；
        #   列表中每一项组成如下= [子音频数据，标签，文件名，当前文件名中的第几个子音频，自带的占位０]　
        #   idx, 列表中第几项，　每一项的内容如下：
        # 　 sample = [原始数据，　标签，　文件名，　第几个呼吸周期，　自带的占位０]

        for idx, sample in enumerate(self.cycle_list):
            output = split_and_pad(sample, self.desired_length, self.sample_rate, types=1)
            self.audio_data.extend(output)

        # note 这是初始化后，　第一次给self.audio_data 赋值；
        # output =  padded: 补齐后音频＝　原始音频 + 原始音频的重复片段；
        # 　original[1], 2,3,4 = sample[1], 2,3,4;  占位0；  pad_times: 代表补齐长度是原始音频的倍数关系；
        #  self.audio_data 是一个列表;  列表个数便是cycle_list　中所有子音频中的个数，包括新添加的样本；
        #  列表中的每一项都是一个元组，　元组中包含七个成分：　
        #  （0 - 补齐后的子音频数据，1-标签，2 -文件名，　3-文件名称中的第几个子音频，　4-自带占位０，　5-重新增加的占位０，6 - 补齐长度的倍数；）
        # fold=4, 测试集　144４份；


        "part6,　获取音频样本与设备之间的关系"
        self.device_wise = []
        for idx in range(device_id):
            self.device_wise.append([])    # self.device_wise 是四个列表，　每一个列表　
        self.class_probs = np.zeros(4)
        self.identifiers = []
        for idx, sample in enumerate(self.audio_data):
            # sample[1]: 标签　
            # self.class_probs[] 将该类别下， 对应的个数加１ ,
            self.class_probs[sample[1]] += 1.0
            # 将当前的音频的标签添加到其中，　最终形成当前训练，测试阶段下所含的所有的标签，从而统计个数；
            self.labels.append(sample[1])

            # identifiers 是一个列表，　每个列表的组成成分为：
            # identifiers = 　str = 文件名_ 该文件中的第几个子音频_类别标签#
            self.identifiers.append(sample[2] + '_' + str(sample[3]) + '_' + str(sample[1]))

            self.device_wise[self.file_to_device[sample[2]]].append(sample)
            # sample[2] :文件名， 通过文件名取出设备编号;　 file_to_device :　字典中包含920个项， key: 文件名称，　value:　设备编号
            #  file_to_device[文件名]　＝　设备编号， 　
            #  device_wise[设备编号]　＝　将当前从self.audio_data中取出的一个样本子音频　添加到其中；　
            # self.device_wise: 是一个列表，总共４个，　代表了四个设备；　
            # 从而将所有的子音频数据，　按照设备，　划分出四个出来；



        if self.train_flag:
            print("TRAIN DETAILS")
        else:
            print("TEST DETAILS")

        print("CLASSWISE SAMPLE COUNTS:", self.class_probs)
        print("Device to ID", device_to_id)
        for idx in range(device_id):
            print("DEVICE ID", idx, "size", len(self.device_wise[idx]))
        self.class_probs = self.class_probs / sum(self.class_probs)
        print("CLASSWISE PROBS", self.class_probs)
        print("LEN AUDIO DATA", len(self.audio_data))

    def new_augment(self, scale=1):

        # classwise_cycle_list:  是一个列表总共４个，按照标签将四个类的呼吸音周期，　分别划分到其中，　列表中的每一个元素有５个成分组成；
        # 每个元组的组成如下：= [子音频数据，　类别标签，　文件名，文件名中第几个呼吸周期，自带的占位０]

        # augment normal    # classwise_cycle_list[0]: 代表正常类normal的类的列表，　该列表中的classwise_cycle_list[]
        aug_nos = scale * len(self.classwise_cycle_list[0]) - len(self.classwise_cycle_list[0])
        for idx in range(aug_nos):
            # normal_i + normal_j
            i = random.randint(0, len(self.classwise_cycle_list[0]) - 1)
            j = random.randint(0, len(self.classwise_cycle_list[0]) - 1)
            normal_i = self.classwise_cycle_list[0][i]      # 第０个列表，代表normal类别下的子音频，　每个子音频是一个元组，由上面的五个成分组成；
            normal_j = self.classwise_cycle_list[0][j]
            new_sample = np.concatenate([normal_i[0], normal_j[0]])
            self.cycle_list.append((new_sample, 0, normal_i[2] + '-' + normal_j[2],idx, 0))
            # cycle_list:　将新生成的数据加入到其中，　　其对应的格式如下：　
            # ( 拼接的子音频数据，　类别标签，　子音频i所对应的文件名称-子音频j所对应的文件名称, 在当前类别下是第几个新增加的样本, 占位0　)
            # 格式为[new_sample -拼接的子音频数据，　0-类别标签，
            # normal_i[2]　子音频i所对应的文件名称， 中间连接符－normal_j[2]:子音频j所对应的文件名称；
            # 　idx，在当前类别下，是第几个新增加的样本。


            self.filenames_with_labels.append(normal_i[2] + '-' + normal_j[2] + '_' + str(idx) + '_0')
            # filenames_with_labels:  将新生成的样本加入到其中，对应的格式如下：
            # normal_i[2]:　子音频i所对应的文件名称， 中间连接符－　normal_j[2]:子音频j所对应的文件名称；
            # str =  子音频i所对应的文件名称-子音频j所对应的文件名称_在当前类别下是第几个新增加的样本_类别标签；




        # augment crackle   classwise_cycle_list[1]：　代表crackle 类，列表其中的每一项代表该类别下的呼吸音周期，
        aug_nos = scale * len(self.classwise_cycle_list[0]) - len(self.classwise_cycle_list[1])
        for idx in range(aug_nos):
            aug_prob = random.random()

            if aug_prob < 0.6:
                # crackle_i + crackle_j
                i = random.randint(0, len(self.classwise_cycle_list[1]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[1]) - 1)
                sample_i = self.classwise_cycle_list[1][i]
                sample_j = self.classwise_cycle_list[1][j]
            elif aug_prob >= 0.6 and aug_prob < 0.8:
                # crackle_i + normal_j
                i = random.randint(0, len(self.classwise_cycle_list[1]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[0]) - 1)
                sample_i = self.classwise_cycle_list[1][i]
                sample_j = self.classwise_cycle_list[0][j]
            else:
                # normal_i + crackle_j
                i = random.randint(0, len(self.classwise_cycle_list[0]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[1]) - 1)
                sample_i = self.classwise_cycle_list[0][i]     # classwise_cycle_list[０][ｉ] : 代表了该类别下的第i个呼吸音周期；
                sample_j = self.classwise_cycle_list[1][j]

            new_sample = np.concatenate([sample_i[0], sample_j[0]])  # sample_i[0]:　　【０】代表了取出数据项中的，原始音频数据部分，
            self.cycle_list.append((new_sample, 1, sample_i[2] + '-' + sample_j[2], idx, 0))
            # cycle_list:　将新生成的数据加入到其中，　格式为　[　0-新数据，　1-类别标签，　2-新生成数据的合成文件名称，　3-该类别下是第几个新数据，　4-占位０]
            self.filenames_with_labels.append(sample_i[2] + '-' + sample_j[2] + '_' + str(idx) + '_1')
            # filenames_with_labels:  　新数据的文件名称+ 该类别下第几个生成的新数据　+ 类别标签


        # augment wheeze
        aug_nos = scale * len(self.classwise_cycle_list[0]) - len(self.classwise_cycle_list[2])
        for idx in range(aug_nos):
            aug_prob = random.random()

            if aug_prob < 0.6:
                # wheeze_i + wheeze_j
                i = random.randint(0, len(self.classwise_cycle_list[2]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[2]) - 1)
                sample_i = self.classwise_cycle_list[2][i]
                sample_j = self.classwise_cycle_list[2][j]
            elif aug_prob >= 0.6 and aug_prob < 0.8:
                # wheeze_i + normal_j
                i = random.randint(0, len(self.classwise_cycle_list[2]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[0]) - 1)
                sample_i = self.classwise_cycle_list[2][i]
                sample_j = self.classwise_cycle_list[0][j]
            else:
                # normal_i + wheeze_j
                i = random.randint(0, len(self.classwise_cycle_list[0]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[2]) - 1)
                sample_i = self.classwise_cycle_list[0][i]
                sample_j = self.classwise_cycle_list[2][j]

            new_sample = np.concatenate([sample_i[0], sample_j[0]])
            self.cycle_list.append((new_sample, 2, sample_i[2] + '-' + sample_j[2], idx, 0))
            self.filenames_with_labels.append(sample_i[2] + '-' + sample_j[2] + '_' + str(idx) + '_2')



        # augment both
        aug_nos = scale * len(self.classwise_cycle_list[0]) - len(self.classwise_cycle_list[3])
        for idx in range(aug_nos):
            aug_prob = random.random()

            if aug_prob < 0.5:
                # both_i + both_j
                i = random.randint(0, len(self.classwise_cycle_list[3]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[3]) - 1)
                sample_i = self.classwise_cycle_list[3][i]
                sample_j = self.classwise_cycle_list[3][j]
            elif aug_prob >= 0.5 and aug_prob < 0.7:
                # crackle_i + wheeze_j
                i = random.randint(0, len(self.classwise_cycle_list[1]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[2]) - 1)
                sample_i = self.classwise_cycle_list[1][i]
                sample_j = self.classwise_cycle_list[2][j]
            elif aug_prob >= 0.7 and aug_prob < 0.8:
                # wheeze_i + crackle_j
                i = random.randint(0, len(self.classwise_cycle_list[2]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[1]) - 1)
                sample_i = self.classwise_cycle_list[2][i]
                sample_j = self.classwise_cycle_list[1][j]
            elif aug_prob >= 0.8 and aug_prob < 0.9:
                # both_i + normal_j
                i = random.randint(0, len(self.classwise_cycle_list[3]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[0]) - 1)
                sample_i = self.classwise_cycle_list[3][i]
                sample_j = self.classwise_cycle_list[0][j]
            else:
                # normal_i + both_j
                i = random.randint(0, len(self.classwise_cycle_list[0]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[3]) - 1)
                sample_i = self.classwise_cycle_list[0][i]
                sample_j = self.classwise_cycle_list[3][j]

            new_sample = np.concatenate([sample_i[0], sample_j[0]])
            self.cycle_list.append((new_sample, 3, sample_i[2] + '-' + sample_j[2],idx, 0))
            self.filenames_with_labels.append(sample_i[2] + '-' + sample_j[2] + '_' + str(idx) + '_3')


    def __getitem__(self, index):

        audio = self.audio_data[index][0]

        aug_prob = random.random()
        if self.train_flag and aug_prob > 0.5:
            # apply augmentation to audio
            audio = gen_augmented(audio, self.sample_rate)
            # todo  check  0000 ;
            # pad incase smaller than desired length
            audio = split_and_pad([audio, 0, 0, 0, 0], self.desired_length, self.sample_rate, types=1)[0][0]

        # roll audio sample
        roll_prob = random.random()
        if self.train_flag and roll_prob > 0.5:
            audio = rollAudio(audio)


        # convert audio signal to spectrogram
        # spectrograms resized to 3x of original size

        # audio_image = cv2.cvtColor(create_mel_raw(audio, self.sample_rate, f_max=self.f_max,
        #                                         n_mels=self.n_mels, nfft=self.nfft, hop=self.hop, resz=3),
        #                         cv2.COLOR_BGR2RGB)



        #  print(" -----------------------  1. the original torch spectrogram part --------------------")
        #  生成三通道的窄带语谱图，  接口函数 create_narrow_width_band_spectrogram(spectrogram, resz=1):
        #  并且调整语谱图的大小为 224 × 224；

        torch_spec_fun = torchaudio.transforms.Spectrogram(n_fft=self.nfft, win_length= 1024, hop_length = self.hop)
        torch_spectrogram = torch_spec_fun(torch.Tensor(audio))
        # print(" 1.1 . torch_ spectrogram  shape should be [513, 800]: \n", torch_spectrogram.shape)

        # print(" -----------------------  2. the Mel  spectrogram part --------------------")
        stft_len = self.nfft // 2 + 1
        melFilters = F.create_fb_matrix(n_freqs=stft_len, f_min=50, f_max= 2000, n_mels=128, sample_rate= self.sample_rate )
        # print( "2.1 torch mel filters shape should be  [513, 128] \n", melFilters.shape)

        torch_mel_spec = torch.matmul(torch_spectrogram.transpose(0,1), melFilters).transpose(0,1)
        # print("2.2 torch mel spectrogram  shape should be  [128, 800] \n",  torch_mel_spec.shape)

        # 生成单通道的Mel语谱图，  接口函数 create_mel_spectrogram_to_1channel(spectrogram,  resz=0):

        torch_mel_spec = torch_mel_spec[:, : 800]
        torch_mel_spec = create_normalized_spectrogram(torch_mel_spec, resz=0)
        # print("2.2-1 normalize spectrogram: should be  [128, 800]\n", torch_mel_spec.shape)

        audio_image = np.expand_dims(torch_mel_spec, axis=2)
        #audio_image = torch_mel_spec.unsqueeze(dim=2)
        # print("2.3 image_dataloader.py:  After  unsqueeze,   torch_ Mel_spectrogram  shape should be [ 128, 800, 1]: \n",  audio_image.shape)
        # audio_image = audio_image.numpy()
        # print("2.4 After convert,   torch_ Mel_spectrogram  shape should be [ 128, 800, 1]: \n", audio_image.shape)

        # print("2.3 after 3 channel, torch mel spectrogram  shape should be  [128, 800,3] \n", mel_spec_3channel.shape)



        if self.dump_images:
            save_images((audio_image, self.audio_data[index][2], self.audio_data[index][3],
                         self.audio_data[index][5], self.audio_data[index][1]), self.train_flag)

        # label
        label = self.audio_data[index][1]

        # apply image transform
        if self.input_transform is not None:
            audio_image = self.input_transform(audio_image)

        return audio_image, label

    def __len__(self):
        return len(self.audio_data)




class image_loader_Freq_mask_sr20k_11channel(Dataset):
    '''
      use
      1. Freq     mask spec   1 channel + 3 channel
      2. Respire  Transform  spec 1 channel + 3 channel
      2.  original  spec    convert to  the  3 Mel_spec channel
    '''
    def __init__(self, data_dir, folds_file, test_fold, train_flag, params_json, input_transform=None, stetho_id=-1,
                 aug_scale=None):
        '''
        Args:
            data_dir:
            folds_file:
            test_fold:
            train_flag:
            params_json:
            input_transform:
            stetho_id:
            aug_scale:
        '''
        # getting device-wise information
        self.file_to_device = {}  # self.file_to_device: 是一个字典，　920个字典，　代表920个文件，所对应的设备号码；

        device_to_id = {}  # 给四个设备编号；
        device_id = 0
        files = os.listdir(data_dir)  # 该文件路径下，总共文件的个数；
        device_patient_list = []
        pats = []
        for f in files:
            device = f.strip().split('_')[-1].split('.')[0]  # device: 取出文件名中的设备名，　对文件名按下划线划分，　取出最后一个下划线后的位置；
            if device not in device_to_id:
                device_to_id[device] = device_id  # device_to_id[]  是4个字典， key: 设备名称，　value: 设备的编号； 　
                device_id += 1  # 设备名称的编号+１，　代表下一个设备；
                device_patient_list.append([])  # 增加设备与病人的编号之间的关系列表；
            self.file_to_device[f.strip().split('.')[0]] = device_to_id[device]
            # file_to_device :　920个字典， key: 文件名称，　value:　设备编号；　

            pat = f.strip().split('_')[0]  # 病人编号
            if pat not in device_patient_list[device_to_id[device]]:
                device_patient_list[device_to_id[device]].append(pat)
            if pat not in pats:
                pats.append(pat)

        print("DEVICE DICT", device_to_id)
        for idx in range(device_id):
            print("Each　Device cover the numbers of patients", idx, len(device_patient_list[idx]))  # 显示每个设备所对应病人的个数

        # get patients dict in current fold based on train flag
        all_patients = open(folds_file).read().splitlines()  # 5个folds ，每个folds 中的对应的病人编号；
        patient_dict = {}
        for line in all_patients:
            idx, fold = line.strip().split(' ')
            if train_flag and int(fold) != test_fold:
                patient_dict[idx] = fold
            elif train_flag == False and int(fold) == test_fold:
                patient_dict[idx] = fold

        # extracting the audiofilenames and the data for breathing cycle and it's label
        print(" Getting filenames  ...")
        filenames, rec_annotations_dict = get_annotations(
            data_dir)  # 提取出文件名称，　rec_annotations该文件中多个呼吸分段的起始，　终止时间以及对应的label;
        if stetho_id >= 0:
            self.filenames = [s for s in filenames if
                              s.split('_')[0] in patient_dict and self.file_to_device[s] == stetho_id]
        else:
            self.filenames = [s for s in filenames if s.split('_')[0] in patient_dict]

        # each sample is a tuple with id_0: audio_data, id_1: label, id_2: file_name, id_3: cycle id, id_4: aug id, id_5: split id
        self.audio_data = []

        self.labels = []
        self.train_flag = train_flag
        self.data_dir = data_dir
        self.input_transform = input_transform

        # parameters for spectrograms
        self.sample_rate = 20000
        self.desired_length = 8
        self.n_mels = 64
        self.nfft = 512
        self.hop = 512 #  no overlap  between the  frames;
        self.f_max = 2000

        self.dump_images = False  # You can set False if already exits;
        self.filenames_with_labels = []

        # get individual breathing cycles from each audio file
        print("Exracting Individual Cycles")
        self.cycle_list = []
        self.classwise_cycle_list = [[], [], [], []]
        # classwise_cycle_list:  是一个列表总共４个，按照标签将四个类的呼吸音周期，　分别划分到其中，　列表中的每一个元素有５个成分组成，；
        # 每个元组的组成如下：= [原始数据，　标签，　文件名，　第几个呼吸周期，　自带的占位０]

        # for idx, file_name in tqdm(enumerate(self.filenames)):
        for idx, file_name in enumerate(tqdm(self.filenames, desc='get individual cycles from each audio file:')):
            # 　提取每个音频中文件名称，　以及对应的分段信息和标签；
            # d[0]: 原始音频数据，　d[1],d[2]代表起始终止时间；　d[3]:　类别标签　０-normal, 1-crackle, 2-wheeze
            data = get_sound_samples(rec_annotations_dict[file_name], file_name, data_dir, self.sample_rate)
            cycles_with_labels = [(d[0], d[3], file_name, cycle_idx, 0) for cycle_idx, d in enumerate(data[1:])]
            '''
            cycles_with_labels 是一个列表list: list的个数，是通过将各个原始音频中的呼吸音周期切割之后，添加到其中；
            每一个列表是tuple元组，　　每个元组的组成如下：　= 　[原始数据，　标签，　文件名，　第几个呼吸周期，　自带的占位０]
            cycle_idx 代表第几个列表；
            ｄ: 代表了当前列表中的元组，　即该元组中的五个成分；
            '''
            self.cycle_list.extend(cycles_with_labels)  # 每个列表的组成如下：= 　[原始数据，　标签，　文件名，　第几个呼吸周期，　自带的占位０]　　
            for cycle_idx, d in enumerate(cycles_with_labels):
                # 注意，这里的d代表即上述列表中的一个元组中的数据，即ｄ[3]　第几个呼吸周期，ｄ[1]:标签
                self.filenames_with_labels.append(file_name + '_' + str(d[3]) + '_' + str(d[1]))  # 文件名称，+　第几个呼吸周期　+　标签
                self.classwise_cycle_list[d[1]].append(d)  # classwise_cycle_list:  是一个列表总共４个，按照标签将四个类的呼吸音周期，　分别划分到其中，　列表中的每一个元素有５个成分组成，；

        # concatenation based augmentation scheme
        if train_flag and aug_scale:
            self.new_augment_same_class(scale=aug_scale)
            # 　将新生成的数据加入其中，　增加到 cycle_list(),  filenames_with_labels　这两项中的内容；

            '''
            self.cycle_list.append((new_sample, 0, normal_i[2] + '-' + normal_j[2], idx, 0))
            # cycle_list:　将新生成的数据加入到其中，　格式为　[0-新数据，　1-类别标签，　2-新生成数据的合成文件名称，　3-该类别下是第几个新数据，　4-占位０]
            self.filenames_with_labels.append(normal_i[2] + '-' + normal_j[2] + '_' + str(idx) + '_0')
            # filenames_with_labels:  　新数据的文件名称+ 该类别下第几个生成的新数据　+ 类别标签
            '''

        # split and pad each cycle to the desired length
        for idx, sample in enumerate(self.cycle_list):
            output = split_and_pad(sample, self.desired_length, self.sample_rate, types=1)
            self.audio_data.extend(output)
            # sample = [原始数据，　标签，　文件名，　第几个呼吸周期，　自带的占位０]
            # output =  padded: 补齐后音频＝　原始音频 + 原始音频的重复片段；
            # 　original[1], 2,3,4 = sample[1], 2,3,4;  占位0；  pad_times: 代表补齐长度是原始音频的倍数关系；
        '''
        通过 self.cycle_list 生成 self.audio_data(),  具体实现　self.cycle_list　提取每个呼吸音周期，　 输入到 split_and_pad()　函数中去，　生成output；
        self.audio_data 是一个列表，列表个数，即加入到其中所有呼吸音周期的个数；
        每个列表是一个元组，　元组中包含七个成分：　（0-补齐后的呼吸音周期，　1-标签，　2-文件名，　3-第几个呼吸音周期，　4-占位０，　5-占位０，6-补齐长度的倍数；）
        '''

        self.device_wise = []
        for idx in range(device_id):
            self.device_wise.append([])  # self.device_wise 是四个列表，　每一个列表　
        self.class_probs = np.zeros(4)
        self.identifiers = []
        for idx, sample in enumerate(self.audio_data):
            self.class_probs[sample[1]] += 1.0  # self.class_probs[] 将该类别下， 对应的个数加１ , sample[1]: 标签　　　
            self.labels.append(sample[1])
            # identifiers 是一个列表，　每个列表的组成成分为：identifiers = 文件名 + 第几个周期 + 标签#

            self.identifiers.append(sample[2] + '_' + str(sample[3]) + '_' + str(sample[1]))
            # sample[2] 文件名， 通过文件名取出设备编号，　sample 是一个元组，　总共包含７个成分；
            # self.device_wise: 是一个列表，总共４个， 每一个列表, 包含了该设备下所有的呼吸音周期；
            # file_to_device :　920个字典， key: 文件名称，　value:　设备编号；　
            # self.device_wise[self.file_to_device[sample[2]]].append(sample)
            # file_to_device[文件名]　＝　设备编号，  device_wise[设备编号]　＝　第几个列表，　用于确定将数据添加到哪个设备的列表下；

        if self.train_flag:
            print("TRAIN DETAILS")
        else:
            print("TEST DETAILS")

        print("CLASSWISE SAMPLE COUNTS:", self.class_probs)
        print("Device to ID", device_to_id)
        for idx in range(device_id):
            print("DEVICE ID", idx, "size", len(self.device_wise[idx]))
        self.class_probs = self.class_probs / sum(self.class_probs)
        print("CLASSWISE PROBS", self.class_probs)
        print("LEN AUDIO DATA", len(self.audio_data))

    def new_augment_same_class(self, scale=1):

        # classwise_cycle_list:  是一个列表总共４个，按照标签将四个类的呼吸音周期，　分别划分到其中，　列表中的每一个元素有５个成分组成，；
        # 每个元组的组成如下：= [原始数据，　标签，　文件名，　第几个呼吸周期，　自带的占位０]

        # augment normal    # classwise_cycle_list[0]: 代表正常类normal的类的列表，　该列表中的classwise_cycle_list[]
        aug_nos = scale * len(self.classwise_cycle_list[0]) - len(self.classwise_cycle_list[0])
        for idx in range(aug_nos):
            # normal_i + normal_j
            i = random.randint(0, len(self.classwise_cycle_list[0]) - 1)
            j = random.randint(0, len(self.classwise_cycle_list[0]) - 1)
            normal_i = self.classwise_cycle_list[0][i]
            normal_j = self.classwise_cycle_list[0][j]
            new_sample = np.concatenate([normal_i[0], normal_j[0]])
            self.cycle_list.append((new_sample, 0, normal_i[2] + '-' + normal_j[2], idx, 0))
            # cycle_list:　将新生成的数据加入到其中，　格式为[0-新数据，　1-类别标签，　2-新生成数据的合成文件名称，　3-该类别下是第几个新数据，　4-占位０]
            self.filenames_with_labels.append(normal_i[2] + '-' + normal_j[2] + '_' + str(idx) + '_0')
            # filenames_with_labels:  　新数据的文件名称+ 该类别下第几个生成的新数据　+ 类别标签

        # augment crackle   classwise_cycle_list[1]：　代表crackle 类，列表其中的每一项代表该类别下的呼吸音周期，
        aug_nos = scale * len(self.classwise_cycle_list[0]) - len(self.classwise_cycle_list[1])
        for idx in range(aug_nos):
            aug_prob = random.random()

            # aug_prob = 0.5
            if aug_prob < 0.8:
                # crackle_i + crackle_j
                i = random.randint(0, len(self.classwise_cycle_list[1]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[1]) - 1)
                sample_i = self.classwise_cycle_list[1][i]
                sample_j = self.classwise_cycle_list[1][j]
            elif aug_prob >= 0.8 and aug_prob < 0.9:
                # crackle_i + normal_j
                i = random.randint(0, len(self.classwise_cycle_list[1]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[0]) - 1)
                sample_i = self.classwise_cycle_list[1][i]
                sample_j = self.classwise_cycle_list[0][j]
            else:
                # normal_i + crackle_j
                i = random.randint(0, len(self.classwise_cycle_list[0]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[1]) - 1)
                sample_i = self.classwise_cycle_list[0][i]  # classwise_cycle_list[０][ｉ] : 代表了该类别下的第i个呼吸音周期；
                sample_j = self.classwise_cycle_list[1][j]

            new_sample = np.concatenate([sample_i[0], sample_j[0]])  # sample_i[0]:　　【０】代表了取出数据项中的，原始音频数据部分，
            self.cycle_list.append((new_sample, 1, sample_i[2] + '-' + sample_j[2], idx, 0))
            # cycle_list:　将新生成的数据加入到其中，　格式为　[　0-新数据，　1-类别标签，　2-新生成数据的合成文件名称，　3-该类别下是第几个新数据，　4-占位０]
            self.filenames_with_labels.append(sample_i[2] + '-' + sample_j[2] + '_' + str(idx) + '_1')
            # filenames_with_labels:  　新数据的文件名称+ 该类别下第几个生成的新数据　+ 类别标签

        # augment wheeze
        aug_nos = scale * len(self.classwise_cycle_list[0]) - len(self.classwise_cycle_list[2])
        for idx in range(aug_nos):
            aug_prob = random.random()
            #aug_prob = 0.5
            if aug_prob < 0.8:
                # wheeze_i + wheeze_j
                i = random.randint(0, len(self.classwise_cycle_list[2]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[2]) - 1)
                sample_i = self.classwise_cycle_list[2][i]
                sample_j = self.classwise_cycle_list[2][j]
            elif aug_prob >= 0.8 and aug_prob < 0.9:
                # wheeze_i + normal_j
                i = random.randint(0, len(self.classwise_cycle_list[2]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[0]) - 1)
                sample_i = self.classwise_cycle_list[2][i]
                sample_j = self.classwise_cycle_list[0][j]
            else:
                # normal_i + wheeze_j
                i = random.randint(0, len(self.classwise_cycle_list[0]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[2]) - 1)
                sample_i = self.classwise_cycle_list[0][i]
                sample_j = self.classwise_cycle_list[2][j]

            new_sample = np.concatenate([sample_i[0], sample_j[0]])
            self.cycle_list.append((new_sample, 2, sample_i[2] + '-' + sample_j[2], idx, 0))
            self.filenames_with_labels.append(sample_i[2] + '-' + sample_j[2] + '_' + str(idx) + '_2')

        # augment both
        aug_nos = scale * len(self.classwise_cycle_list[0]) - len(self.classwise_cycle_list[3])
        for idx in range(aug_nos):
            aug_prob = random.random()

            #aug_prob = 0.3

            if aug_prob < 0.7:
                # both_i + both_j
                i = random.randint(0, len(self.classwise_cycle_list[3]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[3]) - 1)
                sample_i = self.classwise_cycle_list[3][i]
                sample_j = self.classwise_cycle_list[3][j]
            elif aug_prob >= 0.7 and aug_prob < 0.8:
                # crackle_i + wheeze_j
                i = random.randint(0, len(self.classwise_cycle_list[1]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[2]) - 1)
                sample_i = self.classwise_cycle_list[1][i]
                sample_j = self.classwise_cycle_list[2][j]
            elif aug_prob >= 0.8 and aug_prob < 0.9:
                # wheeze_i + crackle_j
                i = random.randint(0, len(self.classwise_cycle_list[2]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[1]) - 1)
                sample_i = self.classwise_cycle_list[2][i]
                sample_j = self.classwise_cycle_list[1][j]
            elif aug_prob >= 0.9 and aug_prob < 0.95:
                # both_i + normal_j
                i = random.randint(0, len(self.classwise_cycle_list[3]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[0]) - 1)
                sample_i = self.classwise_cycle_list[3][i]
                sample_j = self.classwise_cycle_list[0][j]
            else:
                # normal_i + both_j
                i = random.randint(0, len(self.classwise_cycle_list[0]) - 1)
                j = random.randint(0, len(self.classwise_cycle_list[3]) - 1)
                sample_i = self.classwise_cycle_list[0][i]
                sample_j = self.classwise_cycle_list[3][j]

            new_sample = np.concatenate([sample_i[0], sample_j[0]])
            self.cycle_list.append((new_sample, 3, sample_i[2] + '-' + sample_j[2], idx, 0))
            self.filenames_with_labels.append(sample_i[2] + '-' + sample_j[2] + '_' + str(idx) + '_3')

    def __getitem__(self, index):

        audio = self.audio_data[index][0]
        aug_prob = random.random()
        if self.train_flag and aug_prob > 0.5:
            # apply augmentation to audio
            audio = gen_augmented(audio, self.sample_rate)
            # pad incase smaller than desired length
            audio = split_and_pad([audio, 0, 0, 0, 0], self.desired_length, self.sample_rate, types=1)[0][0]

        # roll audio sample
        roll_prob = random.random()
        if self.train_flag and roll_prob > 0.5:
            audio = rollAudio(audio)

        # convert audio signal to spectrogram

        '''
          use
          1. Freq     mask spec   1 channel + 3 channel 
          2. Respire  Transform  spec 1 channel + 3 channel 
          2.  original  spec    convert to  the  3 Mel_spec channel
        '''

        # print(" -----------------------  PartI: 1.generate the spectrogram part --------------------")
        # 通过窄带语谱图，生成对应的Ｍel 语谱图和 Gamma 语谱图
        torch_spec_fun1 = torchaudio.transforms.Spectrogram(n_fft=self.nfft, win_length=512, hop_length=self.hop)
        torch_spectrogram = torch_spec_fun1(torch.Tensor(audio))
        torch_spectrogram = torch_spectrogram[:, :312]
        #print("1-1.  the  spectrogram  shape should be [257, 312]: \n", torch_spectrogram.shape)


        # print(" -----------------------  1. generate  the Mel spectrogram  part --------------------")
        stft_len = self.nfft // 2 + 1
        melFilters = F.create_fb_matrix(n_freqs=stft_len, f_min=50, f_max=2000, n_mels=64, sample_rate=self.sample_rate)
        # print("1.2  mel filters shape should be  [257, 64] \n", melFilters.shape)

        original_mel_spec = torch.matmul(torch_spectrogram.transpose(0, 1), melFilters).transpose(0, 1)
        # print("1.2 torch mel spectrogram  shape should be  [64, 312] \n", original_mel_spec.shape)
        original_mel_spec = original_mel_spec[:, :312]
        original_mel_spec = create_mel_3channel(original_mel_spec)
        # print("1.3 image_dataloader.py:  original  Mel_spectrogram  shape should be [ 64, 312, 3]: \n", original_mel_spec.shape)



        db_spectrogram = librosa.amplitude_to_db(torch_spectrogram)
        # db_spectrogram = librosa.power_to_db(torch_spectrogram,ref=np.max)

        # print(" -----------------------  2. generate the  freq mask spectrogram  --------------------")
        freq_mask = torch.zeros((257,312))
        freq_mask[7:71, : ] = 1   #　生成在　300Hz~2800Hz 之间生成全１的矩阵；  bin_num = 64,  Freq_bin = 39Hz,

        # note: here  should use the  dB spectrogram
        mask_spec = np.multiply(db_spectrogram, freq_mask)
        mask_spec = mask_spec[7:71,:].numpy()
        mask_spec = np.expand_dims(mask_spec, axis= 2)
        # print("2.1  mask_spectrogram  shape should be [ 64, 312, 1]: \n", mask_spec.shape)

        mask_spec_3channel = gen_3channel(mask_spec)
        # print("2.2  mask_spectrogram_3channel  shape should be [ 64, 312, 3]: \n", mask_spec_3channel.shape)
        mask_spectrogram = np.concatenate((mask_spec, mask_spec_3channel),  axis= 2)
        # print("2.3 the final   mask_spectrogram   shape should be [ 64, 312, 4]: \n", mask_spectrogram.shape)



        # print(" -----------------------  3. generate  the respire  Transform  spectrogram  --------------------")
        rt_mask = torch.zeros((257,312))
        rt_mask[7:71, :] = 1   #　生成在2000Hz~200Hz 之间生成全１的矩阵；

        # note: here  should use the  dB spectrogram
        rt_spec = np.multiply(db_spectrogram, rt_mask)
        rt_spec = rt_spec[7:71,:].numpy()
        rt_spec = np.expand_dims(rt_spec, axis=2)
        # print("3.1 rt_spectrogram  shape should be [ 64, 312, 1]: \n",rt_spec.shape)

        # 从第0列开始, 每隔三列即３帧，每帧25ms，　进行respire 变换：
        #  每隔三帧中，　两边的两帧求和，得到平均值mean ,  再将中间帧的数值减去 -mean 　　　　　　　　　　　　　　　
        for j in range(0, 312 - 3, 3):
            sum_left_right = rt_spec[:, j] + rt_spec[:, j + 2]
            mean_2frame = 0.5 * sum_left_right
            rt_spec[:, j + 1] = rt_spec[:, j + 1] - mean_2frame  # 　中间帧减去两边帧的均值；


        rt_spec_3channel = gen_3channel(rt_spec)
        # print("3.2  rt_spectrogram_3channel  shape should be [ 64, 312, 3]: \n", rt_spec_3channel.shape)
        rt_spectrogram = np.concatenate((rt_spec, rt_spec_3channel),  axis= 2)
        # print("3.3 the final   rt _spectrogram   shape should be [ 64, 312, 4]: \n", rt_spectrogram.shape)

        # print(" -----------------------  4. generate  the 11 channel   spectrogram  --------------------")
        audio_image = np.concatenate((mask_spectrogram, rt_spectrogram), axis=2)
        audio_image = np.concatenate( (audio_image, original_mel_spec), axis=2)

        # print("4.1 after conbine the spectrogram shape should be  [64, 312, 11] \n", audio_image.shape)

        '''
        #print(" -----------------------  3. generate 3 Gamma spectrogram part --------------------")
        gammatone_spec, center_freq = gtg_in_dB_torch(audio, self.sample_rate, n_fft=self.nfft, n_win=512, n_hop=self.hop)
        # print("3.1  gammatone spectrogram  shape should be  [128, 800] \n", gammatone_spec.shape)
        gammatone_spec = gammatone_spec[:, :640]
        audio_image_gamma = create_Gamma_3channel(gammatone_spec)
        # audio_image_gamma = np.expand_dims(gammatone_spec, axis=2)
        # print( "3.2  image_dataloader.py:  After  unsqueeze, torch_  Gamma_spectrogram  shape should be [ 128, 800, 1]: \n",audio_image_gamma.shape)



        # print(" -----------------------  4. Concat the   Mel and  Gamma spectrogram, generate 6 channel  --------------------")
        # 使用numpy 将窄带Mel语谱图和 gamma语谱图进行拼接， 形成2个通道；；
        audio_image = np.concatenate((audio_image_mel, audio_image_gamma), axis=2)
        # print(" 4.1 After combine the narrow  mel and gammma spectrogram image,  should be [128, 800, 6]: \n",  audio_image.shape)



        #print(" -----------------------  9. Concat the narrow  Mel and  Gamma spectrogram, generate 2 channel  --------------------")
        # 使用numpy 将宽带Mel语谱图和 gamma语谱图进行拼接， 形成2个通道；；
        # audio_image  = np.concatenate((narrow_Mel_Gamma_spectrogram, width_Mel_Gamma_spectrogram), axis=2)
        #print(" 9.1 After combine the narrow width  mel and gammma spectrogram image,  should be [128, 800, 4]: \n", audio_image.shape)
        '''

        # blank region clipping
        # audio_raw_gray = cv2.cvtColor(create_mel_raw(audio, self.sample_rate, f_max=self.f_max,
        #                                             n_mels=self.n_mels, nfft=self.nfft, hop=self.hop),
        #                              cv2.COLOR_BGR2GRAY)

        if self.dump_images:
            save_images((audio_image, self.audio_data[index][2], self.audio_data[index][3],
                         self.audio_data[index][5], self.audio_data[index][1]), self.train_flag)

        # label
        label = self.audio_data[index][1]

        # apply image transform
        if self.input_transform is not None:
            audio_image = self.input_transform(audio_image)

        return audio_image, label

    def __len__(self):
        return len(self.audio_data)

    # ----------------------------------para ------------------------------




dq_params = {
        'q1': 4,
        'redundancy_1': 3,
        'stages_1': 55,
        'q2': 1,
        'redundancy_2': 3,
        'stages_2': 10,
        'lambda_1': 0.5,
        'lambda_2': 0.5,
        'mu': 0.7,
        'num_iterations': 30,
        'compute_cost_function': True
    }









