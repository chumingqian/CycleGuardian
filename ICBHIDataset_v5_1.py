"""
1channel mel spectrogram
"""

from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch
import torchaudio
import os
import pandas as pd
import librosa
import random
import nlpaug.augmenter.audio as naa
import cmapy
import cv2

# import sys
# sys.path.append('/media/respecting_god/S1/1_Python_project/05_Multi_view_CL')

from pytorch_wavelets import DWT1DForward, DWT1DInverse
from config.utils import *
from  config.augmentation import GroupSpecAugment, GroupSpecAugment_v2



# v1, 重新将生成语谱图的这一步骤放在 DataLoader 中，
# 不同的是放在了 __getitem__() 中，从而保证了这一操作是针对每个音频都进行的操作

# v2, 增加不同类型的语谱图，以及倒谱系数类型特征；

#v3_1,  相比于2-3 这里，只计算了 spec 语谱图类型的特征， 输入到模型中， 倒谱系数类型的特征，暂时不计算了；

# 3-3 引入语谱图级别的 mask,  freq, time 两个维度都进行mask;
# v4_1:  引入两种采样率， 生成两种尺度下的特征
# sr1= 22k;  sr2 =10k




#  v5_1,  sr=22k,  使用数据扩充的方式，
#  引入 events 事件级别的标签， 扩充各个异常类别的样本；






def Extract_Annotation_Data(file_name, data_dir):
    """
    extract information from file
    :param file_name: file name
    :param data_dir: root data directory
    :return:
        recording_info(DataFrame) : a dict saved all information about file
        recording_annotations(DataFrame) : a dict saved audio clip information
                including : (start time,
                             end time,
                             if Crackle,
                             if Wheezes)
    """
    tokens = file_name.split('_')
    recording_info = pd.DataFrame(data=[tokens],
                                  columns=['Patient Number',
                                           'Recording index',
                                           'Chest location',
                                           'Acquisition mode',
                                           'Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(data_dir, file_name + '.txt'),
                                        names=['Start', 'End', 'Crackles', 'Wheezes'],
                                        delimiter='\t')
    return recording_info, recording_annotations


# Note: get annotations data and filenames
def get_annotations(data_dir):
    """
    Args:
        data_dir: data directory

    Returns:
        filenames: file name
        rec_annotations_dict: audio recording annotation dict
    """
    filenames = [s.split('.')[0] for s in os.listdir(data_dir) if '.txt' in s]
    i_list = []
    rec_annotations_dict = {}
    for s in filenames:
        i, a = Extract_Annotation_Data(s, data_dir)
        i_list.append(i)
        rec_annotations_dict[s] = a

    recording_info = pd.concat(i_list, axis=0)
    recording_info.head()  # desc：打印输出前5行信息

    return filenames, rec_annotations_dict




def Extract_Event_Annotation_Data(file_name, data_dir):
    """
    extract information from file
    :param file_name: file name
    :param data_dir: root data directory
    :return:
        recording_info(DataFrame) : a dict saved all information about file
        recording_annotations(DataFrame) : a dict saved audio clip information
                including : (start time,
                             end time,
                             if Crackle,
                             if Wheezes)
    """
    tokens = file_name.split('_')
    recording_info = pd.DataFrame(data=[tokens],
                                  columns=['Patient Number',
                                           'Recording index',
                                           'Chest location',
                                           'Acquisition mode',
                                           'Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(data_dir, file_name + '.txt'),
                                        names=['Start', 'End', 'Crackles', 'Wheezes'],
                                        delimiter='\t')
    return recording_info, recording_annotations




# Note: get annotations data and filenames
def get_event_annotations(ori_file, data_dir):
    """
    Args:
        data_dir: data directory 表示 event 事件的音频目录

    Returns:
        filenames: file name
        rec_annotations_dict: audio recording annotation dict

    """
    # note, 这里传入ori_file 的作用是，  只从 event 事件音频抽取训练集中包含的音频， 即不能将测试集上音频 从event 提取出来；
    # 6-4 划分下， 训练集中 ori_file: 920 中 包含 551 record, 370 份属于测试集

    event_filenames = [s.split('.')[0] for s in os.listdir(data_dir) if '.txt' in s] # 920 份全的；
    curr_filenames = []  # 只保留当前训练集或者是 测试集上的音频；

    # 只保留event record 属于当前训练集上的音频；
    for s in event_filenames:
        file_name = s.split('_')[:-1]
        file_name = '_'.join(file_name)  # + '.txt'  不用添加后缀名称
        if file_name  in ori_file:
            curr_filenames.append(s)

    print(f'the  file name both in train_data & envent_data ，its the length  {len(curr_filenames)}')
           # 跳过训练集中不存在的音频，  需要保证测试集的音频没有泄漏到训练集中

    rec_annotations_dict = {}
    name_list = []

    # 进一步筛选， 只保留event 中属于异常类的音频，；
    for s in curr_filenames:
        filepath = os.path.join(data_dir,  s + '.txt') #  检查当前文件是否问空文件,
        data_frame = pd.read_csv(filepath,delim_whitespace=True, header=None, names=["start", "end", "category"] )

        if data_frame.empty:  # 如果是空文件，则跳过操作, 因为 normal 类型的数据，标注的是空文件；
            continue
        else:
            rec_annotations_dict[s] = data_frame
            name_list.append(s)

        # 如果为空， 则移除该文件
        # with open(filepath, 'r') as file :
        #     content = file.read().strip()
        #     if not content:
        #         filenames.remove(s)
        #        continue
        #     else:
        #       rec_annotations_dict[s] = data_frame

    return  name_list, rec_annotations_dict






def slice_data(start, end, raw_data, sample_rate):
    max_ind = len(raw_data)
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    return raw_data[start_ind: end_ind]



def get_label(crackle, wheeze):
    if crackle == 0 and wheeze == 0:
        return 0
    elif crackle == 1 and wheeze == 0:
        return 1
    elif crackle == 0 and wheeze == 1:
        return 2
    else:
        return 3


def get_event_label(category):
    if category == 'crackle':
        return 1
    elif category == 'wheeze':
        return 2



def get_sound_samples(recording_annotations,
                      file_name,
                      data_dir,
                      sample_rate):
    
    
    """
    :param recording_annotations:file information -> dirextory [start, end, Crackles, wheezes]
    :param file_name:
    :param data_dir:
    :param sample_rate:
    :return: sample data -> list [audio chunk, start, end, label]
    """
    sample_data = [file_name]
    # load file with specified sample rate (also converts to mono)
    data, rate = librosa.load(os.path.join(data_dir, file_name + '.wav'), sr=sample_rate)
    # print("Sample Rate", rate)

    for i in range(len(recording_annotations.index)):
        row = recording_annotations.loc[i]
        start = row['Start']
        end = row['End']
        crackles = row['Crackles']
        wheezes = row['Wheezes']
        audio_chunk = slice_data(start, end, data, rate)
        sample_data.append((audio_chunk,
                            start,
                            end,
                            get_label(crackles, wheezes)))
    return sample_data


# note, 输入两种采样率， 获取两种采样率下的样本
def get_2sound_samples(recording_annotations,
                      file_name,
                      data_dir,
                      sample_rate_low,
                      sample_rate_high):
    """
    :param recording_annotations:file information -> dirextory [start, end, Crackles, wheezes]
    :param file_name:
    :param data_dir:
    :param sample_rate:
    :return: sample data -> list [audio chunk, start, end, label, audio_chunk2]
    """
    sample_data = [file_name]
    # load file with specified sample rate (also converts to mono)
    data, rate = librosa.load(os.path.join(data_dir, file_name + '.wav'), sr=sample_rate_low)
    data2, rate2 = librosa.load(os.path.join(data_dir, file_name + '.wav'), sr=sample_rate_high)
    
    # print("Sample Rate", rate)

    for i in range(len(recording_annotations.index)):
        row = recording_annotations.loc[i]
        start = row['Start']
        end = row['End']
        crackles = row['Crackles']
        wheezes = row['Wheezes']
        audio_chunk1 = slice_data(start, end, data, rate)
        audio_chunk2 = slice_data(start, end, data2, rate2)
        sample_data.append((audio_chunk1,
                            start,
                            end,
                            get_label(crackles, wheezes),
                             audio_chunk2))
    return sample_data


def get_event_sound_samples(recording_annotations,
                       event_file_name,
                       data_dir,
                       sample_rate,
                      ):
    """
    :param recording_annotations:file information -> dirextory [start, end, Crackles, wheezes]
    :param file_name:
    :param data_dir:
    :param sample_rate:
    :return: sample data -> list [audio chunk, start, end, label, audio_chunk2]
    """

    # original  input string is  '101_1b1_Pr_sc_Meditron_events' , need to remove the events
    file_name = event_file_name.split('_')[:-1]
    file_name = '_'.join(file_name)

    sample_data = [file_name]
    # load file with specified sample rate (also converts to mono)
    # 根据文件名，载入对应的音频文件；
    data, rate = librosa.load(os.path.join(data_dir, file_name + '.wav'), sr=sample_rate)
    #data2, rate2 = librosa.load(os.path.join(data_dir, file_name + '.wav'), sr=sample_rate_high)

    # print("Sample Rate", rate)
    # 逐行 提取当前样本中的 各个 event 的cycle;
    for i in range(len(recording_annotations.index)):
        row = recording_annotations.loc[i]
        start = row['start']
        end = row['end']
        category = row['category']

        # 增加一个持续时间的属性；
        duration = end - start
        audio_chunk1 = slice_data(start, end, data, rate)
        sample_data.append((audio_chunk1,
                            start,
                            end,
                            get_event_label(category),
                            duration))
    return sample_data


def construct_basic_unit( audio_chunk, normal_list,  label,sample_rate, ):

    #
    # 只保留正常样本中 大于1s 的音频， ；
    suit_list = []
    for id, tuple in enumerate(normal_list):
        if  len(tuple[0]) > 1 * sample_rate:
            suit_list.append(tuple[0] )
        else:
            continue

    sample_id = random.randint(0, len(suit_list) -1)
    normal_sample = suit_list[sample_id]




    if label == 1:
        # 代表crackle: 则以0.1s 为基本单元构建出 一份样本长度；
        # 从当前样本中，取出其中一段时间上区间的长度
        # 将他们拼接

        dersired_chunk = int( 0.1 * sample_rate) # 2200
        if len(audio_chunk) > dersired_chunk:
            basic_unit = audio_chunk[:dersired_chunk]

        else:
            need_chunk = dersired_chunk - len(audio_chunk)
            # if need_chunk <0 or need_chunk is not int:
            #     print(f' the need chuunk {need_chunk}, {type(need_chunk)}')
            max_start = len(normal_sample) - need_chunk
            start_ioc = random.randint(0, max_start)
            normal_chunk = normal_sample[start_ioc: start_ioc + need_chunk]
            basic_unit = np.concatenate( [normal_chunk, audio_chunk ] )


    elif label == 2:
            # 代表wheeze : 则以 1s 为基本单元构建出 一份样本长度；
            # 从当前样本中，取出其中一段时间上区间的长度
            # 将他们拼接

            dersired_chunk =  1 * sample_rate
            if len(audio_chunk) > dersired_chunk:
                basic_unit = audio_chunk[:dersired_chunk]
            else:
                need_chunk = dersired_chunk - len(audio_chunk)
                max_start = len(normal_sample) - need_chunk
                # if max_start < 0:
                #     print(f'the normal length {len(normal_sample)}, the need chunk{need_chunk}')
                #     raise ValueError("max_start must be non-negative")
                start_ioc = random.randint(0, max_start)
                normal_chunk = normal_sample[start_ioc: start_ioc + need_chunk]
                basic_unit = np.concatenate([normal_chunk, audio_chunk])

    return basic_unit






def gen_augmented(original, sample_rate):
    # note: list of augmentors available from the nlpaug library
    augment_list = [
        # naa.CropAug(sampling_rate=sample_rate)
        naa.NoiseAug(),
        naa.SpeedAug(),
        naa.LoudnessAug(factor=(0.5, 2)),
        naa.VtlpAug(sampling_rate=sample_rate, zone=(0.0, 1.0)),
        naa.PitchAug(sampling_rate=sample_rate, factor=(-1, 3))
    ]
    # sample augmentation randomly
    aug_idx = random.randint(0, len(augment_list) - 1)
    augmented_data = augment_list[aug_idx].augment(original)

    # note, whach out the  nlpaug  version
    return augmented_data[0]


def generate_padded_samples(original, source, output_length, sample_rate, types, train_flag=True):
    """
    pad source(framed audio data) into output_length
    :param original: original whole audio data（原始整体信号）
    :param source: framed audio data which is to be padded（需要被pad的帧信号）
    :param output_length: output length
    :param sample_rate: sample rate
    :param types: way to pad (1:use original augmentation to pad ; 0:use itself to pad)

    aug: data used to padded source data
    :return: padded data
    """
    copy = np.zeros(output_length, dtype=np.float32)
    src_length = len(source)
    left = output_length - src_length  # amount to be padded
    # pad front or back
    if types == 0:
        aug = np.zeros(output_length)
    elif types == 1:
        aug = original
    else:
        aug = gen_augmented(original, sample_rate)

    while len(aug) < left:
        aug = np.concatenate([aug, aug])

    # note: random pad source data in its left or right
    if train_flag:
        prob = random.random()
    else:
        prob = 1.
    if prob < 0.5:
        # pad is in front, source is in back
        copy[left:] = source
        copy[:left] = aug[len(aug) - left:]
    else:
        # source is in front, pad is in back
        copy[:src_length] = source[:]
        copy[src_length:] = aug[:left]

    return copy


def generate_padded_samples_v1(original, source, output_length, sample_rate, types, pad_value, train_flag=True):
    """
    pad source(framed audio data) into output_length
    :param original: original whole audio data（原始整体信号）
    :param source: framed audio data which is to be padded（需要被pad的帧信号）
    :param output_length: output length
    :param sample_rate: sample rate
    :param types: way to pad (1:use original augmentation to pad ; 0:use itself to pad)

    aug: data used to padded source data
    :return: padded data
    """
    #copy = np.zeros(output_length, dtype=np.float32)
    copy = np.full(output_length, pad_value, dtype=np.float32)
    src_length = len(source)
    left = output_length - src_length  # amount to be padded
    # pad front or back
    if types == 0:
        aug = np.zeros(output_length)
    elif types == 1:
        aug = original
    else:
        aug = gen_augmented(original, sample_rate)

    while len(aug) < left:
        aug = np.concatenate([aug, aug])

    # note: random pad source data in its left or right
    if train_flag:
        prob = random.random()
    else:
        prob = 1.
    if prob < 0.5:
        # pad is in front, source is in back
        copy[left:] = source
        copy[:left] = aug[len(aug) - left:]
    else:
        # source is in front, pad is in back
        copy[:src_length] = source[:]
        copy[src_length:] = aug[:left]

    return copy



def get_audio_dataAndLable(original, desiredLength, sample_rate, pad_value=0, types=0,  train_flag=True):

    """

    分情况实现，根据是否为训练集，使用不同的　time shift 方式：
    １．信号长度大于规定长度的，　直接进行中心裁剪；

    2.针对训练集，　 直接获取对齐后的音频信号；；
    3. 针对测试集，　将原始信号居中，　在两边进行padd ；

    """

    output_buffer_length = int(desiredLength * sample_rate)
    sound_clip = original[0]
    n_samples = len(sound_clip)

    if n_samples > output_buffer_length:
        # 　超出规定的长度截取　，信号的中心；
        # Crop the signal to the central part that matches the target length
        start_ind = (n_samples - output_buffer_length) // 2
        end_ind = start_ind + output_buffer_length
        fixed_audio = sound_clip[start_ind:end_ind]


    if train_flag:
        # 训练集， 直接获取
        fixed_audio = sound_clip

    if not train_flag and n_samples <= output_buffer_length:
          # 如果是测试集，　且长度不足规定的长度时，　则将信号居中，两边补零；
        num_zeros = output_buffer_length - n_samples
        num_zeros_front = num_zeros //2
        num_zeros_end =  num_zeros - num_zeros_front
        fixed_audio =  np.pad( sound_clip, (num_zeros_front, num_zeros_end), 'constant' ) # constant , 默认pad 是0;


    output = (fixed_audio, original[1], original[2], original[3], original[4])

    return output








def split_and_pad_drop_v2(original, desiredLength, sample_rate, pad_value=0, types=0,  train_flag=True):

    """
    无论前面，是否使用了time shift, 在这里，统一对所有的音频，　（原始的音频　和数据扩充后的音频），
    note, 　

    分情况实现，根据是否为训练集，使用不同的　time shift 方式：
    １．信号长度大于规定长度的，　直接进行中心裁剪；

    2.针对训练集，　先使用time shift,  然后在末尾padd 到统一长度；
    3. 针对测试集，　将原始信号居中，　在两边进行padding  zero ；

    """

    output_buffer_length = int(desiredLength * sample_rate)
    sound_clip = original[0]
    n_samples = len(sound_clip)

    if n_samples > output_buffer_length:
        # 　超出规定的长度截取　，信号的中心；
        # Crop the signal to the central part that matches the target length
        start_ind = (n_samples - output_buffer_length) // 2
        end_ind = start_ind + output_buffer_length
        fixed_audio = sound_clip[start_ind:end_ind]


    if train_flag and n_samples <= output_buffer_length:
        # 训练集，　重复堆叠自身，统一到固定长度；
        fixed_audio = padded_byself( sound_clip,
                                       output_buffer_length,)

    if not train_flag and n_samples <= output_buffer_length:
          # 如果是测试集，　且长度不足规定的长度时，　则将信号居中，两边补零；
        num_zeros = output_buffer_length - n_samples
        num_zeros_front = num_zeros //2
        num_zeros_end =  num_zeros - num_zeros_front

        fixed_audio =  np.pad( sound_clip, (num_zeros_front, num_zeros_end), 'constant' ) # constant , 默认pad 是0;


    output = (fixed_audio, original[1], original[2], original[3], original[4])

    return output


def split_and_pad_drop_v3(original, desiredLength, sample_rate, pad_value=0, types=0,  train_flag=True):

    """
    无论前面，是否使用了time shift, 在这里，统一对所有的音频，　（原始的音频　和数据扩充后的音频），
    note, 　v3, add the  DWT decompose for  the audio signal

    分情况实现，根据是否为训练集，使用不同的　time shift 方式：
    １．信号长度大于规定长度的，　直接进行中心裁剪；

    2.针对训练集，　先使用time shift,  然后在末尾padd 到统一长度；
    3. 针对测试集，　将原始信号居中，　在两边进行padding  zero ；

    """

    output_buffer_length = int(desiredLength * sample_rate)
    sound_clip = original[0]
    n_samples = len(sound_clip)

    if n_samples > output_buffer_length:
        # 　超出规定的长度截取　，信号的中心；
        # Crop the signal to the central part that matches the target length
        start_ind = (n_samples - output_buffer_length) // 2
        end_ind = start_ind + output_buffer_length
        fixed_audio = sound_clip[start_ind:end_ind]


    if train_flag and n_samples <= output_buffer_length:
        # 训练集，　重复堆叠自身，统一到固定长度；
        fixed_audio = padded_byself( sound_clip,
                                       output_buffer_length,)

    if not train_flag and n_samples <= output_buffer_length:
          # 如果是测试集，　且长度不足规定的长度时，　则将信号居中，两边补零；
        num_zeros = output_buffer_length - n_samples
        num_zeros_front = num_zeros //2
        num_zeros_end =  num_zeros - num_zeros_front

        fixed_audio =  np.pad( sound_clip, (num_zeros_front, num_zeros_end), 'constant' ) # constant , 默认pad 是0;

    # note: 对信号使用 dwt 分解，然后重构，目的是为了降噪；
    #  use here  epoch 118, acc =55
    dwt = DWT1DForward(wave='db6', J=3)
    idwt = DWT1DInverse(wave='db6')
    # print(m[0].shape)
    audio = torch.tensor(fixed_audio)

    audio = audio.unsqueeze(0).unsqueeze(0)
    yl, yh = dwt(audio)

    audio = idwt((yl, yh))
    audio = audio.squeeze(0).squeeze(0)

    audio = audio.numpy()

    output = (audio, original[1], original[2], original[3], original[4])

    return output

def split_and_pad_drop_v4(original, desiredLength, sample_rate, pad_value=0, types=0,  train_flag=True):

    """
    无论前面，是否使用了time shift, 在这里，统一对所有的音频，　（原始的音频　和数据扩充后的音频），
    note, 　v4,  该版本修改了测试集的对齐方式，和训练集一致；


    分情况实现，根据是否为训练集，使用不同的　time shift 方式：
    １．信号长度大于规定长度的，　直接进行中心裁剪；

    2.针对训练集， 　重复自身，堆叠 到统一长度；
    3. 针对测试集， 　同样重复自身堆叠到统一长度　 ；

    """

    output_buffer_length = int(desiredLength * sample_rate)
    sound_clip = original[0]
    n_samples = len(sound_clip)

    if n_samples > output_buffer_length:
        # 　超出规定的长度截取　，信号的中心；
        # Crop the signal to the central part that matches the target length
        start_ind = (n_samples - output_buffer_length) // 2
        end_ind = start_ind + output_buffer_length
        fixed_audio = sound_clip[start_ind:end_ind]


    if train_flag and n_samples <= output_buffer_length:
        # 训练集，　重复堆叠自身，统一到固定长度；
        fixed_audio = padded_byself( sound_clip,
                                       output_buffer_length,)

    if not train_flag and n_samples <= output_buffer_length:
        fixed_audio = padded_byself( sound_clip,
                                       output_buffer_length,)



    output = (fixed_audio, original[1], original[2], original[3], original[4])

    return output





def keep_audio(original, desiredLength, sample_rate, pad_value=0, types=0,  train_flag=True):

    """

   　用于一个函数包装一下，　直接将原始音频返回；
    """

    # output_buffer_length = int(desiredLength * sample_rate)
    # sound_clip = original[0]

    output = (original[0], original[1], original[2], original[3], original[4])

    return output



def extand_audio(original, desiredLength, sample_rate, pad_value=0, types=0,  train_flag=True):

    """

   　用于一个函数包装一下，　直接将原始音频返回；
    """

    # output_buffer_length = int(desiredLength * sample_rate)
    # sound_clip = original[0]

    output = (original[0], original[1], original[2], original[3], original[4], original[5])

    return output





def hybrid_align_audio(original, desiredLength, sample_rate, ):

    """
    这里使用，混合对齐的方式，　
    ０.５ 概率使用，　　padded_zero 方式，　对齐到统一长度；
    ０.５ 概率使用，　　padded_by self 方式，　对齐到统一长度；
    ０.　time shift  方式，　对齐到统一长度，　暂不使用，原因是后面的数据增强中，使用 rollAudio 这种模式；

    """

    pad_prob = random.random()


    output_buffer_length = int(desiredLength * sample_rate)
    sound_clip = original
    n_samples = len(sound_clip)

    if n_samples > output_buffer_length:
        # 　超出规定的长度截取　，信号的中心；
        # Crop the signal to the central part that matches the target length
        start_ind = (n_samples - output_buffer_length) // 2
        end_ind = start_ind + output_buffer_length
        fixed_audio = sound_clip[start_ind:end_ind]


    elif  n_samples <= output_buffer_length:

        if pad_prob <= 0.5:
            # 重复堆叠自身，统一到固定长度；
            fixed_audio = padded_byself( sound_clip,
                                           output_buffer_length,)

        elif  0.5 < pad_prob <= 1.0:
            # 将信号居中，两边补零；
            num_zeros = output_buffer_length - n_samples
            num_zeros_front = num_zeros //2
            num_zeros_end =  num_zeros - num_zeros_front

            fixed_audio =  np.pad( sound_clip, (num_zeros_front, num_zeros_end), 'constant' ) # constant , 默认pad 是0;




    return fixed_audio




def hybrid_align_audio_v1(original, desiredLength, sample_rate,):

    """
    这里使用，混合对齐的方式，　
    ０  概率使用，　　padded_zero 方式，　对齐到统一长度；
    ０.５ 概率使用，　 padded_by self 方式，对齐到统一长度；
    ０.　time shift 方式，　对齐到统一长度，　暂不使用，原因是后面的数据增强中，使用 rollAudio 这种模式；


    """

    pad_prob = random.random()


    output_buffer_length = int(desiredLength * sample_rate)
    sound_clip = original
    n_samples = len(sound_clip)

    if n_samples > output_buffer_length:
        # 超出规定的长度截取　，截取信号的后一段部分，丢弃开头的部分；
        # Crop the signal to the central part that matches the target length
        # start_ind = (n_samples - output_buffer_length) // 2
        # end_ind = start_ind + output_buffer_length
        # fixed_audio = sound_clip[start_ind:end_ind]

        start_ind = (n_samples - output_buffer_length)
        end_ind = start_ind + output_buffer_length
        fixed_audio = sound_clip[start_ind:end_ind]



    elif  n_samples <= output_buffer_length:

        if pad_prob <= 1:
            # 重复堆叠自身，统一到固定长度；
            fixed_audio = padded_byself( sound_clip,
                                           output_buffer_length,)

        # elif  0.5 < pad_prob <= 1.0:
        #     # 0.5的概率，将信号居中，两边补零；
        #     num_zeros = output_buffer_length - n_samples
        #     num_zeros_front = num_zeros //2
        #     num_zeros_end =  num_zeros - num_zeros_front
        #
        #     fixed_audio =  np.pad( sound_clip, (num_zeros_front, num_zeros_end), 'constant' ) # constant , 默认pad 是0;


    return fixed_audio







def shift_and_padded(original, source, output_length, sample_rate, types, pad_value, train_flag=True):
        """
        pad source(framed audio data) into output_length
        :param original: original whole audio data（原始整体信号）
        :param source: framed audio data which is to be padded（需要被pad的帧信号）
        :param output_length: output length
        :param sample_rate: sample rate
        :param types: way to pad (1:use original augmentation to pad ; 0:use itself to pad)

        aug: data used to padded source data
        :return: padded data
        """


         # 先time shift , 然后末尾padd;
        src_length = len(source)
        shift_ratio = random.uniform(0.01, 0.8)

        shift_len = int(shift_ratio * src_length)
        shifted_signal = source[shift_len:]

        # padding = np.full(shift_len, pad_value, dtype=np.float32)
        pad_len =  output_length - len(shifted_signal)
        padding = np.zeros(pad_len, dtype=np.float32)

        #shifted_padded_signal = np.concatenate( (shifted_signal, padding))
        shifted_padded_signal = np.concatenate([shifted_signal, padding])

        return shifted_padded_signal


def padded_byself( source, output_length, ):
    """
    pad source(framed audio data) into output_length
    :param original: original whole audio data（原始整体信号）
    :param source: framed audio data which is to be padded（需要被pad的帧信号）
    :param output_length: output length
    :param sample_rate: sample rate
    :param types: way to pad (1:use original augmentation to pad ; 0:use itself to pad)

    aug: data used to padded source data
    :return: padded data
    """

    # 先time shift , 然后末尾padd;
    #src_length = len(source)

    pad_signal = np.concatenate( [source, source])
    while len(pad_signal) < output_length:
        pad_signal = np.concatenate( [pad_signal, source])

    if len(pad_signal) > output_length:
        pad_signal = pad_signal[: output_length]

    return  pad_signal




def split_and_pad_drop(original, desiredLength, sample_rate, types=0, train_flag=True):
    """
    :param original: original is a tuple -> (data, label, filename, cycle_index, aug_id)
    :param desiredLength:
    :param sample_rate:
    :param types:
    :return:
    """
    output_buffer_length = int(desiredLength * sample_rate)
    sound_clip = original[0]
    n_samples = len(sound_clip)

    if n_samples > output_buffer_length:
        frame = sound_clip[0:output_buffer_length]
    else:
        frame = generate_padded_samples(sound_clip,
                                        sound_clip,
                                        output_buffer_length,
                                        sample_rate,
                                        types,
                                        train_flag=train_flag)
    output = (frame, original[1], original[2], original[3], original[4])
    return output





# 使用规定的padding 数值；
def split_and_pad_drop_v1(original, desiredLength, sample_rate, pad_value=-1, types=0,  train_flag=True):
    """
    :param original: original is a tuple -> (data, label, filename, cycle_index, aug_id)
    :param desiredLength:
    :param sample_rate:
    :param types:
    :return:
    """
    output_buffer_length = int(desiredLength * sample_rate)
    sound_clip = original[0]
    n_samples = len(sound_clip)

    if n_samples > output_buffer_length:
        frame = sound_clip[0:output_buffer_length]
    else:
        frame = generate_padded_samples_v1(sound_clip,
                                        sound_clip,
                                        output_buffer_length,
                                        sample_rate,
                                        types,
                                        pad_value,
                                        train_flag=train_flag)
    output = (frame, original[1], original[2], original[3], original[4])
    return output


def generate_fbank(audio, sample_rate, n_mels=128):
    """
    use torchaudio library to convert mel fbank for AST model
    """
    assert sample_rate == 16000, 'input audio sampling rate must be 16kHz'
    fbank = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True, sample_frequency=sample_rate, use_energy=False,
                                              window_type='hanning', num_mel_bins=n_mels, dither=0.0, frame_shift=10)

    mean, std = -4.2677393, 4.5689974
    fbank = (fbank - mean) / (std * 2)  # mean / std
    fbank = fbank.unsqueeze(-1).numpy()
    return fbank


def generate_mfcc(audio, sample_rate, n_ceps=42):
    """
    use torchaudio library to convert mel fbank for AST model
    """
    assert sample_rate == 16000, 'input audio sampling rate must be 16kHz'

    # (width_frames, num_ceps )
    mfcc_feature = torchaudio.compliance.kaldi.mfcc(audio, htk_compat=True, sample_frequency=sample_rate, use_energy=False,
                                              window_type='povey', num_ceps=n_ceps,num_mel_bins=128, dither=0.0, frame_shift=10)


    from torchaudio.functional import compute_deltas
    # 从mfcc 中提取，　时序的动态性，　使用 一阶delat, 和二阶 delta 系数；

    # 对mfcc 系数进行归一化,Normalize MFCC features (using either Min-Max scaling or Z-score normalization)
    mfcc_norm = (mfcc_feature - mfcc_feature.mean(dim=-1, keepdim=True)) / mfcc_feature.std(dim=-1, keepdim=True)

    mfcc_norm = mfcc_norm.unsqueeze(0)

    return mfcc_norm


import torch
# 对语谱图中， 或者 倒谱系数 沿着行维度进行标准化，
# 原因是 每一行，代表 同一个频率段上出现的信息； 对每一行 进行标准化， 从而可以凸显该频率段上出现的信息；
# 相反， 不可以对列进行标准化的原因， 每一列中， 大部分信息都是出现在低频部分，
# 如果对列进行标注化， 网络 则多数都是关注的低频部分的信息；

def standardize_rows(input_data):
    # Create an empty tensor to store the standardized data
    standardized_data = torch.empty_like(input_data)

    for i in range(input_data.size(0)):  # Loop through rows
        row = input_data[i, :]  # Get the current row
        mean = row.mean()  # Calculate the mean of the row
        std = row.std()  # Calculate the standard deviation of the row

        # Standardize the row separately
        standardized_row = (row - mean) / std

        # Store the standardized row in the result tensor
        standardized_data[i, :] = standardized_row

    return standardized_data

# Example usage:





import  torchaudio
from torchaudio.transforms import  AmplitudeToDB, MelSpectrogram
from  nnAudio.features import  CQT2010v2, Gammatonegram
import random



class ICBHIDataset_with_event(Dataset):
    def __init__(self,
                 data_dir,
                 event_data_dir,
                 dataset_split_file,
                 train_flag,
                 dataset_split=0,
                 test_fold=4,
                 stetho_id=-1,
                 aug_audio=True,
                 aug_audio_scale=1,

                 aug_feature=False, # choice to use  specAugment
                 desired_time= 8,
                 sample_rate1= 22000,
                 n_fft1=1024,
                 win_len1=1000,
                 hop_len1= 256,  #500,  # 100, # 注意hop_len 选取必须满足 2的 整数次幂， 否则影响到cqt 特征提取时， 出现错误；
                 f_min1=0,
                 f_max1=3000,
                 n_filters1=84,


                 sample_rate2=22000,
                 n_fft2= 512,
                 win_len2= 500,
                 hop_len2= 256,    #220,  # 100,
                 f_min2=0,
                 f_max2=4000,
                 n_filters2=42,


                 input_transform=None,
                 ):
        """
        Args:
            data_dir: audio data directory
            dataset_split_file:
            test_fold:
            train_flag:
            params_json:
            input_transform: use data augmentation at Feature level,  spectrogram  transform
            stetho_id: id of loaded data acquired by stetho_id device
            aug_scale: data augment scale

        Paras:
            self.file_to_device: a directory map: audio_file -> device
            self.audio_data: audio data
            self.labels: audio data label
            self.dump_images: flag if save spectrum image
            self.filenames_with_labels:
            self.cycle_list: a list [audio_data,
                                     label,
                                     file_name,
                                     cycle id,
                                     aug id,
                                     split id]
            self.classwise_cycle_list: cycle_list divided by label, elements similar to cycle_list
            self.class_num: sample number of each class
            self.identifiers: new name of sample which has been preprocessed(include frame、split、pad ect.)
        """
        self.train_flag = train_flag
        self.data_dir = data_dir
        self.event_data_dir = event_data_dir
        self.spec_transform = input_transform  # desc: input tensor transforme

        # NOTE:parameters for spectrograms
        self.sample_rate1 = sample_rate1  # desc:采样频率
        self.sample_rate2 = sample_rate2  # desc:采样频率
        self.desired_time = desired_time  # desc:样本时间长度
        # self.n_mels = n_filters  # desc:mel滤波器个数
        # self.win_len = win_len
        # self.nfft = n_fft  # desc：短时傅里叶变换点数
        # self.hop = hop_len  # desc：帧移
        # self.f_max = f_max  # desc：生成mel滤波器的最高采用频率

        self.dump_images = False  # desc:是否保存产生的mel谱
        self.file_to_device = {}  # desc: a directory map: audio_file -> device

        device_name_to_id, device_num = self.get_info(self.data_dir)
        # desc: device_name_to_id: 将收集肺音的设备转换为编号的map
        #       device_num：每个设备收集的样本的数量

        # note：按照给定的文件划分训练集-测试集（5折划分，可用于6/4，可用于8/2），每个病人属于第几折
        #   dataset_divide为0时为6-6split， 为1时为8-2split
        all_patients = open(dataset_split_file).read().splitlines()  # desc: all_patients:所有的病人编号和病人属于第几折的字符串信息
        patient_dict = {}  # desc: patient_dict: a directory map: patient -> fold
        if dataset_split == 0:
            for line in all_patients:
                file_name, Set = line.strip().split('\t')
                index = file_name.split('_')[0]
                if train_flag and Set == "train":
                    patient_dict[index] = Set
                elif (not train_flag) and Set == "test":
                    patient_dict[index] = Set
        else:
            for line in all_patients:
                idx, fold = line.strip().split(' ')
                if train_flag and int(fold) != test_fold:
                    patient_dict[idx] = fold
                elif (not train_flag) and int(fold) == test_fold:
                    patient_dict[idx] = fold

        # ：extracting the audio filenames and the data for breathing cycle and it's label
        print("Getting filenames ...")  # filenames: {list:920}, rec_ann :{dict: 920 * {DataFrame:(rows, col= 4)}}
        filenames, rec_annotations_dict = get_annotations(
            data_dir)  # desc：annotations一个字典，索引为文件名，内容为DataFrame：开始，结束，是否为crakle, 是否为wheeze，
        print("filenames have been gotten!")

        # 是否取出某一听诊设备录制的所有音频，stetho_id=0就取出所有音频(文件名)
        if stetho_id >= 0:
            self.filenames = [s for s in filenames if
                              s.split('_')[0] in patient_dict and
                              self.file_to_device[s] == stetho_id]
        else:

            #  note, 这里表明， filenames= 920 个所有的整体文件，
            # 而 self.filenames  则只取出当前训练集或者是 测试集上的数据，
            #  即通过前面 patient_dict 来区分当前的样本属于 测试集还是训练集， 去除了后缀名.txt的形式，如'103_2b2_Ar_mc_LittC2SE'
            self.filenames = [s for s in filenames if s.split('_')[0] in patient_dict]

        # :get individual breathing cycles from each audio file
        self.filenames_with_labels = []  # desc:cycle's 携带标签和数据片段编号的文件名
        self.cycle_list = []  # desc：内容为list包括：数据片段、标签、文件名、片段序号、数据增强序号
        self.classwise_cycle_list = [[], [], [], []]
        print("Extracting Individual Cycles")



        for _, file_name in enumerate(tqdm(self.filenames, desc='get individual cycles from each audio file:')):
            data = get_sound_samples(rec_annotations_dict[file_name],
                                     file_name,
                                     data_dir,
                                     self.sample_rate1,
                                     )
            # data: [file_name, audio_chunk,stat, end, label,  ]


            # cycles_with_labels: [ nums * (audio_chunk, label, file_name, cycle_idx, aug_index,)]
            cycles_with_labels = []
            for cycle_idx, d in enumerate(data[1:]):
                #  data[1:] = [  (audio_chunk,stat, end, label,), (),.... ]
                # d 指取出其中的一项 (audio_chunk,stat, end, label,audio_chunk2),
                     cycles_with_labels.append((d[0], d[3], file_name, cycle_idx, 0,))


            self.cycle_list.extend(cycles_with_labels)
            for _, d in enumerate(cycles_with_labels):
                self.filenames_with_labels.append(file_name + '_' + str(d[3]) + '_' + str(d[1]))
                self.classwise_cycle_list[d[1]].append(d)
        print("Extraction has completed!")


        #note,   获取事件级别的片段 # desc：annotations一个字典，索引为文件名，内容为DataFrame：开始，结束，是否为crakle, 是否为wheeze，
        abnormal_filenames, event_rec_annotations_dict = get_event_annotations( self.filenames,event_data_dir,)
        print(" abnormal  event filenames have been gotten!")

        self.event_cycle_list = []  # desc：内容为list包括：数据片段、标签、文件名、片段序号、数据增强序号
        self.classwise_event_list = [[], [], [], []]
        for _, filename in  enumerate(abnormal_filenames, ):
            # filename = '101_1b1_Al_sc_Meditron_events.txt'  # empty file
            data = get_event_sound_samples(  event_rec_annotations_dict[filename],
                                             filename,
                                             data_dir,  # 存放音频的路径
                                             self.sample_rate1)

            # data: [file_name, nums * ( audio_chunk,stat, end, label,  duration 持续时间)... ]
            # data  中第一个数据表示文件名， 后面跟随该样本中被标注的多个呼吸音cycle 的信息

            # note, event_cycles_with_labels: [ nums * (audio_chunk, label, file_name, cycle_idx, aug_index, duration )]
            event_cycles_with_labels = []

            # d = data[1:] = [audio_chunk,stat, end, label, duration ]
            # normal+ 将原始的片段 构建出一个最小的基本单元，
            for cycle_idx, d in enumerate(data[1:]):
                basic_unit =  construct_basic_unit(d[0], self.classwise_cycle_list[0], d[3], sample_rate2)
                event_cycles_with_labels.append((basic_unit, d[3], file_name, cycle_idx, 0, d[4]))


            self.event_cycle_list.extend(event_cycles_with_labels)
            # event_cycles_with_labels [ nums * (audio_chunk, label, file_name, cycle_idx, aug_index, duration )]
            for _ , d in enumerate(event_cycles_with_labels):
                self.classwise_event_list[d[1]].append(d)

        print('\n generate the  basic  abnormal unit done')
        # 先提取每个片段， 将其构成一个基本的单元；
        # 在数据增强中， 从每个类别下 抽取多个基本单元，构成一个基本的样本。





        # 　NOTE： １.　数据扩充， 将每个类别中的样本数量进行扩充，
        # 对样本随机的进行　time shift,  将前一部分数据，搬移到尾部；
        # 将time shift 后的样本与原始样本进行拼接， 从而形成新的样本，
        if train_flag and aug_audio:
            self.extand_with_events( self.desired_time,scale=aug_audio_scale,)
        print(f" \n We  extend  the abnormal  class sample, add it to the cycle_list,  and it's Done! \n")


        self.audio_data = []  # desc：一个list包括：数据片段、标签、文件名、片段序号、数据增强序号
        self.labels = []  # desc：用于记录数据的标签
        #  NOTE 2,  取出 cycle_list中的样本，　将其对应到统一的长度，
        # 然后将放入到 self.audio_data中，　留着后续self.getitem取出来；
        for _, sample in enumerate(tqdm(self.cycle_list, desc=' get the data from cycle list \n')):
            # def split_and_pad_drop_v2(original, desiredLength, sample_rate, pad_value=0, types=0,  train_flag=True):

            output = keep_audio(sample,
                                self.desired_time,
                                self.sample_rate1,
                                pad_value=0,
                                types=1,
                                train_flag=self.train_flag)


            self.audio_data.append(output)
            # note, self.audio_data: [audio_chunk, label, file_name, cycle_idx, aug_index,audio_chunk2  ]

        # 以下的属性，用于从音频中抽取 特征

        self.win_len1 = win_len1
        self.nfft1 = n_fft1
        self.hop_len1 = hop_len1
        self.fmin1 = f_min1
        self.fmax1 = f_max1
        self.filters1 = n_filters1



        self.win_len2 = win_len2
        self.nfft2 = n_fft2
        self.hop_len2 = hop_len2
        self.fmin2 = f_min2
        self.fmax2 = f_max2
        self.filters2 = n_filters2


        self.powerToDB = AmplitudeToDB(stype='power')
        self.amplitudeToDB = AmplitudeToDB(stype='magnitude')


        # note, ---------------- 以下实现语谱图类别的特征，　如 Mel, CQT, Gamma spectrogram -----------------
        #

        # 该方法是调用 torchaudio 中实现的；
        self.mel_spec1 = MelSpectrogram(
            sample_rate=sample_rate1,
            n_fft=n_fft1,
            win_length=win_len1,
            hop_length=hop_len1,
            f_min=32.7,
            f_max=f_max1,
            n_mels=n_filters1,
        )

        # self.cqt_spectrogram =
        # 这里设置的　cqt　内核使用默认的，　trainable=False;
        self.cqt_spectrogram1 = CQT2010v2(
            sr=sample_rate1,
            hop_length= hop_len1,  #hop_len,
            fmin= 32.7,  #32.7,
            fmax= f_max1, #f_max,
            n_bins= 84,  # 这里设置成84 为了和 Mel 滤波器个数对应上；
            trainable=False,
        )

        self.gamma_spec1 = Gammatonegram(
            sr=sample_rate1,
            n_fft=n_fft1,
            hop_length=hop_len1,
            n_bins=84,
            fmin= 32.7,
            fmax=f_max1,
            trainable_bins=False,
        )


        # # ------------------high  sample rate---------
        # self.mel_spec2 = MelSpectrogram(
        #     sample_rate=sample_rate2,
        #     n_fft=n_fft2,
        #     win_length=win_len2,
        #     hop_length=hop_len2,
        #     f_min=32.7,
        #     f_max=f_max2,
        #     n_mels=n_filters2,
        # )
        #
        # # self.cqt_spectrogram =
        # # 这里设置的　cqt　内核使用默认的，　trainable=False;
        # self.cqt_spectrogram2 = CQT2010v2(
        #     sr=sample_rate2,
        #     hop_length=hop_len2,  # hop_len,
        #     fmin=32.7,  # 32.7,
        #     fmax=f_max2,  # f_max,
        #     n_bins= n_filters2,  # 这里设置成84 为了和 Mel 滤波器个数对应上；
        #     trainable=False,
        # )
        #
        # self.gamma_spec2 = Gammatonegram(
        #     sr=sample_rate2,
        #     n_fft=n_fft2,
        #     hop_length=hop_len2,
        #     n_bins= n_filters2,
        #     fmin=32.7,
        #     fmax=f_max2,
        #     trainable_bins=False,
        # )

        self.spec_aug1 = GroupSpecAugment_v2(specaug_policy='group_sr22k')
        #self.spec_aug2 = GroupSpecAugment_v2(specaug_policy='group_sr22k')

        # self.mfcc_fea = torchaudio.transforms.MFCC(sample_rate= self.sample_rate, n_mfcc=42,dct_type=2, log_mels=False,
        #                                         melkwargs={"n_fft":self.nfft, "win_length":self.win_len,"hop_length": self.hop_len,
        #                                                    'f_max':self.fmax, 'n_mels':self.filters, 'center': True,  }) # 　这里center 需要设置成 True, 否则帧数对应不上；
        #
        #
        #
        # self.lfcc_fea = torchaudio.transforms.LFCC(sample_rate=self.sample_rate, n_filter=self.filters, f_max=self.fmax,n_lfcc=42,
        #                                         speckwargs={"n_fft": self.nfft, "win_length":self.win_len, 'hop_length':self.hop_len,
        #                                                     })



        # NOTE:收集分类好了之后的其他信息
        self.device_wise = []  # desc:按照设备对所有数据分类
        self.class_wise_num = np.zeros(4)  # desc:每个类别的样本数量
        self.identifiers = []  # desc:分了片段之后、分了帧之后的每段数据的新名字
        for _ in range(device_num):
            self.device_wise.append([])

        for _, sample in enumerate(self.audio_data):  # sample :[  audio_arr,  label, file_name, cycle_id, aug_id]
            self.class_wise_num[sample[1]] += 1.0
            self.labels.append(sample[1])
            if len(sample[2].split("_")) == 3:
                continue #  跳过合成样本的统计；
            else:
                self.device_wise[self.file_to_device[sample[2]]].append(sample)
                self.identifiers.append(sample[2] + '_' + str(sample[3]) + '_' + str(sample[1]))
            # if len(sample[2].split("-shift")) == 1:
            #     self.device_wise[self.file_to_device[sample[2]]].append(sample)
            # self.identifiers.append(sample[2] + '_' + str(sample[3]) + '_' + str(sample[1]))

#
            # self.cycle_list.append( (new_sample, 0, sample_i[2]+ '-shift'+ str(shift_len), idx,0)   )
            # self.filenames_with_labels.append(sample_i[2]+ '-shift'+ str(shift_len) + '_'+ str(idx)  + '_0')



        self.class_ratio = self.class_wise_num / sum(self.class_wise_num)

        # note: print details about audio dataloader
        if self.train_flag:
            print("TRAIN DETAILS:")
        else:
            print("TEST DETAILS:")

        print("Device to ID\n", device_name_to_id)

        for idx in range(device_num):
            print("DEVICE ID:", idx, "\tsize:", len(self.device_wise[idx]))


        print(f'CLASSWISE RATIOS:{self.class_ratio},\n LENGTH AUDIO DATA: {len(self.audio_data)}')

        print(f'CLASSWISE Number :{self.class_ratio * len(self.audio_data)} ')
        # print("CLASSWISE RATIOS:\n", self.class_ratio)
        # print("LENGTH AUDIO DATA:", len(self.audio_data))

    def get_info(self, data_dir):
        """
        a file name example: 101_1b1_Al_sc_Meditron.txt
        note : name made by"Patient number +
                            Recording index +
                            Chest location +
                            Acquisition mode +
                            Recording equipment"

        Patient number : (101,102,...,226)
        Recording index
        Chest location : (Trachea (Tc),
                {Anterior (A), Posterior (P), Lateral (L)}{left (l), right (r)})
        Acquisition mode : (sequential/single channel (sc), simultaneous/multichannel (mc))
        Device Name(Recording equipment) :
            AKG C417L Microphone,
            3M Littmann Classic II SE Stethoscope,
            3M Litmmann 3200 Electronic Stethoscope,
            WelchAllyn Meditron Master Elite Electronic Stethoscope

        :param data_dir:
        :return:
        """
        device_name_to_id = {}  # device_to_id : 将设备转化为编号的list
        device_num = 0  # device_num : 设备数量，可分给设备作为id
        device_patient_list = []  # 以设备编号为索引，存储每一个设备检测的病人的list
        patients = []  # 病人列表汇总
        device_name = []

        files = os.listdir(data_dir)
        for f in files:
            device = f.strip().split('_')[-1].split('.')[0]
            if device not in device_name_to_id:
                device_name_to_id[device] = device_num
                device_num += 1
                device_patient_list.append([])
                device_name.append(device)
            self.file_to_device[f.strip().split('.')[0]] = device_name_to_id[device]
            pat = f.strip().split('_')[0]  # 病人编号
            if pat not in device_patient_list[device_name_to_id[device]]:
                device_patient_list[device_name_to_id[device]].append(pat)
            if pat not in patients:
                patients.append(pat)

        print("==================================")
        print("Device_name\t\t" + "id\t\t" + "pat_num")
        for idx in range(device_num):
            print(device_name[idx], "\t\t", idx, "\t\t", len(device_patient_list[idx]))
        print("==================================")
        return device_name_to_id, device_num



    def __getitem__(self, index):


        audio_low_sr    =  self.audio_data[index][0]
        # audio_high_sr   =  self.audio_data[index][5]

        # 　将数据对齐的工作放在 getitem() 中完成，
        # audio1 = 10k, audio2 = 22k
        audio1 = hybrid_align_audio_v1(original=audio_low_sr,  desiredLength=self.desired_time, sample_rate=self.sample_rate1)
        # audio2 = hybrid_align_audio_v1(original=audio_high_sr, desiredLength=self.desired_time, sample_rate=self.sample_rate2)




        # note , 这里在getitem () 使用数据层面数据增强，　以及 特征层面的数据增强，
        # 这样确保了， 一个batch 中不同的样本会使用到不同的数据增强；
        # note, ===================  以下部分，实现了在音频数据层面的数据增强 =======================
        # 则意味着，一个batch中样本使用的同一个数据增强，
        # 但是不同的batch之间使用的不同的数据增强，从而提高模型的泛化能力；
        # 需要注意的，数据增强只对训练集使用，　测试集不可以使用；
        # 对测试集的数据增强，包含了三大部分，
        # 1.分解重构，　2.音频的加减速等　3.　roll, time shift 方式；


        reconstruct_prob = random.random() # 对每个训练样本使用DWT进行分解重构的概率；
        if  self.train_flag  and reconstruct_prob > 0.5:
            dwt = DWT1DForward(wave='db6', J=3)
            idwt = DWT1DInverse(wave='db6')
            # print(m[0].shape)
            
            audio1 = torch.tensor(audio1)
            audio1 = audio1.unsqueeze(0).unsqueeze(0)
            yl, yh = dwt(audio1)
            audio1 = idwt((yl, yh))
            audio1 = audio1.squeeze(0).squeeze(0)
            audio1 = audio1.numpy()




        aug_prob = random.random()
        if self.train_flag and aug_prob > 0.5:
            audio1 = gen_augmented(audio1, self.sample_rate1)
            # 由于数据增强后，会改变音频的长度，故需要重新对齐到统一长度；
            audio1 =  split_and_pad([audio1, 0, 0,0,0], self.desired_time, self.sample_rate1, types=1,)[0][0]





        roll_prob = random.random() # 随机的对音频进行roll, 将音频中前一部分，移动到后面；
        if self.train_flag and roll_prob > 0.5:
            audio1 = rollAudio(audio1)

        audio1 = torch.tensor(audio1)




        # =================以下部分完成语谱图， 以及倒谱系数 这两种类型的特征生成；
        # ----------- 生成语谱图类别的特征 ----------------------------
        mel_spec1 = self.mel_spec1(audio1) #(n_filters=84, frames = 121)
        mel_spec1 = self.powerToDB(mel_spec1)
        # note, 对语谱图分别在行维度上进行标准化；
        mel_norm1 = standardize_rows(mel_spec1)
        mel_norm1 = mel_norm1.unsqueeze(0)   #(1, 84, 121)
        

        cur_cqt1 = self.cqt_spectrogram1(audio1)  # (1, bins, frames =626 )
        cur_cqt1 = self.amplitudeToDB(cur_cqt1)
        cur_cqt1 = cur_cqt1.squeeze(0)
        cqt_norm1 = standardize_rows(cur_cqt1)
        cqt_norm1 = cqt_norm1.unsqueeze(0)

        cur_gamma1 = self.gamma_spec1(audio1)  # (1, n_filters, frames= 626)
        cur_gamma1 = self.powerToDB(cur_gamma1)
        cur_gamma1 = cur_gamma1.squeeze(0)
        gamma_norm1 = standardize_rows(cur_gamma1)
        gamma_norm1 = gamma_norm1.unsqueeze(0)



        # ------------------------
        # mel_spec2 = self.mel_spec2(audio2)  # (n_filters=84, frames = 626)
        # mel_spec2 = self.powerToDB(mel_spec2)
        # # note, 对语谱图分别在行维度上进行标准化；
        # mel_norm2 = standardize_rows(mel_spec2)
        # mel_norm2 = mel_norm2.unsqueeze(0)  # (2, 84, 626)
        #
        # cur_cqt2 = self.cqt_spectrogram2(audio2)  # (2, bins, frames =626 )
        # cur_cqt2 = self.amplitudeToDB(cur_cqt2)
        # cur_cqt2 = cur_cqt2.squeeze(0)
        # cqt_norm2 = standardize_rows(cur_cqt2)
        # cqt_norm2 = cqt_norm2.unsqueeze(0)
        #
        # cur_gamma2 = self.gamma_spec2(audio2)  # (2, n_filters, frames= 626)
        # cur_gamma2 = self.powerToDB(cur_gamma2)
        # cur_gamma2 = cur_gamma2.squeeze(0)
        # gamma_norm2 = standardize_rows(cur_gamma2)
        # gamma_norm2 = gamma_norm2.unsqueeze(0)
        #



        spec_Aug = random.random() #note, 随机的对语谱图特征，在频率与时间维度上，进行随机的掩码；
        if self.train_flag and spec_Aug > 0.5:
            mel_norm1   = self.spec_aug1(mel_norm1)
            cqt_norm1   = self.spec_aug1(cqt_norm1)
            gamma_norm1 = self.spec_aug1(gamma_norm1)

  
        spec_feature1 = torch.cat((cqt_norm1, gamma_norm1,  mel_norm1), dim=0)    #(bt, 3, filters= 84,   frames=120)
        # spec_feature2 = torch.cat((cqt_norm2, gamma_norm2,  mel_norm2), dim=0)    #(bt, 3, filters= 42,  frames= 600)
        # 对语谱图特征在通道维度上进行标准化；
        # spec_feature = spec_feature



        # note, ===================  以下部分，实现了在特征层面的数据增强 =======================
        # 注意，这里语谱图的特征实现了数据增强， 对于 mfcc 倒谱系数类型特征，并没有进行数据增强；
        if self.spec_transform is not None:
            # 对于训练集和测试集， 同时会有由于 transform.ToTensor()操作， 会将 channel 维度提前,
            # 并且如果数值是属于[0,255]的numpy or PIL image, 格式则会进行缩放归一化到0-1区间； 否则， 对于数值不属于[0,255]的整数数据不会进行缩放，或者不是numpy类型也不会缩放 ；
            # 对于训练集，  transform 中的 SpecAugment 会随机的对语谱图中横轴或者纵轴进行掩码， 掩码部分使用均值替代；

            spec_feature = self.spec_transform(spec_feature1.numpy())

        label = self.audio_data[index][1]

        # dataset 中提取的每个样本的信息， 之后传入到 DataLoader中，
        # DataLoader其中 collate_fn 函数，会增加batch 这个维度，将每个样本的信息打包成一个batch 组合输出出来，

        return  spec_feature1,  label

    def __len__(self):
        return len(self.audio_data)


    def select_abnormal_audio(self,audio_label1, audio_label2,desired_time ):
        """
        随机在两类音频集合里面各选一段音频
        适用与 两个标签相同的数据扩充 ， 即只用于对 crackle , wheeze 的数据增强；
        :param audio_label1: label of audio1
        :param audio_label2: label of audio2
        :return:
        """

        if audio_label1 == 1 and  audio_label1 == audio_label2:
            nums =  int(desired_time / 0.1 )

        elif audio_label1 == 2 and  audio_label1 == audio_label2:
            nums = int(desired_time / 1)

        else:

            raise AssertionError('the input should be  same class')

        i = random.randint(0, len(self.classwise_event_list[audio_label1]) - 1)
        j = random.randint(0, len(self.classwise_event_list[audio_label2]) - 1)
        sample_i = self.classwise_event_list[audio_label1][i]
        sample_j = self.classwise_event_list[audio_label2][j]

        new_part = np.concatenate([sample_i[0], sample_j[0]])

        for _ in range(2, nums): # 前面使用了两个样本；

            j = random.randint(0, len(self.classwise_event_list[audio_label2]) - 1)
            sample_j = self.classwise_event_list[audio_label2][j]
            new_part = np.concatenate( [ new_part, sample_j[0] ] )


        return new_part

    def select_audio_both(self, audio_label1, audio_label2, desired_time):
        """
        随机在两类音频集合里面各选一段音频
        用于生成 both 类型的数据， 两个标签必须不同
        适用与 两个标签相同的数据扩充 ， 即只用于对 crackle , wheeze 的数据增强；
        :param audio_label1: label of audio1
        :param audio_label2: label of audio2
        :return:
        """

        if audio_label1 == audio_label2:
            raise AssertionError('the input should be different class')


        i = random.randint(0, len(self.classwise_event_list[audio_label1]) - 1)
        j = random.randint(0, len(self.classwise_event_list[audio_label2]) - 1)
        sample_i = self.classwise_event_list[audio_label1][i]
        sample_j = self.classwise_event_list[audio_label2][j]

        new_part = np.concatenate([sample_i[0], sample_j[0]]) # crackle + wheeze

        # 随机生成wheeze 组成的个数，
        # 由于总共时长 8s, wheeze 基本单元1 s；
        # 故wheeze 至少一个， 最多7个，
        wheeze_num =  random.randint(1,desired_time-1)
        crackle_num = int( ( desired_time - wheeze_num ) / 0.1 )   # crackle 基本单元0.1s;


        for _ in range(1, crackle_num): # 起始时，已经使用一个,  故跳过0， 从1开始
            j = random.randint(0, len(self.classwise_event_list[audio_label1]) - 1)
            sample_c = self.classwise_event_list[audio_label1][j]
            new_part = np.concatenate([new_part, sample_c[0]])

        for _ in range(1, wheeze_num): # 起始时，已经使用一个；
            i = random.randint(0, len(self.classwise_event_list[audio_label2]) - 1)
            sample_w = self.classwise_event_list[audio_label2][i]
            new_part = np.concatenate([new_part, sample_w[0]])

        return new_part




    def extand_with_events(self,desired_time= 8,scale=1, ):
        """
        直接在音频数据，层面进行扩充，　扩充的方式，
        将同一类别下的不同样本，进行拼接，从而生成新的样本。
        :param scale: augmention scale
        :return:
        """

        # labels of different audios
        # classwise_cycle_list:  是一个列表总共４个，按照标签将四个类的呼吸音周期，　
        # 分别划分到其中，　列表中的每一项是一个元组；
        # 每个元组的组成如下：= [子音频数据，　类别标签，　文件名，文件名中第几个呼吸周期，自带的占位０]，有５个成分组成；

        normal = 0
        crackle = 1
        wheeze = 2
        both = 3

        # augment normal
        # 对正常样本不进行数据增强；
        # aug_nos = scale * len(self.classwise_cycle_list[0]) - len(self.classwise_cycle_list[0])
        # for idx in range(aug_nos):
        #     # normal_i + normal_j
        #     normal_i, normal_j = self.select_audio(normal, normal)
        #     new_sample = np.concatenate([normal_i[0], normal_j[0]])
        #     # note, todo
        #     self.cycle_list.append((new_sample, 0, normal_i[2] + '-' + normal_j[2], idx, 0))
        #     self.filenames_with_labels.append(normal_i[2] + '-' + normal_j[2] + '_' + str(idx) + '_0')
            # cycle_list:　将新生成的数据加入到其中，　　其对应的格式如下：　
            # ( 拼接的子音频数据，　类别标签，　子音频i所对应的文件名称-子音频j所对应的文件名称, 在当前类别下是第几个新增加的样本, 占位0　)
            # 格式为[new_sample -拼接的子音频数据，　0-类别标签，
            # normal_i[2]　子音频i所对应的文件名称， 中间连接符－normal_j[2]:子音频j所对应的文件名称；
            # 　idx，在当前类别下，是第几个新增加的样本。




        # augment crackle,  由于 crackle 的基础样本数目是1208, 故对其扩充 3倍， 使得其中 2/3 都是生成的数据；
        scale_crakle = 3
        aug_nos = scale_crakle * len(self.classwise_cycle_list[1]) - len(self.classwise_cycle_list[1])
        for idx in range(aug_nos):
            new_sample = self.select_abnormal_audio(crackle, crackle, desired_time)
            # cycle_list: [ (audio_chunk, label, file_name, cycle_idx, aug_index, duration)]
            self.cycle_list.append((new_sample, 1, 'crackle_extand_'+ str(idx), idx, 0))


        # augment wheeze
        aug_nos = scale * len(self.classwise_cycle_list[0]) - len(self.classwise_cycle_list[2])
        for idx in range(aug_nos):
            new_sample = self.select_abnormal_audio(wheeze, wheeze, desired_time)
            self.cycle_list.append((new_sample, 2, 'wheeze_extand_'+ str(idx), idx, 0))


        # augment both
        # 　注意更新 event 级别的标签来看， 确实， crackle ,wheeze 分开来打的标签；
        aug_nos = scale * len(self.classwise_cycle_list[0]) - len(self.classwise_cycle_list[3])
        for idx in range(aug_nos):
            new_sample = self.select_audio_both(crackle, wheeze, desired_time)
            self.cycle_list.append((new_sample, 3,  'both_extand_'+ str(idx), idx, 0))





from torchvision import transforms

if __name__ == "__main__":

    # args.h, args.w = 798, 128
    # train_transform = [transforms.ToTensor(),
    #                    SpecAugment(args),
    #                    transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
    # val_transform = [transforms.ToTensor(),
    #                  transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
    # # train_transform.append(transforms.Normalize(mean=mean, std=std))
    # # val_transform.append(transforms.Normalize(mean=mean, std=std))
    #
    # train_transform = transforms.Compose(train_transform)
    # val_transform = transforms.Compose(val_transform)
    #
    # train_set = ICBHIDataset(data_dir='../data/ICBHI_final_database',
    #                                dataset_split_file='../data/patient_trainTest6_4.txt',
    #                                train_flag=True, test_fold=4,
    #                                input_transform=None, stetho_id=-1,
    #                                aug_audio=True, aug_audio_scale=2, aug_feature=False, aug_feature_scale=1,
    #                                sample_rate=160000, desired_time= 8, n_mels=84, f_max=3000,
    #                                n_fft=1024, win_len=1000, hop=160, dataset_split=0,
    #                                )

    train_dataset = ICBHIDataset_with_event(data_dir='./data/ICBHI_final_database',
                                 event_data_dir= './data/events',
                                 dataset_split_file='./data/patient_trainTest6_4.txt',

                                 stetho_id=-1,
                                 train_flag=True,
                                 aug_audio= True, aug_audio_scale=1, aug_feature=False,
                                 desired_time= 8, sample_rate1=10000, sample_rate2=22000,
                                 n_filters1=84, n_filters2=42,
                                 input_transform=None,  # train_transform,
                                 )

    print('it is ok')