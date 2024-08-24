#!/usr/bin/python
# Author: Siddhartha Gairola (t-sigai at microsoft dot com))
import numpy as np
import os
import io
import math
import random
import pandas as pd

import matplotlib.pyplot as plt
import librosa
import librosa.display
import cv2
import cmapy

import nlpaug
import nlpaug.augmenter.audio as naa

import torch

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lwcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b,a, data)
    return y

def Extract_Annotation_Data(file_name, data_dir):
    tokens = file_name.split('_')
    recording_info = pd.DataFrame(data = [tokens], columns = ['Patient Number', 'Recording index', 'Chest location','Acquisition mode','Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(data_dir, file_name + '.txt'), names = ['Start', 'End', 'Crackles', 'Wheezes'], delimiter= '\t')
    return recording_info, recording_annotations

# get annotations data and filenames
def get_annotations(data_dir):
	filenames = [s.split('.')[0] for s in os.listdir(data_dir) if '.txt' in s]
	i_list = []
	rec_annotations_dict = {}
	for s in filenames:
		i,a = Extract_Annotation_Data(s, data_dir)
		i_list.append(i)
		rec_annotations_dict[s] = a

	recording_info = pd.concat(i_list, axis = 0)
	recording_info.head()

	return filenames, rec_annotations_dict

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
#Used to split each individual sound file into separate sound clips containing one respiratory cycle each
#output: [filename, (sample_data:np.array, start:float, end:float, label:int (...) ]
#label: 0-normal, 1-crackle, 2-wheeze, 3-both
def get_sound_samples(recording_annotations, file_name, data_dir, sample_rate):
    sample_data = [file_name]
    # load file with specified sample rate (also converts to mono)
    data, rate = librosa.load(os.path.join(data_dir, file_name+'.wav'), sr=sample_rate)
    #print("Sample Rate", rate)
    
    for i in range(len(recording_annotations.index)):
        row = recording_annotations.loc[i]
        start = row['Start']
        end = row['End']
        crackles = row['Crackles']
        wheezes = row['Wheezes']
        audio_chunk = slice_data(start, end, data, rate)
        sample_data.append((audio_chunk, start,end, get_label(crackles, wheezes)))
    return sample_data


# split samples according to desired length
'''
types:
* 0: simply pad by zeros
* 1: pad with duplicate on both sides (half-n-half)
* 2: pad with augmented sample on both sides (half-n-half)	
'''
def split_and_pad(original, desiredLength, sample_rate, types=0):
	if types==0:
		return split_and_pad_old(original, desiredLength, sample_rate)

	output_buffer_length = int(desiredLength*sample_rate)
	soundclip = original[0].copy()
	n_samples = len(soundclip)

	output = []
	# if: the audio sample length > desiredLength, then split & pad
	# else: simply pad according to given type 1 or 2
	if n_samples > output_buffer_length:
		frames = librosa.util.frame(soundclip, frame_length=output_buffer_length, hop_length=output_buffer_length//2, axis=0)
		for i in range(frames.shape[0]):
			output.append((frames[i], original[1], original[2], original[3], original[4], i, 0))

		last_id = frames.shape[0]*(output_buffer_length//2)
		last_sample = soundclip[last_id:]; pad_times = (output_buffer_length-len(last_sample))/len(last_sample)
		padded = generate_padded_samples(soundclip, last_sample, output_buffer_length, sample_rate, types)
		output.append((padded, original[1], original[2], original[3], original[4], i+1, pad_times))

	else:
		padded = generate_padded_samples(soundclip, soundclip, output_buffer_length, sample_rate, types); pad_times = (output_buffer_length-len(soundclip))/len(soundclip)
		output.append((padded, original[1], original[2], original[3], original[4], 0, pad_times))

	return output

def split_and_pad_old(original, desiredLength, sample_rate):
    output_buffer_length = int(desiredLength * sample_rate)
    soundclip = original[0].copy()
    n_samples = len(soundclip)
    total_length = n_samples / sample_rate #length of cycle in seconds
    n_slices = int(math.ceil(total_length / desiredLength)) #get the minimum number of slices needed
    samples_per_slice = n_samples // n_slices
    src_start = 0 #Staring index of the samples to copy from the original buffer
    output = [] #Holds the resultant slices
    for i in range(n_slices):
        src_end = min(src_start + samples_per_slice, n_samples)
        length = src_end - src_start
        copy = generate_padded_samples_old(soundclip[src_start:src_end], output_buffer_length)
        output.append((copy, original[1], original[2], original[3], original[4], i))
        src_start += length
    return output





# 这里与上面的区别是，　传入的只有音频数据，　不包括其他类别标签等信息；
def split_and_pad_simple(original, desiredLength, sample_rate, types=0):
    if types==0:
        return split_and_pad_old_simple_v1(original, desiredLength, sample_rate)

    output_buffer_length = int(desiredLength*sample_rate)
    soundclip = original.copy()
    n_samples = len(soundclip)

    output = []


        # 　note: 注意，　这里修改了时间长度为3的时候， 需要进行修改；
        #  原始功能　超过3s的部分，　继续切出来作为通过补齐的方式作为下一个样本，
        #  这样做是不合理的，原因是且出来的这部分音频，有可能不是原来的类别标签；
        # 改进后的方式，　对于超过3s部分的长度，直接切除；　
    if n_samples > output_buffer_length:
        padded = soundclip[:output_buffer_length];pad_times = 0
        output.append((padded, pad_times))


    # if: the audio sample length > desiredLength, then split & pad
    # else: simply pad according to given type 1 or 2
    # if n_samples > output_buffer_length:
    # 	frames = librosa.util.frame(soundclip, frame_length=output_buffer_length, hop_length=output_buffer_length//2, axis=0)
    # 	for i in range(frames.shape[0]):output.append(frames[i])
    #
    # 	last_id = frames.shape[0]*(output_buffer_length//2)
    # 	last_sample = soundclip[last_id:]; pad_times = (output_buffer_length-len(last_sample))/len(last_sample)
    # 	padded = generate_padded_samples(soundclip, last_sample, output_buffer_length, sample_rate, types)
    # 	output.append( (padded, pad_times))

    else:
        padded = generate_padded_samples(soundclip, soundclip, output_buffer_length, sample_rate, types); pad_times = (output_buffer_length-len(soundclip))/len(soundclip)
        output.append((padded, pad_times) )

    return output



# v2:  传入全局最小的padding value ,
def split_and_pad_simple_v2(original, desiredLength, sample_rate, types=0, padding_values=-1.0):
    if types==0:
        return split_and_pad_old_simple_v2(original, desiredLength, sample_rate,padding_values)

    output_buffer_length = int(desiredLength*sample_rate)
    soundclip = original.copy()
    n_samples = len(soundclip)

    output = []


        # 　note: 注意，　这里修改了时间长度为3的时候， 需要进行修改；
        #  原始功能　超过3s的部分，　继续切出来作为通过补齐的方式作为下一个样本，
        #  这样做是不合理的，原因是且出来的这部分音频，有可能不是原来的类别标签；
        # 改进后的方式，　对于超过3s部分的长度，直接切除；　
    if n_samples > output_buffer_length:
        padded = soundclip[:output_buffer_length];pad_times = 0
        output.append((padded, pad_times))


    # if: the audio sample length > desiredLength, then split & pad
    # else: simply pad according to given type 1 or 2
    # if n_samples > output_buffer_length:
    # 	frames = librosa.util.frame(soundclip, frame_length=output_buffer_length, hop_length=output_buffer_length//2, axis=0)
    # 	for i in range(frames.shape[0]):output.append(frames[i])
    #
    # 	last_id = frames.shape[0]*(output_buffer_length//2)
    # 	last_sample = soundclip[last_id:]; pad_times = (output_buffer_length-len(last_sample))/len(last_sample)
    # 	padded = generate_padded_samples(soundclip, last_sample, output_buffer_length, sample_rate, types)
    # 	output.append( (padded, pad_times))

    else:
        padded = generate_padded_samples(soundclip, soundclip, output_buffer_length, sample_rate, types); pad_times = (output_buffer_length-len(soundclip))/len(soundclip)
        output.append((padded, pad_times) )

    return output





def split_and_pad_old_simple(original, desiredLength, sample_rate):
    output_buffer_length = int(desiredLength * sample_rate)
    soundclip = original.copy()
    n_samples = len(soundclip)
    total_length = n_samples / sample_rate #  total_len: 当前音频时间长度多少秒数；　 length of cycle in seconds
    n_slices = int(math.ceil(total_length / desiredLength)) #get the minimum number of slices needed, 当前时间长度是desired_length的倍数，　获取所需的最小的切片数, 是个整数；
    samples_per_slice = n_samples // n_slices
    src_start = 0 #Staring index of the samples to copy from the original buffer
    output = [] #Holds the resultant slices
    for i in range(n_slices):
        src_end = min(src_start + samples_per_slice, n_samples)
        length = src_end - src_start
        copy = generate_padded_samples_old(soundclip[src_start:src_end], output_buffer_length)
        #output.append((copy, original[1], original[2], original[3], original[4], i))
        output.append(copy)
        src_start += length
    return output



# v1 版本实现，　通过最前面补零，　达到统一的数据长度； 超出部分，　直接切片取出前面的统一长度的数据；
def split_and_pad_old_simple_v1(original, desiredLength, sample_rate):
        output_buffer_length = int(desiredLength * sample_rate)
        soundclip = original.copy()
        src_len = len(soundclip)
        total_length = src_len / sample_rate  # total_len: 当前音频时间长度多少秒数；　 length of cycle in seconds

        n_slices = int(math.ceil(total_length / desiredLength))  # get the minimum number of slices needed, 当前时间长度是desired_length的倍数，　获取所需的最小的切片数, 是个整数；

        frac =  src_len / output_buffer_length
        copy = np.zeros(output_buffer_length, dtype= np.float32)  #　先初始化一个全0的数组，　数据长度= 统一长度；
        output = []
        if frac < 1:
            copy[ (output_buffer_length - src_len): ] = soundclip[:]
            output.append(copy)
        elif frac >=1 :
            output.append(soundclip[:src_len])

        return output

# 使用规定的　padding_values  来对齐数据长度；
def split_and_pad_old_simple_v2(original, desiredLength, sample_rate, padding_values):
    output_buffer_length = int(desiredLength * sample_rate)
    soundclip = original.copy()
    src_len = len(soundclip)
    total_length = src_len / sample_rate  # total_len: 当前音频时间长度多少秒数；　 length of cycle in seconds

    n_slices = int(math.ceil(
        total_length / desiredLength))  # get the minimum number of slices needed, 当前时间长度是desired_length的倍数，　获取所需的最小的切片数, 是个整数；

    frac = src_len / output_buffer_length
    #note  先初始化一个全 padding_values的数组，　数据长度= 统一长度；
    copy = np.full(output_buffer_length, padding_values, dtype=np.float32)
    output = []
    if frac < 1:
        copy[(output_buffer_length - src_len):] = soundclip[:]
        output.append(copy)
    elif frac >= 1:
        output.append(soundclip[:src_len])

    return output


# split_and_pad_2sr(audio_raw, self.desired_length, self.sample_rate, types=1)
#  note:  这里根据两个采样率，　生成两个对齐长度的样本；　　每个样本只包含了传入的只有音频数据，pad_times(补齐的倍数)　不包括其他类别标签等信息；
def split_and_pad_2sr(original, desiredLength, sample_rate, types=0):
    output = []
    for i in range(len(original)):
        if types==0:
            return split_and_pad_old_simple(original[i], desiredLength, sample_rate[i])

        output_buffer_length = int(desiredLength*sample_rate[i])
        soundclip = original[i].copy()
        n_samples = len(soundclip)


        # if: the audio sample length > desiredLength, then split & pad
        # else: simply pad according to given type 1 or 2
        # 　note: 注意，　这里修改了时间长度为3的时候， 需要进行修改；
        #  原始功能　超过3s的部分，　继续切出来作为通过补齐的方式作为下一个样本，
        #  这样做是不合理的，原因是且出来的这部分音频，有可能不是原来的类别标签；
        # 改进后的方式，　对于超过3s部分的长度，直接切除；　
        if n_samples > output_buffer_length:
            padded = soundclip[:output_buffer_length]; pad_times = 0
            output.append( (padded, pad_times))
        # if n_samples > output_buffer_length:
        #     frames = librosa.util.frame(soundclip, frame_length=output_buffer_length, hop_length=output_buffer_length//2, axis=0)
        #     for i in range(frames.shape[0]):output.append(frames[i])
        #
        #     last_id = frames.shape[0]*(output_buffer_length//2)
        #     last_sample = soundclip[last_id:]; pad_times = (output_buffer_length-len(last_sample))/len(last_sample)
        #     padded = generate_padded_samples(soundclip, last_sample, output_buffer_length, sample_rate[i], types)
        #     output.append((padded, pad_times))

        else:
            padded = generate_padded_samples(soundclip, soundclip, output_buffer_length, sample_rate[i], types)
            pad_times = (output_buffer_length-len(soundclip))/len(soundclip)
            output.append(  (padded, pad_times))

    # note 此时，output　返回的有两个样本，　是不同采样率的样本；
    return output


#output.append((padded, original[1], original[2], original[3], original[4], i+1, pad_times))



def split_and_pad_sr22k(original, desiredLength, sample_rate, types=0):
        output = []

        if types==0:
            return split_and_pad_old_simple(original, desiredLength, sample_rate)

        output_buffer_length = int(desiredLength*sample_rate)
        soundclip = original.copy()
        n_samples = len(soundclip)


        # if: the audio sample length > desiredLength, then split & pad
        # else: simply pad according to given type 1 or 2
        # note: 注意，　这里修改了时间长度为3的时候， 需要进行修改；
        if n_samples > output_buffer_length:
            padded = soundclip[:output_buffer_length]; pad_times = 0
            output.append((padded, pad_times))
        # if n_samples > output_buffer_length:
        #     frames = librosa.util.frame(soundclip, frame_length=output_buffer_length, hop_length=output_buffer_length//2, axis=0)
        #     for i in range(frames.shape[0]):output.append(frames[i])
        #
        #     last_id = frames.shape[0]*(output_buffer_length//2)
        #     last_sample = soundclip[last_id:]; pad_times = (output_buffer_length-len(last_sample))/len(last_sample)
        #     padded = generate_padded_samples(soundclip, last_sample, output_buffer_length, sample_rate[i], types)
        #     output.append((padded, pad_times))

        else:
            padded = generate_padded_samples(soundclip, soundclip, output_buffer_length, sample_rate, types)
            pad_times = (output_buffer_length-len(soundclip))/len(soundclip)
            output.append((padded, pad_times))

        # note 此时，output　返回的有两个样本，　是不同采样率的样本；
        return output





def generate_padded_samples_old(source, output_length):
    copy = np.zeros(output_length, dtype = np.float32)
    src_length = len(source)
    frac = src_length / output_length     # 原始点数　占比　统一点数的多少倍；
    if(frac < 0.5):
        #tile forward sounds to fill empty space
        cursor = 0        # 当原始点数占比统一点数　不到一半时，使用的方式是将原始点，通过重复堆叠原始数据的方式达到统一长度。
        while(cursor + src_length) < output_length:
            copy[cursor:(cursor + src_length)] = source[:]
            cursor += src_length
    else: # 若frac >0.5, 则原始数据点放在最前面，后面的补零; 即原始数据在前，　后面的全0;
        copy[:src_length] = source[:]
    return copy

def generate_padded_samples(original, source, output_length, sample_rate, types):
	copy = np.zeros(output_length, dtype=np.float32)
	src_length = len(source)
	left = output_length-src_length # amount to be padded
	# pad front or back
	prob = random.random()
	if types == 1:
		aug = original
	else:
		aug = gen_augmented(original, sample_rate)

	while len(aug) < left:
		aug = np.concatenate([aug, aug])

	if prob < 0.5:
		#pad back
		copy[left:] = source
		copy[:left] = aug[len(aug)-left:]
	else:
		#pad front
		copy[:src_length] = source[:]
		copy[src_length:] = aug[:left]

	return copy


#**********************DATA AUGMENTAION***************************
#Creates a copy of each time slice, but stretches or contracts it by a random amount
def gen_augmented(original, sample_rate):
	# list of augmentors available from the nlpaug library
	augment_list = [
	#naa.CropAug(sampling_rate=sample_rate)
	naa.NoiseAug(),
	naa.SpeedAug(),
	naa.LoudnessAug(factor=(0.5, 2)),
	naa.VtlpAug(sampling_rate=sample_rate, zone=(0.0, 1.0)),
	naa.PitchAug(sampling_rate=sample_rate, factor=(-1,3))
	]
	# sample augmentation randomly
	aug_idx = random.randint(0, len(augment_list)-1)
	augmented_data = augment_list[aug_idx].augment(original)
	return augmented_data

#Same as above, but applies it to a list of samples
def augment_list(audio_with_labels, sample_rate, n_repeats):
    augmented_samples = []
    for i in range(n_repeats):
        addition = [(gen_augmented(t[0], sample_rate), t[1], t[2], t[3], t[4]+i+1 ) for t in audio_with_labels]
        augmented_samples.extend(addition)
    return augmented_samples

def create_spectrograms(current_window, sample_rate, n_mels=128, f_min=50, f_max=4000, nfft=2048, hop=512):
    fig = plt.figure(figsize=[1.0, 1.0])
    #fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=current_window, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    # There may be a better way to do the following, skipping it for now.
    buf = io.BytesIO()
    plt.savefig(buf, dpi=800, bbox_inches='tight',pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    plt.close('all')
    return img

def create_spectrograms_raw(current_window, sample_rate, n_mels=128, f_min=50, f_max=4000, nfft=2048, hop=512):
    S = librosa.feature.melspectrogram(y=current_window, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    S = librosa.power_to_db(S, ref=np.max)
    S = (S-S.min()) / (S.max() - S.min())
    S *= 255
    img = cv2.applyColorMap(S.astype(np.uint8), cmapy.cmap('viridis'))
    height, width, _ = img.shape
    img = cv2.resize(img, (width*3, height*3), interpolation=cv2.INTER_LINEAR)
    return img


def create_mel_raw(current_window, sample_rate, n_mels=128, f_min=50, f_max=4000, nfft=2048, hop=512, resz=1):
    S = librosa.feature.melspectrogram(y=current_window, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    S = librosa.power_to_db(S, ref=np.max)
    S = (S-S.min()) / (S.max() - S.min())
    S *= 255
    img = cv2.applyColorMap(S.astype(np.uint8), cmapy.cmap('magma'))
    height, width, _ = img.shape
    if resz > 0:
        img = cv2.resize(img, (width*resz, height*resz), interpolation=cv2.INTER_LINEAR)
    img = cv2.flip(img, 0)
    return img


def create_mel_raw_lab1_TA(current_window, sample_rate, n_mels=128, f_min=50, f_max=4000, nfft=2048, hop=512, resz=1):
    S = librosa.feature.melspectrogram(y=current_window, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    S = librosa.power_to_db(S, ref=np.max)
    S = (S-S.min()) / (S.max() - S.min())
    S *= 255
    img = cv2.applyColorMap(S.astype(np.uint8), cmapy.cmap('magma'))
    height, width, _ = img.shape
    if resz > 0:
        # 列数= 原始帧数， 这里保持不变；   行数= n_mels 乘2， 使用 线性插值；
        img = cv2.resize(img, (width*1, height*2), interpolation=cv2.INTER_LINEAR)
    img = cv2.flip(img, 0)
    return img




#Transpose and wrap each array along the time axis
def rollFFT(fft):
    n_row, n_col = fft.shape[:2]
    pivot = np.random.randint(n_col)
    return np.reshape(np.roll(fft, pivot, axis = 1), (n_row, n_col, 1))

def rollAudio(audio):
    # expect audio to be 1 dimensional
    pivot = np.random.randint(audio.shape[0])
    rolled_audio = np.roll(audio, pivot, axis=0)
    assert audio.shape[0] == rolled_audio.shape[0], "Roll audio shape mismatch"
    return rolled_audio

# others
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        #inputs = inputs[:,0,:,:,:]
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def get_mean_and_std_18channel(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    mean = torch.zeros(18)
    std = torch.zeros(18)
    print('==> Computing the 18 channel  mean and std..')
    for inputs, targets in dataloader:
        # inputs = inputs[:,0,:,:,:]
        for i in range(18):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std



def get_mean_and_std_4channel(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(4)
    std = torch.zeros(4)
    print('==> Computing the 9 channel  mean and std..')
    for inputs, targets in dataloader:
        # inputs = inputs[:,0,:,:,:]
        for i in range(4):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std





def get_mean_and_std_4channel(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(4)
    std = torch.zeros(4)
    print('==> Computing the 9 channel  mean and std..')
    for inputs, targets in dataloader:
        # inputs = inputs[:,0,:,:,:]
        for i in range(4):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def get_mean_and_std_6channel(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers= 5)
    mean = torch.zeros(6)
    std = torch.zeros(6)
    print('==> Computing the 6 channel  mean and std..')
    for inputs, targets in dataloader:
        # inputs = inputs[:,0,:,:,:]
        for i in range(6):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std







def get_mean_and_std_9channel(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(9)
    std = torch.zeros(9)
    print('==> Computing the 9 channel  mean and std..')
    for inputs, targets in dataloader:
        # inputs = inputs[:,0,:,:,:]
        for i in range(9):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def get_mean_and_std_12channel(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=10)
    mean = torch.zeros(12)
    std = torch.zeros(12)
    print('==> Computing the 12 channel  mean and std..')
    for inputs, targets in dataloader:
        # inputs = inputs[:,0,:,:,:]
        for i in range(12):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std







def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

# save images from dataloader in dump_images/train or dump_images/test based on train_flag
# expect images to be in RGB format
# image is a tuple: (spectrogram, filename, label, cycle, split)
def save_images(image, train_flag):
    save_dir = 'dump_image'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    if train_flag:
        save_dir = os.path.join(save_dir, 'train')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, image[1]+'_'+str(image[2])+'_'+str(image[3])+'_'+str(image[4])+'.jpg'), cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR))
    else:
        save_dir = os.path.join(save_dir, 'test')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, image[1]+'_'+str(image[2])+'_'+str(image[3])+'_'+str(image[4])+'.jpg'), cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR))




