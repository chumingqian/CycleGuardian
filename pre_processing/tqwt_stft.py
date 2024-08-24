import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
import shutil
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter, firwin
from scipy import stats
import librosa.display as libdisplay
from pydub import AudioSegment
import multiprocessing as mp

from  datetime import  datetime

import sys
import argparse
sys.path.append('../tqwt_tools-master/')
from tqwt_tools import DualQDecomposition




dq_params = {
    'q1': 4,
    'redundancy_1': 3,
    'stages_1': 30,
    'q2': 1,
    'redundancy_2': 3,
    'stages_2': 10,
    'lambda_1': 0.1, # regularization parameter
    'lambda_2': 0.1, #
    'mu': 0.1, # affect convergence speed
    'num_iterations': 100,
    'compute_cost_function': True
}




def clip_test(dir):
    """seprate trainset and testset"""
    # 根据 train_test.txt官方提供的，划分训练集和测试集；
    txt_dir = dir + 'train_test.txt'

    with open(txt_dir, 'r') as f:
        name = []
        set_type = []
        for row in f.readlines():
            row = row.strip('\n') # 去除换行标记符
            row = row.split('\t') # 去除\t 标记符, 至此，列表中两个元素

            name.append(row[0])  # row[0]: 文件名称，　row[1]: 训练集，测试集
            set_type.append(row[1])

    for i in range(len(name)):   #根据文件名索引到，该文件对应的是训练还是测试;
        if set_type[i] == 'test':
            shutil.move(dir + 'ICBHI_final_database/' + name[i] + '.wav', dir + 'testset/' + name[i] + '.wav')
        if set_type[i] == 'train':
            shutil.move(dir + 'ICBHI_final_database/' + name[i] + '.wav', dir + 'trainset/' + name[i] + '.wav')

def clip_cycle(dir, new_dir):
    """clip the record into breath cycle
    dir : trainset/testset record path
    new_dir:breath cycle save path
    """
    t1 = datetime.now()
    for file in os.listdir(dir):# 取出训练集中，当前的音频文件
        # txt_name = '../ICBHI/' + file[:-4] + '.txt'
        txt_name = '../data/ICBHI/' + file[:-4] + '.txt'# 将该音频文件名的后缀换成.txt，　从而查找该record对应的标注文本；
        time = np.loadtxt(txt_name)[:, 0:2]  #将当前record 中的分段时间信息，按照起始，终止时间一组；　提取多个cycle的起始,终止时间。
        sound = AudioSegment.from_wav(dir + file)  # 将训练集中，载入当前文件名对应的音频数据；
        for i in range(time.shape[0]):  # shape0 代表了该record中划分出的子音频的个数；
            start_time = time[i, 0] * 1000  # 求出当前子音频的起始，终止时间；　乘以1000因为标签达到了ms；
            stop_time = time[i, 1] * 1000
            word = sound[start_time:stop_time]
            word.export(new_dir + file[:-4] + str(i) + '.wav', format="wav")
        #　将切分的子音频按照，  　文件名＋该record中的第几个片段＋　.wav 的形式保存到new_dir　路径下面；
    t2 = datetime.now()
    duration = t2 - t1
    print(f"{dir} file clip cycle done, and cost {duration} time  ")

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def Normalization(x, pattern='min_max'):
    x = x.astype(float)
    max_x = max(x)
    min_x = min(x)
    if pattern == 'min_max':
        for i in range(len(x)):
            x[i] = float(x[i] - min_x) / (max_x - min_x)
    else:  # z-score
        x = stats.zscore(x)
    return x

# stft_and_save(sig, fs, dir_ori, 0.02, file)   fft= int(44100 *0.02) = 882; 　hop=int(0.01* 44100)= 441;
def stft_and_save(sig, fs, dir, win_len, file):
    # (1+nfft/2 , nframes) = (442,  sig/hop ) =
    # 　这里输出的类型是complex128;
    stft = librosa.stft(sig, n_fft=int(win_len * fs), hop_length=int(win_len/2 * fs), window='hann')
    # 在行维度上进行切片，　取出前百分之　(4k/fs)　的行，　列保持不变；
    sub_stft = stft[0:int(len(stft) * 4000 / fs), :]

    librosa.display.specshow(librosa.amplitude_to_db(np.abs(sub_stft), ref=np.max), y_axis='log', x_axis='time',
                             sr=fs)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(dir + file[:-3] + 'png', cmap='Greys_r')
    plt.close()

def save_pic(file):
    dir_ck = save_dir + '/ck/'
    dir_wh = save_dir + '/wh/'
    dir_res = save_dir + '/res/'
    dir_ori = save_dir + '/ori/'
    if os.path.exists(dir_ck + file[:-3] + 'png'):
        return
    # wav_dir = " ../data/test_cycle" 全局变量;
    fs, sig = wav.read(wav_dir + '/' + file)  # 读取出音频数据和　采样率
    sig = Normalization(sig)  # 将数据范围归一化到0-1　之间;
    if fs > 4000:   # 对采样率大于4k　的信号，通过50－２k的　带通滤波器
        sig = butter_bandpass_filter(sig, 50, 2000, fs, order=3)


    stft_and_save(sig, fs, dir_ori, 0.02, file)  # 先将信号，没有进行tqwt　分解之前, 原始子音频对应的语谱图保存到对应的文件夹中。

    if sig.size%2 !=0:   # 补齐到2的整数倍;
        sig = np.append(sig,[0])

    dq = DualQDecomposition(**dq_params)
    sig_high, sig_low = dq(sig)
    sig_res = sig - sig_high - sig_low

    stft_and_save(sig_low, fs, dir_ck, 0.02, file)
    stft_and_save(sig_high, fs, dir_wh, 0.08, file)
    stft_and_save(sig_res, fs, dir_res, 0.2, file)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)




parser = argparse.ArgumentParser()
parser.add_argument('--ncpu', type=int, default=10)
parser.add_argument('--savedir', default="tqwt", type=str, help='save directory')
parser.add_argument('--wavedir', default="test_cycle", type=str, help=' the path stored  the train or test breath cycles ')
args = parser.parse_args()


# 　使用tqwt　将每个子音频cycle,分解成三个成分， ori = cr + wheeze + res
# save_dir = " ../analysis/tqwt/train_cycle"
# save_dir = " ../analysis/tqwt/test_cycle"
save_dir = "../analysis/"+args.savedir+"/"+args.wavedir

# 待分解的音频所在的文件路径
# wav_dir = " ../data/train_cycle"
# wav_dir = " ../data/test_cycle"
wav_dir = '../data/'+args.wavedir



if __name__ == '__main__':


    fs = '../data/'
    # data文件下包括了ICBHI_final_database文件夹，新建trainset, testset两个文件夹；
    # 1. 先使用 clip_test()　将920份record划分为训练集和测试集;
    # clip_test(fs)


    # 2. 再使用clip_cycle ,将训练集和测试集中的每个record 中呼吸音子片段切割出来;
    # 需要新建两个文件夹，train_cycle/test_cycle 用于存放切割出来的子音频文件；
    # clip_cycle('../data/trainset/', "../data/train_cycle/")
    # clip_cycle('../data/testset/', "../data/test_cycle/")


    # 3. 　在该路径下 save_dir = " ../analysis/tqwt/train"　分别新建四个文件夹，用于保存　每个cycle　分解出来的３个分量音频；
    #  save_dir = " ../analysis/tqwt/test" ，　训练集和测试集　分别对子音频进行tqwt　分解；
    makedirs(save_dir + '/ori/')
    makedirs(save_dir + '/ck/')
    makedirs(save_dir + '/wh/')
    makedirs(save_dir + '/res/')
    pool = mp.Pool(processes=args.ncpu)


    t1 = datetime.now()
    file_list = []
    for file in os.listdir(wav_dir):
        if os.path.splitext(file)[1] == '.wav':# 　取出后缀为.wav文件名
            file_list.append(file)# 　将该文件下的所有文件名保存到file_list　列表中去；
    pool.map(save_pic, file_list)  # file_list是一个列表，　map(fun, para): 是多进程实现该函数的运行，　将para中参数传入到fun函数
    pool.close()
    pool.join()
    t2 = datetime.now()
    dur = t2 - t1

    print(f'Using tqwt decompose on the {wav_dir} and cost {dur}  time')
