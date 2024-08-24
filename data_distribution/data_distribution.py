from torch.utils.data import Dataset

import stats as sts
import librosa
from tqdm import tqdm
from config.utils import *
import  argparse
import torch
import torchaudio
from  torchaudio import  functional as F
from config.gtg import gen_gamma_3channel, gammatonegram_torch




class image_loader(Dataset):
    def __init__(self, data_dir, folds_file, test_fold, train_flag, params_json, input_transform=None, stetho_id=-1, aug_scale=None):

        # getting device-wise information
        self.file_to_device = {}
        device_to_id = {}
        device_id = 0
        files = os.listdir(data_dir)
        device_patient_list = []
        pats = []
        for f in files:
            device = f.strip().split('_')[-1].split('.')[0]
            if device not in device_to_id:
                device_to_id[device] = device_id
                device_id += 1
                device_patient_list.append([])
            self.file_to_device[f.strip().split('.')[0]] = device_to_id[device]
            pat = f.strip().split('_')[0]
            if pat not in device_patient_list[device_to_id[device]]:
                device_patient_list[device_to_id[device]].append(pat)
            if pat not in pats:
                pats.append(pat)

        print("DEVICE DICT", device_to_id)
        for idx in range(device_id):
            print("Device", idx, len(device_patient_list[idx]))

        # get patients dict in current fold based on train flag
        all_patients = open(folds_file).read().splitlines()  # 列表， 126个病人;  每个病人所对应的第几折中；
        patient_dict = {}
        for line in all_patients:   # 即当在当test_fold =4 时，  此时， patient_dict[] 只会有101 项病人， 不会包含第四折中的病人；
            idx, fold = line.strip().split(' ')
            if train_flag and int(fold) != test_fold:
                patient_dict[idx] = fold
            elif train_flag == False and int(fold) == test_fold:
                patient_dict[idx] = fold

        #extracting the audiofilenames and the data for breathing cycle and it's label
        print("Getting filenames ...")   # filenames: 920 份文件名称， rec_annotation_dict: 920  个字典， 每个包含了该音频的分段标注信息， 即起始 ，终止时间， 标签类别；
        filenames, rec_annotations_dict = get_annotations(data_dir)
        if stetho_id >= 0:
            self.filenames = [s for s in filenames if s.split('_')[0] in patient_dict and self.file_to_device[s] == stetho_id]
        else:
            self.filenames = [s for s in filenames if s.split('_')[0] in patient_dict]
            # self.filenames = 722:   从filenames(920)分 中取出  patient_dict 的病人， 即只会取出 训练集中的编号病人，  即第4折中的病人数据 都没有包含，
        self.audio_data = [] # each sample is a tuple with id_0: audio_data, id_1: label, id_2: file_name, id_3: cycle id, id_4: aug id, id_5: split id
        self.labels = []
        self.train_flag = train_flag
        self.data_dir = data_dir
        self.input_transform = input_transform

        # parameters for spectrograms
        self.sample_rate = 4000
        self.desired_length = 8
        self.n_mels = 64
        self.nfft = 256
        self.hop = self.nfft//2
        self.f_max = 2000

        self.dump_images = False
        self.filenames_with_labels = []

        # get individual breathing cycles from each audio file
        print("Exracting Individual Cycles")
        self.cycle_list = []   # self.cycle_list: 从训练集中的总共4折的病人中， 生成5454份，呼吸音音频；
        self.classwise_cycle_list = [[], [], [], []]

        self.classes_with_duration_list = [[], [], [], []]
        # 按照类别将，　　每个类别下各个子音频的持续时间添加到其中；


        for idx, file_name in tqdm(enumerate(self.filenames)):
            data = get_sound_samples(rec_annotations_dict[file_name], file_name, data_dir, self.sample_rate)
            cycles_with_labels = [(d[0], d[3], file_name, cycle_idx, 0) for cycle_idx, d in enumerate(data[1:])]
            self.cycle_list.extend(cycles_with_labels)
            for cycle_idx, d in enumerate(cycles_with_labels):
                self.filenames_with_labels.append(file_name+'_'+str(d[3])+'_'+str(d[1]))
                self.classwise_cycle_list[d[1]].append(d)


            # 1. 统计出四个类别下， 每个类别下, 各自样本所持续的时间； dur =  end - start;
            for cycle_in_curr_record, cur_data in  enumerate( data[1:]  ):
                cycle_dur =  cur_data[2] - cur_data[1]
                #　由于此时的 cur_data[3] 代表的是子音频的标签，　所以范围0-3　符合四个列表的范围；
                self.classes_with_duration_list[cur_data[3]].append(cycle_dur)


        if train_flag:
            print(" in the traindataset :\n")
            print(" the number of normal samples: \n", len(self.classes_with_duration_list[0]))
            print(" the number of crackle samples: \n", len(self.classes_with_duration_list[1]))
            print(" the number of wheeze samples: \n", len(self.classes_with_duration_list[2]))
            print(" the number of both  samples: \n", len(self.classes_with_duration_list[3]))


        if not train_flag:
            print(" in the testdataset :\n")
            print(" the number of normal samples: \n", len(self.classes_with_duration_list[0]))
            print(" the number of crackle samples: \n", len(self.classes_with_duration_list[1]))
            print(" the number of wheeze samples: \n", len(self.classes_with_duration_list[2]))
            print(" the number of both  samples: \n", len(self.classes_with_duration_list[3]))



        # concatenation based augmentation scheme
        if train_flag and aug_scale:
            self.new_augment(scale=aug_scale)

        # split and pad each cycle to the desired length
        for idx, sample in enumerate(self.cycle_list):
            output = split_and_pad(sample, self.desired_length, self.sample_rate, types=1)
            self.audio_data.extend(output)
           # self.audio_data: 生成5471  份， 为什么 ！= 5454 份呼吸音；   从 # self.cycle_list 从训练集中的5454 份，呼吸音音频从
        self.device_wise = []
        for idx in range(device_id):
            self.device_wise.append([])
        self.class_probs = np.zeros(4)
        self.identifiers = []
        for idx, sample in enumerate(self.audio_data):
            self.class_probs[sample[1]] += 1.0
            self.labels.append(sample[1])
            self.identifiers.append(sample[2]+'_'+str(sample[3])+'_'+str(sample[1]))
            self.device_wise[self.file_to_device[sample[2]]].append(sample)

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


    def __getitem__(self, index):

        audio = self.audio_data[index][0]

        aug_prob = random.random()
        if self.train_flag and aug_prob > 0.5:
            # apply augmentation to audio
            audio = gen_augmented(audio, self.sample_rate)

            # pad incase smaller than desired length
            audio = split_and_pad([audio, 0,0,0,0], self.desired_length, self.sample_rate, types=1)[0][0]

        # roll audio sample
        roll_prob = random.random()
        if self.train_flag and roll_prob > 0.5:
            audio = rollAudio(audio)

        # convert audio signal to spectrogram
        # spectrograms resized to 3x of original size
        audio_image = cv2.cvtColor(create_mel_raw(audio, self.sample_rate, f_max=self.f_max,
            n_mels=self.n_mels, nfft=self.nfft, hop=self.hop, resz=3), cv2.COLOR_BGR2RGB)

        # blank region clipping
        audio_raw_gray = cv2.cvtColor(create_mel_raw(audio, self.sample_rate, f_max=self.f_max,
            n_mels=self.n_mels, nfft=self.nfft, hop=self.hop), cv2.COLOR_BGR2GRAY)
        # print("\n  audio raw_gray  shape ", audio_raw_gray.shape )
        # ;torch.Size([3, 192, 753])   audio raw_gray  shape  (64, 251)


        audio_raw_gray[audio_raw_gray < 10] = 0
        for row in range(audio_raw_gray.shape[0]):
            black_percent = len(np.where(audio_raw_gray[row,:]==0)[0])/len(audio_raw_gray[row,:])
            if black_percent < 0.80:
                break

        # 　如果此时的行数减少了，　行数变成 row + 1;
        if (row+1)*3 < audio_image.shape[0]:
            # 　此时将剩余的行，代表了频域信息，　乘以３倍， 扩张后的mel 语谱图仍然< n_mels*3 时，则使用线性插值的方式，统一语谱图的大小；
            audio_image = audio_image[(row+1)*3:, :, :]
        audio_image = cv2.resize(audio_image, (audio_image.shape[1], self.n_mels*3), interpolation=cv2.INTER_LINEAR)

        if self.dump_images:
            save_images((audio_image, self.audio_data[index][2], self.audio_data[index][3],
                self.audio_data[index][5], self.audio_data[index][1]), self.train_flag)

        # label
        label = self.audio_data[index][1]

        # apply image transform
        if self.input_transform is not None:
            audio_image = self.input_transform(audio_image)

        # print("\n  audio image  shape ", audio_image.shape )
        # ;torch.Size([3, 192, 753])

        return audio_image, label

    def __len__(self):
        return len(self.audio_data)
#%%

# --data_dir ./data/ICBHI_final_database/ --folds_file ./data/patient_list_foldwise.txt --model_path models_out --lr 1e-3 --batch_size 1 --num_worker 8 --start_epochs 0 --epochs 200 --test_fold 4 --checkpoint ./models/ckpt_best.pkl

data_dir = '../data/ICBHI_final_database/'
folds_file = '../data/patient_list_foldwise.txt'
test_fold = 4



import ipdb

for i in range(5):
    print(i)
    ipdb.set_trace()



audio_image = image_loader(data_dir, folds_file,test_fold, False, "Params_json", input_transform=None, stetho_id=-1)

