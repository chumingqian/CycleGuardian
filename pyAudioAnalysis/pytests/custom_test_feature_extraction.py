import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from pyAudioAnalysis.pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis.pyAudioAnalysis import ShortTermFeatures
from pyAudioAnalysis.pyAudioAnalysis import MidTermFeatures


def test_feature_extraction_short():
    [fs, x] = audioBasicIO.read_audio_file("test_data/1_sec_wav.wav")
    F, f_names = ShortTermFeatures.feature_extraction(x, fs, 
                                                      0.050 * fs, 0.050 * fs)# 注意，这里使用 win_len = hop_len ，　即帧与帧之间没有重叠；
    assert F.shape[1] == 20, "Wrong number of mid-term windows"
    assert F.shape[0] == len(f_names), "Number of features and feature " \
                                       "names are not the same"
    # 返回的是特征(n_features, frames)，　和特征名字;



# def mid_feature_extraction(signal, sampling_rate, mid_window, mid_step,
#                            short_window, short_step):
#     """
def test_feature_extraction_segment():
    print("Short-term feature extraction")
    [fs, x] = audioBasicIO.read_audio_file("test_data/5_sec_wav.wav")
    mt, st, mt_names = MidTermFeatures.mid_feature_extraction(x, fs, # 音频信号，　采样率；
                                                              0.5 * fs, # mid_window, mid_step
                                                              0.5 * fs,
                                                              0.04 * fs,  # short_window
                                                              0.02 * fs)  # short_step
    assert mt.shape[1] == 5, "Wrong number of short-term windows"
    assert mt.shape[0] == len(mt_names),  "Number of features and feature " \
                                          "names are not the same"

# 　中期窗口的位移，决定了整个音频被分成多少组；
#  　短期窗口的位移，决定整个音频将被分成多少帧；　
#  　需要明白，　一组中，　共包含了多少帧，是如何确定的, 通过 mid_window_ratio,  mt_step_ratio 求出来的；