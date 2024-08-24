# author: Chu Yun,  contact {chuyunxinlan at gmail dot com}
# time: 2023/8/9
#       上午16:08


from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import  torch.nn.functional as F


# v1, 实现将语谱图类别的特征，　与Mfcc统计类别的特征进行融合；
# Mel 语谱图使用cnn 抽取特征，  MFCC 使用 vit 抽取特征；


# 使用 2帧一组， 20 帧一组， 构成两种尺度下的组特征， 从而总共生成 313 + 31 = 344 组 向量表示；
# 并分别使用 cnn,  与 TCN 提取空间信息与时间信息；


# v3_2 使用kmeans_torch. 完成聚类功能；
#v3_4 : 只使用20 帧一组的分辨率，进行实验；  cnn + TCN ; 然后融合之后进行分组；
#v3_6: 使用DEC 深度编码聚类的方式，对所有的组进行聚类，



#v3_7:  将聚类的个数 从3变成五类；
#v3_7_4: 将簇表示向量维度， 在生成簇向量表示后，此时维度是768，
# 将簇向量表示维度降低到 128 ；

# v3_10-1:  直接使用 20 帧一组的语谱图视图，  丢弃倒谱系数视图；
# v3_10-2:  直接使用 10 帧一组的语谱图视图

#cycleGuardian:v1_1:
# 2帧位移， 3帧一组 cnn,    用于捕获crackle,  312组使用 transEncode, 生成全局向量， 对transEncode 使用对比学习；
# 5帧位移， 15帧一组，ctn;  123组使用聚类生成全局向量；  两类全局向量，门控融合；


#cycleGuardian:v1_2:
# 2帧位移， 3帧一组 cnn,   用于捕获crackle,  312组使用 transEncode, 生成全局向量， 对transEncode 使用对比学习；
# 5帧位移， 25帧一组，ctn;  125组使用聚类生成全局向量；  两类全局向量，门控融合；

#cycleGuardian:v2_1
# 4帧位移， 4帧一组 cnn,  用于捕获crackle,  156组使用 transEncode, 生成全局向量， 对transEncode 使用对比学习；
# 10帧位移， 15帧, 20, 25一组，ctn;  三个61组使用聚类生成全局向量，生成4种全局向量，  两类全局向量，门控融合；


#cycleGuardian:v2_2
# 4帧位移， 4帧一组 cnn,  用于捕获crackle,  156组使用 transEncode, 生成全局向量， 对transEncode 使用对比学习；
# 10帧位移， 20帧, 30, 40一组，ctn;  三个61组使用聚类生成全局向量，生成4种全局向量，  两类全局向量，门控融合；



#cycleGuardian:v2_3
# 4帧位移，  4帧一组 cnn,  用于捕获crackle, 156组使用 transEncode,  生成全局向量， 对transEncode 使用对比学习；
# 15帧位移， 20帧一组, cnn;  41组使用聚类生成全局向量，生成2种全局向量， 两类全局向量，门控融合；


#cycleGuardian:v2_4
# 直接使用 15帧位移， 20帧一组, cnn;  41组使用聚类生成全局向量，然后直接分类；


# cycleGuardian:v5_1   引入 event 数据扩充形成的样本；
# 并 对聚类中心使用余弦相似度，  对四个 glo_vec  之间则使用 软余弦相似度；
# 直接使用 15帧位移， 20帧一组, cnn;  41组使用聚类生成全局向量，然后直接分类；

#v5-1-1 ， 使用 簇融合的方式， 丢弃 门控网络的方式；
#


# #v5-1-2
# ， 去除 聚类中心的损失， 进行对比；



from torch.cuda.amp import autocast
import os
import wget
import timm
from copy import deepcopy
from timm.models.layers import to_2tuple, trunc_normal_



# 用于实现 聚类损失；
def kld_loss_function(q, p):
    kld = p * torch.log(p / (q+1e-10))
    return kld.sum()

# 用于实现相似度约束损失，用于实现辅助聚类；
from itertools import combinations
def cos_sim_loss_fun_v2(vec_a, vec_b, vec_c, vec_d,):
    vectors = [vec_a, vec_b, vec_c, vec_d]
    cos_loss = 0
    # Iterate over all unique pairs of vectors
    for vec1, vec2 in combinations(vectors, 2):
        cos_sim = torch.nn.functional.cosine_similarity(vec1, vec2, dim=1)
        cos_loss += torch.mean(torch.abs(cos_sim))

    return cos_loss


def normalized_cosine_similarity(x, y):
    """
    Compute the normalized cosine similarity between two vectors x and y, scaled to [0, 1].
    Parameters:
    - x: Tensor of shape (reduced_dim,)
    - y: Tensor of shape (reduced_dim,)
    Returns:
    - Normalized cosine similarity: Scalar
    """
    cos_sim = F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0))
    return (cos_sim + 1) / 2


def abs_soft_cosine_similarity(x, y, similarity_matrix):
    """
    Compute the normalized soft cosine similarity between two vectors x and y using a similarity matrix, scaled to [0, 1].
    Parameters:
    - x: Tensor of shape (batch_size, embed_dim)
    - y: Tensor of shape (batch_size, embed_dim)
    - similarity_matrix: Tensor of shape (embed_dim, embed_dim)
    Returns:
    - Normalized soft cosine similarity: Tensor of shape (batch_size,)
    """
    x_norm = torch.matmul(x, similarity_matrix)
    y_norm = torch.matmul(y, similarity_matrix)

    numerator = torch.sum(x * y_norm, dim=1)
    denominator = torch.norm(x_norm, dim=1) * torch.norm(y_norm, dim=1)

    soft_cos_sim = numerator / (denominator + 1e-10)

    return  torch.abs(soft_cos_sim)






# 卷积的输入的第一层使用 reflect 填充， 确保输入的原始特征的模式不被改变；
# 而最后一层卷积层可以使用 circular 填充模式， 此时提取的抽象级别的特征；




# todo :  需要 实现（84， 2），  （84， 20） 两种尺度下的卷积公式计算；
class Basic_slide_conv_scale20(nn.Module):
    def __init__(self, embed_dim):
        super(Basic_slide_conv_scale20, self).__init__()
        self.current_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0,bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True), #PReLU(32),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), #PReLU(64),

            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  #PReLU(128),

            nn.Conv2d(64,128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  #PReLU(256),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1,padding_mode='circular'),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # PReLU(256),
        )


        self.spec_embed_dim  = embed_dim
        self.linear_proj = nn.Sequential(
            # 是将256个通道的,15*1大小的特征图， 映射编码成512维度的编码向量，
            # 每组使用512维度的向量表示，共35组；
            nn.Linear(256 * 10 * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(True),
        )


    def forward(self, x):
        # torch.Size([1, 4, 84, 25])
        # print('the current group input: \n', x.shape)
        # 每组的输出是256通道, 　15*1大小的特征图；
        group_conv_out = self.current_conv(x)
        # torch.Size [bt, 256, 10, 2]

        # 将256通道的，　15*1大小的特征图，　在通道维度上展平；
        # print('the current group output: \n', group_conv_out.shape)
        # torch.Size([bt, channel, 15, 1])
        group_conv_out = torch.flatten(group_conv_out, start_dim=1)
        # torch.Size([1, 3840])
        # print('the current group output after flatten: \n',group_conv_out.shape)
        cur_group_embed = self.linear_proj(group_conv_out)  # (bt, 1024)
        return cur_group_embed




import  torchaudio
# from torchaudio.transforms import  AmplitudeToDB, MelSpectrogram
# from  nnAudio.spatial_time_state import  CQT2010v2, Gammatonegram
# import random



class Autoencoder(nn.Module):
    # Define your autoencoder for DEC here,
    # 该类用于实现将输入的编码维度，重新降维 和升高维度；
    # 聚类的时候，使用降维后的编码向量；
    # 而重构损失，是为了确保
    # ...

    def __init__(self, embed_dim, reduced_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, reduced_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(reduced_dim,embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z) #对降维后的编码向量，进行还原到原始的编码向量维度；

        return x_recon, z


#  增加了软余弦相似度；
class DEC(nn.Module):
    # DEC implementation, including soft_assignment and target_distribution methods
    # 这里实现软分配函数 以及 辅助目标分配函数； 用于计算聚类损失；
    # ...
    def __init__(self, embed_dim, reduced_dim,  num_clusters):
        super(DEC, self).__init__()
        self.autoencoder = Autoencoder(embed_dim=embed_dim, reduced_dim=reduced_dim)
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, reduced_dim))
        self.alpha = 1.0
        self.similarity_matrix = nn.Parameter(torch.eye(reduced_dim))




    def forward(self,x):
        bt, groups, embed_dim = x.shape
        x = x.view(-1, x.size(-1))# Flatten to (bt * num_groups, embed_dim)
        x_recon, z = self.autoencoder(x)
        z = z.view(bt, groups, -1)
        # x_recon (bt*groups,   embed_dim);   z:(bt, groups,reduced_dim)
        return  x_recon, z

    def soft_assignment(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(2) - self.cluster_centers) ** 2, dim=3) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=2, keepdim=True)
        return q

    def target_distribution(self, q):
        # weight = q ** 2 / q.sum(0)
        # return (weight.t() / weight.sum(1)).t()
        # Square the probabilities and sum across the 'groups' dimension
        weight = q ** 2 / q.sum(1, keepdim=True)

        # Normalize across the 'max_cluster' dimension
        return (weight / weight.sum(2, keepdim=True))

    def compute_similarity_loss(self, group_fused_vecs):
        """
        Compute the soft cosine similarity between two vectors x and y using a similarity matrix.

        Parameters:
        - x: Tensor of shape (batch_size, embed_dim)
        - y: Tensor of shape (batch_size, embed_dim)
        - similarity_matrix: Tensor of shape (embed_dim, embed_dim)

        Returns:
        - Soft cosine similarity: Tensor of shape (batch_size,)
        """
        center_loss = 0.0
        fused_vec_loss = 0

        # Penalize high similarity between cluster centers
        # num_centers = self.cluster_centers.size(0)
        # for i in range(num_centers):
        #     for j in range(i + 1, num_centers):
        #         similarity = normalized_cosine_similarity(self.cluster_centers[i],
        #                                                   self.cluster_centers[j])
        #         center_loss += similarity  #.mean()  # Minimize similarity (encourage dissimilarity)

        # Minimize soft cosine similarity between fused vectors
        num_vectors = len(group_fused_vecs)
        for i in range(num_vectors):
            for j in range(i + 1, num_vectors):
                similarity = abs_soft_cosine_similarity(group_fused_vecs[i], group_fused_vecs[j],
                                                               self.similarity_matrix)

                fused_vec_loss +=  similarity.mean() # Minimize similarity (encourage dissimilarity)

        return  fused_vec_loss





class  Group_Cluster_FusionAndProj(nn.Module):
    def __init__(self, in_channels=31, out_channels=1, kernel_size=1, hid_dim= 768,out_dim=128 ):
        super(Group_Cluster_FusionAndProj,self).__init__()

        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                nn.GELU(),  # nn.Tanh(),
                nn.LayerNorm(normalized_shape=[1, hid_dim, 1]),
        )

        self.layer2 = nn.Sequential(
                nn.Linear(hid_dim, out_dim),
                nn.GELU(),  # You can choose another activation function if needed
                nn.LayerNorm(out_dim),)

    def forward(self,x):
        #  #(bt,1, embed_dim, 1)
        out1 = self.layer1(x.unsqueeze(3))
        out2 = self.layer2(out1.squeeze(1).squeeze(2)) #(bt, dim)
        return  out2


class Cluster_Repre_FusionAndProj(nn.Module):
    def __init__(self,num_clusters=5, out_channel=1, cluster_dim=128,  out_dim= 768 ):
        super(Cluster_Repre_FusionAndProj, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels= num_clusters, out_channels=out_channel, kernel_size=1),
            nn.GELU(),  # nn.Tanh(),
            nn.LayerNorm(normalized_shape=[1, cluster_dim, 1]),
        )
        # 用于将融合后簇向量映射到， 768维度；
        # self.layer2 = nn.Sequential(
        #     nn.Linear(cluster_dim, out_dim),
        #     nn.LayerNorm(out_dim),
        # )

    def forward(self, x):
        #  # 5* (bt,embed_dim,) -->(bt, 5, embed_dim)
        comb_cluster = torch.stack(x, dim=1)
        out1 = self.layer1(comb_cluster.unsqueeze(3)) # (bt,5, embed_dim, 1) --->(bt, 1, embed_dim,1)
        # out2 = self.layer2(out1.squeeze(1).squeeze(2))  # (bt, out_dim)
        out2 = out1.squeeze(1).squeeze(2)
        return out2



import numpy as np
# from kmeans_pytorch import kmeans
class GuardianNet(nn.Module):
    def __init__(self,
                 # 语谱图的特征属性
                 spec_freq_height= 128, # 设置Mel, cqt, gamma 中滤波器的个数为84;
                 frame_width= 687,
                 spec_in_c=3,  # 输入特征图的大小
                 spec_embed_dim = 768,
                 group_embed_dim = 768,


                 # 统一的组数
                 num_groups_scale2=312,
                 num_groups_scale4=156,
                 num_groups_scale10=62,
                 num_groups_scale15= 45,  #345,
                 num_groups_scale20=31,
                 num_groups_scale25=25,


                 max_clusters = 4,
                 mix_beta=1.0,
                 # 将融合组，进行统一编码;
                 uni_state_embed= 768,
                 cluster_dim = 128,
                 num_classes=4,
                 distilled=False,):

        super(GuardianNet, self).__init__()



        self.num_classes = num_classes
        self.spec_embed_dim =  spec_embed_dim
        self.mix_beta = mix_beta
        self.num_tokens = 2 if distilled else 1
        # note,------------ 构建cnn 模块，用于处理语谱图特征 ------------------------------
        # self.conv_block_scale4 = Basic_Slide4_Conv(embed_dim=spec_embed_dim)
        # 为了确保两种尺度下  spatial 与 time 编码维度之和是 (1152) = 768+384 = 1024 + 128
        self.conv_block_scale20 = Basic_slide_conv_scale20(embed_dim=1024)
        self.spatial_time_proj = nn.Sequential(
                                     nn.Linear(1024, uni_state_embed),
                                     nn.Softplus(),)


        # note, 不使用预训练的位置编码权重，随机初始化位置编码权重， 然后更新其位置编码权重；


        self.max_clusters = max_clusters
        self.s20_dec_module = DEC(embed_dim= group_embed_dim, reduced_dim=cluster_dim,num_clusters= max_clusters)
        self.s20_group_cluster_fusion = Group_Cluster_FusionAndProj(in_channels=num_groups_scale15, hid_dim=group_embed_dim, out_dim=cluster_dim)



        self.s20_cluster_rep_fusion = Cluster_Repre_FusionAndProj(num_clusters=max_clusters, cluster_dim=cluster_dim,out_dim= cluster_dim)

        self.gate_fusion = GateFusion_v2(representation_size= cluster_dim)
        self.classifier_head = nn.Sequential( #nn.LayerNorm(fusion_embed_dim),
                                      nn.Linear(cluster_dim, self.num_classes))


        # note: 初始化token
        # nn.init.trunc_normal_(self.position_token_group_scale2, std=0.02)
        # nn.init.trunc_normal_(self.position_token_group_scale20, std=0.02)
        # nn.init.trunc_normal_(self.group_token, std=0.02)
        self.apply(_init_vit_weights)


    def group_mix(self, group_spec, target, time_domain=False, hw_num_patch=None):
        """
        group_spec: tensor,  (bt, num_group, embed_dim):  num_pt, 即group的个数， 每个group使用 768 维度的编码向量表示的；
        target: tensor,  (bt),  batch 中每个样本的标签，0，1，2，3；
        hw_num_patch: list  [12, 79]
        """
        if self.mix_beta > 0:  # 1-lam用来表示patch 混合的比例；
            lam = np.random.beta(self.mix_beta, self.mix_beta)
        else:
            lam = 1

        batch_size, num_patch, dim = group_spec.size()
        device = group_spec.device

        # 生成batch 个索引index, 但是索引的顺序打乱；
        index = torch.randperm(batch_size).to(device)

        if not time_domain:  # （1 -lam）:用于表示num_patch 中  将会被替换掉的比例；
            num_mask = int(num_patch * (1. - lam))  # num_mask 代表一个样本中， 有多少个groups(这里是patch) 将会被替换；

            # 随机产生（一个样本中的 patch 的总数）num_patch 个数目，并将其顺序打乱， 取出前面num_mask（将被替换的个数） 个patch；
            # 此时的mask 代表了一个样本中， 具体哪些位置上的 patch 将会被替换掉， 注意，由于是随机生成的位置，所以替换的patch位置不是顺序的；
            mask = torch.randperm(num_patch)[:num_mask].to(device)

            # 将原始的 group_spec (bt, num_pt, embed_dim) ， index = [3,4, 0,2,5,1] ， 假设bt=6;
            # 在batch 维度上首先换成 [index], 在样本层面进行，

            group_spec[:, mask, :] = group_spec[index][:, mask, :]
            # 所以，该行表达的意思是，每个样本中，mask 以外的位置仍然使用原始group_spec 样本中的patch；
            # 而原始的一个batch 中， 每个样本中 mask 位置上的patch,  将会被替换掉;
            # 替换的方式，用随机batch 中的每个样本， 对应的 mask 上的 patch

            lam = 1 - (num_mask / num_patch)  # lam 代表了一个样本中，保留了原始自身patch的比例；
        else:
            squared_1 = self.square_patch(group_spec, hw_num_patch)
            squared_2 = self.square_patch(group_spec[index], hw_num_patch)

            w_size = squared_1.size()[2]
            num_mask = int(w_size * (1. - lam))
            mask = torch.randperm(w_size)[:num_mask].to(device)

            squared_1[:, :, mask, :] = squared_2[:, :, mask, :]
            group_spec = self.flatten_patch(squared_1)
            lam = 1 - (num_mask / w_size)

        y_a, y_b = target, target[index]
        # 此时， group_spec 表示原始样本被混合后，新生成的样本特征；
        # y_a, 原始bt 个的样本标签；   y_b： 混合样本中 被替换掉的patch 来自于哪个类别下的patch,
        # index 代表了，原始每个group_spec 样本中被替换掉的patch 来自于原始batch 中的第几个 样本索引；
        # 即每个混合group_spec 样本中，新来的patch 对应来自于batch 中的第几个样本。
        return group_spec, y_a, y_b, lam, index


    def forward_fea_cluster(self, mel_spec, group_mix=False, y=None,):

        # mel spec : (bt, channel, n_mels, width_frames,  )
        # mfcc cof: (bt, ch,  n_ceps,frames,)

        # 将帧数和高度维度进行交换， 高度放在前面


        # 检查帧数是否正确，以及组数是否正确；
        # (bt, ch, height, width)
        #bt_spec = bt_spec[:, :, :, :(self.num_group_scale2 * )]  # 确保输入 frames 帧数正确;
        bt_spec = mel_spec # (bt, ch, n_mels, width_frames)
        bt, ch, height, width = bt_spec.shape
        cnn_input = bt_spec  # (bt, ch=3, H_freq=84, W_frames = 626 ) #note,  这里需要对应get_item 中通道叠加的顺序；


        stride15 = 15  # 15, 20,25 帧他们的位移都是10 帧；
        window_size20 = 20  # As the longest symptom duration is 20 frames
        group20_feature = []  # groups * (bt, 1, embed_dim)
        for i in range(0, width - window_size20 + 1, stride15):
            window = cnn_input[:, :, :, i:i + window_size20]  # tcn input: input should be (bt, embed_dim, seq_len=15)
            window_fea = self.conv_block_scale20(window).unsqueeze(dim=1)  # (bt, 1, embed_dim=1)
            window_fea = self.spatial_time_proj(window_fea)
            group20_feature.append(window_fea)

        scale20_tcn_state = torch.cat(group20_feature, dim=1)  # groups * (bt, 1, embed_dim)-->(bt, groups=61, embed_dim=252)
        # scale30_tcn_state = torch.cat(group30_feature, dim=1) #(bt, 60, 336)
        # scale40_tcn_state = torch.cat(group40_feature, dim=1) #(bt, 59, 420 )

        if group_mix:
            scale20_tcn_state, ori_label, s20_new_bt_label, lam_s20, s20_new_bt_index = self.group_mix(group_spec=scale20_tcn_state, target=y, time_domain=False, hw_num_patch=None)



        s20_recon, s20_red = self.s20_dec_module(scale20_tcn_state)
        s20_q = self.s20_dec_module.soft_assignment(s20_red)  # (bt, groups, max_cluster)
        s20_cluster_ids = torch.argmax(s20_q, dim=2)  # (bt,  num_groups)
        s20_p = self.s20_dec_module.target_distribution(s20_q)


        s20_rep_vec = []
        for i in range(self.max_clusters):  # (bt, num_groups, embed_dim) 该向量中是一个布尔类型，表明的是属于当前簇上group 置位True;
            s20_mask = (s20_cluster_ids == i).unsqueeze(2).expand_as(scale20_tcn_state).cuda()
            s20_cluster_features = torch.where(s20_mask, scale20_tcn_state, torch.tensor(0.).cuda())
            s20_clu_fea = self.s20_group_cluster_fusion(s20_cluster_features)
            s20_rep_vec.append(s20_clu_fea)  # s20_cluster_features,全0 输入到 其中之后，出现tensor(1135651., grad_fn=<MaxBackward1>) tensor(-828582.3125, grad_fn=<MinBackward1>)


        #   note,  以下两个余弦相似度都重新归一化到 (0,1)  区间上了
        # 计算4 个聚类中心之间的相似度损失， 使用余弦相似度， 计算4 簇表示向量之间的相似度损失，使用软余弦相似度
        # 获取一个全局向量；
        ori_s20_clu_loss = kld_loss_function(s20_q, s20_p)
        cluster4_fusion_vec_loss =  self.s20_dec_module.compute_similarity_loss(s20_rep_vec)

        #s20_glo_vec = self.gate_fusion(s20_rep_vec[0], s20_rep_vec[1], s20_rep_vec[2], s20_rep_vec[3])
        s20_glo_vec = self.s20_cluster_rep_fusion(s20_rep_vec)



        if not group_mix: 
            class_out = self.classifier_head(s20_glo_vec)
            return  ori_s20_clu_loss,   cluster4_fusion_vec_loss,  s20_glo_vec, class_out


        else:
            s20_cl_info =  [s20_new_bt_label,  s20_new_bt_index,  lam_s20,  s20_glo_vec]
            return ori_s20_clu_loss,   cluster4_fusion_vec_loss,  s20_cl_info




    def forward(self, mel_spec, group_mix=False, label=None):

        if not  group_mix: #  未使用 group mix
            # 返回的是原始特征的信息，
            # 依次用于求出，聚类损失；分类损失； 5个簇表示向量之间的余弦相似度损失；   特征重构损失； 可视化簇中心向量
            ori_s20_clu_loss,    cluster4_fusion_vec_loss,  s20_glo_vec, class_out = self.forward_fea_cluster(mel_spec, group_mix=False, y=None,)
            return  ori_s20_clu_loss,  cluster4_fusion_vec_loss,  s20_glo_vec, class_out

        else:
            # 当使用 group mix , 调用以下过程；
            # 如果调用了group mix 生成混合特征， 则需要返回新batch 的标签，以及对应的 index 索引；
            ori_s20_clu_loss, cluster4_fusion_vec_loss,  s20_cl_info = self.forward_fea_cluster(mel_spec,group_mix=group_mix,y=label)
            return  ori_s20_clu_loss,  cluster4_fusion_vec_loss,  s20_cl_info



import torch.nn as nn
class Projector(nn.Module):
    def __init__(self, in_dim, out_dim=128, apply_bn=True):
        super(Projector, self).__init__()
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(in_dim)
        self.relu = nn.ReLU()
        if apply_bn:
            self.projector = nn.Sequential(self.linear1, self.bn, self.relu, self.linear2)
        else:
            self.projector = nn.Sequential(self.linear1, self.relu, self.linear2)

    def forward(self, x):
        return self.projector(x)


class GroupMixConLoss(nn.Module):
    def __init__(self, temperature=0.06, negative_pair="all"):
        super().__init__()
        self.temperature = temperature
        self.negative_pair = negative_pair

    def forward(self, projection1, projection2, labels_a, labels_b, lam, index,):
        batch_size = projection1.shape[0]
        # proj: (bt, embed_dim)
        projection1, projection2 = F.normalize(projection1), F.normalize(projection2)

        # 先进行归一化， 然后计算两个 proj 的之间的相似性， (bt, bt ) = （bt, embed_dim）矩阵乘  (embed_dim, bt)
        anchor_dot_contrast = torch.div(torch.matmul(projection2, projection1.T), self.temperature) # 除以温度系数；

        mask_a = torch.eye(batch_size).cuda() # 创建单位矩阵， 即对角线矩阵为1， 其余为0；
        mask_b = torch.zeros(batch_size, batch_size).cuda()  # 初始化0矩阵；
        mask_b[torch.arange(batch_size).unsqueeze(1), index.view(-1, 1)] = 1 # 根据 index 张量， mask_b 被修改为在特定位置具有1。
        # 此时的mask_b  每一行， 代表的原始batch 中， 第几个样本；
        # 即mask_b 中第i 行，代表原始batch 中第 i 个样本； 而其中的每一行中，第j列数据为1，表示的是使用原始batch 中第j个样本；
        # 基于 lam 的值，使用 mask_a 和 mask_b 之间的线性插值创建组合掩码。

        # mask_a,即对角线位置上表示的是原始的样本； lam * mask_a，用于表示原始样本中保留patch 的比率；
        # （1-lam）* mask_b,代表使用另外一个样本中 patch 的比率；
        # 从而mask 中的第i行，表示了新生成的batch中第i个样本，
        # 此时的mask 每行中，只有两个位置上存在数值，其余位置都是0，
        # 一个是对角线位置上代表了保留原始样本中patch 的比率，另一个位置上表示使用另外一个样本中patch 的比率；
        mask = lam * mask_a + (1 - lam) * mask_b

        # Logits 是通过从 anchor_dot_contrast 中减去最大值来计算的，以保证数值稳定性。计算 logits 的指数, logits_max 代表每一行中的最大值；
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) #(bt, 1)
        logits = anchor_dot_contrast - logits_max.detach() # for numerical stability  (bt, bt)
        #此时， logits 代表的是原始特征与 混合特征之间的相似度；



        exp_logits = torch.exp(logits)  #(bt, bt)  # 其中的每一行，代表的该样本 与其他样本的相似度，

        # 如果 args 中的 negative_pair 设置为“diff_label”，则会应用附加掩码来排除具有相同标签的投影对。
        if self.negative_pair == 'diff_label':
            labels_a = labels_a.contiguous().view(-1, 1)
            logits_mask = torch.ne(labels_a, labels_a.T).cuda() + (mask_a.bool() + mask_b.bool())
            exp_logits *= logits_mask.float()


        # eq2  = log_prob  计算对数概率，并根据掩码计算正对的平均对数概率。 (bt, bt)
        # exp_logits.sum(1,) 在行维度上求和，将该样本与其他所有样本的相似度求和；
        # 本行上每个样本的相似度 减去 - (本行中该样本与其他样本的相似度总和)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # 最终损失是该平均对数概率的负数，然后将其重塑为大小为 (1, batch_size) 并取其平均值以获得最终损失值。
        loss = -mean_log_prob_pos
        loss = loss.view(1, batch_size)

        loss = loss.mean()
        return loss


# mask_b[]  = 1 的运作方式， 例如，如果 index 包含 [3, 7, 2] ，则这行代码会将 mask_b 中以下位置的元素设置为 1：
#
# The element at row 0, column 3.第 0 行第 3 列的元素。
# The element at row 1, column 7.第 1 行第 7 列的元素。
# The element at row 2, column 2.第 2 行第 2 列的元素。
# 结果是，对于每个批处理元素，一个特定列在 mask_b 中标记为 1。
# 这通常用于对比损失计算中，以指示哪些样本是正样本（属于同一类别或类别），并且在计算损失时应区别对待。





class GateFusion_v2(nn.Module):
    def __init__(self,  representation_size=1024, ):
        super(GateFusion_v2, self).__init__()

        W1 = torch.empty(representation_size, representation_size, requires_grad=True)
        W2 = torch.empty(representation_size, representation_size, requires_grad=True)

        W3 = torch.empty(representation_size, representation_size, requires_grad=True)
        W4 = torch.empty(representation_size, representation_size, requires_grad=True)
        #
        # W5 = torch.empty(representation_size, representation_size, requires_grad=True)
        # W6 = torch.empty(representation_size, representation_size, requires_grad=True)


        nn.init.kaiming_normal_(W1)
        nn.init.kaiming_normal_(W2)
        nn.init.kaiming_normal_(W3)
        nn.init.kaiming_normal_(W4)

        # nn.init.kaiming_normal_(W5)
        # nn.init.kaiming_normal_(W6)

        self.W1 = nn.Parameter(W1)
        self.W2 = nn.Parameter(W2)

        self.W3 = nn.Parameter(W3)
        self.W4 = nn.Parameter(W4)
        # self.W5 = nn.Parameter(W5)
        # self.W6 = nn.Parameter(W6)


        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2, x3, x4,  ):
        G1 = torch.sigmoid(x1 @ self.W1)
        G2 = torch.sigmoid(x2 @ self.W2)

        G3 = torch.sigmoid(x3 @ self.W3)
        G4 = torch.sigmoid(x4 @ self.W4)
        #
        # G5 = torch.sigmoid(x5 @ self.W5)
        # G6 = torch.sigmoid(x6 @ self.W6)

        F = G1 * x1 + G2 * x2  + G3 * x3 + G4 * x4


        return  F




def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def group_uni_net(num_classes: int = 4, mix_beta: float=1.0):

    model = GuardianNet( mix_beta=mix_beta,
                         # seq_len 序列的的编码维度
                        num_classes=num_classes,)
    return model



import  operator
if __name__ == "__main__":
    model = group_uni_net(num_classes=4,).cuda()
    # 相对高一点的学习率 3e-3, 使用以下参数
    cof_params = []
    # 使用较低的学习率， 1e-3,
    spec_params = []
    fusion_params = []

    spec_name = []
    cof_name = []
    fusion_name = []

    unconsider_params = []

    net = model
    for name, param in net.named_parameters():
        if param.requires_grad == True:
            if operator.contains(name, "conv_") or operator.contains(name, "spec_"):
                spec_params.append(param)
                spec_name.append(name)

        if param.requires_grad == True:
            if operator.contains(name, "gru_") or operator.contains(name, "cof_"):
                cof_params.append(param)
                cof_name.append(name)

            # print("\n ----cof params ", name, "\t", param.size())

        if param.requires_grad == True:

            if operator.contains(name, "uni_") or operator.contains(name, "fusion_"):
                fusion_params.append(param)
                fusion_name.append(name)

    print("\nHere is the spec name \n", spec_name)

    print("\nHere is the cof name \n", cof_name)

    print("\nHere is the fusion name \n", fusion_name)

    #input = torch.rand(6, 1, 96, 800).cuda()
    input = torch.rand(3, 80000).cuda()


    input = torch.rand(3, 80000).cuda()
    label = torch.randint(0,4, (2,)).cuda()

    mel_spec = torch.rand(2,3, 84, 687).cuda()
    mfcc_cof = torch.rand(2,1, 84, 687 ).cuda()



    print(" \n ================the original spec feature info =========== ")
    ori_s20_clu_loss,  cluster4_fusion_vec_loss, s20_glo_vec, class_out  = model(mel_spec,)
    print(f'5 culter center {class_out.shape}'
          f's4 trans global  vec {s20_glo_vec[0].shape}')


    print(" \n ================the hybrid spec feature info ===========")
    hyb_s20_clu_loss, hyb_cluster4_fusion_vec_loss, s20_cl_info = model(mel_spec,group_mix= True, label=label)
    print(f" s20_new_bt_label shape {s20_cl_info[0].shape} \t, s20_new_bt_index shape{s20_cl_info[1].shape}, "
          f"mix_s20_glo_vec {s20_cl_info[3].shape}  "
          f"hy s10 glo vec \t ")