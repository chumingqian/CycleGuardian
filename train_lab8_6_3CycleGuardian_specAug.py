#!/usr/bin/python






import argparse

import torch.nn  as nn
import torch.nn.functional
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import matplotlib
matplotlib.use('Agg')

# load external modules

import  operator
from  copy import deepcopy
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from  torch.utils.tensorboard import SummaryWriter
from  torchvision import  transforms
#from config.augmentation import SpecAugment
from timm.utils.model_ema import ModelEmaV2, ModelEmaV3

from ICBHIDataset_v5_1 import *
from nets.CycleGuardian_v5_1_3 import group_uni_net as create_model
from  nets.CycleGuardian_v5_1_3 import  Projector, GroupMixConLoss



#lab1,  使用 cnn, + vit,  vit 替代gru 的原因是， gru 各组之间会受到时序信息的影响；

print ("Train import done successfully")
# input argmuments
parser = argparse.ArgumentParser(description='Temporal_convolution_transformer: Lung Sound Classification')
parser.add_argument('--lr_h', default=1e-2, type=float, help='High learning rate')
parser.add_argument('--lr_m', default=1e-3, type=float, help='middle learning rate')
parser.add_argument('--lr_l', default=5e-4, type=float, help='low learning rate')


parser.add_argument('--weight_decay', default=0.0005,help='weight decay value')
parser.add_argument('--gpu_ids', default=[0], help='a list of gpus')
parser.add_argument('--num_worker', default=4, type=int, help='numbers of worker')
parser.add_argument('--batch_size', default=4, type=int, help='bacth size')
parser.add_argument('--epochs', default=10, type=int, help='epochs')
parser.add_argument('--start_epochs', default=0, type=int, help='start epochs')

parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--event_dir', type=str, default= './data/events')
parser.add_argument('--split_method', default=0, type=int, help='0: official 6-4 split; 1: five folds split, 2: random 8-2 split')

# if the official 6-4 split, provide the  train- test  split file;
parser.add_argument('--dataset_split_file', type=str, default=None)

# if  tqwt 3 componment ,use following parameter
parser.add_argument('--train_dir', type=str, help='data directory')
parser.add_argument('--test_dir', type=str, help='data directory')

# if 5 fold split , provide  5 folds split file.
parser.add_argument('--folds_file', type=str, help='folds text file')
parser.add_argument('--test_fold', default=4, type=int, help='Test Fold ID')


parser.add_argument('--aug_scale', default=None, type=int, help='Augmentation multiplier')
parser.add_argument('--specaug_policy', default='icbhi_ast_sup', type=str, help='policy for spec augemnt')
parser.add_argument('--specaug_mask', default='mean', type=str, help='spec aug mask value', choices=['mean', 'zero'])
parser.add_argument('--model_path',type=str, help='model saving directory')
parser.add_argument('--checkpoint', default=None, type=str, help='load checkpoint')
parser.add_argument('--stetho_id', default=-1, type=int, help='Stethoscope device id')
parser.add_argument("--annealing_epoch", type=int, default=50)



# use for  constrative learning
# 设置用于 对比学习的参数，  temperature,  args.target_type= grad_flow,  p_cl;
parser.add_argument('--proj_dim', type=int, default=768)
parser.add_argument('--negative_pair', type=str, default='all',
                    help='the method for selecting negative pair', choices=['all', 'diff_label'])

parser.add_argument('--mix_beta', default=1.0, type=float,    #用于生成  设置每个样本 group 的混合比例
                    help='patch-mix interpolation coefficient')
parser.add_argument('--temperature', type=float, default=0.06)

# use  for  different loss  part  weight,
parser.add_argument('--p_cluster', default=  1.0 , type=float, help='loss weight  for  the global cluster loss')
parser.add_argument('--p_cos_sim', default= 0.2 , type=float, help='loss weight  for  the cos sim loss')
parser.add_argument('--p_contra',  default= 0.500, type=float,)
parser.add_argument('--p_class', default= 1.0, type=float, help='loss weight  for  the global fusion loss')



parser.add_argument('--target_type', type=str, default='project_flow',
                    help='how to make target representation',
                    choices=['grad_block', 'grad_flow', 'project_block', 'project_flow'])

args = parser.parse_args()

################################MIXUP#####################################
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

##############################################################################
#@torch.compile
def get_score(hits, counts, pflag=False):
    sp = hits[0] / counts[0]
    se = (hits[1] + hits[2] + hits[3]) / (counts[1] + counts[2] + counts[3])
    sc = (se+sp) / 2.0

    # normal accuracy
    int_sp = hits[0] / (counts[0] + 1e-10) * 100
    # abnormal accuracy
    int_se = sum(hits[1:]) / (sum(counts[1:]) + 1e-10) * 100
    int_sc = (int_sp + int_se) / 2.0


    if pflag:
        print("*************The official Metrics******************")
        print("The frac format Sp: {}, Se: {}, Score: {}".format(sp, se, sc))
        print("The int  format S_p: {}, S_e: {}, Score: {}".format(int_sp, int_se, int_sc))

        print("Normal: {}, Crackle: {}, Wheeze: {}, Both: {} \n ".format(hits[0]/counts[0], hits[1]/counts[1],
            hits[2]/counts[2], hits[3]/counts[3]))




from itertools import combinations
#@torch.compile
def cos_sim_loss_fun_v2(vec_a, vec_b, vec_c, vec_d, vec_e):
    vectors = [vec_a, vec_b, vec_c, vec_d, vec_e]
    cos_loss = 0
    # Iterate over all unique pairs of vectors
    for vec1, vec2 in combinations(vectors, 2):
        cos_sim = torch.nn.functional.cosine_similarity(vec1, vec2, dim=1)
        cos_loss += torch.mean(torch.abs(cos_sim))

    return cos_loss


# @torch.compile
def kld_loss_function(q, p):
    kld = p * torch.log(p / (q+1e-10))
    return kld.sum()



class Trainer:
    def __init__(self):
        self.args = args

        if self.args.split_method  == 0:
            print(" this modeling  on  the  official split methon  6-4 split: \n ", )

        elif self.args.split_method  == 1:
            print(" this modeling  on  the  Five  folds  split: \n ", )
        else:
            print(" this modeling  on  the  random 8-2  split: \n ", )


        self.writter_train = SummaryWriter("runs_log/Train")
        self.writter_vaild = SummaryWriter("runs_log/Vaild")


        # args.h, args.w = 798, 128
        # args.resz = 1
        # train_transform = [transforms.ToTensor(),
        #                    SpecAugment(args),
        #                    transforms.Resize(antialias=True,size=(int(args.h * args.resz), int(args.w * args.resz)))]
        # val_transform = [transforms.ToTensor(),
        #                  transforms.Resize(antialias=True,size=(int(args.h * args.resz), int(args.w * args.resz)))]
        # # train_transform.append(transforms.Normalize(mean=mean, std=std))
        # # val_transform.append(transforms.Normalize(mean=mean, std=std))
        #
        # train_transform = transforms.Compose(train_transform)
        # val_transform = transforms.Compose(val_transform)

        train_dataset = ICBHIDataset_with_event(data_dir=self.args.data_dir,
                                     event_data_dir= self.args.event_dir,
                                     dataset_split=self.args.split_method,
                                     dataset_split_file=self.args.dataset_split_file,
                                     test_fold=self.args.test_fold, stetho_id=-1,
                                     train_flag=True,
                                     aug_audio= False, aug_audio_scale=1, aug_feature=False,
                                     desired_time=8, sample_rate1= 22000, sample_rate2=22000,
                                     n_filters1=84, n_filters2=42,
                                     input_transform=None,  # train_transform,
                                     )

        test_dataset = ICBHIDataset_with_event(data_dir=self.args.data_dir,
                                    event_data_dir=self.args.event_dir,
                                    dataset_split=self.args.split_method,
                                    dataset_split_file=self.args.dataset_split_file,
                                    test_fold=self.args.test_fold, stetho_id=-1,
                                    train_flag=False,
                                    aug_audio=False, aug_audio_scale=1, aug_feature=False,
                                    desired_time=8, sample_rate1= 22000, sample_rate2=22000,
                                    n_filters1=84, n_filters2=42,
                                    input_transform=None,  # val_transform,
                                    )

        self.test_ids = np.array(test_dataset.identifiers)
        self.test_paths = test_dataset.filenames_with_labels



        # loading checkpoint
        self.net = create_model(num_classes=4, mix_beta=self.args.mix_beta) #.cuda()


        self.s20_projector = Projector(128, 128).cuda()

        # note,  初始化EmaV2 模型, 并将其移动到与model 同一个设备上；
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        self.net_copy = self.custom_deepcopy(self.net)
        self.ema = ModelEmaV3(self.net_copy, decay=0.1)
        device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
        self.ema.module.to(device)
        self.net.to(device)


        if self.args.checkpoint is not None:
            checkpoint = torch.load(self.args.checkpoint)
            self.net.load_state_dict(checkpoint)
            # uncomment in case fine-tuning, specify block layer
            # before block_layer, all layers will be frozen durin training
            # self.net.fine_tune(block_layer=5)
            print("Pre-trained Model Loaded:", self.args.checkpoint)
        self.net = nn.DataParallel(self.net, device_ids=self.args.gpu_ids)

        # weighted sampler
        # reciprocal_weights = []
        # for idx in range(len(train_dataset)):
        #     reciprocal_weights.append(train_dataset.class_ratio[train_dataset.labels[idx]])
        # weights = (1 / torch.Tensor(reciprocal_weights))
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_dataset))

        # note 1:don't use the sample;
        sampler = None
        # dataLoader 　用于一次取出batch size 个数据，送到网络中，　
        # 注意 如果指定sampler，　则表明使用这种规则的方式获取样本的索引，　则此时，　shuffle 使用默认值 False;
        # shuffle 为False 时，且没有指定sampler时，　按照顺序采样样本；　
        self.train_data_loader = DataLoader(train_dataset, num_workers=self.args.num_worker,
                batch_size=self.args.batch_size, sampler=sampler, shuffle= True)
        self.val_data_loader = DataLoader(test_dataset, num_workers=self.args.num_worker, 
                batch_size=self.args.batch_size, shuffle=False)
        print("DATA LOADED")



        params_to_update = []
        for name,param in self.net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print("\n ", name, param.size())



        # Observe that all parameters are being optimized
        #self.optimizer = optim.SGD(params_to_update, lr=self.args.lr_m, momentum=0.9, weight_decay=self.args.weight_decay)
        self.optimizer = optim.Adam(params_to_update, lr = self.args.lr_h)
        # self.cl_optimizer = optim.Adam(cl_params, lr=self.args.lr_l)
        # self.cl_optimizer = optim.SGD(params_to_update, lr=self.args.lr_l, momentum=0.9, weight_decay=self.args.weight_decay)

        self.exp_lr_scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[200,350,450,550],gamma=0.33,last_epoch= -1)





        # weights for the loss function
        #weights = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        # weights = torch.tensor(train_dataset.class_ratio, dtype=torch.float32)
        # weights = weights / weights.sum()
        # weights = 1.0 / weights
        # weights = weights / weights.sum()
        # weights = weights.cuda()
        
        weights = None
        self.loss_func = nn.CrossEntropyLoss(weight=weights)
        self.loss_nored = nn.CrossEntropyLoss(reduction='none')
        self.cl_criterion =  GroupMixConLoss(temperature=self.args.temperature, negative_pair= self.args.negative_pair).cuda()



        self.mix_beta = self.args.mix_beta
        self.p_cluster =  self.args.p_cluster
        self.p_sim  = self.args.p_cos_sim
        self.p_contra = self.args.p_contra
        self.p_class  = self.args.p_class



    def custom_deepcopy(self, model):
        model_copy = type(model)()
        model_copy.load_state_dict(model.state_dict())
        return  model_copy



    def train(self):
        train_losses = []
        test_losses = []

        test_acc = []
        best_acc = -1

        best_Se = -1
        best_Sp = -1
        best_Sc = -1
        best_Confusion_Matrix = []


        tb_writer = SummaryWriter()

        # 　开始一个epoch 的训练；
        for _, epoch in enumerate(range(self.args.start_epochs, self.args.epochs)):



            cla_losses = []
            cluster_losses = []
            fusion_rep_losses = []
            contra_losses = []


            losses = []
            class_hits = [0.0, 0.0, 0.0, 0.0]
            class_counts = [0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7]


            # 以下两个参数用于计算已经遍历过了的batch上的正确率
            running_corrects = 0.0   # 已经遍历了的batch上正确数量
            denom = 0.0 # 已经遍历了的batch的样本数量

            classwise_train_losses = [[], [], [], []]  # 每个类别的损失；
                
            # 从dataloader 中读取一个batch的数据， 并且通过enumerate() 逐个取出该batch　中的每个样本到网络中；
            for i, (spec, label) in enumerate(tqdm(self.train_data_loader,  desc=' training process')):
                spec_data, label = spec.cuda().float(),  label.long().cuda()
                # label = label.long() # ;

                # in case using mixup, uncomment 2 lines below
                # image, label_a, label_b, lam = mixup_data(image, label, alpha=0.5)
                # image, label_a, label_b = map(Variable, (image, label_a, label_b))

                ori_s20_clu_loss, cluster4_fusion_vec_loss, s20_glo_vec, class_out = self.net(spec_data, group_mix=False, label=None)

                cla_loss = self.loss_func(class_out, label)  #  融合特征的分类损失；

                if args.target_type == 'grad_block':
                    proj1 = deepcopy(s20_glo_vec[0].detach())
                elif args.target_type == 'grad_flow':
                    proj1 =s20_glo_vec[0]
                elif args.target_type == 'project_block':
                    proj1 = deepcopy(self.s20_projector(s20_glo_vec[0]).detach())
                elif args.target_type == 'project_flow':
                    proj1_s20 = self.s20_projector(s20_glo_vec)


                # mix_features  相比与原始 的features,  需要经过一个 projector() 层；
                # cl_info: list[0]:bt_label, list[1]:bt_index, [2]: lam_ratio;  [3]: mix_glo_vec
                hyb_s20_clu_loss,  hyb_cluster4_fusion_vec_loss, s20_cl_info = self.net(spec_data, group_mix=True,label=label)
                proj2_s20 = self.s20_projector(s20_cl_info[3])
                cl_s20_loss = self.cl_criterion(proj1_s20, proj2_s20, label, s20_cl_info[0], s20_cl_info[2], s20_cl_info[1])


                clu_loss =  self.p_cluster *  (ori_s20_clu_loss +  hyb_s20_clu_loss)  # 1000
                contra_loss =  self.p_contra * cl_s20_loss
                fusion_rep_loss =  self.p_sim  *  (cluster4_fusion_vec_loss + hyb_cluster4_fusion_vec_loss)   #


                loss  =  cla_loss +  clu_loss   + contra_loss  + fusion_rep_loss


                fus_nored = self.loss_nored(class_out, label)
                loss_nored =  fus_nored

                # spec_pred = torch.argmax(g_spec_out, 1)
                prob_fus,  fus_pred = torch.max(class_out, 1)
                preds = fus_pred
                preds = preds.cuda()
                
                running_corrects += torch.sum(preds == label.data)
                denom += len(label.data)


                #class 计算在训练数据中（真实值）每一类对应样本数量和每一类预测正确的样本数量
                for idx in range(preds.shape[0]):
                    class_counts[label[idx].item()] += 1.0
                    if preds[idx].item() == label[idx].item():
                         class_hits[label[idx].item()] += 1.0
                    classwise_train_losses[label[idx].item()].append(loss_nored[idx].item())

                self.optimizer.zero_grad()
                # self.cl_optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #self.cl_optimizer.step()

                # note, 在模型完成反向传播之后使用， 这里更新ema 的模型
                self.ema.update(self.net)


                cluster_losses.append(clu_loss.data.cpu().numpy())
                fusion_rep_losses.append(fusion_rep_loss.data.cpu().numpy())
                contra_losses.append(contra_loss.data.cpu().numpy()) # 对比损失；
                cla_losses.append(cla_loss.data.cpu().numpy())  # 分类损失
                losses.append(loss.data.cpu().numpy())


                if i % 10000 == self.train_data_loader.__len__()-1:
                    print(" \n ==================================================")
                    print("epoch {} iter {}/{} Train Total loss: {}".format(epoch,i, len(self.train_data_loader), np.mean(losses)))

                    print("Train Accuracy: {}".format(running_corrects.double() / denom))
                    print("Classwise_Losses Normal: {}, Crackle: {}, Wheeze: {}, Both: {}".format(
                        np.mean(classwise_train_losses[0]),
                        np.mean(classwise_train_losses[1]),
                        np.mean(classwise_train_losses[2]),
                        np.mean(classwise_train_losses[3])))

                    print("\n -----show the training info -----------")
                    get_score(class_hits, class_counts, True)


                    print("testing......")
                    acc, test_loss, conf_mat, Sp, Se, Sc = self.evaluate(self.net, epoch, i)
                    self.writter_vaild.add_scalar("Sc", Sc, epoch)
                    self.writter_vaild.add_scalar("Se", Se, epoch)
                    self.writter_vaild.add_scalar("Sp", Sp, epoch)


                    if epoch > 10 and  best_acc < acc:
                        best_acc = acc
                        best_Se = Se
                        best_Sp = Sp
                        best_Sc = Sc
                        best_Confusion_Matrix = conf_mat
                        self.writter_vaild.add_scalar("best_acc", best_acc, epoch)  # desc:可视化

                        torch.save(self.net.module.state_dict(), args.model_path + '/lab8-6CycGuradin' +'epoch_'+ str(epoch)+'Acc_'+str(best_acc)  +'.pkl')
                        print("Best ACC achieved......", best_acc.item())
                    print("BEST ACCURACY TILL NOW", best_acc)

                    train_losses.append(np.mean(losses))
                    test_losses.append(test_loss)
                    test_acc.append(acc)

            train_acc = running_corrects.double() / denom
            train_loss = np.mean(losses)

            tags = ["train_acc", "train_loss", "val_acc", "val_loss", ]

            self.writter_train.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], epoch)

            self.writter_train.add_scalar("train_acc", train_acc, epoch)
            self.writter_train.add_scalars("Train_Class_acc",
                                            {"Noraml": class_hits[0] / class_counts[0],
                                             "Crackle": class_hits[1] / class_counts[1],
                                             "Wheeze": class_hits[2] / class_counts[2],
                                             "Both": class_hits[3] / class_counts[3]},
                                            epoch)
            
            # 可视化每个epoch 上总损失 ，以及 四个分量上的损失；
            self.writter_train.add_scalar("Train_total_Loss", train_loss, epoch)
            self.writter_train.add_scalars("Train_multi_loss",
                                            {
                                                " cluster loss": np.mean(cluster_losses),
                                                "fusion rep vec loss": np.mean(fusion_rep_losses),
                                                "constrstive learning loss ": np.mean(contra_losses),
                                                "classification item": np.mean(cla_losses),
                                            },
                                            epoch)

            self.exp_lr_scheduler.step()
            print(f"best_Se{best_Se}\tbest_Sp{best_Sp}\tbest_Sc{best_Sc}\tbest_Acc{best_acc}")
            print(f"ds combine best_Confusion_matrix:\n{best_Confusion_Matrix}")


    def evaluate(self, net, epoch, iteration):

        self.net.eval()
        test_losses = []

        denom = 0.0
        running_corrects = 0.0
        class_hits = [0.0, 0.0, 0.0, 0.0]  # normal, crackle, wheeze, both
        class_counts = [0.0, 0.0, 0.0 + 1e-7, 0.0 + 1e-7]  # normal, crackle, wheeze, both
        classwise_test_losses = [[], [], [], []]
        conf_label, conf_pred = [], []



        # for i, (image, label) in tqdm(enumerate(self.val_data_loader)):
        for i, (spec,label) in enumerate(self.val_data_loader, ):
            spec_data,  label = spec.cuda().float(),   label.long().cuda()
            # label = label.long() # ;

            # in case using mixup, uncomment 2 lines below
            # image, label_a, label_b, lam = mixup_data(image, label, alpha=0.5)
            # image, label_a, label_b = map(Variable, (image, label_a, label_b))

            ori_s20_clu_loss,  cluster4_fusion_vec_loss, s20_glo_vec, class_out = self.ema.module(spec_data,  group_mix=False,label=None)

            cla_loss = self.loss_func(class_out, label)
            clu_loss = self.p_cluster * (ori_s20_clu_loss )  # 1000
            fusion_rep_loss = self.p_sim * (cluster4_fusion_vec_loss )  #
            loss =  cla_loss +  clu_loss  + fusion_rep_loss



            fus_nored = self.loss_nored(class_out, label)
            loss_nored = fus_nored

            # spec_pred = torch.argmax(g_spec_out, 1)
            prob_fus, fus_pred = torch.max(class_out, 1)
            preds = fus_pred
            preds = preds.cuda()


            # calculate loss from output
            # in case using mixup, uncomment line below and comment the next line
            # loss = mixup_criterion(self.loss_func, output, label_a, label_b, lam)

            test_losses.append(loss.data.cpu().numpy())
            running_corrects += torch.sum(preds == label.data)

            # updating denom
            denom += len(label.data)

            # class
            for idx in range(preds.shape[0]):
                class_counts[label[idx].item()] += 1.0
                conf_label.append(label[idx].item())
                conf_pred.append(preds[idx].item())
                if preds[idx].item() == label[idx].item():
                    class_hits[label[idx].item()] += 1.0

                classwise_test_losses[label[idx].item()].append(loss_nored[idx].item())


        print("Val Accuracy by  fusion result : {}".format(running_corrects.double() / denom))
        print("epoch {}, Validation BCE loss: {}".format(epoch, np.mean(test_losses)))
        print("Classwise_Losses Normal: {}, Crackle: {}, Wheeze: {}, Both: {}".format(
            np.mean(classwise_test_losses[0]),
            np.mean(classwise_test_losses[1]),
            np.mean(classwise_test_losses[2]),
            np.mean(classwise_test_losses[3])))

        print("\n -----show the validation info -----------")
        get_score(class_hits, class_counts, True)


        Se = (class_hits[1] + class_hits[2] + class_hits[3]) / (class_counts[1] + class_counts[2] + class_counts[3])
        Sp = class_hits[0] / class_counts[0]
        Sc = (Se + Sp) / 2
        Acc = sum(class_hits) / sum(class_counts)

        self.writter_vaild.add_scalar("Acc_test", running_corrects / denom, epoch)
        self.writter_vaild.add_scalars("Class_acc_test",
                                      {"Noraml": class_hits[0] / class_counts[0],
                                       "Crackle": class_hits[1] / class_counts[1],
                                       "Wheeze": class_hits[2] / class_counts[2],
                                       "Both": class_hits[3] / class_counts[3]},
                                      epoch)


        self.writter_vaild.add_scalars("Class_Losss_test",
                                      {"Noraml": np.mean(classwise_test_losses[0]),
                                       "Crackle": np.mean(classwise_test_losses[1]),
                                       "Wheeze": np.mean(classwise_test_losses[2]),
                                       "Both": np.mean(classwise_test_losses[3])},
                                      epoch)

        self.writter_vaild.add_scalar("Total_Loss_test", np.mean(test_losses), epoch)
        # 验证集上， 四个部分的损失

        # aggregating same id, majority voting
        conf_label = np.array(conf_label)
        conf_pred = np.array(conf_pred)

        # the following  code relize the for the exceed  part,
        # 以下情况是当输入超过8s的部分， 将超过的部分通过重叠的方式，重新组成一个新的样本，
        # If a cycle  exceed the 8s , like 9.6s,
        # the over part 1.6s  will  also be used padded byself  and generate the new sample,
        # and the generate new sample's label  will  use the  same label;
        # y_pred, y_true = [], []
        # for pt in self.test_paths:
        #     y_pred.append(np.argmax(np.bincount(conf_pred[np.where(self.test_ids == pt)])))
        #     y_true.append(int(pt.split('_')[-1]))

        conf_matrix = confusion_matrix(conf_label, conf_pred)
        acc = accuracy_score(conf_label, conf_pred)


        print("*************The Helper Metrics******************")
        print("Confusion Matrix \n", conf_matrix)
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        print("Classwise Scores -->: ", conf_matrix.diagonal())

        print("Accuracy Score ---> :", acc)
        print(f" Micro F1 Score: {f1_score(conf_label, conf_pred, average= 'micro')}")
        prec = precision_score(conf_label, conf_pred, average='weighted', zero_division= np.nan)
        rec = recall_score(conf_label, conf_pred, average='weighted')
        print(f" weighted Precision: {prec}")
        print(f" weighted Recall: {rec}")


        self.net.train()
        return acc, np.mean(test_losses), conf_matrix, Sp, Se, Sc,


if __name__ == "__main__":

    '''
    for test_id in range(0, 5):
        args.test_fold =  test_id
        args.epochs = 30
        args.lr = 3e-3
        args.arg_scale = 1
        args.checkpoint = './models_out/pitch_lab2_6ch_best_acc.pkl'
    '''
    trainer = Trainer()
    trainer.train()

"""
python  train_lab3_1GuardianUni.py    --data_dir ./data/ICBHI_final_database --dataset_split_file ./data/patient_trainTest6_4.txt --model_path ./models_out --lr_h 0.001 --lr_l 0.001 --batch_size 8 --num_worker 0 --start_epochs 0 --epochs 700
"""
