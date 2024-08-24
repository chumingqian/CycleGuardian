# author: Chu Yun,  contact {chuyunxinlan at gmail dot com}
# time: 2023/4/20
#       下午6:26

import numpy as np
import torch

size = 512
atten_shape = (1, size, size)

mask = np.triu(np.ones(atten_shape), k=1).astype("uint8")
print(mask)


def subsequent_mask(size):
    atten_shape = (1, size, size)

    mask = np.triu(np.ones(atten_shape), k=1).astype("uint8")

    return  torch.from_numpy((mask) == 0).cuda()