# author: Chu Yun,  contact {chuyunxinlan at gmail dot com}
# time: 2023/9/4
#       下午6:36



import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn.functional as F

# Generate or load your encoded vectors with shape (batch_size, nums_embed, embed_dim)
# Replace this with your data
batch_size = 2
nums_embed = 25
embed_dim = 256
encoded_vectors = torch.randn(batch_size, nums_embed, embed_dim)

# Calculate Cosine Dissimilarity for each embedding vector
mean_dissimilarity_scores = []

for i in range(batch_size):
    batch_dissimilarity_scores = torch.zeros(nums_embed)
    for j in range(nums_embed):
        # Calculate cosine dissimilarity (1 - cosine similarity)
        a = encoded_vectors[i, j] # (embed_dim), 当前batch 下， 第i份数据， 第j 组；
        a = a.unsqueeze(0)   # (embed_dim ) ---> (1, embed_dim)
        b = encoded_vectors[i]  # 代表在当前batch 下， 第i 组数据，（groups,  embed_dim）

        cos_similarities = 1 - F.cosine_similarity(a, b, dim=1)  #（groups） 在编码维度上求余弦相异性；


        # cos_similarities = 1 - F.cosine_similarity(
        #     encoded_vectors[i, j].unsqueeze(0), encoded_vectors[i], dim=1)

        # Calculate the mean dissimilarity value
        mean_dissimilarity = cos_similarities.mean()
        batch_dissimilarity_scores[j] = mean_dissimilarity

    mean_dissimilarity_scores.append(batch_dissimilarity_scores)

# Convert the list of mean dissimilarity scores to a tensor
mean_dissimilarity_scores = torch.stack(mean_dissimilarity_scores)

# Print the mean dissimilarity scores for each embedding vector
print("Mean Dissimilarity scores for each embedding vector:")
print(mean_dissimilarity_scores.shape)



import torch

# Example mean dissimilarity scores and encoded_vectors tensor
mean_dissimilarity_scores = torch.randn(batch_size, nums_embed)  # Replace with your data
encoded_vectors = torch.randn(batch_size, nums_embed, embed_dim)  # Replace with your data

# Multiply the mean dissimilarity scores by encoded_vectors with broadcasting
weighted_embed_vectors = mean_dissimilarity_scores.unsqueeze(2) * encoded_vectors

# Sum along the groups dimension to obtain the weighted sum for each embed_dim
weighted_sum_embed_dim = weighted_embed_vectors.sum(dim=1)

# Print the result
print("Weighted sum of embed_dim for each group:")
print(weighted_sum_embed_dim)

import torch


def find_repeated_value_or_third_value(my_list):
    # Convert the list to a PyTorch tensor
    tensor_list = torch.tensor(my_list)

    # Check if there are any repeated values
    unique_values, counts = torch.unique(tensor_list, return_counts=True)

    # Find the repeated value (if any)
    repeated_value = unique_values[counts > 1]

    if repeated_value.numel() > 0:
        return repeated_value.item()  # Return the repeated value
    else:
        # If all values are different, return the third value
        return tensor_list[2].item()


# Example usage:
my_list = [0, 0, 3]  # Replace this with your list
result = find_repeated_value_or_third_value(my_list)
print("Result:", result)
