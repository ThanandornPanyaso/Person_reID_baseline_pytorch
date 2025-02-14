import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=330, type=int, help='test_image_index')
parser.add_argument('--test_dir',default='/home/big/Github/dataset_prepare/config-roi/cam1/dataset',type=str, help='./test_data')
opts = parser.parse_args()

data_dir = opts.test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in ['gallery','query']}


result = scipy.io.loadmat('features_deep.mat')
query_feature = torch.FloatTensor(result['query_f'])
gallery_feature = torch.FloatTensor(result['gallery_f'])
ids = result['person_Id'][0]
print(type(ids))
query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

#######################################################################
# sort the images
def sort_img(qf, gf):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    return index, score

i = np.where(ids == opts.query_index)[0]
# Check if query_index was found
if len(i) == 0:
    print(f"Query index {opts.query_index} not found in the array.")
else:
    print(f"Query index {opts.query_index} found at position {i[0]}.")
    index, scores = sort_img(query_feature[i], gallery_feature)
    for i in range(10):
        print(ids[index[i]])
        print(scores[index[i]])

