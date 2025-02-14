import argparse
import scipy.io
import torch
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd

def get_cropped_image_path(metadata_csv, class_id, current_frame):
    df = pd.read_csv(metadata_csv)
    df.columns = df.columns.str.strip()
    result = df[(df["class_id"] == class_id) & (df["current_frame"] == current_frame)]
    if not result.empty:
        relative_path = result.iloc[0]["image_cropped_obj_path_saved"]
        return os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(metadata_csv)), relative_path))
    return None
def get_cropped_image_path_loose(metadata_csv, class_id):
    df = pd.read_csv(metadata_csv)
    df.columns = df.columns.str.strip()
    result = df[(df["class_id"] == class_id)]
    if not result.empty:
        relative_path = result.iloc[-1]["image_cropped_obj_path_saved"]
        return os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(metadata_csv)), relative_path))
    return None
def imshow(path, title=None):
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def sort_img(qf, gf):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query).squeeze(1).cpu().numpy()
    index = np.argsort(score)[::-1]  # Sort descending
    return index, score

parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--output_dir', default='/home/big/Github/dataset_prepare/get_img_cosine_renew/output', type=str, help='./img_data')
opts = parser.parse_args()

data_dir = opts.output_dir
result = scipy.io.loadmat('features_deep_renew_split.mat')
meta_csv = os.path.join(data_dir, "cleaned_metadata.csv")
placeholderpath = "placeholder.png"
query_feature = torch.FloatTensor(result['query_f']).cuda()
gallery_feature = torch.FloatTensor(result['gallery_f']).cuda()
query_ids = result['query_ids'][0]
query_frames = result['query_frames'][0]
gallery_ids = result['gallery_ids'][0]
gallery_frames = result['gallery_frames'][0]

for query_id in query_ids:
    query_index = np.where(query_ids == query_id)[0][0]
    print(f"Processing Query ID: {query_id}")
    index, scores = sort_img(query_feature[query_index], gallery_feature)
    query_path = get_cropped_image_path(meta_csv, query_id, int(query_frames[query_index]))
    print('Top 10 images are as follows:')
    
    fig = plt.figure(figsize=(16, 4))
    ax = plt.subplot(1, 11, 1)
    ax.axis('off')
    img_title = f'{query_id}\nquery\n{query_frames[query_index]}'
    if query_path is not None:
        imshow(query_path, img_title)
    else:
        img_title += '\nloose'
        imshow(get_cropped_image_path_loose(meta_csv, query_id), img_title)
    
    for i in range(10):
        ax = plt.subplot(1, 11, i + 2)
        ax.axis('off')
        img_path = get_cropped_image_path(meta_csv, int(gallery_ids[index[i]]), int(gallery_frames[index[i]]))
        img_title = f'{gallery_ids[index[i]]}\n{scores[index[i]]:.3f}\n{gallery_frames[index[i]]}'
        if img_path is not None:
            imshow(img_path, img_title)
        else:
            img_title += '\nloose'
            imshow(get_cropped_image_path_loose(meta_csv, int(gallery_ids[index[i]])), img_title)
    
    fig.savefig(f"./img_output/show_{query_id}.png")
    plt.close(fig)
