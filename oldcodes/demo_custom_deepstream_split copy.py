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
    # Load the CSV file
    df = pd.read_csv(metadata_csv)

    # Drop any leading/trailing spaces in column names
    df.columns = df.columns.str.strip()

    # Filter based on class_id and current_frame
    result = df[(df["class_id"] == class_id) & (df["current_frame"] == current_frame)]

    # If a match is found, construct the absolute path
    if not result.empty:
        relative_path = result.iloc[0]["image_cropped_obj_path_saved"]
        absolute_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(metadata_csv)), relative_path))
        return absolute_path

    return None
#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_id', default=290, type=int, help='test_image_index')
parser.add_argument('--output_dir',default='/home/big/Github/dataset_prepare/get_img_cosine_renew800/output',type=str, help='./img_data')
opts = parser.parse_args()

data_dir = opts.output_dir
#image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in ['gallery','query']}


result = scipy.io.loadmat('features_deep_renew_split800.mat')
meta_csv = os.path.join(data_dir,"cleaned_metadata800.csv")
placeholderpath = "placeholder.png"
query_feature = torch.FloatTensor(result['query_f'])
gallery_feature = torch.FloatTensor(result['gallery_f'])
query_ids = result['query_ids'][0]
query_frames = result['query_frames'][0]
gallery_ids = result['gallery_ids'][0]
gallery_frames = result['gallery_frames'][0]

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

#####################################################################
#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

######################################################################
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

query_index = int(np.where(query_ids == opts.query_id)[0][0])
# Check if query_index was found
if query_index == 0:
    print(f"Query index {opts.query_id} not found in the array.")
else:
    print(f"Query index {opts.query_id} found at position {query_index}.")
    index, scores = sort_img(query_feature[query_index], gallery_feature)
    for i in range(10):
        print(gallery_ids[index[i]])
        print(scores[index[i]])
    query_path = get_cropped_image_path(meta_csv,opts.query_id,int(query_frames[query_index]))
    print('Top 10 images are as follow:')
    try: # Visualize Ranking Result 
        # Graphical User Interface is needed
        fig = plt.figure(figsize=(16,4))
        ax = plt.subplot(1,11,1)
        ax.axis('off')
        imshow(query_path,'query')
        for i in range(5):
            ax = plt.subplot(1,11,i+2)
            ax.axis('off')
            img_path = get_cropped_image_path(meta_csv,int(gallery_ids[index[i]]),int(gallery_frames[index[i]]))
            # label = gallery_label[index[i]]
            print(img_path)
            img_title = f'{gallery_ids[index[i]]}\n{scores[index[i]]:.3f}\n{gallery_frames[index[i]]}'
            if img_path is not None:
                imshow(img_path, img_title)
            else:
                imshow(placeholderpath, img_title)
            # if label == query_label:
            #     ax.set_title('%d'%(i+1), color='green')
            # else:
            #     ax.set_title('%d'%(i+1), color='red')
    
    except RuntimeError:
        # for i in range(10):
        #     img_path = image_datasets.imgs[index[i]]
        #     print(img_path[0])
        print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

    fig.savefig("./img_output/show.png")

