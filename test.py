import torch
import json
import joblib
import pickle
import torch.utils.data as data_utils
import numpy as np
import scipy.sparse as sp
import pandas as pd
from neg_sampler import *
from pathlib import Path
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from model.ctr.inputs import *

# df = pd.read_csv('/root/autodl-fs/Tenrec/ctr_data_1M.csv', engine='python', usecols=["user_id", "item_id", "click", "like", "video_category", "gender", "age", "hist_1", "hist_2",
#                    "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"])

# print(df.head())
# print(df.shape)





chunks = pd.read_csv('/root/autodl-fs/Tenrec/cold_data_1.csv', chunksize=100, engine='python' )
print('chunk done')
list = []
columns=["user_id", "item_id", "click", "like", "video_category", "gender", "age", "hist_1", "hist_2",
                       "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10", "hist_11", "hist_12", "hist_13"]
for chunk in chunks:   
    # print('chunk+1')     
    # concat
    list = list + chunk.values.tolist()  
    # df.append(chunk)
    # print('chunk+1done')
    # print(len(df))
    # print(chunk)
# print(df.head())
# print(df.shape)
df = pd.DataFrame(list, columns=columns)
print(df.head())
print(len(df))





# chunks = pd.read_csv('/root/autodl-fs/Tenrec/cold_data_1.csv', chunksize=100, engine='python' )
# print('chunk done')
# df = pd.DataFrame(columns=["user_id", "item_id", "click", "like", "video_category", "gender", "age", "hist_1", "hist_2",
#                        "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10", "hist_11", "hist_12", "hist_13"]
#                 )
# for chunk in chunks:   
#     # print('chunk+1')
#     # concat
#     df = pd.concat([df, chunk], ignore_index=True)        
#     # print('chunk+1done')
#     # print(len(df))
#     # print(chunk)
# # print(df.head())
# # print(df.shape)
# print(len(df))
# print(len(df[0]))
