# encoding: utf-8
"""
@author: vinklibrary
@contact: shenxvdong1@gmail.com
@site: Todo

@version: 1.0
@license: None
@file: basic_feature.py
@time: 2019/7/8 14:15

这一行开始写关于本文件的说明与解释
"""
import pandas as pd
import numpy as np

train_data = pd.read_csv("./dataset/raw_data/train.csv",na_values="\\N")
testt_data = pd.read_csv("./data/raw_date/test.csv")
user_info = pd.read_csv("./dataset/raw_data/user_info.csv")
listing_info = pd.read_csv("./dataset/raw_data/listing_info.csv")

# 加入用户基础特征
train_data = pd.merge(train_data,user_info,on='user_id',how='left')
train_data = train_data[train_data.auditing_date > train_data.insertdate]
filters = train_data.groupby(['user_id','listing_id','due_date'])['insertdate'].max().reset_index()
train_data = pd.merge(filters,train_data,on=['user_id','listing_id','due_date','insertdate'],how='left')

testt_data = pd.merge(testt_data, user_info, on='user_id', how='left')
testt_data = testt_data[testt_data.auditing_date > testt_data.insertdate]
filters = testt_data.groupby(['user_id','listing_id','due_date'])['insertdate'].max().reset_index()
testt_data = pd.merge(filters,testt_data,on=['user_id','listing_id','due_date','insertdate'],how='left')

# 加入标的基础特征
train_data = pd.merge(train_data,listing_info,on=['user_id','listing_id','auditing_date'],how='left')
testt_data = pd.merge(testt_data,listing_info,on=['user_id','listing_id','auditing_date'],how='left')

