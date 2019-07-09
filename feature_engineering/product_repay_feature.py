# encoding: utf-8
"""
@author: vinklibrary
@contact: shenxvdong1@gmail.com
@site: Todo

@version: 1.0
@license: None
@file: product_repay_feature.py
@time: 2019/7/9 12:40

这一行开始写关于本文件的说明与解释
"""
import pandas as pd
import numpy as np
from utils.functions import get_converation_rate
from functools import reduce

train_data = pd.read_csv("./train.csv",na_values="\\N")
testt_data = pd.read_csv("./test.csv")
train_data['flag'] = 'train'
testt_data['flag'] = 'testt'
data = pd.concat([train_data[['user_id','listing_id','auditing_date','due_date','flag']],testt_data[['user_id','listing_id','auditing_date','due_date','flag']]])

listing_info = pd.read_csv("../dataset/raw_date/listing_info.csv")
user_pay_logs_fe1 = pd.read_csv("./data/samples/user_pay_logs_fe1.csv")
user_pay_logs_fe1 = user_pay_logs_fe1.rename(columns={'listing_id_x':'listing_id'})
user_pay_logs_fe1 = pd.merge(user_pay_logs_fe1,listing_info,on=['user_id','listing_id','auditing_date'],how='left')

product_daoqi_in15days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore15].groupby(['term','rate','principal','auditing_date']).size()\
.reset_index().rename(columns={0:'product_daoqi_in15days'})
product_daoqi_in31days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore31].groupby(['term','rate','principal','auditing_date']).size()\
.reset_index().rename(columns={0:'product_daoqi_in31days'})
product_daoqi_in90days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore90].groupby(['term','rate','principal','auditing_date']).size()\
.reset_index().rename(columns={0:'product_daoqi_in90days'})
product_daoqi_in180days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore180].groupby(['term','rate','principal','auditing_date']).size()\
.reset_index().rename(columns={0:'product_daoqi_in180days'})

product_yuqi_in15days = user_pay_logs_fe1[(user_pay_logs_fe1.repay_date=='2200-01-01')&(user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore15)].groupby(['term','rate','principal','auditing_date']).size()\
.reset_index().rename(columns={0:'product_yuqi_in15days'})
product_yuqi_in31days = user_pay_logs_fe1[(user_pay_logs_fe1.repay_date=='2200-01-01')&(user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore31)].groupby(['term','rate','principal','auditing_date']).size()\
.reset_index().rename(columns={0:'product_yuqi_in31days'})
product_yuqi_in90days = user_pay_logs_fe1[(user_pay_logs_fe1.repay_date=='2200-01-01')&(user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore90)].groupby(['term','rate','principal','auditing_date']).size()\
.reset_index().rename(columns={0:'product_yuqi_in90days'})
product_yuqi_in180days = user_pay_logs_fe1[(user_pay_logs_fe1.repay_date=='2200-01-01')&(user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore180)].groupby(['term','rate','principal','auditing_date']).size()\
.reset_index().rename(columns={0:'product_yuqi_in180days'})

product_u_count_in15days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore15].groupby(['term','rate','principal','auditing_date'])['user_id'].nunique()\
.reset_index().rename(columns={'user_id':'product_u_count_in15days'})
product_u_count_in31days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore31].groupby(['term','rate','principal','auditing_date'])['user_id'].nunique()\
.reset_index().rename(columns={'user_id':'product_u_count_in31days'})
product_u_count_in90days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore90].groupby(['term','rate','principal','auditing_date'])['user_id'].nunique()\
.reset_index().rename(columns={'user_id':'product_u_count_in90days'})
product_u_count_in180days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore180].groupby(['term','rate','principal','auditing_date'])['user_id'].nunique()\
.reset_index().rename(columns={'user_id':'product_u_count_in180days'})

product_days_in15days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore15].groupby(['term','rate','principal','auditing_date'])['due_date'].nunique()\
.reset_index().rename(columns={'due_date':'product_days_in15days'})
product_days_in31days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore31].groupby(['term','rate','principal','auditing_date'])['due_date'].nunique()\
.reset_index().rename(columns={'due_date':'product_days_in31days'})
product_days_in90days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore90].groupby(['term','rate','principal','auditing_date'])['due_date'].nunique()\
.reset_index().rename(columns={'due_date':'product_days_in90days'})
product_days_in180days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore180].groupby(['term','rate','principal','auditing_date'])['due_date'].nunique()\
.reset_index().rename(columns={'due_date':'product_days_in180days'})

product_repay_features1 = reduce(lambda x, y: pd.merge(x, y, on=['term','rate','principal','auditing_date'], how='outer'),
                                   [product_daoqi_in15days,product_daoqi_in31days,product_daoqi_in90days,product_daoqi_in180days,
product_yuqi_in15days,product_yuqi_in31days,product_yuqi_in90days,product_yuqi_in180days,
product_u_count_in15days,product_u_count_in31days,product_u_count_in90days,product_u_count_in180days,
product_days_in15days,product_days_in31days,product_days_in90days,product_days_in180days])
product_repay_features1 = product_repay_features1.fillna(0)

# 逾期率的计算 (轻微过拟合 获得训练集的转化率可防止)
product_repay_features1['product_yuqi_rate_in15days'] = get_converation_rate(product_repay_features1,'product_yuqi_in15days','product_daoqi_in15days')[0]
product_repay_features1['product_yuqi_rate_in31days'] = get_converation_rate(product_repay_features1,'product_yuqi_in31days','product_daoqi_in31days')[0]
product_repay_features1['product_yuqi_rate_in90days'] = get_converation_rate(product_repay_features1,'product_yuqi_in90days','product_daoqi_in90days')[0]
product_repay_features1['product_yuqi_rate_in180days'] = get_converation_rate(product_repay_features1,'product_yuqi_in180days','product_daoqi_in180days')[0]
product_repay_features1.to_csv("../dataset/gen_features/product_repay_features1.csv", index=None)

user_pay_logs_fe2 = pd.read_csv("./data/samples/user_pay_logs_fe2.csv")
user_pay_logs_fe2 = user_pay_logs_fe2.rename(columns={'listing_id_x':'listing_id'})
user_pay_logs_fe2 = pd.merge(user_pay_logs_fe2,listing_info,on=['user_id','listing_id','auditing_date'],how='left')

# 该产品 历史上首期的平均还款时间
user_pay_logs_fe2_order1 = user_pay_logs_fe2[user_pay_logs_fe2.order_id==1]

product_repay1_backward_mean_in15days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore15].groupby(['term','rate','principal','auditing_date'])['backward_days'].mean()\
    .reset_index().rename(columns={'backward_days':'product_repay1_backward_mean_in15days'})
produce_repay1_backward_mean_in31days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore31].groupby(['term','rate','principal','auditing_date'])['backward_days'].mean()\
    .reset_index().rename(columns={'backward_days':'produce_repay1_backward_mean_in31days'})
product_repay1_backward_mean_in90days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore90].groupby(['term','rate','principal','auditing_date'])['backward_days'].mean()\
    .reset_index().rename(columns={'backward_days':'product_repay1_backward_mean_in90days'})
product_repay1_backward_mean_in180days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore180].groupby(['term','rate','principal','auditing_date'])['backward_days'].mean()\
    .reset_index().rename(columns={'backward_days':'product_repay1_backward_mean_in180days'})

product_repay1_backward_median_in15days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore15].groupby(['term','rate','principal','auditing_date'])['backward_days'].median()\
    .reset_index().rename(columns={'backward_days':'product_repay1_backward_median_in15days'})
produce_repay1_backward_median_in31days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore31].groupby(['term','rate','principal','auditing_date'])['backward_days'].median()\
    .reset_index().rename(columns={'backward_days':'produce_repay1_backward_median_in31days'})
product_repay1_backward_median_in90days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore90].groupby(['term','rate','principal','auditing_date'])['backward_days'].median()\
    .reset_index().rename(columns={'backward_days':'product_repay1_backward_median_in90days'})
product_repay1_backward_median_in180days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore180].groupby(['term','rate','principal','auditing_date'])['backward_days'].median()\
    .reset_index().rename(columns={'backward_days':'product_repay1_backward_median_in180days'})

product_repay1_backward_max_in15days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore15].groupby(['term','rate','principal','auditing_date'])['backward_days'].max()\
    .reset_index().rename(columns={'backward_days':'product_repay1_backward_max_in15days'})
produce_repay1_backward_max_in31days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore31].groupby(['term','rate','principal','auditing_date'])['backward_days'].max()\
    .reset_index().rename(columns={'backward_days':'produce_repay1_backward_max_in31days'})
product_repay1_backward_max_in90days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore90].groupby(['term','rate','principal','auditing_date'])['backward_days'].max()\
    .reset_index().rename(columns={'backward_days':'product_repay1_backward_max_in90days'})
product_repay1_backward_max_in180days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore180].groupby(['term','rate','principal','auditing_date'])['backward_days'].max()\
    .reset_index().rename(columns={'backward_days':'product_repay1_backward_max_in180days'})

product_repay1_backward_min_in15days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore15].groupby(['term','rate','principal','auditing_date'])['backward_days'].min()\
    .reset_index().rename(columns={'backward_days':'product_repay1_backward_min_in15days'})
produce_repay1_backward_min_in31days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore31].groupby(['term','rate','principal','auditing_date'])['backward_days'].min()\
    .reset_index().rename(columns={'backward_days':'produce_repay1_backward_min_in31days'})
product_repay1_backward_min_in90days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore90].groupby(['term','rate','principal','auditing_date'])['backward_days'].min()\
    .reset_index().rename(columns={'backward_days':'product_repay1_backward_min_in90days'})
product_repay1_backward_min_in180days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore180].groupby(['term','rate','principal','auditing_date'])['backward_days'].min()\
    .reset_index().rename(columns={'backward_days':'product_repay1_backward_min_in180days'})

product_repay1_backward_std_in15days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore15].groupby(['term','rate','principal','auditing_date'])['backward_days'].std()\
    .reset_index().rename(columns={'backward_days':'product_repay1_backward_std_in15days'})
produce_repay1_backward_std_in31days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore31].groupby(['term','rate','principal','auditing_date'])['backward_days'].std()\
    .reset_index().rename(columns={'backward_days':'produce_repay1_backward_std_in31days'})
product_repay1_backward_std_in90days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore90].groupby(['term','rate','principal','auditing_date'])['backward_days'].std()\
    .reset_index().rename(columns={'backward_days':'product_repay1_backward_std_in90days'})
product_repay1_backward_std_in180days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore180].groupby(['term','rate','principal','auditing_date'])['backward_days'].std()\
    .reset_index().rename(columns={'backward_days':'product_repay1_backward_std_in180days'})

# 该产品 历史上首期的平均还款时间
# user_pay_logs_fe2_order1 = user_pay_logs_fe2[user_pay_logs_fe2.order_id==1]
user_pay_logs_fe2_order2 = user_pay_logs_fe2[user_pay_logs_fe2.backward_days<=31]

product_repay2_backward_mean_in15days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore15].groupby(['term','rate','principal','auditing_date'])['backward_days'].mean()\
    .reset_index().rename(columns={'backward_days':'product_repay2_backward_mean_in15days'})
produce_repay2_backward_mean_in31days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore31].groupby(['term','rate','principal','auditing_date'])['backward_days'].mean()\
    .reset_index().rename(columns={'backward_days':'produce_repay2_backward_mean_in31days'})
product_repay2_backward_mean_in90days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore90].groupby(['term','rate','principal','auditing_date'])['backward_days'].mean()\
    .reset_index().rename(columns={'backward_days':'product_repay2_backward_mean_in90days'})
product_repay2_backward_mean_in180days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore180].groupby(['term','rate','principal','auditing_date'])['backward_days'].mean()\
    .reset_index().rename(columns={'backward_days':'product_repay2_backward_mean_in180days'})

product_repay2_backward_median_in15days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore15].groupby(['term','rate','principal','auditing_date'])['backward_days'].median()\
    .reset_index().rename(columns={'backward_days':'product_repay2_backward_median_in15days'})
produce_repay2_backward_median_in31days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore31].groupby(['term','rate','principal','auditing_date'])['backward_days'].median()\
    .reset_index().rename(columns={'backward_days':'produce_repay2_backward_median_in31days'})
product_repay2_backward_median_in90days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore90].groupby(['term','rate','principal','auditing_date'])['backward_days'].median()\
    .reset_index().rename(columns={'backward_days':'product_repay2_backward_median_in90days'})
product_repay2_backward_median_in180days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore180].groupby(['term','rate','principal','auditing_date'])['backward_days'].median()\
    .reset_index().rename(columns={'backward_days':'product_repay2_backward_median_in180days'})

product_repay2_backward_max_in15days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore15].groupby(['term','rate','principal','auditing_date'])['backward_days'].max()\
    .reset_index().rename(columns={'backward_days':'product_repay2_backward_max_in15days'})
produce_repay2_backward_max_in31days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore31].groupby(['term','rate','principal','auditing_date'])['backward_days'].max()\
    .reset_index().rename(columns={'backward_days':'produce_repay2_backward_max_in31days'})
product_repay2_backward_max_in90days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore90].groupby(['term','rate','principal','auditing_date'])['backward_days'].max()\
    .reset_index().rename(columns={'backward_days':'product_repay2_backward_max_in90days'})
product_repay2_backward_max_in180days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore180].groupby(['term','rate','principal','auditing_date'])['backward_days'].max()\
    .reset_index().rename(columns={'backward_days':'product_repay2_backward_max_in180days'})

product_repay2_backward_min_in15days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore15].groupby(['term','rate','principal','auditing_date'])['backward_days'].min()\
    .reset_index().rename(columns={'backward_days':'product_repay2_backward_min_in15days'})
produce_repay2_backward_min_in31days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore31].groupby(['term','rate','principal','auditing_date'])['backward_days'].min()\
    .reset_index().rename(columns={'backward_days':'produce_repay2_backward_min_in31days'})
product_repay2_backward_min_in90days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore90].groupby(['term','rate','principal','auditing_date'])['backward_days'].min()\
    .reset_index().rename(columns={'backward_days':'product_repay2_backward_min_in90days'})
product_repay2_backward_min_in180days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore180].groupby(['term','rate','principal','auditing_date'])['backward_days'].min()\
    .reset_index().rename(columns={'backward_days':'product_repay2_backward_min_in180days'})

product_repay2_backward_std_in15days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore15].groupby(['term','rate','principal','auditing_date'])['backward_days'].std()\
    .reset_index().rename(columns={'backward_days':'product_repay2_backward_std_in15days'})
produce_repay2_backward_std_in31days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore31].groupby(['term','rate','principal','auditing_date'])['backward_days'].std()\
    .reset_index().rename(columns={'backward_days':'produce_repay2_backward_std_in31days'})
product_repay2_backward_std_in90days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore90].groupby(['term','rate','principal','auditing_date'])['backward_days'].std()\
    .reset_index().rename(columns={'backward_days':'product_repay2_backward_std_in90days'})
product_repay2_backward_std_in180days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore180].groupby(['term','rate','principal','auditing_date'])['backward_days'].std()\
    .reset_index().rename(columns={'backward_days':'product_repay2_backward_std_in180days'})

# 该商品提前还款的单数
user_pay_logs_fe2_before = user_pay_logs_fe2[user_pay_logs_fe2.backward_days>31]

product_repay_before_in15days = user_pay_logs_fe2_before[user_pay_logs_fe2_before.repay_date>user_pay_logs_fe2_before.daybefore15].groupby(['term','rate','principal','auditing_date']).size()\
    .reset_index().rename(columns={0:'product_repay_before_in15days'})
product_repay_before_in31days = user_pay_logs_fe2_before[user_pay_logs_fe2_before.repay_date>user_pay_logs_fe2_before.daybefore31].groupby(['term','rate','principal','auditing_date']).size()\
    .reset_index().rename(columns={0:'product_repay_before_in31days'})
product_repay_before_in90days = user_pay_logs_fe2_before[user_pay_logs_fe2_before.repay_date>user_pay_logs_fe2_before.daybefore90].groupby(['term','rate','principal','auditing_date']).size()\
    .reset_index().rename(columns={0:'product_repay_before_in90days'})
product_repay_before_in180days = user_pay_logs_fe2_before[user_pay_logs_fe2_before.repay_date>user_pay_logs_fe2_before.daybefore180].groupby(['term','rate','principal','auditing_date']).size()\
    .reset_index().rename(columns={0:'product_repay_before_in180days'})

product_repay_by_dow = user_pay_logs_fe2.groupby(['term','rate','principal','auditing_date','repay_dow']).size().reset_index().rename(columns={0:'product_repay_by_dow'})
product_repay_by_dow = product_repay_by_dow.pivot_table(index=['term','rate','principal','auditing_date'],columns='repay_dow',values='product_repay_by_dow').reset_index()
product_repay_by_dow.columns = ['term','rate','principal','auditing_date','p_repay_dow1','p_repay_dow2','p_repay_dow3','p_repay_dow4','p_repay_dow5','p_repay_dow6','p_repay_dow7']
product_repay_counts = user_pay_logs_fe2.groupby(['term','rate','principal','auditing_date']).size().reset_index().rename(columns={0:'product_repay_counts'})
product_repay_distribution = pd.merge(product_repay_by_dow,product_repay_counts,on=['term','rate','principal','auditing_date'],how='left')
product_repay_distribution = product_repay_distribution.fillna(0)
product_repay_distribution['p_repay_dow1_rate'] = get_converation_rate(product_repay_distribution,'p_repay_dow1','product_repay_counts')[0]
product_repay_distribution['p_repay_dow2_rate'] = get_converation_rate(product_repay_distribution,'p_repay_dow2','product_repay_counts')[0]
product_repay_distribution['p_repay_dow3_rate'] = get_converation_rate(product_repay_distribution,'p_repay_dow3','product_repay_counts')[0]
product_repay_distribution['p_repay_dow4_rate'] = get_converation_rate(product_repay_distribution,'p_repay_dow4','product_repay_counts')[0]
product_repay_distribution['p_repay_dow5_rate'] = get_converation_rate(product_repay_distribution,'p_repay_dow5','product_repay_counts')[0]
product_repay_distribution['p_repay_dow6_rate'] = get_converation_rate(product_repay_distribution,'p_repay_dow6','product_repay_counts')[0]
product_repay_distribution['p_repay_dow7_rate'] = get_converation_rate(product_repay_distribution,'p_repay_dow7','product_repay_counts')[0]

product_repay_features2 = reduce(lambda x, y: pd.merge(x, y, on=['term','rate','principal','auditing_date'], how='outer'),
                                   [product_repay1_backward_mean_in15days,produce_repay1_backward_mean_in31days,product_repay1_backward_mean_in90days,product_repay1_backward_mean_in180days,
product_repay1_backward_median_in15days,produce_repay1_backward_median_in31days,product_repay1_backward_median_in90days,product_repay1_backward_median_in180days,
product_repay1_backward_max_in15days,produce_repay1_backward_max_in31days,product_repay1_backward_max_in90days,product_repay1_backward_max_in180days,
product_repay1_backward_min_in15days,produce_repay1_backward_min_in31days,product_repay1_backward_min_in90days,product_repay1_backward_min_in180days,
product_repay1_backward_std_in15days,produce_repay1_backward_std_in31days,product_repay1_backward_std_in90days,product_repay1_backward_std_in180days,
product_repay2_backward_mean_in15days,produce_repay2_backward_mean_in31days,product_repay2_backward_mean_in90days,product_repay2_backward_mean_in180days,
product_repay2_backward_median_in15days,produce_repay2_backward_median_in31days,product_repay2_backward_median_in90days,product_repay2_backward_median_in180days,
product_repay2_backward_max_in15days,produce_repay2_backward_max_in31days,product_repay2_backward_max_in90days,product_repay2_backward_max_in180days,
product_repay2_backward_min_in15days,produce_repay2_backward_min_in31days,product_repay2_backward_min_in90days,product_repay2_backward_min_in180days,
product_repay2_backward_std_in15days,produce_repay2_backward_std_in31days,product_repay2_backward_std_in90days,product_repay2_backward_std_in180days,
product_repay_before_in15days,product_repay_before_in31days,product_repay_before_in90days,product_repay_before_in180days,
product_repay_distribution])
product_repay_features2 = product_repay_features2.fillna(0)
product_repay_features2.to_csv("../dataset/gen_features/product_repay_features2.csv",index=None)