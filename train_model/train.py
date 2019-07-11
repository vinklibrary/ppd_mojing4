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
from utils.functions import Caltime, get_date_before
from functools import reduce
import tqdm
import lightgbm as lgb
import math

train_data = pd.read_csv("../data/train.csv")
testt_data = pd.read_csv("../data/test.csv")
train_data['repay_amt'] = train_data.repay_amt.map(lambda x: np.nan if x == "\\N" else float(x))
train_data['is_overdue'] = train_data.repay_amt.map(lambda x:1 if np.isnan(x) else 0)
train_data['backward_days'] = train_data.apply(lambda x: Caltime(x['repay_date'],x['due_date']), axis=1)
train_data['forward_days'] = train_data.apply(lambda x: Caltime(x['auditing_date'],x['repay_date']), axis=1)
train_data['backward_days'] = train_data.backward_days.map(lambda x:32 if x==-1 else x)
train_data['forward_days'] = train_data.forward_days.map(lambda x:32 if x==-1 else x)

# 读取特征
basic_feature = pd.read_csv("../data/features/basic_features.csv")
user_behavier_features = pd.read_csv("../data/features/user_behavier_features.csv")
user_repay_features1 = pd.read_csv("../data/features/user_repay_features1.csv")
user_repay_features2 = pd.read_csv("../data/features/user_repay_features2.csv")
user_repay_features3 = pd.read_csv("../data/features/user_repay_features3.csv")

#加入基础特征
train_data = pd.merge(train_data,basic_feature,on=['user_id','listing_id','auditing_date','due_date'])
testt_data = pd.merge(testt_data,basic_feature,on=['user_id','listing_id','auditing_date','due_date'])
train_data = pd.merge(train_data,user_repay_features1,on=['user_id','auditing_date'],how='left')
testt_data = pd.merge(testt_data,user_repay_features1,on=['user_id','auditing_date'],how='left')
train_data = pd.merge(train_data,user_repay_features2,on=['user_id','auditing_date'],how='left')
testt_data = pd.merge(testt_data,user_repay_features2,on=['user_id','auditing_date'],how='left')
train_data = pd.merge(train_data,user_repay_features3,on=['user_id','auditing_date'],how='left')
testt_data = pd.merge(testt_data,user_repay_features3,on=['user_id','auditing_date'],how='left')
train_data = pd.merge(train_data,user_behavier_features,on=['user_id','listing_id','auditing_date'],how='left')
testt_data = pd.merge(testt_data,user_behavier_features,on=['user_id','listing_id','auditing_date'],how='left')

user_taglist = pd.read_csv("../samples/features/user_taglist_fea_v1.csv")
train_data = pd.merge(train_data,user_taglist,on=['user_id'],how='left')
testt_data = pd.merge(testt_data,user_taglist,on=['user_id'],how='left')

product_repay_features1 = pd.read_csv("../data/features/product_repay_features1.csv")
product_repay_features2 = pd.read_csv("../data/features/product_repay_features2.csv")
train_data = pd.merge(train_data,product_repay_features1,on=['term','rate','principal','auditing_date'],how='left')
testt_data = pd.merge(testt_data,product_repay_features1,on=['term','rate','principal','auditing_date'],how='left')
train_data = pd.merge(train_data,product_repay_features2,on=['term','rate','principal','auditing_date'],how='left')
testt_data = pd.merge(testt_data,product_repay_features2,on=['term','rate','principal','auditing_date'],how='left')

train_data['user_yuqi_rate_in15days'] = train_data['user_yuqi_in15days']/(train_data['user_daoqi_in15days']+1)
train_data['user_yuqi_rate_in31days'] = train_data['user_yuqi_in31days']/(train_data['user_daoqi_in31days']+1)
train_data['user_yuqi_rate_in90days'] = train_data['user_yuqi_in90days']/(train_data['user_daoqi_in90days']+1)
train_data['user_yuqi_rate_in180days'] = train_data['user_yuqi_in180days']/(train_data['user_daoqi_in180days']+1)
testt_data['user_yuqi_rate_in15days'] = testt_data['user_yuqi_in15days']/(testt_data['user_daoqi_in15days']+1)
testt_data['user_yuqi_rate_in31days'] = testt_data['user_yuqi_in31days']/(testt_data['user_daoqi_in31days']+1)
testt_data['user_yuqi_rate_in90days'] = testt_data['user_yuqi_in90days']/(testt_data['user_daoqi_in90days']+1)
testt_data['user_yuqi_rate_in180days'] = testt_data['user_yuqi_in180days']/(testt_data['user_daoqi_in180days']+1)

train_data['due_date_diff_influce'] = (train_data['farthest_due_date_diff']-train_data['latest_due_date_diff'])/(train_data['farthest_due_date_diff']+train_data['latest_due_date_diff'])
testt_data['due_date_diff_influce'] = (testt_data['farthest_due_date_diff']-testt_data['latest_due_date_diff'])/(testt_data['farthest_due_date_diff']+testt_data['latest_due_date_diff'])
train_data['user_repay_before_rate_in15days'] = train_data['user_repay_before_in15days']/(train_data['user_repay_in15days']+1)
train_data['user_repay_before_rate_in31days'] = train_data['user_repay_before_in31days']/(train_data['user_repay_in31days']+1)
train_data['user_repay_before_rate_in90days'] = train_data['user_repay_before_in90days']/(train_data['user_repay_in90days']+1)
train_data['user_repay_before_rate_in180days'] = train_data['user_repay_before_in180days']/(train_data['user_repay_in180days']+1)
testt_data['user_repay_before_rate_in15days'] = testt_data['user_repay_before_in15days']/(testt_data['user_repay_in15days']+1)
testt_data['user_repay_before_rate_in31days'] = testt_data['user_repay_before_in31days']/(testt_data['user_repay_in31days']+1)
testt_data['user_repay_before_rate_in90days'] = testt_data['user_repay_before_in90days']/(testt_data['user_repay_in90days']+1)
testt_data['user_repay_before_rate_in180days'] = testt_data['user_repay_before_in180days']/(testt_data['user_repay_in180days']+1)

#User Repay Features2
features = ['user_repay_in15days','user_repay_in31days','user_repay_in90days','user_repay_in180days','user_repay_amt_in15days','user_repay_amt_in31days','user_repay_amt_in90days','user_repay_amt_in180days','user_repay1_backward_mean_in15days','user_repay1_backward_mean_in31days','user_repay1_backward_mean_in90days','user_repay1_backward_mean_in180days','user_repay1_backward_median_in15days','user_repay1_backward_median_in31days','user_repay1_backward_median_in90days','user_repay1_backward_median_in180days','user_repay1_backward_max_in15days','user_repay1_backward_max_in31days','user_repay1_backward_max_in90days','user_repay1_backward_max_in180days','user_repay1_backward_min_in15days','user_repay1_backward_min_in31days','user_repay1_backward_min_in90days','user_repay1_backward_min_in180days','user_repay1_backward_std_in15days','user_repay1_backward_std_in31days','user_repay1_backward_std_in90days','user_repay1_backward_std_in180days','user_repay2_backward_mean_in15days','user_repay2_backward_mean_in31days','user_repay2_backward_mean_in90days','user_repay2_backward_mean_in180days','user_repay2_backward_median_in15days','user_repay2_backward_median_in31days','user_repay2_backward_median_in90days','user_repay2_backward_median_in180days','user_repay2_backward_max_in15days','user_repay2_backward_max_in31days','user_repay2_backward_max_in90days','user_repay2_backward_max_in180days','user_repay2_backward_min_in15days','user_repay2_backward_min_in31days','user_repay2_backward_min_in90days','user_repay2_backward_min_in180days','user_repay2_backward_std_in15days','user_repay2_backward_std_in31days','user_repay2_backward_std_in90days','user_repay2_backward_std_in180days','user_repay_before_in15days','user_repay_before_in31days','user_repay_before_in90days','user_repay_before_in180days','user_repay_amt_before_in15days','user_repay_amt_before_in31days','user_repay_amt_before_in90days','user_repay_amt_before_in180days','latest_repay_date_diff','farthest_repay_date_diff','user_repay_counts','repay_dow1rate','repay_dow2rate','repay_dow3rate','repay_dow4rate','repay_dow5rate','repay_dow6rate','repay_dow7rate','mon_code0rate','mon_code1rate','mon_code2rate','repay_date_numeric_diff_mean','repay_date_numeric_diff_min','repay_date_numeric_diff_max','repay_date_numeric_diff_median','repay_date_numeric_diff_std']
# User Basic Features
features.extend(['gender','age','rate','principal','reg_mon_diff','cell_same_id','rate65','rate69','rate72','rate76','rate80','rate83','rate86','term3','term6','term9','term12','backward_days_max','due_dow','auditing_dow','due_dom','auditing_dom','due_dom_reverse','auditing_dom_reverse'])
# User Repay Features 1
features.extend(['user_daoqi_in15days','user_daoqi_in31days','user_daoqi_in90days','user_daoqi_in180days','user_daoqi_amt_in15days','user_daoqi_amt_in31days','user_daoqi_amt_in90days','user_daoqi_amt_in180days','user_yuqi_in15days','user_yuqi_in31days','user_yuqi_in90days','user_yuqi_in180days','due_amt_x','due_amt_y','due_amt_x.1','due_amt_y.1','latest_due_date_diff','farthest_due_date_diff','due_date_diff_mean_order1','due_date_diff_median_order1','due_date_diff_min_order1','due_date_diff_max_order1','due_date_diff_std_order1','due_date_diff_mean','due_date_diff_median','due_date_diff_min','due_date_diff_max','due_date_diff_std'])
# User Behavier Features
features.extend(['user_behavier1_days_in_6month','user_behavier1_days_in_3month','user_behavier1_days_in_1month','user_behavier1_in_6month','user_behavier1_in_3month','user_behavier1_in_1month','user_behavier1_latest_date_diff','user_behavier2_days_in_6month','user_behavier2_days_in_3month','user_behavier2_days_in_1month','user_behavier2_in_6month','user_behavier2_in_3month','user_behavier2_in_1month','user_behavier2_latest_date_diff','user_behavier3_days_in_6month','user_behavier3_days_in_3month','user_behavier3_days_in_1month','user_behavier3_in_history','user_behavier3_in_6month','user_behavier3_in_3month','user_behavier3_in_1month','user_behavier3_latest_date_diff','user_behavier_counts_x','hour00rate','hour01rate','hour02rate','hour03rate','hour04rate','hour05rate','hour06rate','hour07rate','hour08rate','hour09rate','hour10rate','hour11rate','hour12rate','hour13rate','hour14rate','hour15rate','hour16rate','hour17rate','hour18rate','hour19rate','hour20rate','hour21rate','hour22rate','hour23rate','hourcode0','hourcode0rate','hourcode1rate','hourcode2rate','hourcode3rate'])
# User taglist Featuresr
features.extend(['taglist_svd0', 'taglist_svd1', 'taglist_svd2', 'taglist_svd3', 'taglist_svd4', 'taglist_svd5', 'taglist_svd6','taglist_svd7', 'taglist_svd8', 'taglist_svd9'])
# User Location Features
features.extend(['id_city_counts','mean_backward_days_of_id_city','mean_due_amt_of_id_city','mean_term_of_id_city','mean_is_overdue_id_city','mean_reg_mon_diff_overdue_id_city','mean_age_diff_overdue_id_city','id_province_counts','mean_backward_days_of_id_province','mean_due_amt_of_id_province','mean_term_of_id_province','mean_is_overdue_id_province','mean_reg_mon_diff_overdue_id_province','mean_age_diff_overdue_id_province','cell_province_counts','mean_backward_days_of_cell_province','mean_due_amt_of_cell_province','mean_term_of_cell_province','mean_is_overdue_cell_province','mean_reg_mon_diff_overdue_cell_province','mean_age_diff_overdue_cell_province'])
# Product Features 1
features.extend(['product_daoqi_in15days','product_daoqi_in31days','product_daoqi_in90days','product_daoqi_in180days','product_yuqi_in15days','product_yuqi_in31days','product_yuqi_in90days','product_yuqi_in180days','product_u_count_in15days','product_u_count_in31days','product_u_count_in90days','product_u_count_in180days','product_days_in15days','product_days_in31days','product_days_in90days','product_days_in180days','product_yuqi_rate_in15days','product_yuqi_rate_in31days','product_yuqi_rate_in90days','product_yuqi_rate_in180days'])
# Product Features 2
features.extend(['product_repay1_backward_mean_in15days','produce_repay1_backward_mean_in31days','product_repay1_backward_mean_in90days','product_repay1_backward_mean_in180days','product_repay1_backward_median_in15days','produce_repay1_backward_median_in31days','product_repay1_backward_median_in90days','product_repay1_backward_median_in180days','product_repay1_backward_max_in15days','produce_repay1_backward_max_in31days','product_repay1_backward_max_in90days','product_repay1_backward_max_in180days','product_repay1_backward_min_in15days','produce_repay1_backward_min_in31days','product_repay1_backward_min_in90days','product_repay1_backward_min_in180days','product_repay1_backward_std_in15days','produce_repay1_backward_std_in31days','product_repay1_backward_std_in90days','product_repay1_backward_std_in180days','product_repay2_backward_mean_in15days','produce_repay2_backward_mean_in31days','product_repay2_backward_mean_in90days','product_repay2_backward_mean_in180days','product_repay2_backward_median_in15days','produce_repay2_backward_median_in31days','product_repay2_backward_median_in90days','product_repay2_backward_median_in180days','product_repay2_backward_max_in15days','produce_repay2_backward_max_in31days','product_repay2_backward_max_in90days','product_repay2_backward_max_in180days','product_repay2_backward_min_in15days','produce_repay2_backward_min_in31days','product_repay2_backward_min_in90days','product_repay2_backward_min_in180days','product_repay2_backward_std_in15days','produce_repay2_backward_std_in31days','product_repay2_backward_std_in90days','product_repay2_backward_std_in180days','product_repay_before_in15days','product_repay_before_in31days','product_repay_before_in90days','product_repay_before_in180days','p_repay_dow1','p_repay_dow2','p_repay_dow3','p_repay_dow4','p_repay_dow5','p_repay_dow6','p_repay_dow7','product_repay_counts','p_repay_dow1_rate','p_repay_dow2_rate','p_repay_dow3_rate','p_repay_dow4_rate','p_repay_dow5_rate','p_repay_dow6_rate','p_repay_dow7_rate'])
# expand
features.extend(['user_yuqi_rate_in15days','user_yuqi_rate_in31days','user_yuqi_rate_in90days','user_yuqi_rate_in180days','due_date_diff_influce','user_repay_before_rate_in15days','user_repay_before_rate_in31days','user_repay_before_rate_in90days','user_repay_before_rate_in180days'])

# 给样本设置权重
train_data['weights'] = train_data.backward_days.map(lambda x:0.9 if x == 32 else 1)

train_sample = train_data[train_data.auditing_date<'2018-10-01']
testt_sample = train_data[train_data.auditing_date>='2018-11-01']

# 加入放穿越统计特征
cell_province_counts = train_sample.groupby('cell_province').size().reset_index().rename(columns = {0:'cell_province_counts'})
mean_backward_days_of_cell_province = train_sample[train_sample.is_overdue==0].groupby('cell_province')['backward_days'].mean().reset_index().rename(columns={'backward_days':'mean_backward_days_of_cell_province'})
mean_due_amt_of_cell_province = train_sample.groupby('cell_province')['due_amt'].mean().reset_index().rename(columns={'due_amt':'mean_due_amt_of_cell_province'})
mean_term_of_cell_province = train_sample.groupby('cell_province')['term'].mean().reset_index().rename(columns={'term':'mean_term_of_cell_province'})
mean_is_overdue_cell_province = train_sample.groupby('cell_province')['is_overdue'].mean().reset_index().rename(columns={'is_overdue':'mean_is_overdue_cell_province'})
mean_regmon_cell_province = train_sample.groupby('cell_province')['reg_mon_diff'].mean().reset_index().rename(columns={'reg_mon_diff':'mean_reg_mon_diff_overdue_cell_province'})
mean_age_cell_province = train_sample.groupby('cell_province')['age'].mean().reset_index().rename(columns={'age':'mean_age_diff_overdue_cell_province'})
cell_province_feature = reduce(lambda x, y: pd.merge(x, y, on='cell_province', how='outer'),
                                   [cell_province_counts,mean_backward_days_of_cell_province,mean_due_amt_of_cell_province,mean_term_of_cell_province,\
                                   mean_is_overdue_cell_province,mean_regmon_cell_province,mean_age_cell_province])
# 加入放穿越统计特征
id_province_counts = train_sample.groupby('id_province').size().reset_index().rename(columns = {0:'id_province_counts'})
mean_backward_days_of_id_province = train_sample[train_sample.is_overdue==0].groupby('id_province')['backward_days'].mean().reset_index().rename(columns={'backward_days':'mean_backward_days_of_id_province'})
mean_due_amt_of_id_province = train_sample.groupby('id_province')['due_amt'].mean().reset_index().rename(columns={'due_amt':'mean_due_amt_of_id_province'})
mean_term_of_id_province = train_sample.groupby('id_province')['term'].mean().reset_index().rename(columns={'term':'mean_term_of_id_province'})
mean_is_overdue_id_province = train_sample.groupby('id_province')['is_overdue'].mean().reset_index().rename(columns={'is_overdue':'mean_is_overdue_id_province'})
mean_regmon_id_province = train_sample.groupby('id_province')['reg_mon_diff'].mean().reset_index().rename(columns={'reg_mon_diff':'mean_reg_mon_diff_overdue_id_province'})
mean_age_id_province = train_sample.groupby('id_province')['age'].mean().reset_index().rename(columns={'age':'mean_age_diff_overdue_id_province'})
id_province_feature = reduce(lambda x, y: pd.merge(x, y, on='id_province', how='outer'),
                                   [id_province_counts,mean_backward_days_of_id_province,mean_due_amt_of_id_province,mean_term_of_id_province,\
                                   mean_is_overdue_id_province,mean_regmon_id_province,mean_age_id_province])
# 加入放穿越统计特征
id_city_counts = train_sample.groupby('id_city').size().reset_index().rename(columns = {0:'id_city_counts'})
mean_backward_days_of_id_city = train_sample[train_sample.is_overdue==0].groupby('id_city')['backward_days'].mean().reset_index().rename(columns={'backward_days':'mean_backward_days_of_id_city'})
mean_due_amt_of_id_city = train_sample.groupby('id_city')['due_amt'].mean().reset_index().rename(columns={'due_amt':'mean_due_amt_of_id_city'})
mean_term_of_id_city = train_sample.groupby('id_city')['term'].mean().reset_index().rename(columns={'term':'mean_term_of_id_city'})
mean_is_overdue_id_city = train_sample.groupby('id_city')['is_overdue'].mean().reset_index().rename(columns={'is_overdue':'mean_is_overdue_id_city'})
mean_regmon_id_city = train_sample.groupby('id_city')['reg_mon_diff'].mean().reset_index().rename(columns={'reg_mon_diff':'mean_reg_mon_diff_overdue_id_city'})
mean_age_id_city = train_sample.groupby('id_city')['age'].mean().reset_index().rename(columns={'age':'mean_age_diff_overdue_id_city'})
id_city_feature = reduce(lambda x, y: pd.merge(x, y, on='id_city', how='outer'),
                                   [id_city_counts,mean_backward_days_of_id_city,mean_due_amt_of_id_city,mean_term_of_id_city,\
                                   mean_is_overdue_id_city,mean_regmon_id_city,mean_age_id_city])
train_sample = pd.merge(train_sample,id_city_feature,on='id_city',how='left')
testt_sample = pd.merge(testt_sample,id_city_feature,on='id_city',how='left')

train_sample = pd.merge(train_sample,id_province_feature,on='id_province',how='left')
testt_sample = pd.merge(testt_sample,id_province_feature,on='id_province',how='left')

train_sample = pd.merge(train_sample,cell_province_feature,on='cell_province',how='left')
testt_sample = pd.merge(testt_sample,cell_province_feature,on='cell_province',how='left')

train_datas=lgb.Dataset(data=train_sample[features].values,label=train_sample.backward_days,feature_name=features,weight=train_sample.weights)
testt_datas=lgb.Dataset(data=testt_sample[features].values,label=testt_sample.backward_days,feature_name=features,weight=testt_sample.weights)

params = {
    'nthread': 3,  # 进程数
    'max_depth': 4,  # 最大深度
    'learning_rate': 0.1,  # 学习率
    'bagging_fraction': 1,  # 采样数
    'num_leaves': 13,  # 终点节点最小样本占比的和
    'feature_fraction': 0.3,  # 样本列采样
    'objective': 'multiclass',
    'lambda_l1': 10,  # L1 正则化
    'lambda_l2': 1,  # L2 正则化
    'bagging_seed': 100,  # 随机种子,light中默认为100
    'verbose': 0,
    'num_class': 33,
    'min_data_in_leaf':1000
}

model_offline = lgb.train(params,train_datas,num_boost_round=1000,valid_sets=[testt_datas,train_datas],early_stopping_rounds=15)

result1 = model_offline.predict(testt_sample[features])

test_result = {
    'listing_id':[],
    'repay_date':[],
    'backward_day':[],
    'repay_amt_predict':[],
    'due_amt':[],
    'repay_amt':[]
}
for i,values in tqdm.tqdm_notebook(enumerate(testt_sample.values)):
    real_backward_day = testt_sample.backward_days.values[i]
    backward_days_max = Caltime(values[2],values[3])

    for j in range(backward_days_max+1):
        test_result['listing_id'].append(values[1])
        test_result['repay_date'].append(get_date_before(values[3],j))
        test_result['backward_day'].append(j)
        test_result['repay_amt_predict'].append(values[4]*result1[i][j])
        test_result['due_amt'].append(values[4])
        if j == real_backward_day:
            test_result['repay_amt'].append(values[4])
        else:
            test_result['repay_amt'].append(0)
test_result = pd.DataFrame(test_result)

result_data = testt_sample
result_data = result_data.reset_index(drop=True)
result_data = result_data[['listing_id','auditing_date']]
result_data['listing_group'] = 0
from sklearn.model_selection import KFold
kf = KFold(n_splits=30, shuffle=True, random_state=1)
idxs = []
for idx in kf.split(result_data):
    idxs.append(idx[1])
i = 0
for idx in idxs:
    result_data.loc[idx,'listing_group']  = i
    i+=1

test_result = pd.merge(test_result[['listing_id','repay_date','repay_amt_predict','repay_amt']], result_data, on='listing_id',how='inner')

from sklearn.metrics import mean_squared_error
print(mean_squared_error(test_result.repay_amt_predict,test_result.repay_amt))
print(math.sqrt(mean_squared_error(result1.repay_amt_predict,result1.repay_amt)))
print(math.sqrt(mean_squared_error(result2.repay_amt_predict_adjust,result2.repay_amt)))