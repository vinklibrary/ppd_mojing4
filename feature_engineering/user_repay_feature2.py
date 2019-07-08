# encoding: utf-8
"""
@author: vinklibrary
@contact: shenxvdong1@gmail.com
@site: Todo

@version: 1.0
@license: None
@file: user_repay_feature2.py.py
@time: 2019/7/8 15:12

这一行开始写关于本文件的说明与解释
"""
import pandas as pd
import numpy as np
from utils.functions import Caltime, get_date_before
from functools import reduce

user_pay_logs_fe2 = pd.read_csv("../dataset/gen_data/user_pay_logs_fe2.csv")
# 统计最近15,31,90,180天内的情况
user_pay_logs_fe2['daybefore15'] = user_pay_logs_fe2.auditing_date.map(lambda x:get_date_before(x,15))
user_pay_logs_fe2['daybefore31'] = user_pay_logs_fe2.auditing_date.map(lambda x:get_date_before(x,31))
user_pay_logs_fe2['daybefore90'] = user_pay_logs_fe2.auditing_date.map(lambda x:get_date_before(x,90))
user_pay_logs_fe2['daybefore180'] = user_pay_logs_fe2.auditing_date.map(lambda x:get_date_before(x,180))
print("#"*10 + "user_pay_logs_fe2 sample preprocess finished"+"#"*10)

# 用户还款特征,最近yi段时间内的还款标的数
user_repay_in15days = user_pay_logs_fe2[user_pay_logs_fe2.repay_date>user_pay_logs_fe2.daybefore15].groupby(['user_id','auditing_date']).size()\
    .reset_index().rename(columns={0:'user_repay_in15days'})
user_repay_in31days = user_pay_logs_fe2[user_pay_logs_fe2.repay_date>user_pay_logs_fe2.daybefore31].groupby(['user_id','auditing_date']).size()\
    .reset_index().rename(columns={0:'user_repay_in31days'})
user_repay_in90days = user_pay_logs_fe2[user_pay_logs_fe2.repay_date>user_pay_logs_fe2.daybefore90].groupby(['user_id','auditing_date']).size()\
    .reset_index().rename(columns={0:'user_repay_in90days'})
user_repay_in180days = user_pay_logs_fe2[user_pay_logs_fe2.repay_date>user_pay_logs_fe2.daybefore180].groupby(['user_id','auditing_date']).size()\
    .reset_index().rename(columns={0:'user_repay_in180days'})

user_repay_amt_in15days = user_pay_logs_fe2[user_pay_logs_fe2.repay_date>user_pay_logs_fe2.daybefore15].groupby(['user_id','auditing_date'])['due_amt'].sum()\
    .reset_index().rename(columns={'due_amt':'user_repay_amt_in15days'})
user_repay_amt_in31days = user_pay_logs_fe2[user_pay_logs_fe2.repay_date>user_pay_logs_fe2.daybefore31].groupby(['user_id','auditing_date'])['due_amt'].sum()\
    .reset_index().rename(columns={'due_amt':'user_repay_amt_in31days'})
user_repay_amt_in90days = user_pay_logs_fe2[user_pay_logs_fe2.repay_date>user_pay_logs_fe2.daybefore90].groupby(['user_id','auditing_date'])['due_amt'].sum()\
    .reset_index().rename(columns={'due_amt':'user_repay_amt_in90days'})
user_repay_amt_in180days = user_pay_logs_fe2[user_pay_logs_fe2.repay_date>user_pay_logs_fe2.daybefore180].groupby(['user_id','auditing_date'])['due_amt'].sum()\
    .reset_index().rename(columns={'due_amt':'user_repay_amt_in180days'})
print("#"*10 + "user_pay_fea2 class1 finished"+"#"*10)

# 用户历史上首期的平均还款时间
user_pay_logs_fe2_order1 = user_pay_logs_fe2[user_pay_logs_fe2.order_id==1]

user_repay1_backward_mean_in15days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore15].groupby(['user_id','auditing_date'])['backward_days'].mean()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_mean_in15days'})
user_repay1_backward_mean_in31days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore31].groupby(['user_id','auditing_date'])['backward_days'].mean()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_mean_in31days'})
user_repay1_backward_mean_in90days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore90].groupby(['user_id','auditing_date'])['backward_days'].mean()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_mean_in90days'})
user_repay1_backward_mean_in180days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore180].groupby(['user_id','auditing_date'])['backward_days'].mean()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_mean_in180days'})

user_repay1_backward_median_in15days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore15].groupby(['user_id','auditing_date'])['backward_days'].median()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_median_in15days'})
user_repay1_backward_median_in31days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore31].groupby(['user_id','auditing_date'])['backward_days'].median()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_median_in31days'})
user_repay1_backward_median_in90days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore90].groupby(['user_id','auditing_date'])['backward_days'].median()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_median_in90days'})
user_repay1_backward_median_in180days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore180].groupby(['user_id','auditing_date'])['backward_days'].median()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_median_in180days'})

user_repay1_backward_max_in15days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore15].groupby(['user_id','auditing_date'])['backward_days'].max()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_max_in15days'})
user_repay1_backward_max_in31days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore31].groupby(['user_id','auditing_date'])['backward_days'].max()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_max_in31days'})
user_repay1_backward_max_in90days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore90].groupby(['user_id','auditing_date'])['backward_days'].max()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_max_in90days'})
user_repay1_backward_max_in180days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore180].groupby(['user_id','auditing_date'])['backward_days'].max()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_max_in180days'})

user_repay1_backward_min_in15days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore15].groupby(['user_id','auditing_date'])['backward_days'].min()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_min_in15days'})
user_repay1_backward_min_in31days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore31].groupby(['user_id','auditing_date'])['backward_days'].min()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_min_in31days'})
user_repay1_backward_min_in90days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore90].groupby(['user_id','auditing_date'])['backward_days'].min()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_min_in90days'})
user_repay1_backward_min_in180days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore180].groupby(['user_id','auditing_date'])['backward_days'].min()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_min_in180days'})

user_repay1_backward_std_in15days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore15].groupby(['user_id','auditing_date'])['backward_days'].std()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_std_in15days'})
user_repay1_backward_std_in31days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore31].groupby(['user_id','auditing_date'])['backward_days'].std()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_std_in31days'})
user_repay1_backward_std_in90days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore90].groupby(['user_id','auditing_date'])['backward_days'].std()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_std_in90days'})
user_repay1_backward_std_in180days = user_pay_logs_fe2_order1[user_pay_logs_fe2_order1.repay_date>user_pay_logs_fe2_order1.daybefore180].groupby(['user_id','auditing_date'])['backward_days'].std()\
    .reset_index().rename(columns={'backward_days':'user_repay1_backward_std_in180days'})
print("#"*10 + "user_pay_fea2 class2 finished"+"#"*10)

# 所有的在还款期间的还款行为特征
user_pay_logs_fe2_order2 = user_pay_logs_fe2[user_pay_logs_fe2.backward_days<=31]

user_repay2_backward_mean_in15days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore15].groupby(['user_id','auditing_date'])['backward_days'].mean()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_mean_in15days'})
user_repay2_backward_mean_in31days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore31].groupby(['user_id','auditing_date'])['backward_days'].mean()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_mean_in31days'})
user_repay2_backward_mean_in90days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore90].groupby(['user_id','auditing_date'])['backward_days'].mean()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_mean_in90days'})
user_repay2_backward_mean_in180days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore180].groupby(['user_id','auditing_date'])['backward_days'].mean()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_mean_in180days'})

user_repay2_backward_median_in15days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore15].groupby(['user_id','auditing_date'])['backward_days'].median()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_median_in15days'})
user_repay2_backward_median_in31days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore31].groupby(['user_id','auditing_date'])['backward_days'].median()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_median_in31days'})
user_repay2_backward_median_in90days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore90].groupby(['user_id','auditing_date'])['backward_days'].median()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_median_in90days'})
user_repay2_backward_median_in180days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore180].groupby(['user_id','auditing_date'])['backward_days'].median()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_median_in180days'})

user_repay2_backward_max_in15days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore15].groupby(['user_id','auditing_date'])['backward_days'].max()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_max_in15days'})
user_repay2_backward_max_in31days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore31].groupby(['user_id','auditing_date'])['backward_days'].max()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_max_in31days'})
user_repay2_backward_max_in90days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore90].groupby(['user_id','auditing_date'])['backward_days'].max()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_max_in90days'})
user_repay2_backward_max_in180days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore180].groupby(['user_id','auditing_date'])['backward_days'].max()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_max_in180days'})

user_repay2_backward_min_in15days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore15].groupby(['user_id','auditing_date'])['backward_days'].min()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_min_in15days'})
user_repay2_backward_min_in31days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore31].groupby(['user_id','auditing_date'])['backward_days'].min()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_min_in31days'})
user_repay2_backward_min_in90days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore90].groupby(['user_id','auditing_date'])['backward_days'].min()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_min_in90days'})
user_repay2_backward_min_in180days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore180].groupby(['user_id','auditing_date'])['backward_days'].min()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_min_in180days'})

user_repay2_backward_std_in15days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore15].groupby(['user_id','auditing_date'])['backward_days'].std()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_std_in15days'})
user_repay2_backward_std_in31days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore31].groupby(['user_id','auditing_date'])['backward_days'].std()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_std_in31days'})
user_repay2_backward_std_in90days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore90].groupby(['user_id','auditing_date'])['backward_days'].std()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_std_in90days'})
user_repay2_backward_std_in180days = user_pay_logs_fe2_order2[user_pay_logs_fe2_order2.repay_date>user_pay_logs_fe2_order2.daybefore180].groupby(['user_id','auditing_date'])['backward_days'].std()\
    .reset_index().rename(columns={'backward_days':'user_repay2_backward_std_in180days'})
print("#"*10 + "user_pay_fea2 class3 finished"+"#"*10)

# 用户提前还款的单数，和金额
user_pay_logs_fe2_before = user_pay_logs_fe2[user_pay_logs_fe2.backward_days > 31]

user_repay_before_in15days = user_pay_logs_fe2_before[
    user_pay_logs_fe2_before.repay_date > user_pay_logs_fe2_before.daybefore15].groupby(
    ['user_id', 'auditing_date']).size() \
    .reset_index().rename(columns={0: 'user_repay_before_in15days'})
user_repay_before_in31days = user_pay_logs_fe2_before[
    user_pay_logs_fe2_before.repay_date > user_pay_logs_fe2_before.daybefore31].groupby(
    ['user_id', 'auditing_date']).size() \
    .reset_index().rename(columns={0: 'user_repay_before_in31days'})
user_repay_before_in90days = user_pay_logs_fe2_before[
    user_pay_logs_fe2_before.repay_date > user_pay_logs_fe2_before.daybefore90].groupby(
    ['user_id', 'auditing_date']).size() \
    .reset_index().rename(columns={0: 'user_repay_before_in90days'})
user_repay_before_in180days = user_pay_logs_fe2_before[
    user_pay_logs_fe2_before.repay_date > user_pay_logs_fe2_before.daybefore180].groupby(
    ['user_id', 'auditing_date']).size() \
    .reset_index().rename(columns={0: 'user_repay_before_in180days'})

user_repay_amt_before_in15days = \
user_pay_logs_fe2_before[user_pay_logs_fe2_before.repay_date > user_pay_logs_fe2_before.daybefore15].groupby(
    ['user_id', 'auditing_date'])['due_amt'].sum() \
    .reset_index().rename(columns={'due_amt': 'user_repay_amt_before_in15days'})
user_repay_amt_before_in31days = \
user_pay_logs_fe2_before[user_pay_logs_fe2_before.repay_date > user_pay_logs_fe2_before.daybefore31].groupby(
    ['user_id', 'auditing_date'])['due_amt'].sum() \
    .reset_index().rename(columns={'due_amt': 'user_repay_amt_before_in31days'})
user_repay_amt_before_in90days = \
user_pay_logs_fe2_before[user_pay_logs_fe2_before.repay_date > user_pay_logs_fe2_before.daybefore90].groupby(
    ['user_id', 'auditing_date'])['due_amt'].sum() \
    .reset_index().rename(columns={'due_amt': 'user_repay_amt_before_in90days'})
user_repay_amt_before_in180days = \
user_pay_logs_fe2_before[user_pay_logs_fe2_before.repay_date > user_pay_logs_fe2_before.daybefore180].groupby(
    ['user_id', 'auditing_date'])['due_amt'].sum() \
    .reset_index().rename(columns={'due_amt': 'user_repay_amt_before_in180days'})

latest_repay_date = user_pay_logs_fe2.groupby(['user_id', 'auditing_date'])['repay_date'].max().reset_index().rename(
    columns={'repay_date': 'latest_repay_date'})
farthest_repay_date = user_pay_logs_fe2.groupby(['user_id', 'auditing_date'])['repay_date'].min().reset_index().rename(
    columns={'repay_date': 'farthest_repay_date'})
latest_repay_date['latest_repay_date_diff'] = latest_repay_date.apply(
    lambda x: Caltime(x['latest_repay_date'], x['auditing_date']), axis=1)
farthest_repay_date['farthest_repay_date_diff'] = farthest_repay_date.apply(
    lambda x: Caltime(x['farthest_repay_date'], x['auditing_date']), axis=1)
user_repay_by_dow = user_pay_logs_fe2.groupby(['user_id', 'auditing_date', 'repay_dow']).size().reset_index().rename(
    columns={0: 'user_repay_by_dow'})
user_repay_by_dow = user_repay_by_dow.pivot_table(index=['user_id', 'auditing_date'], columns='repay_dow',
                                                  values='user_repay_by_dow').reset_index()
user_repay_by_dow.columns = ['user_id', 'auditing_date', 'repay_dow1', 'repay_dow2', 'repay_dow3', 'repay_dow4',
                             'repay_dow5', 'repay_dow6', 'repay_dow7']
user_repay_counts = user_pay_logs_fe2.groupby(['user_id', 'auditing_date']).size().reset_index().rename(
    columns={0: 'user_repay_counts'})

user_repay_distribution = pd.merge(user_repay_by_dow, user_repay_counts, on=['user_id', 'auditing_date'], how='left')
user_repay_distribution = user_repay_distribution.fillna(0)
for column in ['repay_dow1', 'repay_dow2', 'repay_dow3', 'repay_dow4', 'repay_dow5', 'repay_dow6', 'repay_dow7']:
    mean_c = user_repay_distribution['user_repay_counts'].mean()
    mean_rate = user_repay_distribution[column].sum() / user_repay_distribution['user_repay_counts'].sum()
    user_repay_distribution[column + "rate"] = (user_repay_distribution[column] + mean_c * mean_rate) / (
                user_repay_distribution['user_repay_counts'] + mean_c)


def get_mon_code(inte):
    if inte <= 10:
        return 0
    elif inte >= 20:
        return 2
    else:
        return 1


user_pay_logs_fe2['mon_code'] = user_pay_logs_fe2.repay_dom.map(lambda x: get_mon_code(x))
user_repay_by_moncode = user_pay_logs_fe2.groupby(['user_id', 'auditing_date', 'mon_code']).size().reset_index().rename(
    columns={0: 'user_repay_by_moncode'})
user_repay_by_moncode = user_repay_by_moncode.pivot_table(index=['user_id', 'auditing_date'], columns='mon_code',
                                                          values='user_repay_by_moncode').reset_index()
user_repay_by_moncode.columns = ['user_id', 'auditing_date', 'mon_code0', 'mon_code1', 'mon_code2']
user_repay_by_moncode = user_repay_by_moncode.fillna(0)
user_repay_distribution2 = pd.merge(user_repay_by_moncode, user_repay_counts, on=['user_id', 'auditing_date'],
                                    how='left')
for column in ['mon_code0', 'mon_code1', 'mon_code2']:
    mean_c = user_repay_distribution2['user_repay_counts'].mean()
    mean_rate = user_repay_distribution2[column].sum() / user_repay_distribution2['user_repay_counts'].sum()
    user_repay_distribution2[column + "rate"] = (user_repay_distribution2[column] + mean_c * mean_rate) / (
                user_repay_distribution2['user_repay_counts'] + mean_c)


# 用户还款时间的diff的差距
def get_diff_mean(x):
    return pd.Series(np.sort(np.unique(x))).diff().mean()
def get_diff_min(x):
    return pd.Series(np.sort(np.unique(x))).diff().min()
def get_diff_max(x):
    return pd.Series(np.sort(np.unique(x))).diff().max()
def get_diff_median(x):
    return pd.Series(np.sort(np.unique(x))).diff().median()
def get_diff_std(x):
    return pd.Series(np.sort(np.unique(x))).diff().std()

repay_date_numeric_diff_mean = user_pay_logs_fe2.groupby(['user_id', 'auditing_date'])['repay_date_numeric'].apply(
    lambda x: get_diff_mean(x)) \
    .reset_index().rename(columns={'repay_date_numeric': 'repay_date_numeric_diff_mean'})

repay_date_numeric_diff_min = user_pay_logs_fe2.groupby(['user_id', 'auditing_date'])['repay_date_numeric'].apply(
    lambda x: get_diff_min(x)) \
    .reset_index().rename(columns={'repay_date_numeric': 'repay_date_numeric_diff_min'})

repay_date_numeric_diff_max = user_pay_logs_fe2.groupby(['user_id', 'auditing_date'])['repay_date_numeric'].apply(
    lambda x: get_diff_max(x)) \
    .reset_index().rename(columns={'repay_date_numeric': 'repay_date_numeric_diff_max'})

repay_date_numeric_diff_median = user_pay_logs_fe2.groupby(['user_id', 'auditing_date'])['repay_date_numeric'].apply(
    lambda x: get_diff_median(x)) \
    .reset_index().rename(columns={'repay_date_numeric': 'repay_date_numeric_diff_median'})

repay_date_numeric_diff_std = user_pay_logs_fe2.groupby(['user_id', 'auditing_date'])['repay_date_numeric'].apply(
    lambda x: get_diff_std(x)) \
    .reset_index().rename(columns={'repay_date_numeric': 'repay_date_numeric_diff_std'})

print("#"*10 + "user_pay_fea2 class4 finished"+"#"*10)

user_repay_features2 = reduce(lambda x, y: pd.merge(x, y, on=['user_id', 'auditing_date'], how='outer'),
                              [user_repay_in15days, user_repay_in31days, user_repay_in90days, user_repay_in180days,
                               user_repay_amt_in15days, user_repay_amt_in31days, user_repay_amt_in90days,
                               user_repay_amt_in180days,
                               user_repay1_backward_mean_in15days, user_repay1_backward_mean_in31days,
                               user_repay1_backward_mean_in90days, user_repay1_backward_mean_in180days,
                               user_repay1_backward_median_in15days, user_repay1_backward_median_in31days,
                               user_repay1_backward_median_in90days, user_repay1_backward_median_in180days,
                               user_repay1_backward_max_in15days, user_repay1_backward_max_in31days,
                               user_repay1_backward_max_in90days, user_repay1_backward_max_in180days,
                               user_repay1_backward_min_in15days, user_repay1_backward_min_in31days,
                               user_repay1_backward_min_in90days, user_repay1_backward_min_in180days,
                               user_repay1_backward_std_in15days, user_repay1_backward_std_in31days,
                               user_repay1_backward_std_in90days, user_repay1_backward_std_in180days,
                               user_repay2_backward_mean_in15days, user_repay2_backward_mean_in31days,
                               user_repay2_backward_mean_in90days, user_repay2_backward_mean_in180days,
                               user_repay2_backward_median_in15days, user_repay2_backward_median_in31days,
                               user_repay2_backward_median_in90days, user_repay2_backward_median_in180days,
                               user_repay2_backward_max_in15days, user_repay2_backward_max_in31days,
                               user_repay2_backward_max_in90days, user_repay2_backward_max_in180days,
                               user_repay2_backward_min_in15days, user_repay2_backward_min_in31days,
                               user_repay2_backward_min_in90days, user_repay2_backward_min_in180days,
                               user_repay2_backward_std_in15days, user_repay2_backward_std_in31days,
                               user_repay2_backward_std_in90days, user_repay2_backward_std_in180days,
                               user_repay_before_in15days, user_repay_before_in31days, user_repay_before_in90days,
                               user_repay_before_in180days,
                               user_repay_amt_before_in15days, user_repay_amt_before_in31days,
                               user_repay_amt_before_in90days, user_repay_amt_before_in180days,
                               latest_repay_date[['user_id', 'auditing_date', 'latest_repay_date_diff']],
                               farthest_repay_date[['user_id', 'auditing_date', 'farthest_repay_date_diff']],
                               user_repay_distribution[
                                   ['user_id', 'auditing_date', 'repay_dow1', 'repay_dow2', 'repay_dow3', 'repay_dow4',
                                    'repay_dow5', 'repay_dow6', 'repay_dow7', 'user_repay_counts', 'repay_dow1rate',
                                    'repay_dow2rate', 'repay_dow3rate', 'repay_dow4rate', 'repay_dow5rate',
                                    'repay_dow6rate', 'repay_dow7rate']],
                               user_repay_distribution2[
                                   ['user_id', 'auditing_date', 'mon_code0', 'mon_code1', 'mon_code2', 'mon_code0rate',
                                    'mon_code1rate', 'mon_code2rate']],
                               repay_date_numeric_diff_mean, repay_date_numeric_diff_min, repay_date_numeric_diff_max,
                               repay_date_numeric_diff_median, repay_date_numeric_diff_std])

user_repay_features2.to_csv("../dataset/gen_features/user_repay_features2.csv",index=None)
print("#"*10 + "user_repay_features2 gen and save finished"+"#"*10)
