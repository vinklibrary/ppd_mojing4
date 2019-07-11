# encoding: utf-8
"""
@author: vinklibrary
@contact: shenxvdong1@gmail.com
@site: Todo

@version: 1.0
@license: None
@file: user_repay_feature1.py
@time: 2019/7/8 16:15

这一行开始写关于本文件的说明与解释
"""
import pandas as pd
import numpy as np
from utils.functions import get_date_before, Caltime
from functools import reduce

user_pay_logs_fe1 = pd.read_csv("../samples/user_pay_logs_fe1.csv")
# 统计最近15,31,90,180天内的情况
user_pay_logs_fe1['daybefore15'] = user_pay_logs_fe1.auditing_date.map(lambda x:get_date_before(x,15))
user_pay_logs_fe1['daybefore31'] = user_pay_logs_fe1.auditing_date.map(lambda x:get_date_before(x,31))
user_pay_logs_fe1['daybefore90'] = user_pay_logs_fe1.auditing_date.map(lambda x:get_date_before(x,90))
user_pay_logs_fe1['daybefore180'] = user_pay_logs_fe1.auditing_date.map(lambda x:get_date_before(x,180))

user_daoqi_in15days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore15].groupby(['user_id','auditing_date']).size()\
.reset_index().rename(columns={0:'user_daoqi_in15days'})
user_daoqi_in31days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore31].groupby(['user_id','auditing_date']).size()\
.reset_index().rename(columns={0:'user_daoqi_in31days'})
user_daoqi_in90days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore90].groupby(['user_id','auditing_date']).size()\
.reset_index().rename(columns={0:'user_daoqi_in90days'})
user_daoqi_in180days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore180].groupby(['user_id','auditing_date']).size()\
.reset_index().rename(columns={0:'user_daoqi_in180days'})

user_daoqi_amt_in15days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore15].groupby(['user_id','auditing_date'])['due_amt'].sum()\
.reset_index().rename(columns={'due_amt':'user_daoqi_amt_in15days'})
user_daoqi_amt_in31days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore31].groupby(['user_id','auditing_date'])['due_amt'].sum()\
.reset_index().rename(columns={'due_amt':'user_daoqi_amt_in31days'})
user_daoqi_amt_in90days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore90].groupby(['user_id','auditing_date'])['due_amt'].sum()\
.reset_index().rename(columns={'due_amt':'user_daoqi_amt_in90days'})
user_daoqi_amt_in180days = user_pay_logs_fe1[user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore180].groupby(['user_id','auditing_date'])['due_amt'].sum()\
.reset_index().rename(columns={'due_amt':'user_daoqi_amt_in180days'})

user_yuqi_in15days = user_pay_logs_fe1[(user_pay_logs_fe1.repay_date=='2200-01-01')&(user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore15)].groupby(['user_id','auditing_date']).size()\
.reset_index().rename(columns={0:'user_yuqi_in15days'})
user_yuqi_in31days = user_pay_logs_fe1[(user_pay_logs_fe1.repay_date=='2200-01-01')&(user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore31)].groupby(['user_id','auditing_date']).size()\
.reset_index().rename(columns={0:'user_yuqi_in31days'})
user_yuqi_in90days = user_pay_logs_fe1[(user_pay_logs_fe1.repay_date=='2200-01-01')&(user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore90)].groupby(['user_id','auditing_date']).size()\
.reset_index().rename(columns={0:'user_yuqi_in90days'})
user_yuqi_in180days = user_pay_logs_fe1[(user_pay_logs_fe1.repay_date=='2200-01-01')&(user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore180)].groupby(['user_id','auditing_date']).size()\
.reset_index().rename(columns={0:'user_yuqi_in180days'})

user_yuqi_amt_in15days = user_pay_logs_fe1[(user_pay_logs_fe1.repay_date=='2200-01-01')&(user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore15)].groupby(['user_id','auditing_date'])['due_amt'].sum()\
.reset_index().rename(columns={0:'user_yuqi_amt_in15days'})
user_yuqi_amt_in31days = user_pay_logs_fe1[(user_pay_logs_fe1.repay_date=='2200-01-01')&(user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore31)].groupby(['user_id','auditing_date'])['due_amt'].sum()\
.reset_index().rename(columns={0:'user_yuqi_amt_in31days'})
user_yuqi_amt_in90days = user_pay_logs_fe1[(user_pay_logs_fe1.repay_date=='2200-01-01')&(user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore90)].groupby(['user_id','auditing_date'])['due_amt'].sum()\
.reset_index().rename(columns={0:'user_yuqi_amt_in90days'})
user_yuqi_amt_in180days = user_pay_logs_fe1[(user_pay_logs_fe1.repay_date=='2200-01-01')&(user_pay_logs_fe1.due_date>user_pay_logs_fe1.daybefore180)].groupby(['user_id','auditing_date'])['due_amt'].sum()\
.reset_index().rename(columns={0:'user_yuqi_amt_in180days'})

latest_due_date = user_pay_logs_fe1.groupby(['user_id','auditing_date'])['due_date'].max().reset_index().rename(columns={'due_date':'latest_due_date'})
farthest_due_date = user_pay_logs_fe1.groupby(['user_id','auditing_date'])['due_date'].min().reset_index().rename(columns={'due_date':'farthest_due_date'})

latest_due_date['latest_due_date_diff'] = latest_due_date.apply(lambda x:Caltime(x['latest_due_date'],x['auditing_date']),axis=1)
farthest_due_date['farthest_due_date_diff'] = farthest_due_date.apply(lambda x:Caltime(x['farthest_due_date'],x['auditing_date']),axis=1)

user_repay_features2 = reduce(lambda x, y: pd.merge(x, y, on=['user_id','auditing_date'], how='outer'),
                                   [user_daoqi_in15days,user_daoqi_in31days,user_daoqi_in90days,user_daoqi_in180days,
user_daoqi_amt_in15days,user_daoqi_amt_in31days,user_daoqi_amt_in90days,user_daoqi_amt_in180days,
user_yuqi_in15days,user_yuqi_in31days,user_yuqi_in90days,user_yuqi_in180days,
user_yuqi_amt_in15days,user_yuqi_amt_in31days,user_yuqi_amt_in90days,user_yuqi_amt_in180days,
latest_due_date,farthest_due_date])

user_repay_features2.to_csv("../dataset/gen_features/user_repay_features2.csv",index=None)