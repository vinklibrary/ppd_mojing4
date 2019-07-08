# encoding: utf-8
"""
@author: vinklibrary
@contact: shenxvdong1@gmail.com
@site: Todo

@version: 1.0
@license: None
@file: user_repay_samples.py
@time: 2019/7/8 15:13

这一行开始写关于本文件的说明与解释
"""
import pandas as pd
from utils.functions import Caltime

user_pay_logs = pd.read_csv("../dataset/raw_data/user_repay_logs.csv")
# 加入两个距离最早日期的天数，方便之后计算日子之间的diff特征
user_pay_logs['due_date_numeric'] = user_pay_logs.due_date.map(lambda x:Caltime('2017-07-05',x))
user_pay_logs['repay_date_numeric'] = user_pay_logs.repay_date.map(lambda x:Caltime('2017-07-05',x))
# 用户提前还的天数
user_pay_logs['backward_days'] = user_pay_logs.apply(lambda x:Caltime(x['repay_date'],x['due_date']),axis=1)
# 标的到期的时间特征
user_pay_logs['due_dom'] = user_pay_logs.due_date.map(lambda x:int(x[8:10]))
user_pay_logs['due_dow'] = user_pay_logs.due_date.map(lambda x:pd.to_datetime(x).weekday()+1)
user_pay_logs['repay_dom'] = user_pay_logs.repay_date.map(lambda x:int(x[8:10]))
user_pay_logs['repay_dow'] = user_pay_logs.repay_date.map(lambda x:pd.to_datetime(x).weekday()+1)
print("#"*10 + "user_repay_logs preprocess finished"+"#"*10)

train_data = pd.read_csv("../dataset/raw_data/train.csv")
testt_data = pd.read_csv("../dataset/raw_data/test.csv")
train_data['flag'] = 'train'
testt_data['flag'] = 'testt'
data = pd.concat([train_data[['user_id','listing_id','auditing_date','due_date','flag']],testt_data[['user_id','listing_id','auditing_date','due_date','flag']]])
# 给每一标的join 还款信息
user_pay_logs_fe =  pd.merge(data[['user_id','auditing_date']],user_pay_logs,on=['user_id'],how='left')
print("#"*10 + "data join user_repay_logs  finished"+"#"*10)

# 用户历史上标的到期日在当前时间之前的数据
user_pay_logs_fe1 = user_pay_logs_fe[user_pay_logs_fe.due_date < user_pay_logs_fe.auditing_date]
# 用户历史上标的还款日在当前时间之间的数据
user_pay_logs_fe2 = user_pay_logs_fe[user_pay_logs_fe.repay_date < user_pay_logs_fe.auditing_date]
user_pay_logs_fe1.to_csv("../dataset/gen_data/user_pay_logs_fe1.csv",index=None)
user_pay_logs_fe2.to_csv("../dataset/gen_data/user_pay_logs_fe2.csv",index=None)
print("#"*10 + "fe filter and save finished"+"#"*10)
