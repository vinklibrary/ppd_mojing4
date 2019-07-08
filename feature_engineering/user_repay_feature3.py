# encoding: utf-8
"""
@author: vinklibrary
@contact: shenxvdong1@gmail.com
@site: Todo

@version: 1.0
@license: None
@file: user_repay_feature3.py
@time: 2019/7/8 14:37

用户还款特征第三类特征。
"""
import pandas as pd
import tqdm
from utils.functions import get_month_after, Caltime
from functools import reduce

train_data = pd.read_csv("../dataset/raw_data/train.csv")
testt_data = pd.read_csv("../dataset/raw_data/test.csv")
train_data['flag'] = 'train'
testt_data['flag'] = 'testt'
data = pd.concat([train_data[['user_id','listing_id','auditing_date','due_date','flag']],testt_data[['user_id','listing_id','auditing_date','due_date','flag']]])

listing_info = pd.read_csv("../dataset/raw_data/listing_info.csv")
data = pd.merge(data[['user_id','auditing_date']],listing_info,on=['user_id'],how='left')
data = data[data.auditing_date_x>data.auditing_date_y]
print("#"*10 + "raw data preprocess finished"+"#"*10)

# 历史借贷数
tmp1 = data.groupby(['user_id','auditing_date_x']).size().reset_index()
# 历史本金的情况
tmp2 = data.groupby(['user_id','auditing_date_x'])['principal'].agg(['max','mean','median','min','std']).reset_index()
tmp1.columns = ['user_id','auditing_date','listnum_in_histoy']
tmp2.columns = ['user_id','auditing_date','principal_his_max','principal_his_mean','principal_his_median','principal_his_min','principal_his_std']

listing_info_feature = {
    'user_id': [],
    'listing_id': [],
    'auditing_date': [],
    'order_id': [],
    'order_date': [],
    'due_date': [],
    'order_due_amt': []
}
for i, value in tqdm.tqdm(enumerate(data.values)):
    tmp_user_id = value[0]
    tmp_listing_id = value[2]
    tmp_auditing_date1 = value[1]
    tmp_term = value[4]
    tmp_auditing_date2 = value[3]

    for j in range(tmp_term):
        if get_month_after(tmp_auditing_date2, j + 1) > tmp_auditing_date1:
            listing_info_feature['user_id'].append(tmp_user_id)
            listing_info_feature['listing_id'].append(tmp_listing_id)
            listing_info_feature['auditing_date'].append(tmp_auditing_date1)
            listing_info_feature['order_id'].append(j + 1)
            listing_info_feature['order_date'].append(get_month_after(tmp_auditing_date2, j))
            listing_info_feature['due_date'].append(get_month_after(tmp_auditing_date2, j + 1))
            listing_info_feature['order_due_amt'].append(int(value[6] / tmp_term))
listing_info_feature = pd.DataFrame(listing_info_feature)
print("#"*10 + "basic sample preprocess finished"+"#"*10)

# 加入未来6个月的时间点
listing_info_feature['month1'] = listing_info_feature.auditing_date.map(lambda x:get_month_after(x,1))
listing_info_feature['month3'] = listing_info_feature.auditing_date.map(lambda x:get_month_after(x,3))
listing_info_feature['month6'] = listing_info_feature.auditing_date.map(lambda x:get_month_after(x,6))

# 用户在当前标的生效的时间点 还有多少单没有还（listing_id的auditing_date<当前标的的auditing_date）
future_list_counts = listing_info_feature.groupby(['user_id','auditing_date'])['listing_id'].nunique().reset_index().rename(columns={'listing_id':'future_list_counts'})
future_list_nums = listing_info_feature.groupby(['user_id','auditing_date']).size().reset_index().rename(columns={0:'future_list_nums'})

# 未来第一个到期单子到现在的距离。 如果那时候用户经济状态良好，极有可能让本单提早还。
latest_come_date  = listing_info_feature.groupby(['user_id','auditing_date'])['due_date'].min().reset_index().rename(columns={'due_date':'latest_come_date'})
latest_come_date['latest_day_diff'] = latest_come_date.apply(lambda x:Caltime(x['auditing_date'],x['latest_come_date']),axis=1)

# 未来三个月内的订单情况 非穿越。
future_list_nums_in1month = listing_info_feature[listing_info_feature.due_date<listing_info_feature.month1].groupby(['user_id','auditing_date']).size().reset_index().rename(columns={0:'future_list_nums_in1month'})
future_list_nums_in3month = listing_info_feature[listing_info_feature.due_date<listing_info_feature.month3].groupby(['user_id','auditing_date']).size().reset_index().rename(columns={0:'future_list_nums_in3month'})
future_list_nums_in6month = listing_info_feature[listing_info_feature.due_date<listing_info_feature.month6].groupby(['user_id','auditing_date']).size().reset_index().rename(columns={0:'future_list_nums_in6month'})
future_amt_in1month = listing_info_feature[listing_info_feature.due_date<listing_info_feature.month1].groupby(['user_id','auditing_date'])['order_due_amt'].sum().reset_index().rename(columns={0:'future_amt_in1month'})
future_amt_in3month = listing_info_feature[listing_info_feature.due_date<listing_info_feature.month3].groupby(['user_id','auditing_date'])['order_due_amt'].sum().reset_index().rename(columns={0:'future_amt_in3month'})
future_amt_in6month = listing_info_feature[listing_info_feature.due_date<listing_info_feature.month6].groupby(['user_id','auditing_date'])['order_due_amt'].sum().reset_index().rename(columns={0:'future_amt_in6month'})

# 将所有特征合并起来，并持久化到中间文件
user_repay_features3 = reduce(lambda x, y: pd.merge(x, y, on=['user_id','auditing_date'], how='outer'),
                                   [tmp1,tmp2,future_list_counts,future_list_nums,
                                    latest_come_date[['user_id','auditing_date','latest_day_diff']],
                                    future_list_nums_in1month, future_list_nums_in1month, future_list_nums_in1month,
                                    future_amt_in1month, future_amt_in1month, future_amt_in1month])
print("#"*10 + "user_repay_fea3 generate finished"+"#"*10)

user_repay_features3.to_csv("../dataset/gen_features/user_repay_features3.csv",index=None)