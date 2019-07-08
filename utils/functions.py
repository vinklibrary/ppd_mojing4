# encoding: utf-8
"""
@author: vinklibrary
@contact: shenxvdong1@gmail.com
@site: Todo

@version: 1.0
@license: None
@file: functions.py
@time: 2019/7/8 13:50

这一行开始写关于本文件的说明与解释
"""
import time
import datetime

# 计算两个日期相差天数，自定义函数名，和两个日期的变量名。 小的日期在前面
def Caltime(date1,date2):
    # 跟项目相关，pandas 空值类型为float
    if type(date1)!=str or type(date2)!=str:
        return -1
    if date1 =="\\N" or date2=="\\N":
        return -1
    date1=time.strptime(date1,"%Y-%m-%d")
    date2=time.strptime(date2,"%Y-%m-%d")
    date1=datetime.datetime(date1[0],date1[1],date1[2])
    date2=datetime.datetime(date2[0],date2[1],date2[2])
    #返回两个变量相差的值，就是相差天数
    return (date2-date1).days

# 获得n天前（后）的日期
def get_date_before(date1, days):
    date1=time.strptime(date1,"%Y-%m-%d")
    date1=datetime.datetime(date1[0],date1[1],date1[2])
    return str(date1-datetime.timedelta(days=days))[0:10]

def get_date_after(date1, days):
    date1=time.strptime(date1,"%Y-%m-%d")
    date1=datetime.datetime(date1[0],date1[1],date1[2])
    return str(date1+datetime.timedelta(days=days))[0:10]

# 计算两个月份之间相差几个月 mon1>mon2
def CalMon(mon1,mon2):
    mon1_y = int(mon1[0:4])
    mon1_m = int(mon1[5:7])
    mon2_y = int(mon2[0:4])
    mon2_m = int(mon2[5:7])
    return (mon1_y-mon2_y)*12+(mon1_m-mon2_m)

# 获得当前日期n个月前的时间
def get_month_before(tmp_auditing_date,num):
    # 项目相关。闰年未考虑
    month_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if int(tmp_auditing_date[5:7]) <= num:
        month = 12-num+int(tmp_auditing_date[5:7])
        day = int(tmp_auditing_date[8:10]) if int(tmp_auditing_date[8:10])<=month_days[month] else month_days[month]
        before_date = datetime.date(int(tmp_auditing_date[0:4])-1,12-num+int(tmp_auditing_date[5:7]),day)
    else:
        month = int(tmp_auditing_date[5:7]) - num
        day = int(tmp_auditing_date[8:10]) if int(tmp_auditing_date[8:10])<=month_days[month] else month_days[month]
        before_date = datetime.date(int(tmp_auditing_date[0:4]),int(tmp_auditing_date[5:7]) - num,day)
    return str(before_date)

def get_month_after(tmp_auditing_date,num):
    month_days = [0,31,28,31,30,31,30,31,31,30,31,30,31]
    if int(tmp_auditing_date[5:7]) + num <= 12:
        month = num+int(tmp_auditing_date[5:7])
        day = int(tmp_auditing_date[8:10]) if int(tmp_auditing_date[8:10])<=month_days[month] else month_days[month]
        before_date = datetime.date(int(tmp_auditing_date[0:4]),month,day)
    else:
        month = int(tmp_auditing_date[5:7]) + num - 12
        day = int(tmp_auditing_date[8:10]) if int(tmp_auditing_date[8:10])<=month_days[month] else month_days[month]
        before_date = datetime.date(int(tmp_auditing_date[0:4])+1, month ,day)
    return str(before_date)

# 获得曜日dow信息
def get_day_of_week(date1):
    date1 = datetime.date(int(date1[0:4]),int(date1[5:7]),int(date1[8:10]))
    return date1.weekday()

# 转化率平滑计算 col1/col2
def get_converation_rate(df,col1,col2,mean=None,mean_rate=None):
    if mean == None:
        mean = df[col2].mean()
    if mean_rate == None:
        mean_rate = df[col1].sum()/df[col2].sum()
    return (df[col1]+mean*mean_rate)/(df[col2]+mean),mean,mean_rate