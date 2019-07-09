# encoding: utf-8
"""
@author: vinklibrary
@contact: shenxvdong1@gmail.com
@site: Todo

@version: 1.0
@license: None
@file: user_tag_embedding.py.py
@time: 2019/7/9 17:33

这一行开始写关于本文件的说明与解释
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

user_tag = pd.read_csv("../dataset/raw_data/user_taglist.csv")
user_tag['taglist'] = user_tag.taglist.map(lambda x:' '.join(x.split("|")))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(user_tag.taglist.values)
svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
mode_svd = svd_enc.fit_transform(X)
mode_svd = pd.DataFrame(mode_svd)

mode_svd.columns = ['taglist_svd0','taglist_svd1','taglist_svd2','taglist_svd3','taglist_svd4','taglist_svd5','taglist_svd6','taglist_svd7','taglist_svd8','taglist_svd9']
user_tag = pd.concat([user_tag,mode_svd],axis=1)

user_tag[['user_id','insertdate','taglist_svd0','taglist_svd1','taglist_svd2','taglist_svd3','taglist_svd4','taglist_svd5','taglist_svd6','taglist_svd7','taglist_svd8','taglist_svd9']].to_csv("../dataset/gen_features/user_taglist_fea.csv",index=None)