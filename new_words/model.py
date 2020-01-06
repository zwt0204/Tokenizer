# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2020/1/6 10:00
@Author  : zwt
@git   : https://kexue.fm/archives/3491
@Software: PyCharm
"""
import numpy as np
import pandas as pd
import re
from numpy import log


class new_words:

    def __init__(self):
        self.file = 'D:\mygit\Tokenizer\data\data.txt'
        self.result_file = 'D:\mygit\Tokenizer\data\\result.txt'
        # 用于挑选是几字词
        self.myre = {2: '(..)', 3: '(...)', 4: '(....)', 5: '(.....)', 6: '(......)', 7: '(.......)'}
        # 需要删除的字符
        self.drop_dict = [u'，', u'\n', u'。', u'、', u'：', u'(', u')', u'[', u']', u'.', u',', u' ', u'\u3000', u'”', u'“', u'？', u'?',
             u'！', u'‘', u'’', u'…']

        # 词语最小出现次数
        self.min_count = 10
        # 录取词语最低支持度，1代表着随机组合
        self.min_support = 30
        # 录取词语最低信息熵，越大说明越有可能独立成词
        self.min_s = 3
        # 候选词语的最大字数
        self.max_sep = 4
        self.data = self.read_data()

    def read_data(self):
        """读取数据"""
        with open(self.file, 'r', encoding='utf8') as f:
            data = f.read()
        for i in self.drop_dict:
            data = data.replace(i, '')
        return data

    def cal_S(self, sl):
        """信息熵计算函数
        P = -(log(n/N) * (n/N))
        """
        return -((sl / sl.sum()).apply(log) * sl / sl.sum()).sum()

    def run(self):
        # 保存结果
        rt = []
        # 保存结果
        t = []
        # 逐字统计,统计list中各个元素出现次数，返回两列，第一列为字，第二列为对应出现次数
        t.append(pd.Series(list(self.data)).value_counts())
        # 统计总字数
        tsum = t[0].sum()
        for m in range(2, self.max_sep + 1):
            print(u'正在生成%s字词...' % m)
            t.append([])
            # 生成所有可能的m字词
            for i in range(m):
                t[m - 1] = t[m - 1] + re.findall(self.myre[m], self.data[i:])
            # 逐词统计
            t[m - 1] = pd.Series(t[m - 1]).value_counts()
            # 最小次数筛选
            t[m - 1] = t[m - 1][t[m - 1] > self.min_count]
            tt = t[m - 1][:]
            for k in range(m - 1):
                # 最小支持度筛选
                # PMI:链接：http://note.youdao.com/noteshare?id=366f2a08729e91b92d2b01e854ea1f0b&sub=D9CFF2172C0D4749B843441DB95FC132
                # t[m - 1]表示当前生成的是几字词
                # t[m - 2 - k]，加入当前是2字词则表示一字词
                qq = np.array(
                    list(map(lambda ms: tsum * t[m - 1][ms] / t[m - 2 - k][ms[:m - 1 - k]] / t[k][ms[m - 1 - k:]],
                             tt.index))) > self.min_support
                tt = tt[qq]
            rt.append(tt.index)

        for i in range(2, self.max_sep + 1):
            print(u'正在进行%s字词的最大熵筛选(%s)...' % (i, len(rt[i - 2])))
            # 保存所有的左右邻结果
            pp = []
            for j in range(i + 2):
                pp = pp + re.findall('(.)%s(.)' % self.myre[i], self.data[j:])
            # 先排序，这个很重要，可以加快检索速度
            pp = pd.DataFrame(pp).set_index(1).sort_index()
            # 作交集
            index = np.sort(np.intersect1d(rt[i - 2], pp.index))
            # 下面两句分别是左邻和右邻信息熵筛选
            print(list(map(lambda s: self.cal_S(pd.Series(pp[0][s]).value_counts()), index)))
            # 设置最小信息熵阈值
            index = index[np.array(list(map(lambda s: self.cal_S(pd.Series(pp[0][s]).value_counts()), index))) > self.min_s]
            rt[i - 2] = index[np.array(list(map(lambda s: self.cal_S(pd.Series(pp[2][s]).value_counts()), index))) > self.min_s]

        # 下面都是输出前处理
        for i in range(len(rt)):
            t[i + 1] = t[i + 1][rt[i]]
            t[i + 1].sort_index(ascending=False)

        # 保存结果并输出
        pd.DataFrame(pd.concat(t[1:])).to_csv(self.result_file, header=False)


if __name__ == '__main__':
    model = new_words()
    model.run()