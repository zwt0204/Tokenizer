# -*- encoding: utf-8 -*-
"""
@File    : search.py
@Time    : 2019/12/12 11:13
@Author  : zwt
@参考   : https://kexue.fm/archives/3908
@Software: PyCharm
"""
import ahocorasick
from math import log


class ac_auto:

    def __init__(self, file):
        super().__init__()
        self.file = file

    def load_dic(self):
        dic = ahocorasick.Automaton()
        total = 0.0
        with open(self.file, 'r', encoding='utf8') as file:
            words = []
            for line in file.readlines():
                line = line.split(' ')
                words.append((line[0], int(line[1])))
                total += int(line[1])
        for i, j in words:
            # 使用log，防止溢出
            dic.add_word(i, (log(j / total)))
        dic.make_automaton()
        return dic

    def all_cut(self, sentence):
        """
        AC自动机实现全模式分词
        """
        dic = self.load_dic()
        words = []
        for i, j in dic.iter(sentence):
            words.append(j[0])
        return words

    def max_match_cut(self, sentence):
        """
        AC自动机实现最大匹配法
        """
        dic = self.load_dic()
        words = ['']
        for i in sentence:
            if dic.match(words[-1] + i):
                words[-1] += i
            else:
                words.append(i)
        return words

    def max_proba_cut(self, sentence):
        """
        AC自动机，结合动态规划实现最大概率组合
        """
        dic = self.load_dic()
        paths = {0: ([], 0)}
        end = 0
        for i, j in dic.iter(sentence):
            start, end = 1 + i - len(j[0]), i + 1
            if start not in paths:
                last = max([i for i in paths if i < start])
                paths[start] = (paths[last][0] + [sentence[last:start]], paths[last][1] - 10)
            proba = paths[start][1] + j[1]
            if end not in paths or proba > paths[end][1]:
                paths[end] = (paths[start][0] + [j[0]], proba)
        if end < len(sentence):
            return paths[end][0] + [sentence[end:]]
        else:
            return paths[end][0]


demo = ac_auto('test.txt')
a = demo.max_match_cut('肯德基你好')
print(a)