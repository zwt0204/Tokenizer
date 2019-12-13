# -*- encoding: utf-8 -*-
"""
@File    : model_hmm.py
@Time    : 2019/12/6 9:50
@Author  : zwt
@git   :  https://github.com/CQUPT-Wan/HMMwordseg
@Software: PyCharm
"""
from numpy import *
import numpy as np

STATES = ['B', 'M', 'E', 'S']
array_A = {}  # 状态转移概率矩阵
array_B = {}  # 发射概率矩阵
array_Pi = {}  # 初始状态分布
word_set = set()  # 训练数据集中所有字的集合
count_dic = {}  # ‘B,M,E,S’每个状态在训练集中出现的次数
line_num = 0  # 训练集语句数量


# 初始化所有概率矩阵
def Init_Array():
    for state0 in STATES:
        array_A[state0] = {}
        for state1 in STATES:
            array_A[state0][state1] = 0.0
    for state in STATES:
        array_Pi[state] = 0.0
        array_B[state] = {}
        count_dic[state] = 0


# 对训练集获取状态标签
def get_tag(word):
    tag = []
    if len(word) == 1:
        tag = ['S']
    elif len(word) == 2:
        tag = ['B', 'E']
    else:
        num = len(word) - 2
        tag.append('B')
        tag.extend(['M'] * num)
        tag.append('E')
    return tag


# 将参数估计的概率取对数，对概率0取无穷小-3.14e+100
def Prob_Array():
    for key in array_Pi:
        if array_Pi[key] == 0:
            array_Pi[key] = -3.14e+100
        else:
            array_Pi[key] = log(array_Pi[key] / line_num)
    for key0 in array_A:
        for key1 in array_A[key0]:
            if array_A[key0][key1] == 0.0:
                array_A[key0][key1] = -3.14e+100
            else:
                array_A[key0][key1] = log(array_A[key0][key1] / count_dic[key0])
    # print(array_A)
    for key in array_B:
        for word in array_B[key]:
            if array_B[key][word] == 0.0:
                array_B[key][word] = -3.14e+100
            else:
                array_B[key][word] = log(array_B[key][word] / count_dic[key])


# 将字典转换成数组
def Dic_Array(array_b):
    tmp = np.empty((4, len(array_b['B'])))
    for i in range(4):
        for j in range(len(array_b['B'])):
            tmp[i][j] = array_b[STATES[i]][list(word_set)[j]]
    return tmp


# Viterbi算法求测试集最优状态序列
def Viterbi(sentence, array_pi, array_a, array_b):
    # 动态规划表
    tab = [{}]
    path = {}
    # 初始化状态
    if sentence[0] not in array_b['B']:
        for state in STATES:
            if state == 'S':
                array_b[state][sentence[0]] = 0
            else:
                array_b[state][sentence[0]] = -3.14e+100

    for state in STATES:
        # 已经求过log所以现在是加法，基于前向算法
        tab[0][state] = array_pi[state] + array_b[state][sentence[0]]
        # tab[t][state]表示时刻t到达state状态的所有路径中，概率最大路径的概率值
        path[state] = [state]
    # {'B': -8.708450118874179, 'M': -3.14e+100, 'E': -3.14e+100, 'S': -12.172927140576377}
    # print(tab[0])
    for i in range(1, len(sentence)):
        tab.append({})
        new_path = {}
        for state in STATES:
            if state == 'B':
                array_b[state]['begin'] = 0
            else:
                array_b[state]['begin'] = -3.14e+100
        for state in STATES:
            if state == 'E':
                array_b[state]['end'] = 0
            else:
                array_b[state]['end'] = -3.14e+100
        for state0 in STATES:
            items = []
            for state1 in STATES:
                # 所有在测试集出现但没有在训练集中出现的字符
                if sentence[i] not in array_b[state0]:
                    if sentence[i - 1] not in array_b[state0]:
                        prob = tab[i - 1][state1] + array_a[state1][state0] + array_b[state0]['end']
                    else:
                        prob = tab[i - 1][state1] + array_a[state1][state0] + array_b[state0]['begin']
                else:
                    # 计算每个字符对应STATES的概率
                    prob = tab[i - 1][state1] + array_a[state1][state0] + array_b[state0][
                        sentence[i]]
                items.append((prob, state1))
            # bset:(prob,state)
            # [(-354.30264436332897, 'B'), (-357.85488557051366, 'M'), (-3.14e+100, 'E'), (-3.14e+100, 'S')]
            # print(items)
            best = max(items)
            tab[i][state0] = best[0]
            new_path[state0] = path[best[1]] + [state0]
        path = new_path

    # path 中存储的是最后一个状态在最佳状态下的前面状态的最佳路径
    prob, state = max([(tab[len(sentence) - 1][state], state) for state in STATES])
    return path[state]


# 根据状态序列进行分词
def tag_seg(sentence, tag):
    word_list = []
    start = -1
    started = False
    if len(tag) != len(sentence):
        return None
    # 语句只有一个字，直接输出
    if len(tag) == 1:
        word_list.append(sentence[0])

    else:
        # 最后一个字状态不是'S'或'E'则修改
        if tag[-1] == 'B' or tag[-1] == 'M':
            if tag[-2] == 'B' or tag[-2] == 'M':
                tag[-1] = 'S'
            else:
                tag[-1] = 'E'

        for i in range(len(tag)):
            if tag[i] == 'S':
                if started:
                    started = False
                    word_list.append(sentence[start:i])
                word_list.append(sentence[i])
            elif tag[i] == 'B':
                if started:
                    word_list.append(sentence[start:i])
                start = i
                started = True
            elif tag[i] == 'E':
                started = False
                word = sentence[start:i + 1]
                word_list.append(word)
            elif tag[i] == 'M':
                continue

    return word_list


if __name__ == '__main__':
    trainset = open('..\data\HMMwordseg-master\HMMseg\CTBtrainingset.txt', encoding='utf-8')   # 读取训练集
    testset = open('..\data\HMMwordseg-master\HMMseg\CTBtestingset.txt', encoding='utf-8')   # 读取测试集

    Init_Array()

    for line in trainset:
        line = line.strip()
        line_num += 1

        word_list = []
        for k in range(len(line)):
            if line[k] == ' ': continue
            word_list.append(line[k])
        # 训练集所有字的集合
        word_set = word_set | set(word_list)

        line = line.split(' ')
        # 这句话的状态序列
        line_state = []

        for i in line:
            line_state.extend(get_tag(i))
        # array_Pi用于计算初始状态分布概率
        array_Pi[line_state[0]] += 1

        for j in range(len(line_state) - 1):
            # count_dic[line_state[j]] += 1   #记录每一个状态的出现次数
            # array_A计算状态转移概率
            array_A[line_state[j]][line_state[j + 1]] += 1

        for p in range(len(line_state)):
            # 记录每一个状态的出现次数
            count_dic[line_state[p]] += 1
            for state in STATES:
                if word_list[p] not in array_B[state]:
                    # 保证每个字都在STATES的字典中
                    array_B[state][word_list[p]] = 0.0
            # array_B用于计算发射概率
            array_B[line_state[p]][word_list[p]] += 1

    Prob_Array()  # 对概率取对数保证精度

    print('参数估计结果')
    print('初始状态分布')
    print(array_Pi)
    print('状态转移矩阵')
    print(array_A)
    print('发射矩阵')
    print(array_B)

    output = ''

    for line in testset:
        line = line.strip()
        tag = Viterbi(line, array_Pi, array_A, array_B)
        seg = tag_seg(line, tag)
        list = ''
        for i in range(len(seg)):
            list = list + seg[i] + ' '
        output = output + list + '\n'
    # print(output)
    # outputfile = open('output.txt', mode='w', encoding='utf-8')
    # outputfile.write(output)
