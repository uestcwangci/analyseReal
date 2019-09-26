#!/usr/bin/env python
# encoding: utf-8
"""
@Author: WangCi
@Contact: 420197925@qq.com
@Software: PyCharm
@File : onlineAnalyse.py 
@Time: 2019/9/18 13:35
@Desc:
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sqlite3 as sq

# plt可以显示中文
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

def readDB():
    """
    读取数据库信息
    :return:
    finger 指纹库信息
    onlineDict 在线字典 <(x, y), [rssi..]>
    """
    conn = sq.connect('esp_buy_map.db')
    csor = conn.cursor()
    temp = csor.execute('select * from ave_24g')
    finger = {}
    for row in temp:
        finger[(row[0], row[1])] = row[2:]
    conn = sq.connect('bleInfo.db')
    csor = conn.cursor()
    online = csor.execute('select * from ble_table')
    onlineDict = {}
    for row in online:
        x = row[1]
        y = row[2]
        if not onlineDict.__contains__((x, y)):
            onlineDict[(x, y)] = []
        onlineDict[(x, y)].append(row[3:-1])
    conn.close()
    return finger, onlineDict

def myBubbleSort(validFoot, validRssi, A):
    # 把RSSI前A个最大值放在前面
    if A == 1:
        return
    flag = True
    for i in range(A):
        if not flag:
            break
        flag = False
        for j in range(len(validRssi) - 1, i, -1):
            if validRssi[j] > validRssi[j - 1]:
                validRssi[j], validRssi[j - 1] = validRssi[j - 1], validRssi[j]
                validFoot[j], validFoot[j - 1] = validFoot[j - 1], validFoot[j]
                flag = True

def rssiToOne(validRssi, A):
    # 把有效的rssi做归一化处理
    sumWeight = 0.0
    for i in range(A):
        sumWeight += -1.0 / validRssi[i]
    Aweight = np.zeros(len(validRssi))
    for i in range(A):
        Aweight[i] = (-1.0 / validRssi[i]) / sumWeight
    return Aweight

def AWKNN(finger, online, K, A):
    """
    AWKNN算法，根据指纹库估计在线位置
    :param K: 取前K个最近距离
    :param minA: 取前A个有效AP
    :param finger：指纹库
    :param online: 在线数据
    :return realDict: key：真实位置  value：计算出的位置
    """
    realCurDict = {}
    for (realLoc, rssis) in online.items():
        realCurDict[realLoc] = []
        validRssi = []
        validFoot = []
        for rssiArray in rssis:
            foot = 0
            for rssi in rssiArray:
                if rssi != -100:
                    validRssi.append(rssi)
                    validFoot.append(foot)
                foot += 1
            minA = min(A, len(validRssi))
            if minA == 0:
                continue
            # 找到前A个最小的RSSI及下标
            myBubbleSort(validFoot, validRssi, minA)
            Aweight = rssiToOne(validRssi, minA)
            dis = {}
            # 只对前A个进行计算
            for (loc, fingerRssi) in finger.items():
                temp = 0.0
                for a in range(minA):
                    # 计算欧式距离并赋予权值
                    temp += Aweight[a] * np.sqrt(np.sum(np.square(np.array(validRssi) - np.array(fingerRssi[validFoot[a]]))))
                dis[loc] = temp
            # 对距离进行排序
            dis_order = dict(sorted(dis.items(), key=lambda x: x[1], reverse=False))
            disSum = 0.0
            k = 0
            for distance in dis_order.values():
                if k >= K:
                    break
                k += 1
                disSum += 1.0 / distance
            k = 0
            curLoc = []
            for (loc, distance) in dis_order.items():
                if k >= K:
                    break
                k += 1
                curLoc = [i / distance / disSum for i in loc]
            realCurDict[realLoc].append(curLoc)
    return realCurDict


"""
把在线的RSSI转化为坐标
K 
A 
"""
K = 3  # 前K个最近距离
A = 3  # 前A个有效AP
xMax = 5
yMax = 6
finger, online = readDB()
locDict = AWKNN(finger, online, K, A)
ax = plt.gca()
# 画坐标及实际观测点
for loc in finger.keys():
    plt.plot(loc[0], loc[1], 'gx')
plt.plot(list(locDict.keys())[0][0], list(locDict.keys())[0][1], 'r.')
for results in locDict.values():
    for result in results:
        plt.plot(result[0], result[1], 'b+')
# 画误差圆
error = {}
for (real, results) in locDict.items():
    error[real] = []
    for result in results:
        error[real].append(np.sqrt(np.sum(np.square(np.array(real) - np.array(result)))))
    error[real] = np.sort(error[real])
    errorMax = error[real][-1]
    errorAvg = np.average(error[real])
    print('点%s，平均误差=%f米, 最大误差=%f米' % real, errorAvg, errorMax)

plt.title('误差分析')
plt.show()
