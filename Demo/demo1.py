import numpy as np
import csv
from ssvepdetect1 import ssvepDetect

if __name__ == '__main__':
    # 实验数据路径
    datapath = r'../ExampleData/D1.csv'

    # 实验参数
    srate = 250  # 采样率250Hz
    dataLen = 4  # example数据长度为4秒

    # 实例化ssvep检测器
    sd = ssvepDetect(srate, [8, 9, 10, 11, 12, 13, 14, 15], dataLen)

    # 读取数据（支持空值处理）
    data = []
    with open(datapath, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳过表头

        for row in csv_reader:
            # 处理空值：空字符串转为NaN
            rowvalue = []
            for val in row:
                if val == '' or val is None:
                    rowvalue.append(np.nan)
                else:
                    rowvalue.append(float(val))
            data.append(rowvalue)

    data = np.array(data, dtype=np.float64)

    points = int(dataLen * srate)
    results = []
    stimIDs = []
    corr = []

    # 每个数据中都有48个片段
    for i in range(48):
        epoch = data[i*points:(i+1)*points, :6]  # 把这一段的6个通道信号片段取出
        epoch = epoch.transpose()  # 以行来组织，每一行是一个通道的数据
        res = sd.detect(epoch)  # 识别，得到的结果res取值范围是0-7
        results.append(res)
        
        # 读取标签（处理-1、空值、NaN等情况）
        try:
            stim_val = data[i*points, -1]
            if not np.isnan(stim_val) and 0 <= stim_val <= 7:
                stim = int(stim_val)
                stimIDs.append(stim)
                corr.append(1 if res == stim else 0)
            else:
                stimIDs.append(-1)
                corr.append(0)
        except:
            stimIDs.append(-1)
            corr.append(0)

    # 计算正确率（只统计有效标签）
    valid_labels = [c for c in corr if c >= 0]  # 过滤掉无效标签
    if len(valid_labels) > 0:
        print("正确率： %.2f" % (sum(valid_labels) / len(valid_labels)))
    else:
        print("无有效标签，无法计算正确率")

    # results里面包含了所有的预测值
    for i in range(48):
        print("task%d预测值：%d" % (i, results[i]))