import numpy as np
import csv
from ssvepdetect import ssvepDetect


def load_data(filepath, has_label=True):
    """加载数据（支持空值和-1处理）"""
    data = []
    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳过表头
        
        for row in csv_reader:
            # 处理空值：空字符串转为NaN
            row_values = []
            for val in row:
                if val == '' or val is None:
                    row_values.append(np.nan)
                else:
                    row_values.append(float(val))
            data.append(row_values)
    data = np.array(data, dtype=np.float64)
    
    srate, dataLen = 250, 4
    points = int(dataLen * srate)
    n_epochs = len(data) // points
    
    X, y = [], []
    has_valid_label = False
    
    for i in range(n_epochs):
        epoch = data[i*points:(i+1)*points, :6].transpose()
        X.append(epoch)
        
        if has_label:
            # 尝试读取标签（处理-1、空值、NaN等情况）
            try:
                stim_val = data[i*points, -1]
                # 检查是否为有效标签（0-7），-1和NaN视为无效
                if not np.isnan(stim_val) and 0 <= stim_val <= 7:
                    y.append(int(stim_val))
                    has_valid_label = True
                else:
                    y.append(-1)
            except:
                y.append(-1)
        else:
            y.append(-1)
    
    if has_label and not has_valid_label:
        print("未检测到有效标签（stimID为-1或空值），将按无标签模式处理")
        y = [-1] * n_epochs
    
    return np.array(X), np.array(y) if has_label and has_valid_label else None


if __name__ == '__main__':
    # 实验数据路径
    datapath = r'../ExampleData/D1.csv'

    # 实验参数
    srate = 250  # 采样率250Hz
    dataLen = 4  # example数据长度为4秒

    # 实例化ssvep检测器
    sd = ssvepDetect(srate, [8, 9, 10, 11, 12, 13, 14, 15], dataLen)
    
    # 如果有多个数据集，可以合并训练以获得更高准确率
    # 例如：合并D1和D2训练
    try:
        X1, y1 = load_data(r'../ExampleData/D1.csv', has_label=True)
        X2, y2 = load_data(r'../ExampleData/D2.csv', has_label=True)
        
        if y1 is not None and y2 is not None:
            X_combined = np.concatenate([X1, X2], axis=0)
            y_combined = np.concatenate([y1, y2], axis=0)
            sd.fit(X_combined, y_combined)
            print("使用D1+D2合并训练")
        else:
            print("训练数据无有效标签，将使用纯FBCCA模式")
    except Exception as e:
        print(f"训练失败: {e}")
        print("仅使用当前数据集（无模板匹配）")

    # 读取测试数据
    X_test, y_test = load_data(datapath, has_label=True)
    n_epochs = len(X_test)
    
    results = []
    stimIDs = []
    corr = []

    # 进行检测
    for i in range(n_epochs):
        res = sd.detect(X_test[i])
        results.append(res)
        
        if y_test is not None:
            stim = y_test[i]
            stimIDs.append(stim)
            if res == stim:
                corr.append(1)
            else:
                corr.append(0)

    # 输出总体正确率
    if y_test is not None:
        print("正确率： %.2f" % (sum(corr) / n_epochs))

        # 输出各频率识别详情
        print("\n各频率识别详情：")
        freq_names = [8, 9, 10, 11, 12, 13, 14, 15]
        for freq_idx in range(8):
            total = 0
            correct_count = 0
            for i in range(n_epochs):
                if stimIDs[i] == freq_idx:
                    total += 1
                    if results[i] == freq_idx:
                        correct_count += 1
            if total > 0:
                print("%dHz: %d/%d = %.2f%%" % (freq_names[freq_idx], correct_count, total, correct_count / total * 100))
    else:
        print("\n无标签数据，跳过准确率计算")

    # results里面包含了所有的预测值
    print()
    for i in range(n_epochs):
        print("task%d预测值：%d" % (i, results[i]))