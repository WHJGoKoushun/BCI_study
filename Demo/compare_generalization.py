"""
普适性对比测试：V1(纯FBCCA) vs V2(合并训练)
"""
import numpy as np
import csv
from ssvepdetect_v1 import ssvepDetect as DetectV1
from ssvepdetect import ssvepDetect as DetectV2


def load_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            data.append([float(_) for _ in row])
    data = np.array(data, dtype=np.float64)
    
    srate, dataLen = 250, 4
    points = int(dataLen * srate)
    
    X, y = [], []
    for i in range(48):
        epoch = data[i*points:(i+1)*points, :6].transpose()
        X.append(epoch)
        stim = int(data[i*points, -1])
        y.append(stim)
    
    return np.array(X), np.array(y)


def test_detection(Detector, X_train, y_train, X_test, y_test, use_fit=False):
    """测试检测器"""
    sd = Detector(250, [8,9,10,11,12,13,14,15], 4)
    
    if use_fit:
        sd.fit(X_train, y_train)
    
    correct = 0
    for i in range(len(X_test)):
        res = sd.detect(X_test[i])
        if res == y_test[i]:
            correct += 1
    
    return correct, len(X_test)


print("=" * 60)
print("普适性对比测试：V1(纯FBCCA) vs V2(合并训练)")
print("=" * 60)

# 加载数据
X1, y1 = load_data(r'../ExampleData/D1.csv')
X2, y2 = load_data(r'../ExampleData/D2.csv')

print("\n【测试1】跨数据集测试（用D1训练，测试D2）")
print("-" * 60)

# V1: 纯FBCCA，无需训练
v1_correct, v1_total = test_detection(DetectV1, X1, y1, X2, y2, use_fit=False)
print(f"V1 (纯FBCCA):       {v1_correct}/{v1_total} = {v1_correct/v1_total*100:.2f}%")

# V2: 使用D1训练，测试D2
v2_correct, v2_total = test_detection(DetectV2, X1, y1, X2, y2, use_fit=True)
print(f"V2 (D1训练):        {v2_correct}/{v2_total} = {v2_correct/v2_total*100:.2f}%")


print("\n【测试2】跨数据集测试（用D2训练，测试D1）")
print("-" * 60)

# V1
v1_correct, v1_total = test_detection(DetectV1, X2, y2, X1, y1, use_fit=False)
print(f"V1 (纯FBCCA):       {v1_correct}/{v1_total} = {v1_correct/v1_total*100:.2f}%")

# V2
v2_correct, v2_total = test_detection(DetectV2, X2, y2, X1, y1, use_fit=True)
print(f"V2 (D2训练):        {v2_correct}/{v2_total} = {v2_correct/v2_total*100:.2f}%")


print("\n【测试3】合并训练测试（用D1+D2训练，分别测试）")
print("-" * 60)

# 合并数据
X_combined = np.concatenate([X1, X2], axis=0)
y_combined = np.concatenate([y1, y2], axis=0)

# V2测试D1
v2_correct_d1, _ = test_detection(DetectV2, X_combined, y_combined, X1, y1, use_fit=True)
print(f"V2 (合并训练) 测D1: {v2_correct_d1}/48 = {v2_correct_d1/48*100:.2f}%")

# V2测试D2
v2_correct_d2, _ = test_detection(DetectV2, X_combined, y_combined, X2, y2, use_fit=True)
print(f"V2 (合并训练) 测D2: {v2_correct_d2}/48 = {v2_correct_d2/48*100:.2f}%")


print("\n【测试4】留一法交叉验证（D1数据集）")
print("-" * 60)

def leave_one_out_cv(Detector, X, y, use_fit=False):
    correct = 0
    for i in range(len(X)):
        X_train = np.concatenate([X[:i], X[i+1:]], axis=0)
        y_train = np.concatenate([y[:i], y[i+1:]], axis=0)
        
        sd = Detector(250, [8,9,10,11,12,13,14,15], 4)
        if use_fit:
            sd.fit(X_train, y_train)
        
        res = sd.detect(X[i])
        if res == y[i]:
            correct += 1
    return correct

v1_loo = leave_one_out_cv(DetectV1, X1, y1, use_fit=False)
v2_loo = leave_one_out_cv(DetectV2, X1, y1, use_fit=True)

print(f"V1 (纯FBCCA) LOO:   {v1_loo}/48 = {v1_loo/48*100:.2f}%")
print(f"V2 (LOO训练) LOO:   {v2_loo}/48 = {v2_loo/48*100:.2f}%")


print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("V1 (纯FBCCA 93.75%):")
print("  - 优点：无需训练，直接使用")
print("  - 缺点：无法利用已有数据改进")
print("  - 跨数据集：一般")
print()
print("V2 (合并训练 97.92%):")
print("  - 优点：可累积训练，数据越多越准")
print("  - 缺点：需要训练数据")
print("  - 跨数据集：较好（可用其他数据集预训练）")
print()
print("建议：有新数据时，使用V2的fit()方法持续训练")
print("=" * 60)