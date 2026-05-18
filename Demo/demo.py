import numpy as np
import csv
from ssvepdetect import ssvepDetect


def load_data(filepath, has_label=True):
    """
    加载数据
    参数:
        filepath: 数据文件路径
        has_label: 是否包含标签（stimID列）
    返回:
        X: 数据 (n_epochs, channels, samples)
        y: 标签 (n_epochs,) 或 None（如果无标签）
    """
    data = []
    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # 读取表头
        
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
                    y.append(-1)  # -1或其他无效值标记为-1
            except:
                # 空值或转换失败
                y.append(-1)
        else:
            y.append(-1)
    
    if has_label and not has_valid_label:
        print("未检测到有效标签（stimID为-1或空值），将按无标签模式处理")
        y = [-1] * n_epochs
    
    return np.array(X), np.array(y) if has_label and has_valid_label else None


if __name__ == '__main__':
    # 实验数据路径（真实数据集时修改此处）
    datapath = r'../ExampleData/D1.csv'

    # 实验参数
    srate = 250  # 采样率250Hz
    dataLen = 4  # 数据长度4秒

    # 加载测试数据
    X_test, y_test = load_data(datapath, has_label=True)
    n_epochs = len(X_test)
    print(f"加载了 {n_epochs} 个样本")

    # 实例化检测器
    sd = ssvepDetect(srate, [8, 9, 10, 11, 12, 13, 14, 15], dataLen)
    
    # 优先加载预训练模型（无需D1D2数据）
    model_loaded = sd.load_model('model.pkl')
    
    # 如果模型加载失败，尝试用D1+D2训练（备用方案）
    if not model_loaded:
        print("尝试使用D1+D2进行训练...")
        try:
            X1, y1 = load_data(r'../ExampleData/D1.csv', has_label=True)
            X2, y2 = load_data(r'../ExampleData/D2.csv', has_label=True)
            
            if y1 is not None and y2 is not None:
                X_combined = np.concatenate([X1, X2], axis=0)
                y_combined = np.concatenate([y1, y2], axis=0)
                sd.fit(X_combined, y_combined)
                print("使用D1+D2训练完成")
            else:
                print("警告：训练数据无有效标签，将使用纯FBCCA模式")
        except Exception as e:
            print(f"训练失败: {e}")
            print("将使用纯FBCCA模式（无数据模板）")

    # 进行检测
    print(f"\n开始检测 {n_epochs} 个样本...")
    results = []
    
    for i in range(n_epochs):
        res = sd.detect(X_test[i])
        results.append(res)
    
    # 如果有标签，计算准确率
    if y_test is not None:
        correct = sum(1 for i in range(n_epochs) if results[i] == y_test[i])
        accuracy = correct / n_epochs
        print(f"\n正确率： {accuracy:.2f}")
        
        # 输出各频率识别详情
        print("\n各频率识别详情：")
        freq_names = [8, 9, 10, 11, 12, 13, 14, 15]
        for freq_idx in range(8):
            total = sum(1 for y in y_test if y == freq_idx)
            if total > 0:
                correct_count = sum(1 for i in range(n_epochs) 
                                  if y_test[i] == freq_idx and results[i] == freq_idx)
                print(f"{freq_names[freq_idx]}Hz: {correct_count}/{total} = {correct_count/total*100:.2f}%")
    else:
        print("\n无标签数据，跳过准确率计算")

    # 输出预测结果（用于提交）
    print(f"\n预测结果（共{n_epochs}个）：")
    for i in range(n_epochs):
        print(f"task{i}预测值：{results[i]}")
    
    # 保存到result.csv（48行，每行一个数字，无末尾空行）
    try:
        with open('result.csv', 'w', newline='') as f:
            # 每行一个预测值，共48行，最后一行不换行
            for i, pred in enumerate(results):
                if i < len(results) - 1:
                    f.write(f"{pred}\n")
                else:
                    f.write(f"{pred}")
        print(f"\n结果已保存到 result.csv")
    except Exception as e:
        print(f"\n保存结果失败: {e}")
