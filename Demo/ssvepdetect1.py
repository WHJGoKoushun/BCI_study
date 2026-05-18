import numpy as np
from scipy import signal as scipysignal
from sklearn.cross_decomposition import CCA

# 本示例使用cca典型相关分析算法对信号进行分析
# cca的基本原理是计算信号和参考信号之间的相关性，和谁的相关性最大则判定受哪个频率的刺激

class ssvepDetect:
    def __init__(self, srate, freqs, dataLen):
        self.cca = CCA(n_components=1)
        self.srate = srate
        templLen = int(dataLen * srate)
        self.TemplateSet = [] # 信号模板
        sample = np.linspace(0, (templLen - 1) / srate, templLen, endpoint=True) # 时间点

        # 每个频率都要构造一个模板
        for freq in freqs:
            _ = 2*np.pi*freq*sample
            sintemp = np.sin(_) # 正弦参考
            costemp = np.cos(_) # 余弦参考
            tempset = np.vstack((sintemp,costemp))
            self.TemplateSet.append(tempset)

    def detect(self, data):  #识别
        data = self.pre_filter(data) # 预处理

        # 将信号和每一组模板进行进行相关性计算，得到相关系数
        p = []
        cdata = data.transpose()
        for template in self.TemplateSet:
            ctemplate = template.transpose()
            # 计算相关系数
            self.cca.fit(cdata,ctemplate)
            datatran, templatetran = self.cca.transform(cdata,ctemplate)
            # coe为最相关的系数
            coe = np.corrcoef(datatran[:,0],templatetran[:,0])[0,1]
            p.append(coe)

        # 找到相关系数最大值的索引，它就是最有可能的目标
        return p.index(max(p))

    def pre_filter(self,data):
        # 将data为chs×N形式，即每一行是一个通道的数据
        # 对原始信号进行预处理，主要是陷波和带通滤波
        b, a = scipysignal.iircomb(50, 35, ftype='notch', fs=self.srate) # 50Hz波
        fs = self.srate / 2
        N, Wn = scipysignal.ellipord([6 / fs, 90 / fs], [2 / fs, 100 / fs], 3, 40)
        b1, a1 = scipysignal.ellip(N, 1, 90, Wn, 'bandpass') # 带通滤波
        filter_data = scipysignal.filtfilt(b1, a1, scipysignal.filtfilt(b, a, data))
        return filter_data