import numpy as np
from scipy import signal as scipysignal
from sklearn.cross_decomposition import CCA


class ssvepDetect:
    """
    SSVEP检测器 - 纯FBCCA版本（无数据模板）
    - 原始93.75%准确率版本
    - 5个子频带
    - 3次谐波
    - 通道归一化
    """
    
    def __init__(self, srate, freqs, dataLen):
        self.srate = srate
        self.freqs = freqs
        self.dataLen = dataLen
        self.cca = CCA(n_components=1)
        self._build_reference_templates()
        self._build_subband_filters()
    
    def _build_reference_templates(self):
        """构建含谐波的参考信号模板（基频+2次+3次谐波）"""
        templLen = int(self.dataLen * self.srate)
        sample = np.linspace(0, (templLen - 1) / self.srate, templLen, endpoint=True)
        self.TemplateSet = []
        
        for freq in self.freqs:
            ref_signals = []
            for h in range(1, 4):  # h=1,2,3
                omega = 2 * np.pi * freq * h * sample
                ref_signals.append(np.sin(omega))
                ref_signals.append(np.cos(omega))
            tempset = np.vstack(ref_signals)
            self.TemplateSet.append(tempset)
    
    def _build_subband_filters(self):
        """构建5个FBCCA子频带滤波器"""
        self.subband_filters = []
        nyq = self.srate / 2
        
        subband_ranges = [(6, 40), (8, 50), (10, 60), (14, 70), (18, 90)]
        
        for low, high in subband_ranges:
            Wn = [low / nyq, high / nyq]
            b, a = scipysignal.butter(4, Wn, btype='band')
            self.subband_filters.append((b, a))
        
        self.subband_weights = np.array([1.0, 1.2, 1.0, 0.8, 0.6])
        self.subband_weights /= self.subband_weights.sum()
    
    def _normalize_channels(self, data):
        """通道z-score归一化"""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        std[std == 0] = 1
        return (data - mean) / std
    
    def detect(self, data):
        """FBCCA检测"""
        data = self.pre_filter(data)
        data = self._normalize_channels(data)
        
        n_freqs = len(self.freqs)
        fused_rho = np.zeros(n_freqs)
        
        for subband_idx, (b, a) in enumerate(self.subband_filters):
            filtered_data = scipysignal.filtfilt(b, a, data, axis=1)
            cdata = filtered_data.transpose()
            
            for freq_idx, template in enumerate(self.TemplateSet):
                ctemplate = template.transpose()
                self.cca.fit(cdata, ctemplate)
                datatran, templatetran = self.cca.transform(cdata, ctemplate)
                rho = np.corrcoef(datatran[:, 0], templatetran[:, 0])[0, 1]
                
                weight = self.subband_weights[subband_idx]
                fused_rho[freq_idx] += weight * abs(rho)
        
        return int(np.argmax(fused_rho))
    
    def pre_filter(self, data):
        """预处理滤波: 50Hz陷波 + 6-90Hz带通"""
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        b_notch, a_notch = scipysignal.iircomb(50, 35, ftype='notch', fs=self.srate)
        data_notched = scipysignal.filtfilt(b_notch, a_notch, data, axis=1)
        
        nyq = self.srate / 2
        low, high = 6 / nyq, 90 / nyq
        b_bp, a_bp = scipysignal.butter(4, [low, high], btype='band')
        return scipysignal.filtfilt(b_bp, a_bp, data_notched, axis=1)