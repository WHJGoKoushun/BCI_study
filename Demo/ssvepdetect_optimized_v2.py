import numpy as np
from scipy import signal as scipysignal
from sklearn.cross_decomposition import CCA


class ssvepDetect:
    """
    优化版V2 - 针对D1的9-10Hz错误优化
    """
    
    def __init__(self, srate, freqs, dataLen):
        self.srate = srate
        self.freqs = freqs
        self.dataLen = dataLen
        self.cca = CCA(n_components=1)
        
        # 针对8-15Hz优化的频带
        self._build_optimized_filters()
        self._build_templates()
    
    def _build_optimized_filters(self):
        """优化子频带 - 针对9-10Hz问题"""
        self.subband_filters = []
        nyq = self.srate / 2
        
        # 更精细的频带划分，增强基频检测
        subband_ranges = [
            (7, 20),    # 基频优化
            (8, 22),    # 基频+谐波
            (14, 40),   # 2次谐波
            (20, 60),   # 3次谐波
            (28, 80),   # 4次谐波
        ]
        
        for low, high in subband_ranges:
            Wn = [low / nyq, high / nyq]
            # 使用Chebyshev II型滤波器，更好的阻带衰减
            b, a = scipysignal.cheby2(4, 40, Wn, btype='band')
            self.subband_filters.append((b, a))
        
        # 优化权重
        self.subband_weights = np.array([1.5, 1.3, 1.0, 0.8, 0.6])
        self.subband_weights /= self.subband_weights.sum()
    
    def _build_templates(self):
        """构建参考信号 - 使用3次谐波"""
        templLen = int(self.dataLen * self.srate)
        sample = np.linspace(0, (templLen - 1) / self.srate, templLen, endpoint=True)
        self.sine_templates = []
        
        for freq in self.freqs:
            ref_signals = []
            for h in range(1, 4):  # h=1,2,3
                omega = 2 * np.pi * freq * h * sample
                ref_signals.append(np.sin(omega))
                ref_signals.append(np.cos(omega))
            tempset = np.vstack(ref_signals)
            self.sine_templates.append(tempset)
    
    def _normalize_channels(self, data):
        """通道归一化"""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        std[std == 0] = 1
        return (data - mean) / std
    
    def detect(self, data):
        """检测"""
        data = self.pre_filter(data)
        data = self._normalize_channels(data)
        
        n_freqs = len(self.freqs)
        fused_rho = np.zeros(n_freqs)
        
        # 多频带CCA
        for subband_idx, (b, a) in enumerate(self.subband_filters):
            filtered_data = scipysignal.filtfilt(b, a, data, axis=1)
            cdata = filtered_data.transpose()
            
            for freq_idx, template in enumerate(self.sine_templates):
                ctemplate = template.transpose()
                self.cca.fit(cdata, ctemplate)
                datatran, templatetran = self.cca.transform(cdata, ctemplate)
                
                # 计算相关系数
                rho = np.corrcoef(datatran[:, 0], templatetran[:, 0])[0, 1]
                weight = self.subband_weights[subband_idx]
                fused_rho[freq_idx] += weight * abs(rho)
        
        return int(np.argmax(fused_rho))
    
    def pre_filter(self, data):
        """预处理"""
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # 50Hz陷波
        b_notch, a_notch = scipysignal.iircomb(50, 35, ftype='notch', fs=self.srate)
        data_notched = scipysignal.filtfilt(b_notch, a_notch, data, axis=1)
        
        # 7-90Hz带通
        nyq = self.srate / 2
        low, high = 7 / nyq, 90 / nyq
        b_bp, a_bp = scipysignal.butter(4, [low, high], btype='band')
        return scipysignal.filtfilt(b_bp, a_bp, data_notched, axis=1)