import numpy as np
from scipy import signal as scipysignal
from sklearn.cross_decomposition import CCA


class ssvepDetect:
    """
    优化版SSVEP检测器 - 不训练，纯FBCCA
    尝试各种优化策略达到100%准确率
    """
    
    def __init__(self, srate, freqs, dataLen):
        self.srate = srate
        self.freqs = freqs
        self.dataLen = dataLen
        self.cca = CCA(n_components=1)
        
        # 策略1: 优化频带选择（针对8-15Hz定制）
        self._build_optimized_filters()
        
        # 策略2: 使用4次谐波（增加高频信息）
        self._build_enhanced_templates()
    
    def _build_optimized_filters(self):
        """优化子频带选择 - 针对8-15Hz SSVEP"""
        self.subband_filters = []
        nyq = self.srate / 2
        
        # 策略1a: 更精确的频带划分
        # 针对每个目标频率优化
        subband_ranges = [
            (6, 18),    # 基频范围
            (8, 24),    # 基频+2次谐波
            (12, 36),   # 2次+3次谐波
            (16, 48),   # 3次+4次谐波
            (20, 60),   # 高频谐波
        ]
        
        for low, high in subband_ranges:
            Wn = [low / nyq, high / nyq]
            b, a = scipysignal.butter(6, Wn, btype='band')  # 6阶滤波器
            self.subband_filters.append((b, a))
        
        # 策略1b: 优化的权重分配
        self.subband_weights = np.array([1.2, 1.5, 1.0, 0.8, 0.5])
        self.subband_weights /= self.subband_weights.sum()
    
    def _build_enhanced_templates(self):
        """增强参考信号 - 使用4次谐波"""
        templLen = int(self.dataLen * self.srate)
        sample = np.linspace(0, (templLen - 1) / self.srate, templLen, endpoint=True)
        self.sine_templates = []
        
        for freq in self.freqs:
            ref_signals = []
            # 策略2: 使用4次谐波（原为3次）
            for h in range(1, 5):  # h=1,2,3,4
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
    
    def _adaptive_weighting(self, rhos):
        """
        策略3: 自适应加权
        根据信噪比动态调整权重
        """
        # 如果某个频率的相关系数明显高于其他，增加其权重
        max_rho = np.max(rhos)
        min_rho = np.min(rhos)
        diff = max_rho - min_rho
        
        if diff > 0.1:  # 如果区分度足够大
            # 增强最大值的权重
            adaptive_weights = np.ones_like(rhos)
            max_idx = np.argmax(rhos)
            adaptive_weights[max_idx] = 1.5
            return rhos * adaptive_weights
        return rhos
    
    def detect(self, data):
        """
        检测SSVEP频率 - 多策略融合
        """
        # 预处理
        data = self.pre_filter(data)
        data = self._normalize_channels(data)
        
        n_freqs = len(self.freqs)
        fused_rho = np.zeros(n_freqs)
        
        # FBCCA分数计算
        for subband_idx, (b, a) in enumerate(self.subband_filters):
            filtered_data = scipysignal.filtfilt(b, a, data, axis=1)
            cdata = filtered_data.transpose()
            
            for freq_idx, template in enumerate(self.sine_templates):
                ctemplate = template.transpose()
                self.cca.fit(cdata, ctemplate)
                datatran, templatetran = self.cca.transform(cdata, ctemplate)
                
                # 策略4: 使用多种相关系数度量
                rho1 = np.corrcoef(datatran[:, 0], templatetran[:, 0])[0, 1]
                
                # 策略5: 计算特征值比值（信噪比估计）
                try:
                    X = datatran[:, 0].reshape(-1, 1)
                    Y = templatetran[:, 0].reshape(-1, 1)
                    U, s, Vt = np.linalg.svd(np.dot(X.T, Y), full_matrices=False)
                    snr_ratio = s[0] / (np.sum(s) + 1e-10)
                except:
                    snr_ratio = 0
                
                # 融合相关系数和信噪比估计
                rho = abs(rho1) + 0.1 * snr_ratio
                
                weight = self.subband_weights[subband_idx]
                fused_rho[freq_idx] += weight * rho
        
        # 策略3: 自适应加权
        fused_rho = self._adaptive_weighting(fused_rho)
        
        return int(np.argmax(fused_rho))
    
    def pre_filter(self, data):
        """预处理滤波"""
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # 50Hz陷波
        b_notch, a_notch = scipysignal.iircomb(50, 35, ftype='notch', fs=self.srate)
        data_notched = scipysignal.filtfilt(b_notch, a_notch, data, axis=1)
        
        # 6-90Hz带通
        nyq = self.srate / 2
        low, high = 6 / nyq, 90 / nyq
        b_bp, a_bp = scipysignal.butter(4, [low, high], btype='band')
        return scipysignal.filtfilt(b_bp, a_bp, data_notched, axis=1)