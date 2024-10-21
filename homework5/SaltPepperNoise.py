import numpy as np
from PIL import Image


class SaltPepperNoise:
    def __init__(self, snr, p):
        """
        :param snr: signal-to-noise ratio
        :param p: The probability of applying transformation to pixels
        """
        assert 0 <= snr <= 1, 'SNR is out of range'
        assert 0 <= p <= 1, 'Probability of applying transformation to pixels is out of range'
        super(SaltPepperNoise, self).__init__()
        self.snr = snr
        self.p = p


    def __call__(self, img):
        if np.random.uniform(0, 1) >= self.p:
            return img
        img_info = np.array(img).copy()
        h, w, c = img_info.shape
        signal = self.snr
        noise = 1 - signal
        mask = np.random.choice([0, 1, 2], size=(h, w, 1), p=[signal, noise/2, noise/2])
        mask = np.repeat(mask, c, axis=2)
        img_info[mask == 1] = 0
        img_info[mask == 2] = 255
        return Image.fromarray(img_info)
        pass


    def __repr__(self):
        return f"{self.__class__.__name__}(snr={self.snr}, p={self.p})"