import os
import numpy as np
from scipy.interpolate import interp1d


class DataLoader():
    def __init__(self, id_list, base_dir, batch_size, input_shape):
        self.id_list = id_list
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.input_shape = input_shape

        self.indices = np.arange(len(self.id_list))
        self.time_size = int(self.input_shape[0])

    def __len__(self):
        gen_len = len(self.id_list) / self.batch_size
        return int(np.ceil(gen_len))

    def input_len(self):
        return len(self.id_list)

    def getitem(self, idx):
        x_ = np.zeros((self.batch_size, self.time_size), dtype=np.float32)
        inds_ = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        for input_idx, id_idx in enumerate(inds_):
            file_id = self.id_list[id_idx]
            loaded_data = np.load(os.path.join(self.base_dir, file_id+".npy"))
            input_beat_length = len(loaded_data)
            x = np.linspace(0, input_beat_length, num=input_beat_length, endpoint=True)
            f = interp1d(x, loaded_data)
            interpolated_beat = np.linspace(0, input_beat_length, num=self.time_size, endpoint=True)
            processed_sig = self.__amplitude_scale(f(interpolated_beat))
            x_[input_idx,:] = processed_sig
        return x_.reshape(self.batch_size, self.input_shape[0], self.input_shape[1])

    def __amplitude_scale(self, beats):
        minv = min(beats)
        maxv = max(beats)
        midv = (minv+maxv) / 2
        
        beats = beats - midv
        beats = beats * ( 2 / (maxv - minv))
        return beats

