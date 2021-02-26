import torch.utils.data as data
import numpy as np
from torch.utils.data import DataLoader
import torch
from sklearn.preprocessing import robust_scale


class BaseDataset(data.Dataset):
    def __init__(self, mode, flag='pm25'):
        super(BaseDataset).__init__()
        self.mode = mode
        # path = '/lfs1/users/jbyu/cjj/2020data/'
        # path = '/lfs1/users/jbyu/cjj/QA_stage2/data_citys_1920/npy_data/'
        path = 'UrbanModels/Temp/data_citys_1920/npy_data/'
        data = np.load(path + '{}_{}.npy'.format(flag, mode))
        print(data.shape)
        self.min_val, self.max_val = np.min(data), np.max(data)

        
        data = self.norm(data)
        target = data[:, -1]
        data = data[:, :-1]#[:, :, np.newaxis]
        # data = robust_scale(data)
        # data = data[:, :, np.newaxis]
        # data = np.expand_dims(data, axis=-1)
        self.data, self.target = data, target
        self.data = self.data[:, :, np.newaxis]
        print(self.data.shape, self.target.shape)


    def norm(self, data):
        min_val ,max_val = self.min_val, self.max_val
        data = (data-min_val)/(max_val-min_val)
        return data

    def renorm(self, data):
        min_val ,max_val = self.min_val, self.max_val
        return data * (max_val - min_val) + min_val

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # print(self.data[index].shape, self.target[index].shape)
        # print(self.data[index])
        return torch.Tensor(np.array(self.data[index])), torch.Tensor(np.array(self.target[index]))


class dataLoader(DataLoader):
    def __init__(self, mode, flag, batch_size, num_workeres=1, shuffle=False):
        self.dataset = BaseDataset(mode, flag)
        super().__init__(self.dataset, batch_size, shuffle=shuffle, num_workers=num_workeres)


if __name__ == '__main__':
    dataloader = dataLoader('train', 128, 1, False)
    for idx, (data, target) in enumerate(dataloader):
        if idx > 10:
            break
        print(data.shape, target.shape)


