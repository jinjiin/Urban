import torch.utils.data as data
import numpy as np
from torch.utils.data import DataLoader
import torch
from sklearn.preprocessing import scale
import pickle as pk


class BaseDataset(data.Dataset):
    def __init__(self, mode, left_flag, right_flag):
        super(BaseDataset).__init__()
        
        path = '/lfs1/users/jbyu/cjj/QA_stage2/data_citys_1920/npy_data/'
        left_data = np.load(path + '{}_{}.npy'.format(left_flag, mode))
        right_data = np.load(path + '{}_{}.npy'.format(right_flag, mode))

        left_target = left_data[:, -1][:, np.newaxis]
        #left_data = self.norm(left_data[:, :-1])#[:, :, np.newaxis]
        right_target = right_data[:, -1][:, np.newaxis]
        #right_data = self.norm(right_data[:, :-1])#[:, :, np.newaxis]
        left_data = left_data[:, :-1]
        right_data = right_data[:, :-1]
        
        print(left_data.shape)
        length = left_data.shape[0]
        global_input = left_data[:length-length%10, :]
        print(global_input.shape, right_data.shape)
        global_input = np.reshape(global_input, (-1, 10, 23))
        print(global_input.shape)
        global_input = np.swapaxes(global_input, 2, 1)
        self.global_input = np.repeat(global_input, 11, axis=0)


        feature = np.concatenate((left_data[:, :, np.newaxis], right_data[:, :, np.newaxis]), axis=2)
        target = np.concatenate((left_target, right_target), axis=1)

        self.local_input = feature
        self.target = target

        
    def norm(self, data):
        min_val ,max_val = np.min(data), np.max(data)
        data = (data-min_val)/(max_val-min_val)
        return data

    def __len__(self):
        return len(self.local_input)

    def __getitem__(self, index):
        
        return torch.Tensor(self.local_input[index]), \
            torch.Tensor(self.global_input[index]), \
            torch.Tensor(self.target[index])
                # torch.Tensor(self.local_atten_states[index]), \
                    # torch.Tensor(self.global_attn_state[index]), \

class dataLoader(DataLoader):
    def __init__(self, mode, left_flag, right_flag, batch_size, num_workeres=1, shuffle=False):
        self.dataset = BaseDataset(mode, left_flag, right_flag)
        super().__init__(self.dataset, batch_size, shuffle=shuffle, num_workers=num_workeres)


if __name__ == '__main__':
    dataloader = dataLoader('train', 128, 1, False)
    for idx, (local_inputs, global_inputs, labels) in enumerate(dataloader):
        if idx < 10: 
            print(local_inputs.shape, global_inputs.shape, labels.shape)
#torch.Size([128, 24, 2]) torch.Size([128, 24, 26]) torch.Size([128, 1, 48]) torch.Size([128, 13, 2, 24]) torch.Size([128, 1, 2])

