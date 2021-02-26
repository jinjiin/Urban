import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.utils.data as data
# from base.base_data_loader import BaseDataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from torch.utils.data import DataLoader
import sys
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

class BaseDataset(data.Dataset):
    def __init__(self, attribute_feature, label_feature, station='aotizhongxin_aq', mode='train', path='data/single.csv', predict_time_len=8, encoder_time_len=24):
        super(BaseDataset).__init__()
        # self.decoder_feature = ["temperature", "pressure", "humidity", "wind_direction", "wind_speed/kph"]
        self.mode = mode
        self.predict_time_len = predict_time_len
        self.encoder_time_len = encoder_time_len

        data = pd.read_csv(path)
        
        group_data = data.groupby('stationId').get_group(station)
        test = group_data.groupby('year').get_group(2018)  # data of 2018 as test
        train_valid = group_data.groupby('year').get_group(2017)  # last 7 day of one month as valid, data of 2017 and 1<day<23 as train

        self.train_data = train_valid[attribute_feature].values
        self.train_label = train_valid[label_feature].values

        self.test_data = test[attribute_feature].values
        self.test_label = test[label_feature].values
        print(self.train_data.shape, self.train_label.shape, self.test_data.shape, self.test_label.shape)

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)-self.encoder_time_len-self.predict_time_len
        elif self.mode == 'valid':
            return len(self.valid_data)-self.encoder_time_len-self.predict_time_len
        else:
            return len(self.test_data)-self.encoder_time_len-self.predict_time_len

    def __getitem__(self, index):
        if self.mode == 'train':
            data = self.train_data
            label = self.train_label
            feature = data[index: index + self.encoder_time_len]
            label_feature = label[index + self.encoder_time_len: index + self.encoder_time_len + self.predict_time_len]
            return feature, label_feature

        else:
            data = self.test_data
            label = self.test_label
            feature = data[index: index + self.encoder_time_len]
            label_feature = label[index + self.encoder_time_len: index + self.encoder_time_len + self.predict_time_len]
            # target = self.test_target[index + self.encoder_time_len: index + self.encoder_time_len + self.predict_time_len]
           
            return feature, label_feature


class dataLoader(DataLoader):
    def __init__(self, attribute_feature, label_feature, station, mode, path, predict_time_len, encoder_time_len, batch_size, num_workeres=1, shuffle=False):
        self.dataset = BaseDataset(attribute_feature, label_feature, station, mode, path, predict_time_len, encoder_time_len)
        super().__init__(self.dataset, batch_size, shuffle=shuffle, num_workers=num_workeres)



if __name__ == '__main__':
    mode = 'train'
    raw_file_path = "/lfs1/users/jbyu/cjj/baseline/data/raw_data/kdd_beijing_17_18_aq_meo_year.csv"

    aq_pos =[
        'fangshan_aq', 'yizhuang_aq', 'shunyi_aq', 'daxing_aq', 'dingling_aq',
       'nongzhanguan_aq', 'yanqin_aq', 'aotizhongxin_aq', 'gucheng_aq',
       'miyun_aq', 'yufa_aq', 'miyunshuiku_aq', 'badaling_aq', 'pingchang_aq',
       'dongsihuan_aq', 'yongledian_aq', 'tongzhou_aq', 'wanliu_aq',
       'huairou_aq', 'fengtaihuayuan_aq', 'nansanhuan_aq', 'yungang_aq',
       'xizhimenbei_aq', 'beibuxinqu_aq', 'liulihe_aq', 'mentougou_aq',
       'donggaocun_aq', 'pinggu_aq', 'yongdingmennei_aq', 'tiantan_aq',
       'dongsi_aq', 'wanshouxigong_aq', 'guanyuan_aq', 'qianmen_aq',
       'zhiwuyuan_aq'
       ]
    # aq_pos = ['fangshan_aq']
    feature_name = 'PM25'
    data_loader = []
    for aq in aq_pos:
        data_loader.append(dataLoader(attribute_feature=feature_name,
                                        label_feature=feature_name,
                                        station=aq,
                                        mode=mode,
                                        path=raw_file_path,
                                        predict_time_len=1,
                                        encoder_time_len=24,
                                        batch_size=1,
                                        num_workeres=0,
                                        shuffle=False
                                        ))

    feature = [[] for _ in range(len(aq_pos))]
    target = [[] for _ in range(len(aq_pos))]
    decoder_input = [[] for _ in range(len(aq_pos))]
    min_len = sys.maxsize
    for i in range(len(aq_pos)):
        for idx, (data, data_target) in enumerate(data_loader[i]):
            data, data_target = torch.squeeze(data, 0).numpy(), torch.squeeze(data_target,0).numpy()

            data = np.expand_dims(data, axis=1)# pm10: 0 idx in c_feature
            data_target = np.expand_dims(data_target, axis=1)
            print(data.shape, data_target.shape)
            decoder_input[i].append(np.expand_dims(data[-1], axis=0))
            feature[i].append(data)
            target[i].append(data_target)
        min_len = min(min_len, len(target[i]))
    for i in range(len(aq_pos)):
        feature[i] = feature[i][:min_len]
        target[i] = target[i][:min_len]
        decoder_input[i] = decoder_input[i][:min_len]

    local_input = np.concatenate(feature, axis=0)         # (bs, 24, 1)
    target = np.concatenate(target, axis=0)                 # (bs, 1)
    decoder_input = np.concatenate(decoder_input, axis=0)   # (bs, 1)
    global_input = np.concatenate(feature, axis=2)
    print(local_input.shape, target.shape, decoder_input.shape, global_input.shape)
    local_atten_states = np.transpose(local_input, (0, 2, 1))
    global_attn_state = np.expand_dims(np.transpose(global_input, (0, 2, 1)), axis=2)


    repeats = local_input.shape[0] // global_input.shape[0]
    global_input = np.repeat(global_input, repeats, axis=0)
    global_attn_state = np.repeat(global_attn_state, repeats, axis=0)

    
    
    print(local_input.shape, decoder_input.shape, target.shape, global_input.shape, local_atten_states.shape, global_attn_state.shape)
    # (101855, 24, 1) (101855, 1, 1) (101855, 1, 1) (7835, 24, 13) (101855, 1, 24) (7835, 13, 1, 24)
      
    all_input = [local_input, decoder_input, target, global_input, local_atten_states, global_attn_state]
    train_local_input, test_local_input, train_decoder_input, test_decoder_input,\
    train_target, test_target, train_global_input, test_global_input,\
    train_local_atten_states, test_local_atten_states, train_global_attn_state, test_global_attn_state\
        = train_test_split(*all_input, test_size=0.2, random_state=0)
    # train_local_input, test_local_input, train_decoder_input, test_decoder_input = train_test_split(local_input, decoder_input, test_size=0.2, random_state=0)
    # train_target, test_target, train_global_input, test_global_input = train_test_split(target, global_input, test_size=0.2, random_state=0)
    # train_local_atten_states, test_local_atten_states, train_global_attn_state, test_global_attn_state = train_test_split(local_atten_states, global_attn_state, test_size=0.2, random_state=0)
    
    processed_path = '/lfs1/users/jbyu/cjj/baseline/GEOMAN_fromstage1/data/processed/35station/{}/'.format(feature_name)
    for i in ['train', 'test']:
        for j in ['local_input', 'decoder_input', 'target', 'global_input', 'local_atten_states', 'global_attn_state']:
            np.save(processed_path + '{}_{}.npy'.format(i, j), eval('{}_{}'.format(i, j)))

