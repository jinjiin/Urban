import json
from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict
import numpy as np


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()


def build_time():
    x = np.loadtxt('data/single.csv', delimiter=',', dtype=str)
    title = x[0, :]
    title = np.concatenate((title, ['week_idx']))
    x = x[1:, :]
    print(x.shape)
    print(x[0, 0])
    time = [[0] * 1 for _ in range(x.shape[0])]
    for i in range(x.shape[0]):
        time[i] = i // 7

    x = np.column_stack((x, np.array(time)))
    x = np.row_stack((title, x))
    f = open('/home/cjj/KG/build_/lstm/data/single_week_idx.csv', 'wb')

    np.savetxt(f, x, delimiter=',', fmt='%s')
    f.close()

