import argparse
import collections
import torch
import torch.nn as nn
import UrbanModels.lstm_raw.data_loader.data_loaders as module_data
from UrbanModels.lstm_raw.model import loss as module_loss
from UrbanModels.lstm_raw.model import metric as module_metric
from UrbanModels.lstm_raw.model import model as module_arch
from UrbanModels.lstm_raw.parse_config import ConfigParser
from UrbanModels.lstm_raw.trainer import Trainer
from UrbanUtils.IO import FileUtils


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.initialize('train_loader', module_data)
    valid_data_loader = config.initialize('valid_loader', module_data)

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()
    FileUtils.WriteFile("%d,%.2f\n" % (100, 0.0), "UrbanModels/Temp/LSTM-log.txt", "a")


def run():
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='./UrbanModels/lstm_raw/config_pm25.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
    ]
    config = ConfigParser(args, options)
    main(config)
