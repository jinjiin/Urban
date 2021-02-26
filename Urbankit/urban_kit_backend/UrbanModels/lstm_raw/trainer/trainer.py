import numpy as np
import torch
from UrbanModels.lstm_raw.base import BaseTrainer
from UrbanModels.lstm_raw.utils import inf_loop
from UrbanUtils.IO import FileUtils


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.config = config
        # self.predict_len = self.config['arch']['args']['predict_len']
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.log_path = "UrbanModels/Temp/LSTM-log.txt"


    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (encoder_input, target) in enumerate(self.data_loader):
            encoder_input, target = encoder_input.to(self.device), target.to(self.device)
            # target = target.squeeze(1)
            self.optimizer.zero_grad()
            output = self.model(encoder_input)
            loss = self.loss(output, target)

            l2_reg = torch.tensor(0.0).to(self.device)
            if self.config['trainer']['l2_regularization']:
                for param in self.model.parameters():
                    l2_reg += torch.norm(param, p=2)
                loss += self.config['trainer']['l2_lambda'] * l2_reg

            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)

            if batch_idx % self.log_step == 0:
                progress_format, progress_value = self._progress(batch_idx)
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} L2_reg: {:.6f}'.format(
                    epoch,
                    progress_format,
                    loss.item(),
                    l2_reg.item()
                ))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                FileUtils.WriteFile("%d,%.2f\n"%(epoch, progress_value), self.log_path, "a")

            if batch_idx == self.len_epoch:
                break

        log = {
            'loss': np.sqrt(total_loss / self.len_epoch),
            'metrics': (total_metrics / self.len_epoch).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(val_log['val_loss'])

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        
        with torch.no_grad():
            for batch_idx, (encoder_input, target) in enumerate(self.valid_data_loader):
                encoder_input, target = encoder_input.to(self.device), target.to(self.device)
                # target = target.squeeze(1)
                output = self.model(encoder_input)

                target = self.valid_data_loader.dataset.renorm(target)
                output = self.valid_data_loader.dataset.renorm(output)
                loss = self.loss(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
               
        print('valid_data_loader length', len(self.valid_data_loader))
        return {
            'val_loss': np.sqrt(total_val_loss / len(self.valid_data_loader)),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist(),
        }

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total), current / total
