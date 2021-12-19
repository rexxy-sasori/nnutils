import time

from .utils import SuppressPrints
from .. import accuracy_evaluator
from ..utils import AverageMeter, ProgressMeter, EPOCH_FMT_STR


class BaseTrainer:
    def __init__(self, optimizer=None, criterion=None, device=None, lr_scheduler=None, num_epoch=-1, log_update_freq=1):
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.log_update_freq = log_update_freq

        self.batch_time = AverageMeter('Time', ':6.3f')
        self.data_time = AverageMeter('Data Loading', ':6.3f')
        self.data_copy_time = AverageMeter('Data to GPU', ':6.3f')
        self.losses = AverageMeter('Loss', ':.4e')
        self.top1 = AverageMeter('Acc@1', ':6.2f')
        self.num_epoch = num_epoch

    def train(self, model, train_loader, test_loader, mask=None, verbose=True):
        with SuppressPrints(not verbose):
            progress = ProgressMeter(
                len(train_loader),
                [self.batch_time, self.data_time, self.data_copy_time, self.losses, self.top1],
                prefix=EPOCH_FMT_STR.format(0)
            )

            for epoch in range(self.num_epoch):
                cur_lr = self.optimizer.param_groups[0]['lr']
                print('Epoch: {}/{} - LR: {}'.format(epoch, self.num_epoch, cur_lr))

                train_time_start = time.time()

                train_accuracy, train_loss = self.train_single_epoch(epoch, model, progress, train_loader, mask)

                train_time_before_valid = time.time()

                print(
                    "trainers one epoch(s) befor valid: " + "{:.2f}".format(train_time_before_valid - train_time_start))

                benchmarking_acc, benchmarking_loss = accuracy_evaluator.eval(model, self.device, test_loader,
                                                                              self.criterion)
                print('Average train loss: {}, Average train top1 acc: {}'.format(train_loss, train_accuracy))
                print(
                    'Average benchmarking loss: {}, Average benchmarking top1 acc: {}'.format(
                        benchmarking_loss, benchmarking_acc
                    )
                )
                train_time_end = time.time()
                print("trainers one epoch(s): " + "{:.2f}".format(train_time_end - train_time_start))

                # self.lr_scheduler.step()
        return model

    def train_single_epoch(self, epoch, model, progress, train_loader, mask=None):
        progress.prefix = EPOCH_FMT_STR.format(epoch)
        self.batch_time.reset()
        self.data_time.reset()
        self.data_copy_time.reset()
        self.losses.reset()
        self.top1.reset()
        model.train()
        end = time.time()
        for i, data in enumerate(train_loader):
            self.data_time.update(time.time() - end)
            x, y = data
            x = x.to(self.device)
            y = y.to(self.device)
            self.data_copy_time.update(time.time() - end)
            n_data = y.size(0)

            acc, loss = self.single_epoch_compute(model, x, y, mask)

            self.losses.update(loss.item(), n_data)
            self.top1.update(acc.item(), n_data)
            self.batch_time.update(time.time() - end)
            end = time.time()

            if i % self.log_update_freq == 0:
                progress.print2(i)
        train_accuracy, train_loss = self.top1.avg, self.losses.avg
        return train_accuracy, train_loss

    def single_epoch_compute(self, *args, **kwargs):
        return NotImplementedError
