from .base_trainer import BaseTrainer
from .. import accuracy_evaluator


class FineTuner(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(FineTuner, self).__init__(*args, **kwargs)

    def single_epoch_compute(self, model, x, y, mask):
        self.optimizer.zero_grad()
        output = model(x)
        loss = self.criterion(output, y)
        accs, _ = accuracy_evaluator.accuracy(output, y, topk=(1,))
        acc = accs[0]
        loss.backward()
        self.optimizer.prune_step(mask)
        return acc, loss
