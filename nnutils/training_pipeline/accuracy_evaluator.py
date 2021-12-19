import torch

from .utils import AverageMeter


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        if type(output) is tuple:
            _, _, output = output
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res, pred[0, :]


def eval(model, device, data_loader, criterion, print_acc=True):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    losses = AverageMeter('Loss', ':.4e')

    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            n_data = y.size(0)

            output = model(x)
            loss = criterion(output, y)
            accs, predictions = accuracy(output, y, topk=(1,))
            acc = accs[0]

            top1.update(acc.item(), n_data)
            losses.update(loss.item(), n_data)

        if print_acc:
            print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg
