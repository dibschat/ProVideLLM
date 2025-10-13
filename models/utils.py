from __future__ import print_function

import math

import numpy as np
import torch
import torch.optim as optim

from prettytable import PrettyTable
from torchvision import transforms


def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def write_log(content, epoch, filename):
    if not os.path.exists(filename):
        log_file = open(filename, "w")
    else:
        log_file = open(filename, "a")
    log_file.write("## Epoch %d:\n" % epoch)
    log_file.write("time: %s\n" % str(datetime.now()))
    log_file.write(content + "\n\n")
    log_file.close()


def calc_topk_accuracy(output, target, topk=(1,)):
    """
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels,
    calculate top-k accuracies.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


def calc_mask_accuracy(output, target_mask, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)

    zeros = torch.zeros_like(target_mask).long()
    pred_mask = torch.zeros_like(target_mask).long()

    res = []
    for k in range(maxk):
        pred_ = pred[:, k].unsqueeze(1)
        onehot = zeros.scatter(1, pred_, 1)
        pred_mask = onehot + pred_mask  # accumulate
        if k + 1 in topk:
            res.append(((pred_mask * target_mask).sum(1) >= 1).float().mean(0))
    return res


def neq_load_customized(model, pretrained_dict, verbose=True):
    """load pre-trained model in a not-equal way,
    when new model has been partially modified"""
    model_dict = model.state_dict()
    tmp = {}
    if verbose:
        print("\n=======Check Weights Loading======")
        print("Weights not used from pretrained file:")
        for k, v in pretrained_dict.items():
            if k in model_dict:
                tmp[k] = v
            else:
                print(k)
        print("---------------------------")
        print("Weights not loaded into new model:")
        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
        print("===================================\n")
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate**3)
        lr = (
            eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
        )
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate**steps)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warmup and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (
            args.warm_epochs * total_batches
        )
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def count_parameters(model, layers=True):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )

    # Corrected and optimized one-liner for formatting numbers
    format_number = lambda n: (
        f"{n:.0f}"
        if n < 1_000
        else (
            f"{n/1_000:.1f}K ({n})"
            if n < 1_000_000
            else (
                f"{n/1_000_000:.1f}M ({n})"
                if n < 1_000_000_000
                else f"{n/1_000_000_000:.1f}B ({n})"
            )
        )
    )
    if layers:
        # Adding rows to the table
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                table.add_row([name, parameter.numel()])
        print(table)

    print(f"Total Trainable Params: {format_number(total_params)}")
    return total_params


def save_model(out_path, name, state, is_best=False):
    filename = f"{out_path}/{name}"
    torch.save(state, f"{filename}.pth.tar")

    if is_best:
        filename = "_".join(filename.split("_")[:-1])
        torch.save(state, f"{filename}_best.pth.tar")


# ------------------------------------------------------------------------------------------


### from: https://github.com/pytorch/pytorch/issues/15849#issuecomment-518126031
class _RepeatSampler(object):
    """Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


# https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048
class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    """for reusing cpu workers, to save time"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        # self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
