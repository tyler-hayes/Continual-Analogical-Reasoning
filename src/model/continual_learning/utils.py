import torch
import json
import os
from torch.utils.data import DataLoader

from dataset_utility import dataset
from rpm_solver import RPM_Solver


class Counter:
    """
    A counter to track number of updates.
    """

    def __init__(self):
        self.count = 0

    def update(self):
        self.count += 1


class CMA(object):
    """
    A continual moving average for tracking loss updates.
    """

    def __init__(self):
        self.N = 0
        self.avg = 0.0

    def update(self, X):
        self.avg = (X + self.N * self.avg) / (self.N + 1)
        self.N = self.N + 1


def fix_task_order(task_order):
    l = []
    for i in range(len(task_order)):
        curr = task_order[i]
        if i == 0:
            c = curr[1:]
        elif i == len(task_order) - 1:
            c = curr[:-1]
        else:
            c = curr
        l.append(str(c))
    return l


def build_classifier(model, args, ckpt=None):
    model = eval(model)(args=args)

    if ckpt is None:
        print("Will not resume any checkpoints!")
    else:
        resumed = torch.load(ckpt)
        if 'state_dict' in resumed:
            state_dict_key = 'state_dict'
        else:
            state_dict_key = 'model_state'
        print("Resuming with {}".format(ckpt))
        safe_load_dict(model, resumed[state_dict_key])
    return model


def safe_load_dict(model, old_model_state):
    new_model_state = model.state_dict()
    c = 0
    d = 0
    for name, param in old_model_state.items():
        n = name.split('.')
        beg = n[0]
        end = n[1:]
        if beg == 'module':
            name = '.'.join(end)
        if name not in new_model_state:
            print('%s not found in new model.' % name)
            d += 1
            continue
        c += 1
        if new_model_state[name].shape != param.shape:
            print('Shape mismatch...ignoring %s' % name)
            d += 1
            continue
        else:
            new_model_state[name].copy_(param)
    if c == 0:
        raise AssertionError('No previous ckpt names matched and the ckpt was not loaded properly.')
    print('%d parameters not loaded into new model.' % d)


def get_full_train_dataset(args, return_task_label=False):
    print('\nUsing task ordering: ', args.task_order)
    trn_d = dataset(args, "train", args.task_order, return_item_ix=True, return_task_label=return_task_label)
    return trn_d


def get_data_loaders(args, trn_batch_size, shuffle_train=False, return_task_label=False):
    print('\nUsing task ordering: ', args.task_order)

    # Create datasets and their loaders.
    print('Loading train data...')
    trn_d = [(dataset(args, "train", [t], return_item_ix=True, return_task_label=return_task_label), t) for t in
             args.task_order]
    print('Loading val data...')
    val_d = [(dataset(args, "val", [t], return_item_ix=True, return_task_label=return_task_label), t) for t in
             args.task_order]
    print('Loading test data...')
    tst_d = [(dataset(args, "test", [t], return_item_ix=True, return_task_label=return_task_label), t) for t in
             args.task_order]

    # List with each element containing (dataloader, rpm_type)
    trn_ldr = [(DataLoader(d, batch_size=trn_batch_size, shuffle=shuffle_train, num_workers=args.load_workers), t)
               for
               d, t in trn_d]
    val_ldr = [(DataLoader(d, batch_size=trn_batch_size, shuffle=False, num_workers=args.load_workers), t) for
               d, t in val_d]
    tst_ldr = [(DataLoader(d, batch_size=args.test_batch_size, shuffle=False, num_workers=args.load_workers), t) for
               d, t in tst_d]

    return trn_ldr, val_ldr, tst_ldr


def get_cumulative_replay_data_loaders(args, trn_batch_size, shuffle_train=False, return_task_label=False):
    print('\nUsing task ordering: ', args.task_order)

    # Create datasets and their loaders.
    trn_d = []
    for i in range(len(args.task_order)):
        tasks = [t for t in args.task_order[:i + 1]]
        d = (dataset(args, "train", tasks, return_item_ix=True, return_task_label=return_task_label), tasks[-1])
        trn_d.append(d)  # contains all previous data

    val_d = []
    for i in range(len(args.task_order)):
        tasks = [t for t in args.task_order[:i + 1]]
        d = (dataset(args, "val", tasks, return_item_ix=True, return_task_label=return_task_label), tasks[-1])
        val_d.append(d)  # contains all previous data

    tst_d = [(dataset(args, "test", [t], return_item_ix=True, return_task_label=return_task_label), t) for t in
             args.task_order]

    # List with each element containing (dataloader, rpm_type)
    trn_ldr = [(DataLoader(d, batch_size=trn_batch_size, shuffle=shuffle_train, num_workers=args.load_workers), t)
               for
               d, t in trn_d]
    val_ldr = [(DataLoader(d, batch_size=trn_batch_size, shuffle=False, num_workers=args.load_workers), t) for
               d, t in val_d]
    tst_ldr = [(DataLoader(d, batch_size=args.test_batch_size, shuffle=False, num_workers=args.load_workers), t) for
               d, t in tst_d]

    return trn_ldr, val_ldr, tst_ldr


def update_accuracies(args, main_acc_dict, curr_acc_dict):
    for k, v in curr_acc_dict.items():
        main_acc_dict[k].append(v)
    json.dump(main_acc_dict, open(os.path.join(args.save_dir, 'incremental_raven_accuracies.json'), 'w'))
