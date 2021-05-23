import time
import os
import numpy as np

from torch.utils.data import DataLoader, Dataset
import torch.utils.data
import torch
import torch.nn as nn

import continual_learning.utils as utils


class ReplayDataset(Dataset):
    def __init__(self, data, indices, return_item_ix, return_task_labels):
        self.data = data
        self.indices = indices
        self.return_item_ix = return_item_ix
        self.return_task_labels = return_task_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x, y, i, t = self.data[index]

        if self.return_item_ix and self.return_task_labels:
            return x, y, index, t
        elif not self.return_item_ix and self.return_task_labels:
            return x, y, t
        elif self.return_item_ix and not self.return_item_ix:
            return x, y, index
        else:
            return x, y


class IndexSampler(torch.utils.data.Sampler):
    """Samples elements sequentially, always in the same order.
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class ReplayBatchSampler(torch.utils.data.Sampler):
    """
    A sampler that returns a generator object which randomly samples from a list, that holds the indices that are
    eligible for replay.
    The samples that are eligible for replay grows over time, so we want it to be a 'generator' object and not an
    iterator object.
    """

    # See: https://github.com/pytorch/pytorch/issues/683
    def __init__(self, replay_ixs, replay_count_dict, proba_list, task_counter, num_replay_samples,
                 replay_strategy='random'):
        # This makes sure that different workers have different randomness and don't end up returning the same data
        # item!
        self.replay_ixs = replay_ixs  # These are the samples which can be replayed. This list can grow over time.

        np.random.seed(os.getpid())
        self.num_replay_samples = num_replay_samples
        self.replay_strategy = replay_strategy
        self.replay_count_dict = replay_count_dict
        self.proba_list = proba_list
        self.task_counter = task_counter

    def __iter__(self):
        # We are returning a generator instead of an iterator, because the data points we want to sample from, differs
        # every time we loop through the data.
        # e.g., if we are seeing 100th sample, we may want to do a replay by sampling from 0-99 samples. But then,
        # when we see 101th sample, we want to replay from 0-100 samples instead of 0-99.
        while True:
            samples = get_replay_samples(self.replay_strategy, self.num_replay_samples, self.replay_ixs,
                                         replay_count_dict=self.replay_count_dict, proba_list=self.proba_list,
                                         task_counter=self.task_counter)
            yield np.array(samples)

    def __len__(self):
        return 2 ** 64  # Returning a very large number because we do not want it to stop replaying.
        # The stop criteria must be defined in some other manner.

    def update_buffer(self, item_ix, curr_task, val_curr_sample):
        self.replay_ixs.append(item_ix)
        self.task_counter.append(curr_task)
        self.replay_count_dict.append(1)
        if 'proba' in self.replay_strategy:
            self.proba_list.append(val_curr_sample)

    def get_state(self):
        return {'replay_ixs': self.replay_ixs,
                'num_replay_samples': self.num_replay_samples}

    def load_state(self, state):
        replay_ixs = state['replay_ixs']
        while len(self.replay_ixs) > 0:
            self.replay_ixs.pop()
        self.replay_ixs.extend(replay_ixs)
        self.num_replay_samples = state['num_replay_samples']

    def get_replay_ixs(self):
        return self.replay_ixs


def filter_by_task(task_labels, task):
    task_labels = np.array(task_labels)
    ixs = list(np.where(task_labels == task)[0])
    return ixs


def get_replay_data_loader(dataset, idxs, batch_size=32, shuffle=False, sampler=None, batch_sampler=None,
                           return_item_ix=False, return_task_labels=False, num_workers=8):
    if batch_sampler is None and sampler is None:
        if shuffle:
            sampler = torch.utils.data.sampler.SubsetRandomSampler(idxs)
        else:
            sampler = IndexSampler(idxs)
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=batch_size, drop_last=False)

    dataset = ReplayDataset(dataset, idxs, return_item_ix, return_task_labels)
    loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_sampler=batch_sampler)

    return loader


def get_default_replay_loader(dataset, replay_train_ixs, replay_count_dict, proba_list, task_counter,
                              num_replay_samples, replay_strategy='random'):
    replay_batch_sampler = ReplayBatchSampler(replay_train_ixs, replay_count_dict, proba_list, task_counter,
                                              num_replay_samples, replay_strategy=replay_strategy)
    loader = get_replay_data_loader(dataset, replay_train_ixs, batch_size=0, sampler=None,
                                    batch_sampler=replay_batch_sampler, return_item_ix=True, return_task_labels=True)
    return loader


def get_replay_samples(replay_strategy, num_samples, replay_ixs, replay_count_dict=None, proba_list=None,
                       task_counter=None):
    if 'random' in replay_strategy:
        return get_ixs_probas(replay_strategy, list(np.ones(len(replay_ixs))), task_counter, replay_ixs,
                              num_samples, min_val=True)
    elif any(ele in replay_strategy for ele in ['loss_proba']) and proba_list is not None:
        return get_ixs_probas(replay_strategy, proba_list, task_counter, replay_ixs, num_samples, min_val=False)
    elif any(ele in replay_strategy for ele in
             ['time_proba', 'logit_dist_proba', 'confidence_proba', 'margin_proba']) and proba_list is not None:
        return get_ixs_probas(replay_strategy, proba_list, task_counter, replay_ixs, num_samples, min_val=True)
    elif 'replay_count' in replay_strategy and replay_count_dict is not None:
        return get_ixs_probas(replay_strategy, replay_count_dict, task_counter, replay_ixs, num_samples)
    else:
        raise NotImplementedError


def get_ixs_probas(replay_strategy, proba, task_list, replay_ixs, num_samples, min_val=True):
    if 'bal_oversample' in replay_strategy:
        # oversample underrepresented class to get enough samples
        ixs = []
        num_tasks = max(task_list) + 1
        per_task = int(np.floor(num_samples / num_tasks))
        for t in range(num_tasks):
            num = per_task
            i = [j for j in range(len(task_list)) if task_list[j] == t]
            p = [proba[j] for j in i]
            v = [replay_ixs[j] for j in i]
            ixs += list(weighted_choice(replay_strategy, p, v, num, len(i) < num_samples, min_val=min_val))
        return ixs
    else:
        ixs = weighted_choice(replay_strategy, proba, replay_ixs, num_samples, False, min_val=min_val)
        assert (len(ixs) == num_samples)
        return ixs


def weighted_choice(replay_strategy, proba, ixs, num_samples, replace, min_val=True):
    proba = np.array(proba)
    if 'shift_min' in replay_strategy:
        proba = proba + (1 - np.min(proba))
        if min_val:
            proba = 1 / (proba + 1e-7)
        else:
            proba = proba
        proba = proba / np.linalg.norm(proba, ord=1)  # sum to 1
    elif 'random' in replay_strategy:
        return list(np.random.choice(ixs, size=num_samples, replace=replace))
    else:
        if min_val:
            proba = 1 / (proba + 1e-7)
        else:
            proba = proba
        proba = proba / np.linalg.norm(proba, ord=1)  # sum to 1

    return list(np.random.choice(ixs, size=num_samples, replace=replace, p=proba))


def compute_replay_probabilities(replay_strategy, outputs, labels, criterion, start_time=None):
    losses = criterion(outputs, labels)
    if 'loss' in replay_strategy:
        vals = losses
    elif 'logit_dist' in replay_strategy:
        mask = torch.arange(labels.shape[0]).cuda()
        conf = outputs[mask, labels]
        vals = abs(conf)
    elif 'confidence' in replay_strategy:
        mask = torch.arange(labels.shape[0]).cuda()
        softmax = torch.nn.Softmax(dim=1)
        p = softmax(outputs)
        vals = p[mask, labels]
    elif 'margin' in replay_strategy:
        Fs = outputs.clone()
        mask = torch.arange(labels.shape[0]).cuda()
        Fs[mask, labels] = -float('inf')
        s_t = torch.argmax(Fs, dim=1)
        vals = outputs[mask, labels] - outputs[mask, s_t]
    elif 'time' in replay_strategy or 'replay_count' in replay_strategy or 'random' in replay_strategy:
        if start_time is None:
            start_time = time.time()
        vals = torch.empty((labels.shape[0]))
        vals.fill_(start_time)  # base init all at same time
    else:
        raise NotImplementedError
    assert vals.shape[0] == outputs.shape[0]
    return vals.cpu(), losses


def fill_replay_buffer(replay_strategy, model, data_loader, n_s, counter=utils.Counter(), base_epochs=50):
    data_len = len(data_loader.dataset)

    print('\nStoring base init data in replay buffer...')
    start_time = time.time()

    replay_count_dict = []
    proba_list = []
    task_counter = []
    replay_ixs = []
    task = 0  # base init consists of only task 0
    criterion = nn.CrossEntropyLoss(reduction='none')
    model.eval().cuda()

    for batch_ix, (data_batch, batch_y, batch_item_ixs, batch_tasks) in enumerate(data_loader):

        batch_labels = np.atleast_2d(batch_y.numpy().astype(np.int)).transpose()
        batch_item_ixs = np.atleast_2d(batch_item_ixs.numpy().astype(np.int)).transpose()
        labels = torch.from_numpy(batch_labels).long().cuda().squeeze()
        data_batch = data_batch.view(-1, 16, 80, 80)
        model.optimizer.zero_grad()

        outputs = model(data_batch.cuda(), n_s)

        if replay_strategy != 'random':
            vals, _ = compute_replay_probabilities(replay_strategy, outputs, labels, criterion, start_time=start_time)

        # put data into buffer (dictionary)
        for j in range(len(batch_labels)):
            ix = int(batch_item_ixs[j])
            replay_ixs.append(ix)

            if replay_strategy != 'random':
                proba_list.append(float(vals[j].item()))
            replay_count_dict.append(base_epochs)
            task_counter.append(task)
            counter.update()

    print("Completed in {} secs".format(time.time() - start_time))

    assert len(replay_ixs) == data_len
    assert len(replay_count_dict) == data_len
    if replay_strategy != 'random':
        assert len(proba_list) == data_len
    assert len(task_counter) == data_len

    return replay_ixs, replay_count_dict, proba_list, task_counter
