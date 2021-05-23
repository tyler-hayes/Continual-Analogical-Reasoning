import torch.utils.data
import torch.nn as nn
import sys
import time
import math

import continual_learning.utils as utils
import continual_learning.models.PartialReplayUtils as PartialReplayUtils

sys.setrecursionlimit(10000)


class PartialReplay(object):

    def __init__(self, args, classifier='RPM_Solver', num_samples=50, num_channels=1, spatial_feat_dim=80,
                 replay_strategy='random'):

        # make the classifier
        print('\nBuilding classifier for PartialReplay.')
        print('\nUsing replay strategy: ', replay_strategy)
        self.classifier = utils.build_classifier(classifier, args, args.classifier_ckpt)
        self.optimizer = self.classifier.optimizer

        # setup parameters
        self.args = args
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.spatial_feat_dim = spatial_feat_dim
        self.replay_strategy = replay_strategy
        self.replay_loader = None

    def initialize_replay_loader(self, dataset, replay_ixs, replay_count_dict, proba_list, task_counter):
        print('\nInitializing replay loader...')
        self.replay_loader = PartialReplayUtils.get_default_replay_loader(dataset, replay_ixs, replay_count_dict,
                                                                          proba_list, task_counter, self.num_samples,
                                                                          replay_strategy=self.args.replay_strategy)

    def fit_incremental_batch(self, curr_loader, prev_num_samples, verbose=True):
        """
        Fit PartialReplay on samples from a data loader one at a time.
        :param curr_loader: the data loader of new samples to be fit (returns (images, labels, item_ixs)
        :param verbose: true for printing loss to console
        :return: None
        """
        replay_loader_iter = iter(self.replay_loader)

        criterion = nn.CrossEntropyLoss(reduction='none')
        msg = '\rSample %d -- train_loss=%1.6f -- elapsed_time=%d secs'

        # put classifier on GPU and set to train
        classifier = self.classifier.cuda()
        classifier.train()

        start_time = time.time()
        total_loss = utils.CMA()
        iter_num = 0
        for batch_images, batch_labels, batch_item_ixs, batch_task_labels in curr_loader:

            batch_item_ixs += prev_num_samples  # scale by prev_num_samples b/c each task has 0 based item_ixs for raven

            data_batch = batch_images.view(-1, 1, 80, 80)

            # train on one new sample at a time
            for jj in range(len(batch_labels)):
                y = batch_labels[jj]
                curr_task = batch_task_labels[jj].item()
                item_ix = batch_item_ixs[jj].item()
                x = data_batch[jj * 16:jj * 16 + 16]

                # get next data batch from replay iterator
                replay_batch, replay_labels, replay_ixs, replay_task_labels = next(replay_loader_iter)
                replay_batch = replay_batch.view(-1, 1, 80, 80)

                # adaptive batch size
                curr_replay_size = len(replay_batch)

                # gather previous data for replay
                data_codes = torch.empty(
                    ((curr_replay_size + 16), 1, self.spatial_feat_dim, self.spatial_feat_dim))
                data_codes[0:16] = x
                data_codes[16:] = replay_batch
                data_codes = data_codes.view(-1, 16, 80, 80)

                data_labels = torch.empty((curr_replay_size // 16 + 1), dtype=torch.long).cuda()
                data_labels[0] = y
                data_labels[1:] = replay_labels

                # fit on replay mini-batch plus new sample
                if 'proba' in self.replay_strategy:
                    self.optimizer.zero_grad()

                    output = classifier(data_codes.cuda(), self.args.trn_n)
                    losses = criterion(output, data_labels)

                    # eval mode for computing replay scores
                    classifier.eval()
                    with torch.no_grad():
                        output = classifier(data_codes.cuda(), self.args.trn_n)
                        replay_probabilities, _ = PartialReplayUtils.compute_replay_probabilities(self.replay_strategy,
                                                                                                  output, data_labels,
                                                                                                  criterion)

                    # train mode to continue training the model
                    classifier.train()
                    val_curr_sample = replay_probabilities[0].item()

                    loss = losses.mean()
                    if math.isnan(loss.item()):
                        sys.exit('NaN in loss...')
                    loss.backward()
                    self.optimizer.step()
                    loss = loss.item()
                else:
                    loss, _ = classifier.train_(data_codes.cuda(), data_labels, self.args.trn_n)
                    val_curr_sample = 1000  # unused

                total_loss.update(loss)
                if verbose:
                    print(msg % (iter_num, total_loss.avg, time.time() - start_time), end="")

                # update replay probability values for replayed samples
                for ii, v in enumerate(replay_ixs):
                    curr_ix = int(v.item())
                    location = self.replay_loader.batch_sampler.replay_ixs.index(curr_ix)
                    self.replay_loader.batch_sampler.replay_count_dict[location] += 1  # replayed 1 more time
                    if 'proba' in self.replay_strategy:
                        self.replay_loader.batch_sampler.proba_list[location] = float(
                            replay_probabilities[ii + 1].item())

                # Since we have visited item_ix, it is now eligible for replay
                self.replay_loader.batch_sampler.update_buffer(int(item_ix), int(curr_task), float(val_curr_sample))
                iter_num += 1

    def predict(self, tst_ldr, preds=False):
        """
        Perform inference with PartialReplay.
        :param tst_ldr: list of data loaders of test images (images, labels)
        :param preds: true if returning probabilities and true labels
        :return: overall_accuracy, task_accuracies, probabilities, true_labels
        """
        acc_dict = {}
        print('\nComputing predictions...')

        num_samples = 0
        for ldr, _ in tst_ldr:
            num_samples += len(ldr.dataset)
        proba_array = torch.empty((num_samples, 8))  # 8 answer choices
        label_array = torch.empty((num_samples))

        with torch.no_grad():
            self.classifier.eval()
            self.classifier.cuda()

            start_val = 0
            acc_overall = 0
            for ldr, rpm_type in tst_ldr:
                start = time.time()
                acc_all = 0.0
                counter = 0
                for batch_idx, (image, target, _, tasks) in enumerate(ldr):
                    counter += 1
                    image = image.cuda()
                    target = target.cuda()
                    end_val = start_val + len(image)

                    data_batch = image.view(-1, 16, 80, 80)

                    output = self.classifier(data_batch, self.args.tst_n)
                    proba = torch.softmax(output, dim=1)
                    proba_array[start_val:end_val] = proba
                    label_array[start_val:end_val] = target.data
                    pred = output.data.max(1)[1]
                    correct = pred.eq(target.data).cpu().sum().numpy()
                    accuracy = correct * 100.0 / target.size()[0]
                    acc_all += accuracy
                    start_val = end_val

                average = acc_all / float(counter)
                end = time.time()
                if not self.args.silent:
                    print("Total {} acc: {:.4f}. Tested in {:.2f} seconds.".format(rpm_type, average, end - start))
                acc_dict[rpm_type] = average
                acc_overall += average

        average_overall = acc_overall / len(tst_ldr)
        if not self.args.silent:
            print("\nAverage acc: {:.4f}\n".format(average_overall))
        if preds:
            return average_overall, acc_dict, proba_array, label_array
        else:
            return average_overall, acc_dict
