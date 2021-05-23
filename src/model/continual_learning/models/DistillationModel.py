import time
import sys
from copy import deepcopy
import numpy as np
import math

import torch
import torch.optim as optim
import torch.nn as nn

import continual_learning.utils as utils


class DistillationModel(object):

    def __init__(self, args, classifier='RPM_Solver', num_epochs=50, temperature=2, lamda=1, fine_tune=False):

        # make the classifier
        print('\nBuilding classifier for %s.' % args.model_type)
        self.classifier_prev = utils.build_classifier(classifier, args, args.classifier_ckpt)
        self.classifier = utils.build_classifier(classifier, args, args.classifier_ckpt)
        self.freeze_old_model()

        # make the optimizer
        self.classifier.optim_list = [
            {'params': filter(lambda p: p.requires_grad, self.classifier.parameters()), 'lr': args.lr},
        ]
        self.optimizer = optim.Adam(self.classifier.optim_list)

        # setup parameters
        self.args = args
        self.num_epochs = num_epochs
        self.temperature = temperature
        self.lamda = lamda  # weighs the distillation loss
        self.fine_tune = fine_tune

    def freeze_old_model(self):
        self.classifier_prev.load_state_dict(deepcopy(self.classifier.state_dict()))
        for param in self.classifier_prev.parameters():
            param.requires_grad = False
        return self.classifier_prev.eval()

    def fit_incremental_batch(self, curr_loader, prev_num_samples):

        # put classifiers on GPU and set current network to train
        classifier_prev = self.classifier_prev.cuda()
        classifier_prev.eval()
        classifier_curr = self.classifier.cuda()
        classifier_curr.train()

        msg = '\rEpoch (%d/%d) -- Iter (%d/%d) -- total_loss=%1.6f (cls_loss=%1.6f ; dist_loss=%1.6f) -- epoch_time=%d secs'
        num_iters = np.ceil(len(curr_loader.dataset) / curr_loader.batch_size)
        ce_criterion = nn.CrossEntropyLoss()

        for e in range(self.num_epochs):
            start_time = time.time()
            total_loss = utils.CMA()
            total_cls_loss = utils.CMA()
            total_dist_loss = utils.CMA()
            iter_num = 0
            for batch_images, batch_labels, batch_item_ixs in curr_loader:

                batch_item_ixs += prev_num_samples  # scale by prev_num_samples since each task with 0 based item_ixs

                # fit on data
                self.optimizer.zero_grad()
                cls_loss, dist_loss = self.compute_loss(classifier_prev, classifier_curr, batch_images.cuda(),
                                                        batch_labels.cuda(), ce_criterion)
                loss = self.lamda * (self.temperature ** 2) * dist_loss + cls_loss
                loss.backward()
                self.optimizer.step()

                total_loss.update(loss.item())
                total_cls_loss.update(cls_loss.item())
                total_dist_loss.update(dist_loss.item())
                if self.args.verbose:
                    if math.isnan(total_loss.avg):
                        sys.exit('Warning: NaN loss...killing script')

                    print(msg % (
                        e + 1, self.num_epochs, iter_num + 1, num_iters, total_loss.avg, total_cls_loss.avg,
                        total_dist_loss.avg,
                        time.time() - start_time), end="")
                iter_num += 1
            if self.args.verbose:
                print(msg % (
                    e + 1, self.num_epochs, iter_num, num_iters, total_loss.avg, total_cls_loss.avg,
                    total_dist_loss.avg,
                    time.time() - start_time), end="")

        # copy current parameters into previous model for distillation
        self.freeze_old_model()

    def compute_loss(self, prev_model, curr_model, images, targets, ce_criterion):
        """
        Computes the distillation loss (cross-entropy).
        """
        y = curr_model(images, self.args.trn_n)
        cls_loss = ce_criterion(y, targets)
        if self.fine_tune:
            return cls_loss, torch.tensor(0)
        else:
            teacher_scores = prev_model(images, self.args.trn_n).detach()
            dist_loss = self.MultiClassCrossEntropy(y, teacher_scores, self.temperature)
        return cls_loss, dist_loss

    def MultiClassCrossEntropy(self, logits, labels, T):
        outputs = torch.log_softmax(logits / T, dim=1)
        labels = torch.softmax(labels / T, dim=1)
        outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
        outputs = -torch.mean(outputs, dim=0, keepdim=False)
        return outputs

    def predict(self, tst_ldr, preds=False):
        """
        Perform inference with DistillationModel.
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
                for batch_idx, (image, target, _) in enumerate(ldr):
                    counter += 1
                    image = image.cuda()
                    target = target.cuda()
                    end_val = start_val + len(image)

                    output = self.classifier(image, self.args.tst_n)
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
