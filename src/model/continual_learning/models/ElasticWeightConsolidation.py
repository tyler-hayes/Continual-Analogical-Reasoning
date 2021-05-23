import numpy as np
import time

import torch
import torch.nn as nn

import continual_learning.utils as utils
from continual_learning.models.EWCutils import EWC


class EWC_Model(object):
    def __init__(self, args, classifier='RPM_Solver', num_epochs=50, ewc_lambda=1.0, supervised_labels=False):

        # make the classifier
        print('\nBuilding classifier for EWC.')
        self.classifier = utils.build_classifier(classifier, args, args.classifier_ckpt)
        self.ewc = EWC(self.classifier, supervised_labels=supervised_labels)

        # make the optimizer
        self.optimizer = self.classifier.optimizer

        self.args = args
        self.num_epochs = num_epochs
        self.ewc_lambda = ewc_lambda

    def initialize(self, train_loader):
        classifier = self.classifier.cuda()
        classifier.train()
        ce_criterion = nn.CrossEntropyLoss()
        self.ewc.precision_update(classifier, ce_criterion, self.args, train_loader)

    def fit_incremental_batch(self, train_loader, prev_num_samples):

        classifier = self.classifier.cuda()
        classifier.train()

        msg = '\rEpoch (%d/%d) -- Iter (%d/%d) -- total_loss=%1.6f (cls_loss=%1.6f ; ewc_loss=%1.6f) -- epoch_time=%d secs'
        num_iters = np.ceil(len(train_loader.dataset) / train_loader.batch_size)
        ce_criterion = nn.CrossEntropyLoss()

        for e in range(self.num_epochs):
            start_time = time.time()
            total_loss = utils.CMA()
            total_cls_loss = utils.CMA()
            total_ewc_loss = utils.CMA()
            iter_num = 0
            for batch_images, batch_labels, batch_item_ixs in train_loader:

                batch_item_ixs += prev_num_samples  # scale by prev_num_samples since each task with 0 based item_ixs

                # fit on data
                self.optimizer.zero_grad()
                output = classifier(batch_images.cuda(), self.args.trn_n)
                cls_loss = ce_criterion(output, batch_labels.cuda())
                ewc_loss = self.ewc.penalty(classifier)
                loss = cls_loss + self.ewc_lambda * ewc_loss
                loss.backward()
                self.optimizer.step()

                total_loss.update(loss.item())
                total_cls_loss.update(cls_loss.item())
                total_ewc_loss.update(ewc_loss.item())
                if self.args.verbose:
                    print(msg % (
                        e + 1, self.num_epochs, iter_num + 1, num_iters, total_loss.avg, total_cls_loss.avg,
                        total_ewc_loss.avg,
                        time.time() - start_time), end="")
                iter_num += 1
            if self.args.verbose:
                print(msg % (
                    e + 1, self.num_epochs, iter_num, num_iters, total_loss.avg, total_cls_loss.avg,
                    total_ewc_loss.avg,
                    time.time() - start_time), end="")

        # update precision matrix
        self.ewc.precision_update(classifier, ce_criterion, self.args, train_loader)

    def predict(self, tst_ldr, preds=False):
        """
        Perform inference with EWC.
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
