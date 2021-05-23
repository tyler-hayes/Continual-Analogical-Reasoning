import time
import sys
import torch
import continual_learning.utils as utils

sys.setrecursionlimit(10000)


class FineTune(object):

    def __init__(self, args, classifier='RPM_Solver', freeze_bn=True):

        # make the classifier
        print('\nBuilding classifier for Fine-Tune Stream.')
        self.classifier = utils.build_classifier(classifier, args, args.classifier_ckpt)

        if freeze_bn:
            # freeze batch norm parameters since we are updating one sample at a time
            c = 0
            for module in self.classifier.modules():
                classname = module.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    for param in module.parameters():
                        param.requires_grad = False
                        c += 1
            print('Froze %d BatchNorm parameters.' % c)

        self.optimizer = self.classifier.optimizer
        self.args = args

    def fit_incremental_batch(self, curr_loader, prev_num_samples, verbose=True):
        """
        Fit Fine-Tune on samples from a data loader one at a time.
        :param curr_loader: the data loader of new samples to be fit (returns (images, labels, item_ixs)
        :param verbose: true for printing loss to console
        :return: None
        """

        # put classifier on GPU and set to train
        classifier = self.classifier.cuda()
        classifier.train()

        msg = '\rSample %d -- train_loss=%1.6f -- elapsed_time=%d secs'

        start_time = time.time()
        total_loss = utils.CMA()
        iter_num = 0
        for batch_images, batch_labels, batch_item_ixs in curr_loader:

            batch_item_ixs += prev_num_samples  # scale by prev_num_samples since each task with 0 based item_ixs

            data_batch = batch_images.view(-1, 1, 80, 80)

            # train Fine-Tune on one new sample at a time
            for jj in range(len(batch_labels)):
                y = batch_labels[jj]
                x = data_batch[jj * 16:jj * 16 + 16]

                # fit on replay mini-batch plus new sample
                loss, _ = classifier.train_(x.unsqueeze(0).cuda(), y.unsqueeze(0).cuda(), self.args.trn_n)

                total_loss.update(loss)
                if verbose:
                    print(msg % (iter_num, total_loss.avg, time.time() - start_time), end="")
                iter_num += 1

    def predict(self, tst_ldr, preds=False):
        """
        Perform inference with Fine-Tune.
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
