import argparse
import numpy as np
import pickle
import torch
import os
import json
import time

from continual_learning.models.PartialReplayUtils import fill_replay_buffer
from continual_learning.models.PartialReplay import PartialReplay
from continual_learning.models.FineTuneModel import FineTune
from continual_learning.models.DistillationModel import DistillationModel
from continual_learning.models.ElasticWeightConsolidation import EWC_Model
import continual_learning.utils as utils


def save_model_info(args, stream_model, accuracies, probas, true_labels, replay_ixs, replay_count_dict):
    # save replay counts and indices out to files
    if args.model_type == 'partial_replay':
        with open(os.path.join(args.save_dir, 'replay_ixs'), "wb") as fp:
            pickle.dump(replay_ixs, fp)
        with open(os.path.join(args.save_dir, 'replay_counts'), "wb") as fp:
            pickle.dump(replay_count_dict, fp)

    # save probabilities, true labels, and ckpt file
    torch.save(probas, os.path.join(args.save_dir, 'final_probas.pth'))
    torch.save(true_labels, os.path.join(args.save_dir, 'final_true_labels.pth'))

    ta = np.mean(np.array([ta[-1] for ta in accuracies.values()]))
    torch.save({'state_dict': stream_model.classifier.state_dict(), 'optimizer': stream_model.optimizer.state_dict(),
                'acc': ta}, f=os.path.join(args.save_dir, 'final_ckpt.pth'))


def initialize_models_and_loaders(args, stream_model):
    replay_ixs = None
    replay_count_dict = None

    # shuffle data for batch models
    if args.model_type in ['distillation', 'ewc', 'cumulative_replay', 'fine_tune_batch']:
        shuffle_train = True
        print('Using a batch model. Data will be shuffled.')
    else:
        shuffle_train = False
        print('Using a streaming model. Data will not be shuffled.')

    # this is only used by partial replay to color our histogram plots
    if args.model_type in ['partial_replay']:
        task_labels = True
    else:
        task_labels = False

    # make data loaders
    if args.model_type == 'cumulative_replay':
        trn_ldr, val_ldr, tst_ldr = utils.get_cumulative_replay_data_loaders(args, args.train_batch_size,
                                                                             shuffle_train=shuffle_train,
                                                                             return_task_label=task_labels)
    else:
        trn_ldr, val_ldr, tst_ldr = utils.get_data_loaders(args, args.train_batch_size, shuffle_train=shuffle_train,
                                                           return_task_label=task_labels)

    if 'ewc' in args.model_type:
        # initialize omega
        stream_model.initialize(trn_ldr[0][0])
    elif 'partial_replay' in args.model_type:
        print('\nFilling Partial Replay buffer on %s...' % trn_ldr[0][1])
        replay_ixs, replay_count_dict, loss_proba_dict, task_counter = fill_replay_buffer(args.replay_strategy,
                                                                                          stream_model.classifier,
                                                                                          trn_ldr[0][0], args.trn_n)

        full_train_dataset = utils.get_full_train_dataset(args, return_task_label=task_labels)
        stream_model.initialize_replay_loader(full_train_dataset, replay_ixs, replay_count_dict, loss_proba_dict,
                                              task_counter)

    return trn_ldr, val_ldr, tst_ldr, replay_ixs, replay_count_dict


def continual_learning(args, stream_model):
    accuracies = {'cs': [], 'io': [], 'lr': [], 'ud': [], 'd4': [], 'd9': [], '4c': []}

    trn_ldr, val_ldr, tst_ldr, replay_ixs, replay_count_dict = initialize_models_and_loaders(args, stream_model)

    # compute base initialization performance
    test_acc, test_acc_dict, probas, true_labels = stream_model.predict(tst_ldr, preds=True)
    utils.update_accuracies(args, accuracies, test_acc_dict)

    # incremental training
    prev_count = len(trn_ldr[0][0].dataset)  # number of initial samples
    for ix, (curr_trn, curr_val) in enumerate(zip(trn_ldr, val_ldr)):

        # skip first task because it was trained during base initialization
        if ix == 0:
            continue

        # fit model
        curr_trn_ldr, curr_trn_type = curr_trn
        print('\nTraining Task %s.' % curr_trn_type)
        stream_model.fit_incremental_batch(curr_trn_ldr, prev_count)

        # compute test accuracy and save out model information
        test_acc, test_acc_dict, probas, true_labels = stream_model.predict(tst_ldr, preds=True)
        utils.update_accuracies(args, accuracies, test_acc_dict)
        save_model_info(args, stream_model, accuracies, probas, true_labels, replay_ixs, replay_count_dict)

        prev_count += len(curr_trn_ldr.dataset)

    return accuracies


def main():
    # Define RPM arguments.
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--path', type=str, default='/media/tyler/Data/datasets/RAVEN-10000-small')
    parser.add_argument('--model', type=str, default='Rel-Base')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=12346)
    parser.add_argument('--load_workers', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=80)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--dataset', type=str, default="raven")
    parser.add_argument('--objects', type=str, default="attention")
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--val_every', type=int, default=1000)
    parser.add_argument('--test_every', type=int, default=1000)
    parser.add_argument('--percent', type=int, default=100)
    parser.add_argument('--trn_configs', nargs='+', type=str, default="*")
    parser.add_argument('--tst_configs', nargs='+', type=str, default="*")
    parser.add_argument('--silent', type=bool, default=False)
    parser.add_argument('--shuffle_first', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=False)

    # General Continual Learning Parameters
    parser.add_argument('--expt_name', type=str)  # name of the experiment
    parser.add_argument('--save_dir', type=str, required=False)
    parser.add_argument('--classifier', type=str, default='RPM_Solver')
    parser.add_argument('--classifier_ckpt', type=str,
                        default='./checkpoints/single_task_expert_base_init_cs_final.pth')

    parser.add_argument('--task_order',
                        nargs="*",  # 0 or more values expected => creates a list
                        type=str,
                        default=['cs', 'io', 'lr', 'ud', 'd4', 'd9', '4c'])

    # Continual Learning Model
    parser.add_argument('--model_type', type=str, default='partial_replay',
                        choices=['fine_tune', 'distillation', 'ewc', 'cumulative_replay',
                                 'fine_tune_batch', 'partial_replay'])

    # Regularization Model Parameters
    parser.add_argument('--reg_lambda', type=float, default=1.)
    parser.add_argument('--reg_temperature', type=float, default=2.)

    # Partial Replay Model Parameters
    parser.add_argument('--replay_samples', type=int, default=32)  # number of replay samples
    parser.add_argument('--replay_strategy', type=str, default='replay_count_proba_shift_min',
                        choices=['random', 'logit_dist_proba_shift_min', 'confidence_proba_shift_min',
                                 'margin_proba_shift_min', 'time_proba_shift_min', 'loss_proba_shift_min',
                                 'replay_count_proba_shift_min', 'random_bal_oversample',
                                 'logit_dist_proba_shift_min_bal_oversample',
                                 'confidence_proba_shift_min_bal_oversample', 'margin_proba_shift_min_bal_oversample',
                                 'time_proba_shift_min_bal_oversample', 'loss_proba_shift_min_bal_oversample',
                                 'replay_count_proba_shift_min_bal_oversample'
                                 ])

    args = parser.parse_args()

    args.task_order = utils.fix_task_order(args.task_order)
    args.trn_n = args.tst_n = 1  # for Rel-Air and unused here

    if args.save_dir is None:
        args.save_dir = 'streaming_experiments/' + args.expt_name

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("Arguments {}".format(json.dumps(vars(args), indent=4, sort_keys=True)))
    json.dump(vars(args), open(os.path.join(args.save_dir, 'parameter_arguments.json'), 'w'))

    # make and train model
    if args.model_type == 'partial_replay':
        stream_real = PartialReplay(args, classifier=args.classifier, num_samples=args.replay_samples,
                                    replay_strategy=args.replay_strategy)
    elif args.model_type == 'fine_tune':
        stream_real = FineTune(args, classifier=args.classifier)
    elif args.model_type == 'distillation':
        stream_real = DistillationModel(args, classifier=args.classifier, lamda=args.reg_lambda,
                                        num_epochs=args.epochs, temperature=args.reg_temperature)
    elif args.model_type in ['cumulative_replay', 'fine_tune_batch']:
        stream_real = DistillationModel(args, classifier=args.classifier, lamda=args.reg_lambda, fine_tune=True,
                                        num_epochs=args.epochs)
    elif args.model_type == 'ewc':
        stream_real = EWC_Model(args, classifier=args.classifier, ewc_lambda=args.reg_lambda, num_epochs=args.epochs)
    else:
        raise NotImplemented

    start = time.time()
    accuracies = continual_learning(args, stream_real)
    end = time.time()
    print('Total Time (s) ', end - start)

    # save time to json file
    json.dump(end - start, open(os.path.join(args.save_dir, 'total_time.json'), 'w'))


if __name__ == '__main__':
    main()
