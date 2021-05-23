# INFORMATION ------------------------------------------------------------------------------------------------------- #

# Author:  Steven Spratley
# Date:    19/02/2020
# Purpose: Model for solving RPM problems

# Refactors and extends code in the official RAVEN github repo: https://github.com/WellyZhang/RAVEN
# Original author: Chi Zhang

# IMPORTS ----------------------------------------------------------------------------------------------------------- #

import argparse
import torch
import time
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import math

from dataset_utility import dataset
from rpm_solver import RPM_Solver
from continual_learning.logging_utility import logwrapper, plotwrapper


# SCRIPT ------------------------------------------------------------------------------------------------------------ #

# Define train, validate, and test functions.
def train(args, model, trn_ldr, epoch):
    model.train()
    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    enum = enumerate(trn_ldr) if args.silent else tqdm(enumerate(trn_ldr))
    for batch_idx, (image, target, item_ix) in enum:
        counter += 1
        image = (image[0].cuda(), image[1].cuda(), image[2]) if args.model == 'Rel-AIR' else image.cuda()
        target = target.cuda()

        loss, acc = model.train_(image, target, args.trn_n)
        if math.isnan(loss):
            sys.exit('Uh oh...NaN in loss...')
        loss_all += loss
        acc_all += acc
    if not args.silent:
        print("Epoch {}: Avg Training Loss: {:.6f}".format(epoch, loss_all / float(counter)))
    return loss_all / float(counter)


def validate(args, model, val_ldr):
    model.eval()
    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    for batch_idx, (image, target, item_ix) in enumerate(val_ldr):
        counter += 1
        image = (image[0].cuda(), image[1].cuda(), image[2]) if args.model == 'Rel-AIR' else image.cuda()
        target = target.cuda()
        loss, acc = model.train_(image, target, args.trn_n)
        loss_all += loss
        acc_all += acc
    if not args.silent:
        print("Total Validation Loss: {:.6f}, Acc: {:.4f}".format(loss_all / float(counter), acc_all / float(counter)))
    return loss_all / float(counter), acc_all / float(counter)


def test(args, model, tst_ldr):
    model.eval()
    acc_overall = 0
    for ldr, rpm_type in tst_ldr:
        start = time.time()
        acc_all = 0.0
        counter = 0
        for batch_idx, (image, target, item_ix) in enumerate(ldr):
            counter += 1
            image = (image[0].cuda(), image[1].cuda(), image[2]) if args.model == 'Rel-AIR' else image.cuda()
            target = target.cuda()
            acc_all += model.test_(image, target, args.tst_n)
        average = acc_all / float(counter)
        end = time.time()
        if not args.silent:
            print("Total {} acc: {:.4f}. Tested in {:.2f} seconds.".format(rpm_type, average, end - start))
        acc_overall += average
    average_overall = acc_overall / len(tst_ldr)
    if not args.silent:
        print("\nAverage acc: {:.4f}\n".format(average_overall))
    return average_overall


def full_training(args, model, trn_d, val_d, tst_d, trn_ldr, val_ldr, tst_ldr, log):
    print(
        "\nTrain set: {:>10} problems.\nValid set: {:>10} problems.\n Test set: {:>10} problems.\n\nBeginning {} training." \
            .format(len(trn_d.file_names), len(val_d.file_names), len([i for j in tst_d for i in j[0].file_names]),
                    args.model))
    if args.model == 'Rel-AIR':
        print("Model uses {0} object slots.\n".format(args.trn_n))

    lo_trn_los = 10
    lo_val_los = 10
    hi_val_acc = 0
    hi_tst_acc = 0
    pc = 0  # patience counter

    prog_start = time.time()
    for epoch in range(0, args.epochs):

        tl = train(args, model, trn_ldr, epoch)
        lo_trn_los = tl if tl < lo_trn_los else lo_trn_los
        vl, va = validate(args, model, val_ldr)

        if va > hi_val_acc:
            # if best val accuracy, save the checkpoint
            hi_val_acc = va
            torch.save({'state_dict': model.state_dict(), 'optimizer': model.optimizer.state_dict(), 'val_acc': va,
                        'epoch': epoch}, f=args.ckpt_path + '/{}'.format(args.ckpt_name + '_best.pth'))

        if vl < lo_val_los and not math.isnan(vl):
            lo_val_los = vl
            pc = 0
        else:
            pc += 1
            if pc > args.patience and epoch >= 50 and args.early_stop:  # min of 50 epochs
                ta = test(args, model, tst_ldr)
                if ta > hi_tst_acc:
                    hi_tst_acc = ta

                loss = {'train': tl, 'val': vl}
                acc = {'val': va, 'test': ta}
                log.write_scalars('Loss', loss, epoch)
                log.write_scalars('Accuracy', acc, epoch)
                break
            print(' -- Patience=%d/%d' % (pc, args.patience))

        if math.isnan(vl):
            print('\nUH OH...NAN LOSS!')
            ta = test(args, model, tst_ldr)
            if ta > hi_tst_acc:
                hi_tst_acc = ta

            loss = {'train': tl, 'val': vl}
            acc = {'val': va, 'test': ta}
            log.write_scalars('Loss', loss, epoch)
            log.write_scalars('Accuracy', acc, epoch)
            break

        ta = test(args, model, tst_ldr)
        if ta > hi_tst_acc:
            hi_tst_acc = ta

        loss = {'train': tl, 'val': vl}
        acc = {'val': va, 'test': ta}
        log.write_scalars('Loss', loss, epoch)
        log.write_scalars('Accuracy', acc, epoch)

    prog_end = time.time()
    print("Training completed in {:.2f} minutes.".format((prog_end - prog_start) / 60))
    print("\nlo_trn_los: {:.4f}\nlo_val_los: {:.4f}\nhi_val_acc: {:.4f}\nhi_tst_acc: {:.4f}\n" \
          .format(lo_trn_los, lo_val_los, hi_val_acc, hi_tst_acc))

    # save out ckpt
    torch.save(
        {'state_dict': model.state_dict(), 'optimizer': model.optimizer.state_dict(), 'test_acc': ta, 'epoch': epoch},
        f=args.ckpt_path + '/{}'.format(args.ckpt_name + '_final.pth'))
    print('Successfully saved ckpt to %s' % args.ckpt_path + '/{}'.format(args.ckpt_name))


def main():
    # Define arguments.
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--model', type=str, default='Rel-Base')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=12346)
    parser.add_argument('--load_workers', type=int, default=8)
    parser.add_argument('--path', type=str, default='/media/tyler/Data/datasets/RAVEN-10000-small')
    parser.add_argument('--img_size', type=int, default=80)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--dataset', type=str, default="raven")
    parser.add_argument('--objects', type=str, default="attention")
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--test_every', type=int, default=1)
    parser.add_argument('--percent', type=int, default=100)
    parser.add_argument('--trn_configs', nargs='+', type=str, default="*")
    parser.add_argument('--tst_configs', nargs='+', type=str, default="*")
    parser.add_argument('--silent', type=bool, default=False)
    parser.add_argument('--shuffle_first', type=bool, default=False)

    parser.add_argument('--ckpt_path', type=str, default='/home/tyler/codes/RAVEN/src/model/checkpoints')
    parser.add_argument('--ckpt_name', type=str)
    parser.add_argument('--log', type=str, default='/home/tyler/codes/RAVEN/src/model/tensorboard_logs_ste/')
    parser.add_argument('--task_order',
                        nargs="*",  # 0 or more values expected => creates a list
                        type=str,
                        default=['cs', 'io', 'lr', 'ud', 'd4', 'd9', '4c'])

    # Parameters to optionally perform early stopping based on a patience counter
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--patience', type=int, default=10)  # if loss doesn't change for patience epochs, quit

    args = parser.parse_args()

    print('trn configs ', args.trn_configs)
    print('tst configs ', args.tst_configs)

    if not os.path.exists(args.log):
        os.makedirs(args.log)

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)

    # Define shorthand for RAVEN configurations, and the number of AIR steps to model each.
    #             0     1     2     3     4     5     6
    rpm_type = ['cs', 'io', 'lr', 'ud', 'd4', 'd9', '4c']
    air_steps = {'cs': 1, 'io': 2, 'lr': 2, 'ud': 2, 'd4': 4, 'd9': 9, '4c': 2, '*': 9}

    # Set training and test sets. Check configs are valid.
    trn_t = args.trn_configs
    tst_t = args.tst_configs

    if args.dataset == 'pgm':
        trn_t = tst_t = ['*']
    if args.dataset == 'raven':
        trn_t = ['cs', 'io', 'lr', 'ud', 'd4', 'd9', '4c'] if trn_t == '*' else trn_t
        tst_t = ['cs', 'io', 'lr', 'ud', 'd4', 'd9', '4c'] if trn_t == '*' else trn_t
    elif set(args.trn_configs + args.tst_configs) - set(rpm_type):
        print("One or more RAVEN configurations aren't recognised. Check arguments.")
        sys.exit(1)

    # Set max number of Rel-AIR slots necessary to model the sets specified.
    args.trn_n = args.tst_n = max([air_steps[i] for i in trn_t + tst_t])

    if args.objects == "all":
        args.trn_n *= 2
        args.tst_n *= 2

    # Set additional parameters.
    if not args.multi_gpu:
        torch.cuda.set_device(args.device)

    torch.cuda.cudnn_enabled = True

    # Create datasets and their loaders.
    trn_d = dataset(args, "train", trn_t, return_item_ix=True)
    val_d = dataset(args, "val", trn_t, return_item_ix=True)
    tst_d = [(dataset(args, "test", [t], return_item_ix=True), t) for t in tst_t]

    trn_ldr = DataLoader(trn_d, batch_size=args.batch_size, shuffle=True, num_workers=args.load_workers)
    val_ldr = DataLoader(val_d, batch_size=args.batch_size, shuffle=False, num_workers=args.load_workers)
    tst_ldr = [(DataLoader(d, batch_size=args.batch_size, shuffle=False, num_workers=args.load_workers), t) for d, t in
               tst_d]

    # Initialise model.
    model = RPM_Solver(args).cuda()

    log = logwrapper(args.log)

    full_training(args, model, trn_d, val_d, tst_d, trn_ldr, val_ldr, tst_ldr, log)


if __name__ == '__main__':
    main()

# END SCRIPT -------------------------------------------------------------------------------------------------------- #
