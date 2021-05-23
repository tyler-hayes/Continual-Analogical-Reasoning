import argparse
import torch
import sys
import os
from torch.utils.data import DataLoader
from dataset_utility import dataset
from continual_learning.utils import build_classifier
from continual_learning.main_offline_model import test


def main():
    # Define arguments.
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--model', type=str, default='Rel-Base')
    parser.add_argument('--epochs', type=int, default=251)
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

    parser.add_argument('--ckpt_path', type=str,
                        default='/media/tyler/Data/codes/Continual-Analogical-Reasoning/src/model/continual_learning/analogical_reasoning_results/offline_ckpts')
    parser.add_argument('--load_ckpt_name', type=str, default=None)
    parser.add_argument('--task_order',
                        nargs="*",  # 0 or more values expected => creates a list
                        type=str,
                        default=['cs', 'io', 'lr', 'ud', 'd4', 'd9', '4c'])

    args = parser.parse_args()

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
        args.trn_n *= 1
        args.tst_n *= 1

    # Set additional parameters.
    if not args.multi_gpu:
        torch.cuda.set_device(args.device)

    torch.cuda.cudnn_enabled = True

    # Create datasets and their loaders.
    # trn_d = dataset(args, "train", trn_t, return_item_ix=True)
    # val_d = dataset(args, "val", trn_t, return_item_ix=True)
    tst_d = [(dataset(args, "test", [t], return_item_ix=True), t) for t in tst_t]

    # trn_ldr = DataLoader(trn_d, batch_size=args.batch_size, shuffle=True, num_workers=args.load_workers)
    # val_ldr = DataLoader(val_d, batch_size=args.batch_size, shuffle=False, num_workers=args.load_workers)
    tst_ldr = [(DataLoader(d, batch_size=args.batch_size, shuffle=False, num_workers=args.load_workers), t) for d, t in
               tst_d]

    # load and test best model
    ckpt_best_val = os.path.join(args.ckpt_path, args.load_ckpt_name + '_best.pth')
    print('\nLoading ckpt from: %s' % ckpt_best_val)
    model = build_classifier('RPM_Solver', args, ckpt=ckpt_best_val).cuda()
    test(args, model, tst_ldr)


if __name__ == '__main__':
    main()
