import argparse
import os
from dataset_utility import dataset

"""
First, download the original RAVEN dataset here: http://wellyzhang.github.io/project/raven.html#dataset
Then, downsample the dataset images to 80x80 pixels & save them out as suggested by: https://github.com/SvenShade/Rel-AIR
Note: we do not use the original RAVEN code to create a new dataset of 80x80 pixel images since this degrades image quality

This script requires the tqdm and cv2 packages:
pip install tqdm
pip install opencv-python
"""

if __name__ == '__main__':
    # Define RPM arguments.
    parser = argparse.ArgumentParser(description='model')

    parser.add_argument('--path', type=str, default='/media/tyler/Data/datasets/RAVEN-10000')  # original image location
    parser.add_argument('--new_path', type=str,
                        default='/media/tyler/Data/datasets/RAVEN-10000-small')  # new small image location

    parser.add_argument('--task_order',
                        nargs="*",  # 0 or more values expected => creates a list
                        type=str,
                        default=['cs', 'io', 'lr', 'ud', 'd4', 'd9', '4c'])

    parser.add_argument('--classifier', type=str, default='RPM_Solver')
    parser.add_argument('--img_size', type=int, default=80)
    parser.add_argument('--dataset', type=str, default="raven")
    parser.add_argument('--objects', type=str, default="attention")
    parser.add_argument('--percent', type=int, default=100)
    parser.add_argument('--shuffle_first', type=bool, default=False)

    # downsample images to 80x80 and save them out to new folder
    args = parser.parse_args()
    args.trn_n = args.tst_n = 1

    if not os.path.exists(args.new_path):
        os.mkdir(args.new_path)

    print('\nSaving train images...')
    d_train = dataset(args, 'train', args.task_order)
    d_train.save_80x80(args.new_path)

    print('\nSaving val images...')
    d_val = dataset(args, 'val', args.task_order)
    d_val.save_80x80(args.new_path)

    print('\nSaving test images...')
    d_test = dataset(args, 'test', args.task_order)
    d_test.save_80x80(args.new_path)
