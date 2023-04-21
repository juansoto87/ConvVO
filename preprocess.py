import argparse
from PIL import Image
import scipy.misc
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import os
import random

'''
This piece of code preprocesses the training and test images by creating pairs of resized images, 
and returns a list with the names of the generated images.
For the processing of the training images, a training subset and a validation subset are created.
'''

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=False, help="where the dataset is stored")
parser.add_argument("--resize", type=bool, default= False, help="If true remember to define width and height")
parser.add_argument("--dest_dir", type=str, required=True, help="Where to dump the training data")
parser.add_argument("--dataset", type=str, default= 'train', help="train or test")
parser.add_argument("--test_dir", type=str, required=False, help="Where to dump the test data")
parser.add_argument("--type_prep", type=str, default= 'concat', help="Type of data preparation. concat or stack")
parser.add_argument("--img_height", type=int, default=128, help="image height")
parser.add_argument("--img_width", type=int, default=416, help="image width")
parser.add_argument("--num_threads", type=int, default=4, help="number of threads to use")
args = parser.parse_args()

print(args.dataset_dir)
parent_dir = os.getcwd()
print(parent_dir)

# List of training sequences
train_seq =['00', '01', '02','03','04', '05', '06', '07', '08']

# List of test sequences
test_seq = ['09','10']



def folder_list(seq = None):
    if seq == None:
        for dir in os.listdir(args.dataset_dir):
            print(dir, len(os.listdir(os.path.join(args.dataset_dir,dir))))
    else:
        for dir in seq:
            print(dir, len(os.listdir(os.path.join(args.dataset_dir,dir))))

## Horizontal concatenation
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

## This function concat and save pairs of images, in addition it returns a list with the names of files saved.

def concat(folder, destination):
    np_file = np.empty((0,6))
    for f in folder:
        image_list = []
        folder_path = os.path.join(args.dataset_dir, f)

        for im in os.listdir(folder_path):
            image_list.append(im)
        image_list.sort()
        for i in range(len(image_list)-1):
            # print(image_list[:10])
            # print(image_list[-10:])
            img1 = Image.open(os.path.join(folder_path,image_list[i]))
            img2 = Image.open(os.path.join(folder_path, image_list[i + 1]))
            if args.resize:
                img1 = img1.resize((args.img_width, args.img_height))
                img2 = img2.resize((args.img_width, args.img_height))

            img_concat = get_concat_h(img1,img2).save(os.path.join(destination,(f + "_" + image_list[i])))

    final_list=[]
    for im_name in os.listdir(destination):
        if '.png' in im_name:
            final_list.append(im_name)
    
    final_list.sort()
    with open(os.path.join(args.dest_dir, "final_list.txt"), "w") as output:
        output.write(str(final_list))
    output.close()        
    
    return final_list


def shuffle(list_p, array, random_seed =42):
    random.seed(random_seed)
    random.shuffle(list_p)
    random.seed(random_seed)
    random.shuffle(array)
    # print(list_p)
    # print(array)
    return list_p, array



def train_valid_split(list_names, poses, valid_split=0.3):
    if len(list_names) == poses.shape[0]:
        lt = int((1 - valid_split)*len(list_names))
        list_train = list_names[:lt]
        list_valid = list_names[lt:]
        poses_train = poses[:lt,:]
        poses_valid = poses[lt:,:]
        print(lt, len(list_train), len(list_valid))
        print(poses.shape, poses_train.shape, poses_valid.shape)
        return list_train, list_valid, poses_train, poses_valid
    else:
        print("Not the same length")

def main():
    if args.dataset == 'train':
        print(args.dataset)
        if not os.path.exists(args.dest_dir):
            os.makedirs(args.dest_dir)
        if args.type_prep == 'concat':
            list_p = concat(train_seq, args.dest_dir)
            poses = np.load(os.path.join(args.dest_dir, 'poses.npy'))
            list_p, pose_file = shuffle(list_p, poses)
            with open(os.path.join(args.dest_dir, "shuffle_train_list.txt"), "w") as output:
                output.write(str(list_p))
            output.close()
            np.save(os.path.join(args.dest_dir, 'train_pose'), pose_file)

            list_p_train, list_p_valid, poses_train, poses_valid = train_valid_split(list_p, pose_file)

            with open(os.path.join(args.dest_dir, "train_set_list.txt"), "w") as output:
                output.write(str(list_p_train))
            output.close()
            np.save(os.path.join(args.dest_dir, 'poses_train'), poses_train)
            with open(os.path.join(args.dest_dir, "valid_set_list.txt"), "w") as output:
                output.write(str(list_p_valid))
            output.close()
            np.save(os.path.join(args.dest_dir, 'poses_test'), poses_valid)
    elif args.dataset == 'test':
        if not os.path.exists(args.test_dir):
            os.makedirs(args.test_dir)
        for seq in test_seq:
            print(len(seq))
            dest = os.path.join(args.test_dir, seq)
            if not os.path.exists(dest):
                os.makedirs(dest)
            if args.type_prep == 'concat':
                list_p = concat([str(seq)], dest)
                with open(os.path.join(dest, "seq_"+seq+"_test_list_p.txt"), "w") as output:
                    output.write(str(list_p))
                output.close()


main()


