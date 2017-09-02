import numpy as np
import tensorflow as tf
import cv2
import argparse
import os
from glob import glob
from tqdm import tqdm

from Hardnet__ import Hardnet
from preprocess import normalize_depth


parser = argparse.ArgumentParser(description='Tensorflow HardNet')
parser.add_argument('--patch_type', default='dep', help='patch type (gray, rgb, dep, rgbd)')
parser.add_argument('--log_dir', default='./logs', help='folder to output model checkpoints')
args = parser.parse_args()

sequence = "/home/dong/Documents/3D_Matching/Dataset/TUM/rgbd_dataset_freiburg1_desk"
association  = "/home/dong/Documents/3D_Matching/Dataset/TUM/rgbd_dataset_freiburg1_desk/fr1_desk.txt"
model = os.path.join(args.log_dir+'_'+args.patch_type, 'checkpoint-0')
savePath = os.path.join('/home/dong/Documents/3D_Matching/depth_patch_training/pcl_data/data',
           association.split('.')[0].split('/')[-1])

patch_size = 32
num_channels = 1

def loadInfo(file):
    with open(file, 'r') as f:
        data = f.read()
        lines = data.split('\n')
        list = [[line.split(" ")[2], line.split(" ")[3]] for line in lines if len(line)>0]
    return list

def computeDescriptor(sess, dep, tfeat_inputs, inputs_pl, is_training, name):
    # flags = np.zeros([dep.shape[0]*dep.shape[1], 1])
    despMat = np.zeros([dep.shape[0], dep.shape[1], 128])
    pbar = tqdm(range(dep.shape[0]))
    for r in pbar:
        pbar.set_description('%s-->row %s' % (name, r))
        patches = []
        flags = np.zeros(dep.shape[1],dtype=bool)
        for c in range(dep.shape[1]):
            if dep[r, c] == 0 or \
                r < int(patch_size/2) or r > int(dep.shape[0] - 1 - patch_size/2) or\
                c < int(patch_size/2) or c > int(dep.shape[1] - 1 - patch_size/2):
                continue
            else:
                top_left_x = int(c - patch_size / 2)
                top_left_y = int(r - patch_size / 2)
                bottom_right_x = int(c + patch_size / 2)
                bottom_right_y = int(r + patch_size / 2)

                patch = dep[top_left_y:bottom_right_y,
                        top_left_x:bottom_right_x]
                patch = patch.astype(np.float32)
                patch = patch.reshape(1, patch_size, patch_size, 1)
                patches.extend(patch)
                flags[c] = 1
        if not patches:
            continue
        patches = np.asarray(patches)
        patches = normalize_depth(patches)     # preprocessing for depth image
        descp = sess.run(tfeat_inputs, feed_dict={inputs_pl:patches, is_training:False})
        despMat[r][flags] = descp
                # despMat[r,c] = descp
    return despMat

def main(model):
    name_depP = loadInfo(association)
    with tf.name_scope('ipnuts'):
        # Define the input tensor shape
        tensor_shape = (None, patch_size, patch_size, num_channels)
        # Triplet place holders
        inputs_pl = tf.placeholder(dtype=tf.float32, shape=tensor_shape, name='inputs_pl')
        is_training = tf.placeholder(dtype=tf.bool)

    # Creating the architecture
    tfeat_architecture = Hardnet(num_channels)
    tfeat_inputs = tfeat_architecture.build(inputs_pl, is_training)
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model)
        for name, path in name_depP:
            img = os.path.join(sequence, path)
            dep = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            despMat = computeDescriptor(sess, dep, tfeat_inputs, inputs_pl, is_training, name)
            np.save(os.path.join(savePath, name+'.npy'), despMat)

if __name__ == '__main__':
    main(model)