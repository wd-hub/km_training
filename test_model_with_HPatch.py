import tensorflow as tf
import pickle
import numpy as np
from tqdm import tqdm
import argparse
from glob import glob
import shutil
import os
import cv2
import csv
import matplotlib.pyplot as plt

from Hardnet__ import Hardnet
from ubc import UBCDataset

from eval_metrics import ErrorRateAt95Recall
from preprocess import normalize_batch

parser = argparse.ArgumentParser(description='Tensorflow HardNet')
parser.add_argument('--image_size', default=32, help='The size of the images to process')
parser.add_argument('--num_channels', default=1, help="""The number of channels in the images to process""")
parser.add_argument('--batch_size', default=128, help="""The size of the mini-batch""")
parser.add_argument('--data_dir', default='/home/dong/Documents/hpatches-benchmark/data/hpatches-release', help="""The default path to the patches dataset""")
parser.add_argument('--save_dir', default='/home/dong/Documents/hpatches-benchmark/data/descriptors/hardnet', help="""The filename of the model to evaluate""")
parser.add_argument('--model_file', default='./logs_liberty_default', help="""The filename of the model to evaluate""")
# parser.add_argument('--output_name', default='out.txt', help="""The default path to save the descriptor""")
args = parser.parse_args()

# all types of patches
tps = ['ref','e1','e2','e3','e4','e5','h1','h2','h3','h4','h5',\
       't1','t2','t3','t4','t5']

enMetricNet = False
is_write = False

models = glob(os.path.join(args.model_file, 'checkpoint-'+'*.meta'))   # only load the first model
models.sort()
models = [models[0]]

def load_process(img_path):
    # Load an color image in grayscale
    image = cv2.imread(img_path, 0)
    h, w = image.shape
    # print(h, w)
    n_patches = h // w
    patches = np.ndarray((n_patches, 32, 32, 1), dtype=np.float32)
    for i in range(n_patches):
        patch = image[i * w: (i + 1) * w, 0:w]
        patches[i, :, :, 0] = cv2.resize(patch, (32, 32)) / 255.
    patches -= 0.443728476019
    patches /= 0.20197947209
    return patches

def run_evaluation(folders):

    # quick fix in order to have normalized data beforehand
    # patches = normalize_data(patches, mean, std)
    # patches -= 0.443728476019
    # patches /= 0.20197947209

    with tf.name_scope('inputs'):
        # User defined parameters
        BATCH_SIZE = args.batch_size
        NUM_CHANNELS = args.num_channels
        IMAGE_SIZE = args.image_size

        # Define the input tensor shape
        tensor_shape = (None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)

        # Triplet place holders
        inputs1_pl = tf.placeholder(dtype=tf.float32, shape=tensor_shape, name='inputs1_pl')
        inputs2_pl = tf.placeholder(dtype=tf.float32, shape=tensor_shape, name='inputs2_pl')
        is_training = tf.placeholder(dtype=tf.bool)

    # Creating the architecture
    tfeat_architecture = Hardnet(NUM_CHANNELS)
    tfeat_inputs1 = tfeat_architecture.build(inputs1_pl, is_training)
    # tfeat_inputs2 = tfeat_architecture.build(inputs2_pl, is_training, reuse=True)

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    # init the graph variables
    # init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # restore session from file
        for model in models:
            saver.restore(sess, model[:-5])
            # sess.run(init)
            # load data
            pbar = tqdm(folders)
            for data_folder in pbar:
                pbar.set_description("Computing %s" % data_folder.split('/')[-1])
                save_folder = os.path.join(args.save_dir, data_folder.split('/')[-1])
                if os.path.isdir(save_folder):
                    shutil.rmtree(save_folder)  # remove dir and all contains
                os.makedirs(save_folder)
                for t in tps:
                    im_path = os.path.join(data_folder, t+'.png')
                    patches = load_process(im_path)
                    n_patches = patches.shape[0]
                    descriptors_for_net = np.zeros((n_patches, 128))
                    n_batches = n_patches // BATCH_SIZE + 1
                    for batch_idx in range(n_batches):
                        if batch_idx == n_batches - 1:
                            if (batch_idx + 1) * BATCH_SIZE > n_patches:
                                end = n_patches
                            else:
                                end = (batch_idx + 1) * BATCH_SIZE
                        else:
                            end = (batch_idx + 1) * BATCH_SIZE
                        data_a = patches[batch_idx * BATCH_SIZE: end, :, :, :].astype(np.float32)
                        input1 = normalize_batch(data_a)
                        descs1 = sess.run(tfeat_inputs1, feed_dict={inputs1_pl: input1, is_training: False})
                        descriptors_for_net[batch_idx * BATCH_SIZE: end, :] = descs1
                    # save as .csv
                    np.savetxt(os.path.join(save_folder, t + '.csv'), descriptors_for_net, delimiter=";")

def main(_):
    subfolders = [f.path for f in os.scandir(args.data_dir) if f.is_dir()]
    subfolders.sort()
    run_evaluation(subfolders)

if __name__ == '__main__':
    tf.app.run()