import tensorflow as tf
import pickle
import numpy as np
from tqdm import tqdm
import argparse
from glob import glob
import os
import matplotlib.pyplot as plt

from tfeat import TFeat
from ubc_tfeat import UBCDataset

from eval_metrics import ErrorRateAt95Recall
from preprocess import normalize_batch

parser = argparse.ArgumentParser(description='Tensorflow HardNet')
parser.add_argument('--image_size', default=32, help='The size of the images to process')
parser.add_argument('--num_channels', default=1, help="""The number of channels in the images to process""")
parser.add_argument('--batch_size', default=128, help="""The size of the mini-batch""")
parser.add_argument('--test_name', default='liberty', help="""The name of the dataset used to for training""")
parser.add_argument('--data_dir', default='/home/dong/Documents/3D_Matching/Dataset/UBC', help="""The default path to the patches dataset""")
parser.add_argument('--model_file', default='./logs_tfeat_notredame_var', help="""The filename of the model to evaluate""")
parser.add_argument('--output_name', default='out.txt', help="""The default path to save the descriptor""")
args = parser.parse_args()

enMetricNet = False
is_write = True

models = glob(os.path.join(args.model_file, 'checkpoint-'+'*.meta'))
models.sort()
def run_evaluation():
    # draw_fpr95
    # draw_fpr95(os.path.join(args.model_file, args.test_name+'.txt'))
    # load data
    dataset = UBCDataset(args.data_dir)
    dataset.load_by_name(args.test_name)

    # compute mean and std
    # print('Loading training stats:')
    # file = open('./data/stats_%s.pkl' % args.test_name, 'rb')
    # mean, std = pickle.load(file, encoding='latin1')
    # print('-- Mean: %s' % mean)
    # print('-- Std:  %s' % std)

    # get patches
    patches = dataset._get_patches(args.test_name)
    matches = dataset._get_matches(args.test_name)

    # quick fix in order to have normalized data beforehand
    # patches = normalize_data(patches, mean, std)
    patches -= 0.443728476019
    patches /= 0.20197947209

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
    tfeat_architecture = TFeat(NUM_CHANNELS)
    tfeat_inputs1 = tfeat_architecture.build(inputs1_pl, is_training)
    tfeat_inputs2 = tfeat_architecture.build(inputs2_pl, is_training, reuse=True)

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    # init the graph variables
    # init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # restore session from file
        if is_write:
            save_write = open(os.path.join(args.model_file, args.test_name+'.txt'), 'w')
        for model in models:
            saver.restore(sess, model[:-5])
            # sess.run(init)

            offset = 0
            # dists = np.zeros(matches.shape[0], )
            # labels = np.zeros(matches.shape[0], )
            dists = np.zeros(matches.shape[0] // args.batch_size * args.batch_size, )
            labels = np.zeros(matches.shape[0] // args.batch_size * args.batch_size, )

            dists_m = np.zeros(matches.shape[0] // args.batch_size * args.batch_size, )
            for x in tqdm(range(matches.shape[0] // args.batch_size)):
                # get batch ids
                batch = matches[offset:offset + args.batch_size, :]

                # update the batch offset
                offset += args.batch_size

                input1 = patches[batch[:, 0]]
                input2 = patches[batch[:, 1]]

                # import cv2
                # for i in range(input1.shape[0]):
                #     img1 = np.squeeze(input1[i])
                #     img2 = np.squeeze(input2[i])
                #     img = cv2.hconcat((img1, img2))
                #     print("label: ", batch[i, 2])
                #     cv2.imshow("patch-pairs", img)
                #     cv2.waitKey(0)

                input1 = normalize_batch(input1)
                input2 = normalize_batch(input2)

                # fetch the model with data
                descs1, descs2 = sess.run([tfeat_inputs1, tfeat_inputs2],
                                          feed_dict={
                                              inputs1_pl: input1,
                                              inputs2_pl: input2,
                                              is_training: False
                                          })

                # compute euclidean distances between descriptors
                for i in range(args.batch_size):
                    idx = x * args.batch_size + i
                    dists[idx] = np.linalg.norm(descs1[i, :] - descs2[i, :])
                    labels[idx] = batch[i, 2]

            # compute the false positives rate
            fpr95 = ErrorRateAt95Recall(labels, dists)
            print ('%s-->FRP95: %s' % (model, fpr95))
            distDistribution(labels, dists)
            if is_write:
                save_write.write(str(fpr95)+'\n')
        if is_write:
            save_write.close()

def distDistribution(labels, dists):
    indNeg = np.where(labels==0)
    disNeg = dists[indNeg]
    indPos = np.where(labels==1)
    disPos = dists[indPos]
    plt.hist(disPos, 50, normed=1, facecolor='green', alpha=0.5, label='pos')
    plt.hist(disNeg, 50, normed=1, facecolor='red', alpha=0.5, label='neg')
    plt.legend(loc='upper right')
    plt.ylabel('Number samples')
    plt.show(block=False)

def draw_fpr95(fileName):
    with open(fileName, 'r') as f:
        file = open(fileName, 'r')
        data = file.read()
        nlines = data.split("\n")
    list_fpr95 = [line for line in nlines if len(line) > 0]
    x = range(len(list_fpr95))
    y = np.asarray(list_fpr95)
    plt.figure(1)
    plt.plot(x, y, 'r')
    plt.xlabel('Epoches')
    plt.ylabel('False Positive Rate at 95% Recall')
    plt.show()

def main(_):
    run_evaluation()

if __name__ == '__main__':
    tf.app.run()