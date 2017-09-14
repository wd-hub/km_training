import tensorflow as tf
import pickle
import numpy as np
from tqdm import tqdm
import argparse
from glob import glob
import os

from Hardnet__ import Hardnet
from tum import TUMDataset

from eval_metrics import ErrorRateAt95Recall
from preprocess import normalize_batch
from utils.config import get_params

parser = argparse.ArgumentParser(description='Tensorflow HardNet')
parser.add_argument('--patch_type', default='dep', help='patch type (gray, rgb, dep, rgbd)')
parser.add_argument('--model_file', default='./logs_depth', help="""The filename of the model to evaluate""")
parser.add_argument('--train_dir', default='/home/dong/Documents/3D_Matching/Dataset/TUM',
                    help="""The name of the dataset used to for training""")
args = parser.parse_args()

params = get_params(args.patch_type)
models = glob(os.path.join(args.model_file,'checkpoint'+'*.meta'))
models.sort()
test_seq = ['rgbd_dataset_freiburg3_long_office_household']
def run_evaluation():

    # load data
    dataset = TUMDataset(args.train_dir, params)
    dataset.load_by_name('testing', test_seq, params)
    patches_test = dataset._get_patches('testing')

    # get matches for evaluation
    matches_test = dataset._get_matches()
    with tf.name_scope('inputs'):
        # User defined parameters
        BATCH_SIZE = params['batch_size']
        NUM_CHANNELS = params['num_channels']
        PATCH_SIZE = params['patch_size']

        # Define the input tensor shape
        tensor_shape = (None, PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS)

        # Triplet place holders
        inputs1_pl = tf.placeholder(
            dtype=tf.float32, shape=tensor_shape, name='inputs1_pl')
        inputs2_pl = tf.placeholder(
            dtype=tf.float32, shape=tensor_shape, name='inputs2_pl')
        is_training = tf.placeholder(dtype=tf.bool)

    # Creating the architecture
    tfeat_architecture = Hardnet(NUM_CHANNELS)
    tfeat_inputs1 = tfeat_architecture.build(inputs1_pl, is_training)
    tfeat_inputs2 = tfeat_architecture.build(inputs2_pl, is_training, reuse=True)

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    # init the graph variables
    # init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # restore session from file
        # model_set = ['./logs/liberty/checkpoint_liberty-1', './logs/liberty/checkpoint_liberty-2']
        for model in models:
            saver.restore(sess, model[:-5])
            # sess.run(init)

            offset = 0
            # dists = np.zeros(matches.shape[0], )
            # labels = np.zeros(matches.shape[0], )
            dists = np.zeros(matches_test.shape[0] // BATCH_SIZE * BATCH_SIZE, )
            labels = np.zeros(matches_test.shape[0] // BATCH_SIZE * BATCH_SIZE, )

            for x in tqdm(range(matches_test.shape[0] // BATCH_SIZE)):
                # get batch ids
                batch = matches_test[offset:offset + BATCH_SIZE, :]

                # update the batch offset
                offset += BATCH_SIZE

                input1 = patches_test[batch[:, 0]]
                input2 = patches_test[batch[:, 1]]

                # import cv2
                # for i in range(input1.shape[0]):
                #     img1 = np.squeeze(input1[i])
                #     img2 = np.squeeze(input2[i])
                #     img = cv2.hconcat((img1, img2))
                #     print("label: ", batch[i, 2])
                #     cv2.imshow("patch-pairs", img)
                #     cv2.waitKey(0)

                if params['patch_type'] != 'dep':
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
                for i in range(BATCH_SIZE):
                    idx = x * BATCH_SIZE + i
                    dists[idx] = np.linalg.norm(descs1[i, :] - descs2[i, :])
                    labels[idx] = batch[i, 2]

            # compute the false positives rate
            fpr95 = ErrorRateAt95Recall(labels, dists)
            print ('%s-->FRP95: %s' % (model, fpr95))

def main(_):
    run_evaluation()

if __name__ == '__main__':
    tf.app.run()