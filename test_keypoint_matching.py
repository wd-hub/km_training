import cv2
import tensorflow as tf
import numpy as np
import argparse
import os
from Hardnet__ import Hardnet

from preprocess import normalize_batch
from utils.config import get_params

parser = argparse.ArgumentParser(description='Tensorflow HardNet')
parser.add_argument('--patch_type', default='gray', help='patch type (gray, rgb, dep, rgbd)')
parser.add_argument('--model_file', default='./logs_liberty', help="""The filename of the model to evaluate""")
args = parser.parse_args()
params = get_params(args.patch_type)
model = os.path.join(args.model_file,'checkpoint-0')
ratioThd = 0.8
# MODEL = importlib.import_module(FLAGS.architecture)
print ("Loading Model...")

with tf.Graph().as_default():

    with tf.name_scope('inputs'):
        # User defined parameters
        BATCH_SIZE = params['batch_size']
        NUM_CHANNELS = params['num_channels']
        PATCH_SIZE = params['patch_size']

        # Define the input tensor shape
        tensor_shape = (None, PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS)

        # Triplet place holders
        inputs1_pl = tf.placeholder(dtype=tf.float32, shape=tensor_shape, name='inputs1_pl')
        inputs2_pl = tf.placeholder(dtype=tf.float32, shape=tensor_shape, name='inputs2_pl')
        is_training = tf.placeholder(dtype=tf.bool)

        inputs1_m = tf.placeholder(dtype=tf.float32, shape=[None, 128], name='inputs1_m')
        inputs2_m = tf.placeholder(dtype=tf.float32, shape=[None, 128], name='inputs2_m')

    # Creating the architecture
    tfeat_architecture = Hardnet(NUM_CHANNELS)
    tfeat_inputs1 = tfeat_architecture.build(inputs1_pl, is_training)
    tfeat_inputs2 = tfeat_architecture.build(inputs2_pl, is_training, reuse=True)
    mfeat_output = tfeat_architecture.metric(inputs1_m, inputs2_m)

    saver = tf.train.Saver()
    # init = tf.global_variables_initializer()
    with tf.Session() as sess:
        saver.restore(sess, model)
        # sess.run(init)
        print ('Done!')

        # opencv3
        _detector = cv2.ORB_create()
        # _descriptor = cv2.xfeatures2d.SIFT_create()
        _descriptor = cv2.ORB_create()
        bf = cv2.BFMatcher()

        rgb1 = cv2.imread('/home/dong/Documents/3D_Matching/src_demo/pretrained_siamese64_model/f1.png')
        rgb2 = cv2.imread('/home/dong/Documents/3D_Matching/src_demo/pretrained_siamese64_model/f2.png')

        if args.patch_type == "gray":
            img1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(rgb2, cv2.COLOR_BGR2GRAY)
        else:
            img1 = rgb1
            img2 = rgb2
        # compute keypoints and descriptors
        _kp1 = _detector.detect(img1, None)
        _kp2 = _detector.detect(img2, None)

        for kp in _kp1[::-1]:
            if (kp.pt[0] < int(PATCH_SIZE / 2) or kp.pt[0] > int(img1.shape[1] - 1 - PATCH_SIZE / 2) or
                        kp.pt[1] < int(PATCH_SIZE / 2) or kp.pt[1] > int(img1.shape[0] - 1 - PATCH_SIZE / 2)):
                _kp1.remove(kp)
        for kp in _kp2[::-1]:
            if (kp.pt[0] < int(PATCH_SIZE / 2) or kp.pt[0] > int(img1.shape[1] - 1 - PATCH_SIZE / 2) or
                        kp.pt[1] < int(PATCH_SIZE / 2) or kp.pt[1] > int(img1.shape[0] - 1 - PATCH_SIZE / 2)):
                _kp2.remove(kp)

        _kp1.sort(key=lambda x: x.response, reverse=True)
        kp1 = _kp1[:500]
        _kp2.sort(key=lambda x: x.response, reverse=True)
        kp2 = _kp2[:500]

        _, des1 = _descriptor.compute(img1, kp1)
        _, des2 = _descriptor.compute(img2, kp2)
        matches = bf.knnMatch(des1, des2, k=2)
        matches_pos = [match for match in matches if match[0].distance/match[1].distance<ratioThd]
        rgb_patches1 = []
        rgb_patches2 = []

        for kp in kp1:
            top_left_x = int(kp.pt[0] - PATCH_SIZE / 2)
            top_left_y = int(kp.pt[1] - PATCH_SIZE / 2)
            bottom_right_x = int(kp.pt[0] + PATCH_SIZE / 2)
            bottom_right_y = int(kp.pt[1] + PATCH_SIZE / 2)

            patch = img1[top_left_y:bottom_right_y,
                    top_left_x:bottom_right_x]
            patch = patch.astype(np.float32)
            rgb_patches1.extend(patch.reshape(1, PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS))

        for kp in kp2:
            top_left_x = int(kp.pt[0] - PATCH_SIZE / 2)
            top_left_y = int(kp.pt[1] - PATCH_SIZE / 2)
            bottom_right_x = int(kp.pt[0] + PATCH_SIZE / 2)
            bottom_right_y = int(kp.pt[1] + PATCH_SIZE / 2)

            patch = img2[top_left_y:bottom_right_y,
                    top_left_x:bottom_right_x]
            patch = patch.astype(np.float32)
            rgb_patches2.extend(patch.reshape(1, PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS))

        rgb_patches1 = np.array(rgb_patches1) / 255.
        rgb_patches2 = np.array(rgb_patches2) / 255.
        input1 = rgb_patches1
        input2 = rgb_patches2

        feed_dict = {
            inputs1_pl: list(input1),
            inputs2_pl: list(input2),
            is_training: False
        }

        descs1, descs2 = sess.run([tfeat_inputs1, tfeat_inputs2],
                                  feed_dict=feed_dict)
        matches_d = bf.knnMatch(descs1, descs2, k=2)
        matches_d_pos = [match for match in matches_d if match[0].distance / match[1].distance < ratioThd]
        # metric net for decision
        # despSet1, despSet2 = [], []
        # for match in matches_d:
        #     despSet1.append(descs1[match[0].queryIdx])
        #     despSet2.append(descs2[match[0].trainIdx])
        # despSet1 = np.asarray(despSet1)
        # despSet2 = np.asarray(despSet2)
        # dist = np.sum((despSet1 - despSet2) ** 2, axis=1)
        # match_descision = sess.run(mfeat_output, feed_dict={inputs1_m:list(despSet1), inputs2_m:list(despSet2)})
        # ind = np.argsort(match_descision)
        # goodMatch = np.asarray(matches_d)[ind[-50:]]

        # draw matching result
        imgMatch = np.concatenate((rgb1, rgb2), axis=1)
        imgMatch2 = imgMatch.copy()
        imgMatch3 = imgMatch.copy()
        imgMatch4 = imgMatch.copy()

        cv2.drawMatchesKnn(rgb1, kp1, rgb2, kp2, matches[:1000], imgMatch, flags=2)
        cv2.drawMatchesKnn(rgb1, kp1, rgb2, kp2, matches_pos[:1000], imgMatch2, flags=2)
        cv2.drawMatchesKnn(rgb1, kp1, rgb2, kp2, matches_d[:1000], imgMatch3, flags=2)
        cv2.drawMatchesKnn(rgb1, kp1, rgb2, kp2, matches_d_pos[:1000], imgMatch4, flags=2)

        # cv2.imwrite('orbm.jpg', imgMatch2)
        # cv2.imwrite('deep.jpg', imgMatch)
        match_orb = cv2.vconcat((imgMatch, imgMatch2))
        match_learnt = cv2.vconcat((imgMatch3, imgMatch4))
        cv2.imshow('orb Matching', match_orb)
        cv2.imshow('Deep Matching', match_learnt)
        cv2.waitKey(0)
        print ('done...')


