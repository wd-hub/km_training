import tensorflow as tf

import pickle
import numpy as np
from tqdm import tqdm
import argparse
import os
import time
import cv2
from PIL import Image
from imgaug import augmenters as iaa

from Hardnet__ import Hardnet
from ubc import UBCDataset

from Losses import loss_margin_min, loss_metric
from eval_metrics import ErrorRateAt95Recall
from preprocess import normalize_batch
from utils.config import get_params
# from Loggers import Logger

parser = argparse.ArgumentParser(description='Tensorflow HardNet')
parser.add_argument('--patch_type', default='gray', help='patch type (gray, rgb, dep, rgbd)')
parser.add_argument('--train_dir', default='/home/dong/Documents/3D_Matching/Dataset/UBC',
                    help="""The name of the dataset used to for training""")
parser.add_argument('--train_name', default='notredame', help="""The name of the dataset used to for training""")
parser.add_argument('--log_dir', default='./logs', help='folder to output model checkpoints')
parser.add_argument('--mean_image', type=float, default=0.443728476019, help='mean of train dataset for normalization')
parser.add_argument('--std_image', type=float, default=0.20197947209, help='std of train dataset for normalization')
parser.add_argument('--anchorswap', type=bool, default=True, help='turns on anchor swap')
parser.add_argument('--anchorave', type=bool, default=False, help='anchorave')
parser.add_argument('--act_decay', type=float, default=0, help='activity L2 decay, default 0')
parser.add_argument('--optimizer', default='sgd', type=str, metavar='OPT', help='The optimizer to use (default: SGD)')
parser.add_argument('--fliprot', type=bool, default=False, help='turns on flip and 90deg rotation augmentation')
args = parser.parse_args()

params = get_params(args.patch_type)
os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu_id']
# batch_size = params['batch_size']

def main(logger):
    # load data
    dataset = UBCDataset(args.train_dir)
    dataset.load_by_name(args.train_name)

    # get patches
    patches = dataset._get_patches(args.train_name)
    # quick fix in order to have normalized data beforehand
    patches -= args.mean_image
    patches /= args.std_image

    # generate triplets ids
    triplets = dataset.generate_triplets(args.train_name, params['n_triplets'], params['batch_size'])
    # get matches for evaluation
    matches_train = dataset._get_matches(args.train_name)
    with tf.Graph().as_default():
        with tf.name_scope('inputs'):
            # User defined parameters
            BATCH_SIZE = params['batch_size']
            NUM_CHANNELS = params['num_channels']
            PATCH_SIZE = params['patch_size']

            lr = tf.Variable(0.0, trainable=False)
            # used for updating learning rate
            tf_batch_size = tf.constant(BATCH_SIZE, dtype=tf.float32)
            tf_ntriplets = tf.constant(params['n_triplets'], dtype=tf.float32)
            tf_epochs = tf.constant(params['epochs'], dtype=tf.float32)
            # Define the input tensor shape
            tensor_shape = (BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS)
            # Triplet place holders
            inputs1_pl = tf.placeholder(
                dtype=tf.float32, shape=tensor_shape, name='anchors')
            inputs2_pl = tf.placeholder(
                dtype=tf.float32, shape=tensor_shape, name='positives')
            is_training = tf.placeholder(dtype=tf.bool)

        with tf.name_scope('accuracy'):
            accuracy_pl = tf.placeholder(tf.float32)

        # Creating the architecture
        tfeat_architecture = Hardnet(NUM_CHANNELS) #args.model_file)
        # tfeat_inputs1 = tfeat_architecture.model(inputs1_pl, is_training)
        # tfeat_inputs2 = tfeat_architecture.model(inputs2_pl, is_training)
        tfeat_inputs1 = tfeat_architecture.build(inputs1_pl, is_training)
        tfeat_inputs2 = tfeat_architecture.build(inputs2_pl, is_training, reuse=True)
        loss, min_idx, watchParams = loss_margin_min(tfeat_inputs1, tfeat_inputs2, margin=params['margin'], anchor_swap=args.anchorswap, anchor_ave=args.anchorave)

        # Creating the Metric Net
        anchor = tfeat_inputs1
        positive = tfeat_inputs2
        # negative = tf.gather(tfeat_inputs2, min_idx)
        negative = tf.matmul(tf.one_hot(min_idx, depth=BATCH_SIZE), positive)
        mfeat_ap = tfeat_architecture.metric(anchor, positive)
        mfeat_an = tfeat_architecture.metric(anchor, negative, reuse=True)
        loss_m, pos_m, neg_m = loss_metric(mfeat_ap, mfeat_an, margin=params['margin'])

        # loss = param['loss']
        # Defining training parameters
        step = tf.Variable(0, trainable=False, dtype=tf.float32)

        # Create optimizer
        # optimizer = tf.train.AdamOptimizer(1e-2).minimize(loss, global_step=step)
        optimizer = tf.train.MomentumOptimizer( learning_rate=lr, momentum=0.9).minimize(loss, global_step=step)

        var_list = tf.trainable_variables()
        # optimizer_m = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss_m, var_list=var_list[28:])
        optimizer_m = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(loss_m, var_list=var_list)
        train_op = tf.group(optimizer)#, optimizer_m)

        # Build the summary operation based on the TF collection of Summaries.
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('loss_m', loss_m)
        # tf.summary.scalar('pos_m', pos_m)
        # tf.summary.scalar('neg_m', neg_m)
        tf.summary.scalar('lr', lr)
        summary_op = tf.summary.merge_all()
        accuracy_op = tf.summary.scalar('accuracy', accuracy_pl)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(max_to_keep=20)

        # init the graph variables
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            # restore session from file
            sess.run(init)
            summary_writer_train = tf.summary.FileWriter(LOG_DIR, sess.graph)

            def run_model(sess, data, data_ids, params, logger):
                batch_size = params['batch_size']
                num_batch = len(data_ids) // batch_size
                pbar = tqdm(range(num_batch))
                for step_batch in pbar:
                    start_time = time.time()

                    # get data batch
                    batch_anchors, batch_positives, batch_negatives = dataset.get_batch(data, data_ids, step_batch, batch_size)

                    if args.fliprot:
                        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
                        seq = iaa.Sequential([
                            # iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
                            sometimes(iaa.Fliplr(0.5)),  # horizontally flip 50% of the images
                            # iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0
                            # iaa.ContrastNormalization((0.5, 1.5)),
                            # iaa.Grayscale(alpha=(0.0, 1.0)),
                            sometimes(iaa.Affine(rotate=90))
                        ])
                        seq_det = seq.to_deterministic()
                        batch_anchors = seq_det.augment_images(batch_anchors)
                        batch_positives = seq_det.augment_images(batch_positives)

                    # for i in range(10):
                    #     image = cv2.hconcat((cv2.hconcat((batch_anchors[i], batch_positives[i])), batch_negatives[i]))
                    #     cv2.imshow("image", image)
                    #     cv2.waitKey(0)

                    if params['patch_type'] == 'dep':
                        input1 = batch_anchors
                        input2 = batch_positives
                    else:
                        input1 = normalize_batch(batch_anchors)
                        input2 = normalize_batch(batch_positives)

                    feed_dict = {inputs1_pl: input1,
                                 inputs2_pl: input2,
                                 is_training: True}
                    _, loss_value, loss_value_m, watch, summary, global_step = sess.run([train_op, loss, loss_m, watchParams, summary_op, step], feed_dict=feed_dict)
                    # _, loss_value = sess.run([optimizer, loss], feed_dict=feed_dict)
                    # loss_value = loss_param['loss']
                    # Update the events file.
                    # print("step: ", step)
                    logger.add_summary(summary, global_step)
                    # update step
                    # global_step += 1
                    # logger.log_value('loss', loss_value).step()
                    duration = time.time() - start_time
                    # Print status to stdout.
                    if step_batch % 10 == 0:
                        pbar.set_description(
                            'Train Epoch: {} [{}/{} ({:.0f}%)]\t(Loss|Loss_m: {:.4f}|{:.4f})'.format(
                                epoch, (step_batch+1) * batch_size, len(data_ids),
                                       100. * (step_batch+1) / num_batch, loss_value, loss_value_m))
                        # pbar.set_description('loss = %.6f (%.3f sec)' % (loss_value, duration))
                print('Saving')
                # fname = './model/model_%s' % (args.train_name)
                saver.save(sess, '{}/checkpoint'.format(LOG_DIR), global_step=epoch)

            def eval_model(sess, patches, matches, params, logger):
                offset = 0
                batch_size = params['batch_size']
                # dists = np.zeros(matches.shape[0], )
                # labels = np.zeros(matches.shape[0], )
                dists = np.zeros(matches.shape[0] // batch_size * batch_size, )
                labels = np.zeros(matches.shape[0] // batch_size * batch_size, )
                dists_m = np.zeros(matches.shape[0] // batch_size * batch_size, )
                for x in tqdm(range(matches.shape[0] // batch_size)):
                    # get batch ids
                    batch = matches[offset:offset + batch_size, :]
                    # update the batch offset
                    offset += batch_size
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

                    if params['patch_type'] != 'dep':     # normalization is necessary for testing??
                        input1 = normalize_batch(input1)
                        input2 = normalize_batch(input2)

                    # fetch the model with data
                    descision, descs1, descs2 = sess.run([mfeat_ap, tfeat_inputs1, tfeat_inputs2],
                                                          feed_dict={
                                                              inputs1_pl: input1,
                                                              inputs2_pl: input2,
                                                              is_training: False
                                                          })

                    # compute with metric net
                    # judge_result = sess.run(mfeat_ap, feed_dict={inputs1_m:list(despSet1), inputs2_m:list(despSet2)})
                    # compute euclidean distances between descriptors
                    for i in range(batch_size):
                        idx = x * batch_size + i
                        dists[idx] = np.linalg.norm(descs1[i, :] - descs2[i, :])
                        # dists[idx] = np.sum(descs1[i, :] * descs2[i, :])
                        labels[idx] = batch[i, 2]
                        dists_m[idx] = descision[i]

                # compute the false positives rate
                fpr95 = ErrorRateAt95Recall(labels, dists)
                print('FRP95: %s' % fpr95)
                fpr95_m = ErrorRateAt95Recall(labels, dists_m)
                print('FRP95: %s' % fpr95_m)
                # if (args.enable_logging):
                    # logger.log_value(args.train_name + ' fpr95', (fpr95*100))
                acc_val, global_step = sess.run([accuracy_op, step], feed_dict={accuracy_pl: fpr95})
                logger.add_summary(acc_val, global_step)

            # And then after everything is built, start the training loop.
            # global_step = 0
            for epoch in range(params['epochs']):
                print('#############################')
                print('Epoch: %s' % epoch)
                epoch_time = time.time()
                localtime = time.asctime(time.localtime(time.time()))
                print ('Local time :', localtime)

                # train
                print('Training ... ')
                # updating learning rate
                print('step', sess.run(step))
                sess.run(tf.assign(lr, tf.maximum(params['lr'] * (1-step*tf_batch_size/(tf_ntriplets*tf_epochs)), 1e-5)))

                # do training
                run_model(sess, patches, triplets, params, summary_writer_train)

                # update global step
                # global_step += len(triplets) // args.batch_size

                # accuray: testing
                print('Accuracy test ...')
                eval_model(sess, patches, matches_train, params, summary_writer_train)
                # np.random.shuffle(triplets)
                # save the model
                # print('Saving')
                # fname = './model/model_%s' % (args.train_name)
                # saver.save(sess, fname, global_step=epoch)
                # print('Done training for %d epochs, %d steps.' % (epoch, global_step))

# def main(_):
#     run_training()

if __name__ == '__main__':
    LOG_DIR = os.path.join(args.log_dir+'_'+args.train_name+'_GOR')   # e.g. logs_rgb

    if os.path.isfile(LOG_DIR):
        os.remove(LOG_DIR)  # remove the file
    elif os.path.isdir(LOG_DIR):
        import shutil
        shutil.rmtree(LOG_DIR)  # remove dir and all contains
        os.makedirs(LOG_DIR)

    # logger = None
    # if (args.enable_logging):
    #     logger = Logger(LOG_DIR)
    main(LOG_DIR)

    # tf.app.run()