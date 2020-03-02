#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import time
import math
import cv2
import argparse
import xlsxwriter
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append("../")
from tools.common import Notify
from model import TVSNet, TVSNet_base, TVSNet_base_siamese, TVSNet_refine
from model import cost_volume_aggregation, cost_volume_aggregation_refine
from model import output_conv, output_conv_refine, prob2depth, prob2depth_upsample
from eval_errors import calc_error, err_metrics_namelist, acc_metrics_namelist

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FLAGS = tf.app.flags.FLAGS
# params for data
tf.app.flags.DEFINE_string('root_path', '../example/', 
                           """Root path to example data.""")
tf.app.flags.DEFINE_integer('example_index', 2,
                            """Index of example data.""")   

tf.app.flags.DEFINE_integer('view_num', 5,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_string('pretrained_model_ckpt_path',  '../model/model.ckpt',
                           """Path to restore the model.""")

tf.app.flags.DEFINE_integer('max_d', 128,
                            """Maximum depth step when training.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """Number of GPUs.""")
tf.app.flags.DEFINE_integer('gpu_id', 0,
                            """GPU index.""")
tf.app.flags.DEFINE_float('sample_scale', 0.25,
                          """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """training batch size""")
tf.app.flags.DEFINE_boolean('inverse_depth', True,
                            """Use inversed depth.""")


def run_test_multiview(savepath, images_data, cams_data, depth_gt = None):
    assert FLAGS.view_num > 2
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        num_images = FLAGS.view_num
        
        ########## buile model ##########
        print(Notify.INFO, 'building model......', Notify.ENDC)
        for i in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % FLAGS.gpu_id):
                with tf.name_scope('Model_tower%d' % i) as scope:
                    # generate data
                    images = tf.placeholder(tf.float32, shape=[1, num_images, None, None, 3])
                    cams = tf.placeholder(tf.float32, shape=[1, num_images, 2, 4, 4])
                    images.set_shape(tf.TensorShape([None, num_images, None, None, 3]))
                    cams.set_shape(tf.TensorShape([None, num_images, 2, 4, 4]))
                    depth_start = tf.reshape(tf.slice(cams, [0, 0, 1, 3, 0], [
                                             FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
                    depth_interval = tf.reshape(tf.slice(cams, [0, 0, 1, 3, 1], [
                                                FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])

                    # initial
                    view_idx = tf.placeholder(tf.int32)
                    _, prob_volume_b2, filtered_cost_volume, depth_view = TVSNet_base_siamese(
                        images, cams, FLAGS.max_d, depth_start, depth_interval, view_i=view_idx, ref_i=0)

                    # AAM1
                    cost_chan = filtered_cost_volume.get_shape().as_list()[-1]
                    filtered_cost_volumes_in = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.max_d, None, None, cost_chan, FLAGS.view_num-1])
                    filtered_cost_volumes_in.set_shape(tf.TensorShape([FLAGS.batch_size, FLAGS.max_d, None, None, cost_chan, FLAGS.view_num-1]))
                    prob_volumes_in = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.max_d, None, None, FLAGS.view_num-1])
                    prob_volumes_in.set_shape(tf.TensorShape([None, FLAGS.max_d, None, None, FLAGS.view_num-1]))

                    cost_volume_agg = cost_volume_aggregation(filtered_cost_volumes_in, reuse=False, keepchannel=True)
                    prob_volume_agg = output_conv(cost_volume_agg, reuse=False) 
                    depth_agg_init = prob2depth(prob_volume_agg, FLAGS.max_d, depth_start, depth_interval, out_prob_map=False)

                    # refinement
                    depth_ref_in = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, None, None, 1])
                    depth_ref_in.set_shape(tf.TensorShape([FLAGS.batch_size, None, None, 1]))
                    depth_view_in = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, None, None, 1])
                    depth_view_in.set_shape(tf.TensorShape([FLAGS.batch_size, None, None, 1]))
                    prob_agg_in = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.max_d, None, None])
                    prob_agg_in.set_shape(tf.TensorShape([FLAGS.batch_size, FLAGS.max_d, None, None]))
                    cost_agg_in = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.max_d, None, None, cost_chan])
                    cost_agg_in.set_shape(tf.TensorShape([FLAGS.batch_size, FLAGS.max_d, None, None, cost_chan]))

                    refined_prob_volume, refined_cost_volume = TVSNet_refine(
                        depth_ref_in, depth_view_in, prob_agg_in, cost_agg_in, 
                        images, cams, FLAGS.max_d, depth_start, depth_interval, view_i=view_idx, ref_i=0)

                    # AAM2
                    refined_cost_volumes_in = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.max_d, None, None, cost_chan, FLAGS.view_num-1])
                    refined_cost_volumes_in.set_shape(tf.TensorShape([FLAGS.batch_size, FLAGS.max_d, None, None, cost_chan, FLAGS.view_num-1]))
                    refined_prob_volumes_in = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.max_d, None, None, FLAGS.view_num-1])
                    refined_prob_volumes_in.set_shape(tf.TensorShape([None, FLAGS.max_d, None, None, FLAGS.view_num-1]))

                    refined_cost_volume_agg = cost_volume_aggregation_refine(refined_cost_volumes_in, reuse=False, keepchannel=True)
                    refined_prob_volume_agg = output_conv_refine(refined_cost_volume_agg, reuse=False) 
                    _, depth_agg_refined = prob2depth_upsample(refined_prob_volume_agg, FLAGS.max_d, depth_start, depth_interval, out_prob_map=False)

        # initialization option
        # init_op = tf.keras.initializers.he_normal(seed=42)#tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, report_tensor_allocations_upon_oom = True)

        with tf.Session(config=config) as sess:
            ############ load pre-trained model ############
            print(Notify.INFO, 'loading checkpoint......', Notify.ENDC)
            if FLAGS.pretrained_model_ckpt_path is not None:
                restorer = tf.train.Saver(tf.global_variables())
                restorer.restore(
                    sess, FLAGS.pretrained_model_ckpt_path)
                print(Notify.INFO, 'pre-trained model restored from %s' % (FLAGS.pretrained_model_ckpt_path), Notify.ENDC)
            else:
                print('FLAGS.pretrained_model_ckpt_path is None !!')
                exit()

            if os.path.exists(savepath) is False:
                os.makedirs(savepath)

            # load input data
            images_data = np.expand_dims(images_data, axis=0)
            cams_data = np.expand_dims(cams_data, axis=0)
            out_depth_map_gt = depth_gt

            ############ run test ############
            print(Notify.INFO, 'running test......', Notify.ENDC)
            filtered_cost_volumes = []
            prob_volumes = []
            depth_views = []
            # run initial
            for view_i in range(1, FLAGS.view_num):
                out_filtered_cost_volume, out_prob_volume_b2, out_depth_view = sess.run([filtered_cost_volume, prob_volume_b2, depth_view],
                                                                    feed_dict={images: images_data, cams: cams_data, view_idx: view_i}, options=run_options)
                filtered_cost_volumes.append(out_filtered_cost_volume)
                prob_volumes.append(out_prob_volume_b2)
                depth_views.append(out_depth_view)
            filtered_cost_volumes = np.stack(filtered_cost_volumes, axis=-1)
            prob_volumes = np.stack(prob_volumes, axis=-1)

            # run AAM1
            out_depth_map_init, out_prob_volume_agg, out_cost_volume_agg = sess.run([depth_agg_init, prob_volume_agg, cost_volume_agg],
                                                        feed_dict={images: images_data, cams: cams_data,
                                                        filtered_cost_volumes_in: filtered_cost_volumes,
                                                        prob_volumes_in: prob_volumes}, 
                                                        options=run_options)

            # run refinement
            refined_cost_volumes = []
            refined_prob_volumes = []
            for view_i in range(1, FLAGS.view_num):
                out_refined_cost_volume, out_refined_prob_volume = sess.run([refined_cost_volume, refined_prob_volume],
                    feed_dict={images: images_data, cams: cams_data, view_idx: view_i,
                    depth_ref_in: out_depth_map_init,
                    depth_view_in: depth_views[view_i-1],
                    prob_agg_in: out_prob_volume_agg, 
                    cost_agg_in: out_cost_volume_agg},
                    options=run_options)
                refined_cost_volumes.append(out_refined_cost_volume)
                refined_prob_volumes.append(out_refined_prob_volume)
            refined_cost_volumes = np.stack(refined_cost_volumes, axis=-1)
            refined_prob_volumes = np.stack(refined_prob_volumes, axis=-1)

            # run AAM2
            out_depth_map = sess.run(depth_agg_refined,
                                        feed_dict={images: images_data, cams: cams_data,
                                        refined_cost_volumes_in: refined_cost_volumes,
                                        refined_prob_volumes_in: refined_prob_volumes}, 
                                        options=run_options)

            out_disp_map = out_depth_map.copy()
            if FLAGS.inverse_depth:
                out_depth_map[out_depth_map<1e-10] = float("inf")
                out_depth_map = 1.0 / out_depth_map

            # save depthmap
            np.save(os.path.join(savepath, 'pred.npy'), np.squeeze(np.array(out_depth_map)))
            # visualize dispmap
            plt.imsave(os.path.join(savepath, 'pred.jpg'), np.squeeze(np.array(out_disp_map)), cmap='viridis')

            if depth_gt is not None:
                # calc error
                print(Notify.INFO, 'calulating error......', Notify.ENDC)
                error, info = calc_error(np.squeeze(out_depth_map), np.squeeze(out_depth_map_gt))

                # save error
                workbook = xlsxwriter.Workbook(os.path.join(savepath, 'error.xlsx'))
                worksheet = workbook.add_worksheet(str(FLAGS.view_num)+'_view')
                num_err_metrics = len(err_metrics_namelist)
                for row_i, name in enumerate(err_metrics_namelist):
                    worksheet.write(row_i + 1, 0, name) 
                for row_i, name in enumerate(acc_metrics_namelist):
                    worksheet.write(row_i + num_err_metrics + 2, 0, name)
                depth_error_list = error.tolist()
                worksheet.write(0, 1, 'err') 
                worksheet.write(num_err_metrics + 1, 1, 'acc') 
                for row_i in range(num_err_metrics):
                    worksheet.write(row_i + 1, 1, depth_error_list[row_i])
                for row_i in range(num_err_metrics, len(depth_error_list)):
                    worksheet.write(row_i + 2, 1, depth_error_list[row_i])
                workbook.close()

            print(Notify.INFO, "result save to {}.".format(savepath), Notify.ENDC)
    return


def run_test_twoview(savepath, images_data, cams_data, depth_gt=None):
    assert FLAGS.view_num==2
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        num_images = FLAGS.view_num        
        ########## buile model ##########
        print(Notify.INFO, 'building model......', Notify.ENDC)
        for i in xrange(FLAGS.num_gpus): 
            gpu_id = FLAGS.gpu_id if FLAGS.gpu_id >= 0 else i
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('Model_tower%d' % i) as scope:
                    # generate data
                    images = tf.placeholder(tf.float32, shape=[1, num_images, None, None, 3])
                    cams = tf.placeholder(tf.float32, shape=[1, num_images, 2, 4, 4])
                    images.set_shape(tf.TensorShape([None, num_images, None, None, 3]))
                    cams.set_shape(tf.TensorShape([None, num_images, 2, 4, 4]))
                    depth_start = tf.reshape(tf.slice(cams, [0, 0, 1, 3, 0], [
                                                FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
                    depth_interval = tf.reshape(tf.slice(cams, [0, 0, 1, 3, 1], [
                                                FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])

                    refined_prob_volume = TVSNet(images, cams, FLAGS.max_d, depth_start, depth_interval, view_i=1, ref_i=0)
                    _, depth_refined = prob2depth_upsample(refined_prob_volume, FLAGS.max_d, depth_start, depth_interval, out_prob_map=False)                    

        # initialization option
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.95
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, report_tensor_allocations_upon_oom = True)

        with tf.Session(config=config) as sess:
            ############ load pre-trained model ############
            print(Notify.INFO, 'loading checkpoint......', Notify.ENDC)
            if FLAGS.pretrained_model_ckpt_path is not None:
                restorer = tf.train.Saver(tf.global_variables())
                restorer.restore(
                    sess, FLAGS.pretrained_model_ckpt_path)
                print(Notify.INFO, 'pre-trained model restored from %s' % (FLAGS.pretrained_model_ckpt_path), Notify.ENDC)
            else:
                print('FLAGS.pretrained_model_ckpt_path is None !!')
                exit()

            # load input data
            images_data = np.expand_dims(images_data, axis=0)
            cams_data = np.expand_dims(cams_data, axis=0)
            out_depth_map_gt = depth_gt

            ############ test TVSNet ############
            print(Notify.INFO, 'running test......', Notify.ENDC)
            out_depth_map = sess.run(depth_refined, feed_dict={images: images_data, cams: cams_data}, options=run_options)

            out_disp_map = out_depth_map.copy()
            if FLAGS.inverse_depth:
                out_depth_map[out_depth_map<=0] = float("inf")
                out_depth_map = 1.0 / out_depth_map

            # save depthmap
            np.save(os.path.join(savepath, 'pred.npy'), np.squeeze(np.array(out_depth_map)))
            # visualize dispmap
            plt.imsave(os.path.join(savepath, 'pred.jpg'), np.squeeze(np.array(out_disp_map)), cmap='viridis')

            if depth_gt is not None:
                # calc error
                print(Notify.INFO, 'calulating error......', Notify.ENDC)
                error, info = calc_error(np.squeeze(out_depth_map), np.squeeze(out_depth_map_gt))

                # save error
                workbook = xlsxwriter.Workbook(os.path.join(savepath, 'error.xlsx'))
                worksheet = workbook.add_worksheet(str(FLAGS.view_num)+'_view')
                num_err_metrics = len(err_metrics_namelist)
                for row_i, name in enumerate(err_metrics_namelist):
                    worksheet.write(row_i + 1, 0, name) 
                for row_i, name in enumerate(acc_metrics_namelist):
                    worksheet.write(row_i + num_err_metrics + 2, 0, name)
                depth_error_list = error.tolist()
                worksheet.write(0, 1, 'err') 
                worksheet.write(num_err_metrics + 1, 1, 'acc') 
                for row_i in range(num_err_metrics):
                    worksheet.write(row_i + 1, 1, depth_error_list[row_i])
                for row_i in range(num_err_metrics, len(depth_error_list)):
                    worksheet.write(row_i + 2, 1, depth_error_list[row_i])
                workbook.close()

            print(Notify.INFO, "result save to {}.".format(savepath), Notify.ENDC)
    return 


def main(argv=None): 
    # create result folder
    data_root = os.path.join(FLAGS.root_path, str(FLAGS.example_index))
    savepath = os.path.join(data_root, 'result')
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # check input images
    valid_view_num = 0
    for view_i in range(FLAGS.view_num): 
        img_path = os.path.join(data_root, str(view_i)+'.jpg')
        cam_path = os.path.join(data_root, str(view_i)+'_cam.npy') 
        if os.path.exists(img_path) and os.path.exists(cam_path):
            valid_view_num = valid_view_num + 1
        else:
            print("{} or {} not exist. check view_num".format(img_path, cam_path))
    if valid_view_num != FLAGS.view_num:
        print (Notify.INFO, 'only %d views found (FLAGS.view_num = %d), continue with %d views' %
               (valid_view_num, FLAGS.view_num, valid_view_num), Notify.ENDC)
        FLAGS.view_num = valid_view_num

    # load data
    images = []
    cams = []
    for view_i in range(FLAGS.view_num):
        img_path = os.path.join(data_root, str(view_i)+'.jpg')
        cam_path = os.path.join(data_root, str(view_i)+'_cam.npy')
        img = cv2.imread(img_path)
        images.append(img)
        cam = np.load(cam_path)
        cams.append(cam)
    images = np.stack(images, axis=0) # (N, H, W, C) input images, the first one is the reference view
    cams = np.stack(cams, axis=0) # (N, 2, 4, 4) corresponding camera params (extrinsic and intrinsic)
    depth_gt_path = os.path.join(data_root, '0_gt.npy')
    if os.path.exists(depth_gt_path):
        depth_gt = np.load(depth_gt_path) # ground truth depth of the reference view
    else:
        depth_gt = None

    if FLAGS.view_num == 2:
        run_test_twoview(savepath, images, cams, depth_gt)
    else:
        run_test_multiview(savepath, images, cams, depth_gt)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default=FLAGS.root_path)
    parser.add_argument('--example_index', type=int, default=FLAGS.example_index)
    parser.add_argument('--pretrained_model_ckpt_path', type=str, default=FLAGS.pretrained_model_ckpt_path)
    parser.add_argument('--view_num', type=int, default=FLAGS.view_num)
    args = parser.parse_args()

    FLAGS.root_path = args.root_path
    FLAGS.example_index = args.example_index
    FLAGS.pretrained_model_ckpt_path = args.pretrained_model_ckpt_path
    FLAGS.view_num = args.view_num
    assert FLAGS.view_num > 1
    print (Notify.INFO, 'Testing A-TVSNet with %d views' % (FLAGS.view_num), Notify.ENDC)

    tf.app.run()
