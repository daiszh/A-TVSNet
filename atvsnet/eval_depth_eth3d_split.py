#!/usr/bin/env python

from __future__ import print_function

import os
import time
import sys
import math
import argparse
import json
from random import randint
import datetime
from tqdm import tqdm
import simplejson
import xlsxwriter
from shutil import copyfile

import cv2
import numpy as np
import tensorflow as tf
# from tensorflow.python import debug as tf_debug

import matplotlib.pyplot as plt
import imageio

sys.path.append("../")
from tools.common import Notify

from preprocess_mvs_syn import *
import preprocess_colmap as colmap

from model import TVSNet, TVSNet_base, TVSNet_base_siamese, TVSNet_refine
from model import cost_volume_aggregation, cost_volume_aggregation_refine, cost_volume_aggregation_singleW
from model import output_conv, output_conv_refine, prob2depth, prob2depth_upsample


from eval_errors import calc_error, metrics_namelist

FLAGS = tf.app.flags.FLAGS


# params for datasets
tf.app.flags.DEFINE_string('data_type', 'colmap',
                           """demon or colmap(i.e. eth3d).""")
tf.app.flags.DEFINE_string('demon_data_root', '../../../dataset/demon_test/',
                           """Path to demon format dataset.""")
tf.app.flags.DEFINE_string('colmap_data_root', '../../../dataset/',
                           """Path to colmap format dataset.""")

# params for training
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """Number of GPUs.""")
tf.app.flags.DEFINE_integer('gpu_id', 0,
                            """GPU index.""")
tf.app.flags.DEFINE_integer('view_num', 5,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 128,
                            """Maximum depth step when training.""")
# w, h should be multiple of 32
tf.app.flags.DEFINE_integer('max_w', 960, 
                            """Maximum image width when training.""")
tf.app.flags.DEFINE_integer('max_h', 640,  
                            """Maximum image height when training.""")
tf.app.flags.DEFINE_float('sample_scale', 0.25,
                          """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_integer('base_image_size', 32,
                            """Base image size to fit the network.""")
tf.app.flags.DEFINE_float('interval_scale', 1.0,
                          """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """training batch size""")

# params for config
tf.app.flags.DEFINE_boolean('inverse_depth', True,
                            """Use inversed depth.""")

tf.app.flags.DEFINE_string('pretrained_model_ckpt_path',  '../model/model.ckpt',
                           """Path to restore the model.""")
tf.app.flags.DEFINE_integer('ckpt_step', 200000,
                            """ckpt step.""")

tf.app.flags.DEFINE_boolean('save_depths', False,
                            """save output results.""")
tf.app.flags.DEFINE_string('save_path', '',
                            """save output results path.""")
tf.app.flags.DEFINE_boolean('dual', True,
                            """dual AAM flag.""")

# disable CPU supports warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

def get_depth_range(depthmap_in, interval_scale=1.0, percentile=0.9, stretch=1.3):
    depthmap = depthmap_in.copy()
    depthmap = depthmap.flatten()
    depthmap_arr = depthmap[(depthmap < 1e10)*(depthmap > 0.0)]
    depthmap_arr = np.sort(depthmap_arr)
    num_valid = len(depthmap_arr)
    depth_max = depthmap_arr[int(num_valid * percentile)] * stretch
    depth_min = depthmap_arr[int(num_valid * (1.0 - percentile))] / stretch
    # depthmap[depthmap > 1e10] = 0.0
    # depth_max = np.max(depthmap[:]) * stretch
    # depthmap[depthmap < 1e-10] = float('inf')
    # depth_min = np.min(depthmap[:]) / stretch
    depth_interval = (depth_max - depth_min) * interval_scale / float(FLAGS.max_d - 1)
    # depth_min = max(0.1, depth_min - depth_interval)

    return depth_min, depth_interval

def find_substring(substring, string):
    indices = []
    index = -1
    while True:
        index = string.find(substring, index + 1)
        if index == -1:  
            break
        indices.append(index)
    return indices


def load_demon_test_data(demon_folder, num_images, current_i):
    images_data = []
    cams_data = []
    flag_bgr2rgb = False

    # load ground truth depth
    depth_path = os.path.join(demon_folder, "{:04d}".format(current_i), "depths", "{:04d}.exr".format(0))
    depth_data = imageio.imread(depth_path)
    depth_data = depth_data * float(colmap.get_unit_scale())
    if FLAGS.inverse_depth:
        depth_data[depth_data <= 0.0] = float("inf")
        depth_data = 1.0/(depth_data)
    
    depth_min, depth_interval = get_depth_range(depth_data)
    # if FLAGS.fix_depth_range:
    #     if FLAGS.inverse_depth:
    #         depth_min = 0.0
    #         depth_interval = (1.0 / float(FLAGS.min_depth) - depth_min) / float(FLAGS.max_d-1)
    #     else:
    #         depth_min = FLAGS.min_depth
    #         _, depth_interval = get_depth_range(depth_data)
    # else:
    #     depth_min, depth_interval = get_depth_range(depth_data)

    file_list = []
    for view in range(num_images):
        image_path = os.path.join(demon_folder, "{:04d}".format(current_i), "images", "{:04d}.png".format(view))
        image = cv2.imread(image_path)
        file_list.append(image_path)
        # image_file = file_io.FileIO(data[view], mode='rb')
        # image = scipy.misc.imread(image_file, mode='RGB')
        if view==0:
            image_ref = image.copy()
        if flag_bgr2rgb:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        images_data.append(image)
        cam = np.zeros((2, 4, 4))
        posepath = os.path.join(demon_folder, "{:04d}".format(current_i), "poses", "{:04d}.json".format(view))
        with open(posepath) as f:
            r_info = json.load(f)
        cam[1][0][0] = r_info["f_x"]
        cam[1][1][1] = r_info["f_y"]
        cam[1][0][2] = r_info["c_x"]
        cam[1][1][2] = r_info["c_y"]
        cam[1][2][2] = 1.0
        # extrinsic is world to cam
        extrinsic = np.array(r_info["extrinsic"])
        for i in range(0, 3):
            for j in range(0, 3):
                cam[0][i][j] = extrinsic[i, j]
        cam[0][0][3] = extrinsic[0, 3] * float(colmap.get_unit_scale())
        cam[0][1][3] = extrinsic[1, 3] * float(colmap.get_unit_scale())
        cam[0][2][3] = extrinsic[2, 3] * float(colmap.get_unit_scale())
        cam[0][3][3] = 1.0
        cam[1][3][0] = depth_min
        cam[1][3][1] = depth_interval
        cam[1][3][1] = cam[1][3][1] * FLAGS.interval_scale
        cams_data.append(cam)
    
    # determine a proper scale to resize input 
    h_scale = float(FLAGS.max_h) / float(images_data[0].shape[0])
    w_scale = float(FLAGS.max_w) / float(images_data[0].shape[1])
    if h_scale <= 1 and w_scale <= 1:
        scaled_input_images, scaled_input_cams = colmap.scale_mvs_input(images_data, cams_data)
    else:
        # print('image_size({:},{:}) should no smaller than [FLAGS.max_h, FLAGS.max_w]({:},{:})'.format(images_data[0].shape[0], images_data[0].shape[1], FLAGS.max_h, FLAGS.max_w))
        # exit()
        # crop image and depth image to 32x
        # print('h_scale >= 1 or w_scale >= 1, crop image and depth image to 32x')
        scaled_input_images = colmap.crop_image_depth(images_data, factor=32)
        scaled_input_cams = cams_data

    # resize ground truth depth
    file_list.append(depth_path)
    depth_scale = max(float(scaled_input_images[0].shape[0]) / float(
        depth_data.shape[0]), float(scaled_input_images[0].shape[1]) / float(depth_data.shape[1]))
    depth_data = colmap.scale_image(depth_data, scale=depth_scale, interpolation='nearest')
    scaled_depth_data = depth_data[0:scaled_input_images[0].shape[0], 0:scaled_input_images[0].shape[1]]

    # crop to fit network
    croped_images, croped_cams, croped_depth = colmap.crop_mvs_input(scaled_input_images, scaled_input_cams, scaled_depth_data)

    croped_depth = colmap.scale_image(croped_depth, scale=FLAGS.sample_scale, interpolation='nearest')

    # center images
    centered_images = []
    for view in range(num_images):
        centered_images.append(colmap.center_image(croped_images[view]))            
    
    # sample cameras for building cost volume
    # real_cams = np.copy(croped_cams) 
    scaled_cams = colmap.scale_mvs_camera(croped_cams, scale=FLAGS.sample_scale)

    # scaled_images = []
    # for view in range(num_images):
    #     scaled_images.append(scale_image(croped_images[view], scale=FLAGS.sample_scale))
    # scaled_images = np.stack(scaled_images, axis=0)

    croped_images = np.stack(croped_images, axis=0)
    scaled_cams = np.stack(scaled_cams, axis=0)
    if len(croped_depth.shape)==2:
        croped_depth = np.expand_dims(croped_depth, axis=-1)

    croped_depth[croped_depth>10e10] = 0.0
    return np.expand_dims(centered_images, axis=0), np.expand_dims(scaled_cams, axis=0), np.expand_dims(croped_depth, axis=0), image_ref, file_list

def load_colmap_data(mvs_list, colmap_sparse, num_images, current_i):
    images_data = []
    cams_data = []
    flag_bgr2rgb = False
    image = colmap_sparse.image_list.images[current_i]

    ref_image_index = image.id
    data = mvs_list[current_i]
    selected_view_num = int(len(data))
    neigh_index_list = image.neighbor_list
    if len(neigh_index_list) < 1:
        for ni in range(num_images):
            nidx0 = ref_image_index + 1 + ni
            nidx1 = ref_image_index - 1 - ni
            if (colmap_sparse.image_list.get_by_id(nidx0) is not None):
                neigh_index_list.append(nidx0)
            elif (colmap_sparse.image_list.get_by_id(nidx1) is not None):
                neigh_index_list.append(nidx1)
            else:
                neigh_index_list.append(ref_image_index)
    image_index_list = np.append(np.array(ref_image_index), np.array(neigh_index_list))
    for view in range(min(num_images, selected_view_num)):
        image = cv2.imread(data[view])                    
        # image_file = file_io.FileIO(data[view], mode='rb')
        # image = scipy.misc.imread(image_file, mode='RGB')
        if view==0:
            image_ref = image.copy()
        if flag_bgr2rgb:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        camers_index = int(colmap_sparse.image_list.get_by_id(image_index_list[view]).camera_id)
        cam = colmap.load_cam(colmap_sparse.camera_list, camers_index,
                                colmap_sparse.image_list, image_index_list[view])
        cam[1][3][1] = cam[1][3][1] * FLAGS.interval_scale
        images_data.append(image)
        cams_data.append(cam)
    if selected_view_num < num_images:
        print('selected_view_num < num_images', selected_view_num, num_images)
        for view in range(selected_view_num, num_images):
            rand_idx = random.randint(0, selected_view_num - 1)
            # image = cv2.imread(data[rand_idx])
            image_file = file_io.FileIO(data[rand_idx], mode='rb')
            image = scipy.misc.imread(image_file, mode='RGB')
            if flag_bgr2rgb:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            camers_index = int(colmap_sparse.image_list.get_by_id(image_index_list[rand_idx]).camera_id)
            cam = colmap.load_cam(colmap_sparse.camera_list, camers_index,
                            colmap_sparse.image_list, image_index_list[rand_idx])
            cam[1][3][1] = cam[1][3][1] * FLAGS.interval_scale
            images_data.append(image)
            cams_data.append(cam)
    
    # determine a proper scale to resize input 
    h_scale = float(FLAGS.max_h) / float(images_data[0].shape[0])
    w_scale = float(FLAGS.max_w) / float(images_data[0].shape[1])
    if h_scale <= 1 and w_scale <= 1:
        scaled_input_images, scaled_input_cams = colmap.scale_mvs_input(images_data, cams_data)
    else:
        # print('image_size({:},{:}) should no smaller than [FLAGS.max_h, FLAGS.max_w]({:},{:})'.format(images_data[0].shape[0], images_data[0].shape[1], FLAGS.max_h, FLAGS.max_w))
        # exit()
        # crop image and depth image to 32x
        # print('h_scale >= 1 or w_scale >= 1, crop image and depth image to 32x')
        scaled_input_images = colmap.crop_image_depth(images_data, factor=32)
        scaled_input_cams = cams_data

    # load ground truth depth
    depth_path = data[0].replace('/images/', '/depths/')
    depth_path = depth_path[0:depth_path.rfind('.')+1]+'exr'
    depth_data = imageio.imread(depth_path)
    depth_data = depth_data[:,:,0] * float(colmap.get_unit_scale())
    if FLAGS.inverse_depth:
        depth_data[depth_data <= 0.0] = float("inf")
        depth_data = 1.0/(depth_data)
    depth_scale = max(float(scaled_input_images[0].shape[0]) / float(
        depth_data.shape[0]), float(scaled_input_images[0].shape[1]) / float(depth_data.shape[1]))
    depth_data = colmap.scale_image(depth_data, scale=depth_scale, interpolation='nearest')
    scaled_depth_data = depth_data[0:scaled_input_images[0].shape[0], 0:scaled_input_images[0].shape[1]]

    # crop to fit network
    croped_images, croped_cams, croped_depth = colmap.crop_mvs_input(scaled_input_images, scaled_input_cams, scaled_depth_data)

    croped_depth = colmap.scale_image(croped_depth, scale=FLAGS.sample_scale, interpolation='nearest')

    # center images
    centered_images = []
    for view in range(num_images):
        centered_images.append(colmap.center_image(croped_images[view]))            
    
    # sample cameras for building cost volume
    # real_cams = np.copy(croped_cams) 
    scaled_cams = colmap.scale_mvs_camera(croped_cams, scale=FLAGS.sample_scale)

    # scaled_images = []
    # for view in range(num_images):
    #     scaled_images.append(scale_image(croped_images[view], scale=FLAGS.sample_scale))
    # scaled_images = np.stack(scaled_images, axis=0)

    croped_images = np.stack(croped_images, axis=0)
    scaled_cams = np.stack(scaled_cams, axis=0)
    if len(croped_depth.shape)==2:
        croped_depth = np.expand_dims(croped_depth, axis=-1)

    croped_depth[croped_depth>10e10] = 0.0
    return np.expand_dims(centered_images, axis=0), np.expand_dims(scaled_cams, axis=0), np.expand_dims(croped_depth, axis=0), image_ref, data


def run_eval_split_singleagg(debugpath, image_infos):
    assert FLAGS.dual==False
    singleW = 'singleW' in FLAGS.pretrained_model_ckpt_path
    print(Notify.INFO, 'singleW ', singleW, Notify.ENDC)

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        num_queue = len(image_infos)
        num_images = FLAGS.view_num
        
        ########## buile model ##########
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

                    view_idx = tf.placeholder(tf.int32)
                    refined_cost_volume, refined_prob_volume = TVSNet(images, cams, FLAGS.max_d, depth_start, depth_interval, view_i=view_idx, ref_i=0)
                    
                    cost_chan = refined_cost_volume.get_shape().as_list()[-1]
                    refined_cost_volumes_in = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.max_d, None, None, cost_chan, FLAGS.view_num-1])
                    refined_cost_volumes_in.set_shape(tf.TensorShape([FLAGS.batch_size, FLAGS.max_d, None, None, cost_chan, FLAGS.view_num-1]))
                    refined_prob_volumes_in = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.max_d, None, None, FLAGS.view_num-1])
                    refined_prob_volumes_in.set_shape(tf.TensorShape([None, FLAGS.max_d, None, None, FLAGS.view_num-1]))
                    
                    # depth_map_b0
                    prob_vol_mean = tf.reduce_mean(refined_prob_volumes_in, axis=-1, keepdims=False)
                    depth_mean = prob2depth(prob_vol_mean, FLAGS.max_d, depth_start, depth_interval, out_prob_map=False, normalized=False)
                    # depth_map_b1
                    if singleW:
                        prob_vol_agg = cost_volume_aggregation_singleW(refined_cost_volumes_in, reuse=False)
                    else:
                        prob_vol_agg = cost_volume_aggregation_refine(refined_cost_volumes_in, reuse=False)
                    depth_agg = prob2depth(prob_vol_agg, FLAGS.max_d, depth_start, depth_interval, out_prob_map=False, normalized=False)                    

        # initialization option
        # init_op = tf.keras.initializers.he_normal(seed=42)#tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, report_tensor_allocations_upon_oom = True)

        with tf.Session(config=config) as sess:
            # load pre-trained model
            if FLAGS.pretrained_model_ckpt_path is not None:
                restorer = tf.train.Saver(tf.global_variables())
                restorer.restore(
                    sess, '-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)]))
                print(Notify.INFO, 'Pre-trained model restored from %s' %
                      ('-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)])), Notify.ENDC)
            else:
                print('FLAGS.pretrained_model_ckpt_path is None !!')
                exit()

            mean_error_b0 = []
            mean_error_b1 = []
            mean_error_b2 = [] 
            mean_error_b3 = [] 
            for datai in range(num_queue):
                scene_error_b0 = []
                scene_error_b1 = []
                scene_error_b2 = []
                scene_error_b3 = []

                # load data
                format=image_infos[datai][1]
                assert format == 'colmap'

                image_info = image_infos[datai][0]
                print('loading image format (colmap), {:}'.format(image_info[1]))

                sparse_path = image_info[0]
                image_path = image_info[1]
                mvs_list, colmap_sparse = colmap.gen_pipeline_mvs_list(sparse_path, image_path)

                debugpath_current = os.path.join(debugpath, image_info[2])
                if os.path.exists(debugpath_current) is False:
                    os.makedirs(debugpath_current)
                sample_index = -1

                for current_i in tqdm(range(len(mvs_list))):
                    sample_index += 1
                    images_data, cams_data, depth_data, image_ref, file_list = load_colmap_data(
                        mvs_list, colmap_sparse, FLAGS.view_num, current_i)
                    if FLAGS.save_depths:
                        with open(os.path.join(debugpath_current, str(sample_index) + '_file.txt'), "w") as text_file:
                            text_file.write("{:}".format(file_list))
                    out_depth_map_gt = depth_data
                    out_ref_image = images_data[:,0,:,:,:]

                    # test
                    refined_cost_volumes = []
                    refined_prob_volumes = []
                    # basenet
                    for view_i in range(1, FLAGS.view_num):
                        out_refined_cost_volume, out_refined_prob_volume = sess.run([refined_cost_volume, refined_prob_volume],
                                                                            feed_dict={images: images_data, cams: cams_data, view_idx: view_i}, options=run_options)
                        refined_cost_volumes.append(out_refined_cost_volume)
                        refined_prob_volumes.append(out_refined_prob_volume)
                    refined_cost_volumes = np.stack(refined_cost_volumes, axis=-1)
                    refined_prob_volumes = np.stack(refined_prob_volumes, axis=-1)

                    out_depth_map_b2, out_depth_map_b3 = sess.run([depth_mean, depth_agg],
                                                                feed_dict={images: images_data, cams: cams_data,
                                                                refined_cost_volumes_in: refined_cost_volumes,
                                                                refined_prob_volumes_in: refined_prob_volumes}, 
                                                                options=run_options)

                    if FLAGS.inverse_depth:
                        out_disp_map_gt = out_depth_map_gt.copy()
                        # out_disp_map_b0 = out_depth_map_b0.copy()
                        # out_disp_map_b1 = out_depth_map_b1.copy()
                        out_disp_map_b2 = out_depth_map_b2.copy()
                        out_disp_map_b3 = out_depth_map_b3.copy()
                        out_depth_map_gt[out_depth_map_gt<=0] = float("inf")
                        out_depth_map_gt = 1.0 / out_depth_map_gt
                        # out_depth_map_b0[out_depth_map_b0<=0] = float("inf")
                        # out_depth_map_b0 = 1.0 / out_depth_map_b0
                        # out_depth_map_b1[out_depth_map_b1<=0] = float("inf")
                        # out_depth_map_b1 = 1.0 / out_depth_map_b1
                        out_depth_map_b2[out_depth_map_b2<=0] = float("inf")
                        out_depth_map_b2 = 1.0 / out_depth_map_b2
                        out_depth_map_b3[out_depth_map_b3<=0] = float("inf")
                        out_depth_map_b3 = 1.0 / out_depth_map_b3

                    # error_b0, _ = calc_error(out_depth_map_b0, out_depth_map_gt)
                    # error_b1, _ = calc_error(out_depth_map_b1, out_depth_map_gt)
                    error_b0 = [0.0]
                    error_b1 = [0.0]
                    error_b2, _ = calc_error(out_depth_map_b2, out_depth_map_gt)
                    error_b3, info_b3 = calc_error(out_depth_map_b3, out_depth_map_gt)
                    scene_error_b0.append(error_b0)
                    scene_error_b1.append(error_b1)
                    scene_error_b2.append(error_b2)
                    scene_error_b3.append(error_b3)

                    if FLAGS.save_depths:
                        with open(os.path.join(debugpath_current, str(sample_index) + '_loss.txt'), "w") as text_file:
                            text_file.write('0.mae, 1.rmse, 2.inverse_mae, 3.inverse_rmse, 4.log_mae, 5.log_rmse, 6.scale_invariant_log, 7.abs_relative, 8.squared_relative, 9.mae_normalized, 10+.inlier_ratio_normalized(multiple)\n' )
                            # text_file.write('b0:\n')
                            # simplejson.dump((error_b0.tolist()), text_file)
                            # text_file.write('\nb1:\n')
                            # simplejson.dump((error_b1.tolist()), text_file)
                            text_file.write('\nb2:\n')
                            simplejson.dump((error_b2.tolist()), text_file)
                            text_file.write('\nb3:\n')
                            simplejson.dump((error_b3.tolist()), text_file)
                            text_file.write('\ninfo:\ndepth_num = %4.f, depth_interval = %.4f, depth_min = %.4f, depth_max = %.4f\n' % (info_b3[0], info_b3[1], info_b3[2], info_b3[3]))
                            text_file.write('inlier_threshold:\n')
                            simplejson.dump(info_b3[4], text_file)

                        min_depth = np.min(out_ref_image)
                        max_depth = np.max(out_ref_image)
                        # cv2.imwrite(os.path.join(debugpath_current, str(sample_index) + '_out_ref_image.jpg'), 255*(np.squeeze(np.array(out_ref_image))-min_depth)/(max_depth - min_depth))

                        cmap = 'viridis'
                        # save ground truth
                        # np.save(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_gt.npy'), np.array(out_depth_map_gt))
                        min_depth = np.min(out_disp_map_gt)
                        max_depth = np.max(out_disp_map_gt)
                        plt.imsave(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_gt.jpg'), 255*(np.squeeze(np.array(out_disp_map_gt))-min_depth)/(max_depth - min_depth), cmap=cmap)

                        # np.save(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b0.npy'), np.array(out_depth_map_b0))
                        # plt.imsave(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b0.jpg'), 255*(np.squeeze(np.array(out_disp_map_b0))-min_depth)/(max_depth - min_depth), cmap=cmap)

                        # np.save(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b1.npy'), np.array(out_depth_map_b1))
                        # plt.imsave(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b1.jpg'), 255*(np.squeeze(np.array(out_disp_map_b1))-min_depth)/(max_depth - min_depth), cmap=cmap)

                        # np.save(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b2.npy'), np.array(out_depth_map_b2))
                        plt.imsave(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b2.jpg'), 255*(np.squeeze(np.array(out_disp_map_b2))-min_depth)/(max_depth - min_depth), cmap=cmap)

                        # np.save(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b3.npy'), np.array(out_depth_map_b3))
                        plt.imsave(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b3.jpg'), 255*(np.squeeze(np.array(out_disp_map_b3))-min_depth)/(max_depth - min_depth), cmap=cmap)


                scene_error_mean_b0 = np.mean(scene_error_b0, axis=0)
                scene_error_mean_b1 = np.mean(scene_error_b1, axis=0)
                scene_error_mean_b2 = np.mean(scene_error_b2, axis=0)
                scene_error_mean_b3 = np.mean(scene_error_b3, axis=0)
                mean_error_b0.append(scene_error_mean_b0)
                mean_error_b1.append(scene_error_mean_b1)
                mean_error_b2.append(scene_error_mean_b2)
                mean_error_b3.append(scene_error_mean_b3)

                np.save(os.path.join(debugpath_current, 'zz_mean.npy'), np.array([scene_error_b0, scene_error_b1, scene_error_b2, scene_error_b3]))
                with open(os.path.join(debugpath_current, 'zz_mean.txt'), "w") as text_file:
                    text_file.write('0.mae, 1.rmse, 2.inverse_mae, 3.inverse_rmse, 4.log_mae, 5.log_rmse, 6.scale_invariant_log, 7.abs_relative, 8.squared_relative, 9.mae_normalized, 10+.inlier_ratio_normalized(multiple)\n' )
                    # text_file.write('b0:\n')
                    # simplejson.dump((scene_error_mean_b0.tolist()), text_file)
                    # text_file.write('\nb1:\n')
                    # simplejson.dump((scene_error_mean_b1.tolist()), text_file)
                    text_file.write('\nb2:\n')
                    simplejson.dump((scene_error_mean_b2.tolist()), text_file)
                    text_file.write('\nb3:\n')
                    simplejson.dump((scene_error_mean_b3.tolist()), text_file)

    return mean_error_b0, mean_error_b1, mean_error_b2, mean_error_b3


def run_eval_split_dualagg(debugpath, image_infos):
    assert FLAGS.dual == True
    singleW = 'singleW' in FLAGS.pretrained_model_ckpt_path
    print(Notify.INFO, 'singleW ', singleW, Notify.ENDC)
    assert singleW==False

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        num_queue = len(image_infos)
        num_images = FLAGS.view_num
        
        ########## buile model ##########
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
                    depth_agg_b2 = prob2depth(prob_volume_agg, FLAGS.max_d, depth_start, depth_interval, out_prob_map=False, normalized=False)
                    prob_volume_mean = tf.reduce_mean(prob_volumes_in, axis=-1, keepdims=False)
                    depth_mean_b2 = prob2depth(prob_volume_mean, FLAGS.max_d, depth_start, depth_interval, out_prob_map=False, normalized=False)

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
                    depth_agg_refined, depth_agg_refined_up = prob2depth_upsample(refined_prob_volume_agg, FLAGS.max_d, depth_start, depth_interval, out_prob_map=False, normalized=False)
                    refined_prob_volume_mean = tf.reduce_mean(refined_prob_volumes_in, axis=-1, keepdims=False)
                    depth_mean_refined = prob2depth(refined_prob_volume_mean, FLAGS.max_d, depth_start, depth_interval, out_prob_map=False, normalized=False)

        # initialization option
        # init_op = tf.keras.initializers.he_normal(seed=42)#tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, report_tensor_allocations_upon_oom = True)

        with tf.Session(config=config) as sess:
            # load pre-trained model
            if FLAGS.pretrained_model_ckpt_path is not None:
                restorer = tf.train.Saver(tf.global_variables())
                restorer.restore(
                    sess, '-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)]))
                print(Notify.INFO, 'Pre-trained model restored from %s' %
                      ('-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)])), Notify.ENDC)
            else:
                print('FLAGS.pretrained_model_ckpt_path is None !!')
                exit()

            mean_error_b0 = []
            mean_error_b1 = []
            mean_error_b2 = [] 
            mean_error_b3 = [] 
            for datai in range(num_queue):
                scene_error_b0 = []
                scene_error_b1 = []
                scene_error_b2 = []
                scene_error_b3 = []

                # load data
                format=image_infos[datai][1]
                assert format == 'colmap'

                image_info = image_infos[datai][0]
                print('loading image format (colmap), {:}'.format(image_info[1]))

                sparse_path = image_info[0]
                image_path = image_info[1]
                mvs_list, colmap_sparse = colmap.gen_pipeline_mvs_list(sparse_path, image_path)

                debugpath_current = os.path.join(debugpath, image_info[2])
                if os.path.exists(debugpath_current) is False:
                    os.makedirs(debugpath_current)
                sample_index = -1

                for current_i in tqdm(range(len(mvs_list))):
                    sample_index += 1
                    images_data, cams_data, depth_data, image_ref, file_list = load_colmap_data(
                        mvs_list, colmap_sparse, FLAGS.view_num, current_i)
                    if FLAGS.save_depths:
                        with open(os.path.join(debugpath_current, str(sample_index) + '_file.txt'), "w") as text_file:
                            text_file.write("{:}".format(file_list))
                    out_depth_map_gt = depth_data
                    out_ref_image = images_data[:,0,:,:,:]

                    # # run test
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
                    out_depth_map_b0, out_depth_map_b1, out_prob_volume_agg, out_cost_volume_agg = sess.run([depth_mean_b2, depth_agg_b2, prob_volume_agg, cost_volume_agg],
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
                            depth_ref_in: out_depth_map_b1,
                            depth_view_in: depth_views[view_i-1],
                            prob_agg_in: out_prob_volume_agg, 
                            cost_agg_in: out_cost_volume_agg},
                            options=run_options)
                        refined_cost_volumes.append(out_refined_cost_volume)
                        refined_prob_volumes.append(out_refined_prob_volume)
                    refined_cost_volumes = np.stack(refined_cost_volumes, axis=-1)
                    refined_prob_volumes = np.stack(refined_prob_volumes, axis=-1)

                    # run AAM2
                    out_depth_map_b2, out_depth_map_b3, out_depth_map_b3_up = sess.run([depth_mean_refined, depth_agg_refined, depth_agg_refined_up],
                                                feed_dict={images: images_data, cams: cams_data,
                                                refined_cost_volumes_in: refined_cost_volumes,
                                                refined_prob_volumes_in: refined_prob_volumes}, 
                                                options=run_options)


                    if FLAGS.inverse_depth:
                        out_disp_map_gt = out_depth_map_gt.copy()
                        out_disp_map_b0 = out_depth_map_b0.copy()
                        out_disp_map_b1 = out_depth_map_b1.copy()
                        out_disp_map_b2 = out_depth_map_b2.copy()
                        out_disp_map_b3 = out_depth_map_b3.copy()
                        out_disp_map_b3_up = out_depth_map_b3_up.copy()
                        out_depth_map_gt[out_depth_map_gt<=0] = float("inf")
                        out_depth_map_gt = 1.0 / out_depth_map_gt
                        out_depth_map_b0[out_depth_map_b0<=0] = float("inf")
                        out_depth_map_b0 = 1.0 / out_depth_map_b0
                        out_depth_map_b1[out_depth_map_b1<=0] = float("inf")
                        out_depth_map_b1 = 1.0 / out_depth_map_b1
                        out_depth_map_b2[out_depth_map_b2<=0] = float("inf")
                        out_depth_map_b2 = 1.0 / out_depth_map_b2
                        out_depth_map_b3[out_depth_map_b3<=0] = float("inf")
                        out_depth_map_b3 = 1.0 / out_depth_map_b3

                    error_b0, _ = calc_error(out_depth_map_b0, out_depth_map_gt)
                    error_b1, _ = calc_error(out_depth_map_b1, out_depth_map_gt)
                    error_b2, _ = calc_error(out_depth_map_b2, out_depth_map_gt)
                    error_b3, info_b3 = calc_error(out_depth_map_b3, out_depth_map_gt)
                    scene_error_b0.append(error_b0)
                    scene_error_b1.append(error_b1)
                    scene_error_b2.append(error_b2)
                    scene_error_b3.append(error_b3)

                    if FLAGS.save_depths:
                        with open(os.path.join(debugpath_current, str(sample_index) + '_loss.txt'), "w") as text_file:
                            text_file.write('0.mae, 1.rmse, 2.inverse_mae, 3.inverse_rmse, 4.log_mae, 5.log_rmse, 6.scale_invariant_log, 7.abs_relative, 8.squared_relative, 9.mae_normalized, 10+.inlier_ratio_normalized(multiple)\n' )
                            text_file.write('b0:\n')
                            simplejson.dump((error_b0.tolist()), text_file)
                            text_file.write('\nb1:\n')
                            simplejson.dump((error_b1.tolist()), text_file)
                            text_file.write('\nb2:\n')
                            simplejson.dump((error_b2.tolist()), text_file)
                            text_file.write('\nb3:\n')
                            simplejson.dump((error_b3.tolist()), text_file)
                            text_file.write('\ninfo:\ndepth_num = %4.f, depth_interval = %.4f, depth_min = %.4f, depth_max = %.4f\n' % (info_b3[0], info_b3[1], info_b3[2], info_b3[3]))
                            text_file.write('inlier_threshold:\n')
                            simplejson.dump(info_b3[4], text_file)

                        min_depth = np.min(out_ref_image)
                        max_depth = np.max(out_ref_image)
                        # cv2.imwrite(os.path.join(debugpath_current, str(sample_index) + '_out_ref_image.jpg'), 255*(np.squeeze(np.array(out_ref_image))-min_depth)/(max_depth - min_depth))

                        cmap = 'viridis'
                        # save ground truth
                        # np.save(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_gt.npy'), np.array(out_depth_map_gt))
                        min_depth = np.min(out_disp_map_gt)
                        max_depth = np.max(out_disp_map_gt)
                        plt.imsave(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_gt.jpg'), 255*(np.squeeze(np.array(out_disp_map_gt))-min_depth)/(max_depth - min_depth), cmap=cmap)

                        # np.save(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b0.npy'), np.array(out_depth_map_b0))
                        plt.imsave(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b0.jpg'), 255*(np.squeeze(np.array(out_disp_map_b0))-min_depth)/(max_depth - min_depth), cmap=cmap)

                        # np.save(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b1.npy'), np.array(out_depth_map_b1))
                        plt.imsave(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b1.jpg'), 255*(np.squeeze(np.array(out_disp_map_b1))-min_depth)/(max_depth - min_depth), cmap=cmap)

                        # np.save(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b2.npy'), np.array(out_depth_map_b2))
                        plt.imsave(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b2.jpg'), 255*(np.squeeze(np.array(out_disp_map_b2))-min_depth)/(max_depth - min_depth), cmap=cmap)

                        # np.save(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b3.npy'), np.array(out_depth_map_b3))
                        plt.imsave(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b3.jpg'), 255*(np.squeeze(np.array(out_disp_map_b3))-min_depth)/(max_depth - min_depth), cmap=cmap)
                        plt.imsave(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b3_up.jpg'), 255*(np.squeeze(np.array(out_disp_map_b3_up))-min_depth)/(max_depth - min_depth), cmap=cmap)

                scene_error_mean_b0 = np.mean(scene_error_b0, axis=0)
                scene_error_mean_b1 = np.mean(scene_error_b1, axis=0)
                scene_error_mean_b2 = np.mean(scene_error_b2, axis=0)
                scene_error_mean_b3 = np.mean(scene_error_b3, axis=0)
                mean_error_b0.append(scene_error_mean_b0)
                mean_error_b1.append(scene_error_mean_b1)
                mean_error_b2.append(scene_error_mean_b2)
                mean_error_b3.append(scene_error_mean_b3)

                np.save(os.path.join(debugpath_current, 'zz_mean.npy'), np.array([scene_error_b0, scene_error_b1, scene_error_b2, scene_error_b3]))
                with open(os.path.join(debugpath_current, 'zz_mean.txt'), "w") as text_file:
                    text_file.write('0.mae, 1.rmse, 2.inverse_mae, 3.inverse_rmse, 4.log_mae, 5.log_rmse, 6.scale_invariant_log, 7.abs_relative, 8.squared_relative, 9.mae_normalized, 10+.inlier_ratio_normalized(multiple)\n' )
                    text_file.write('b0:\n')
                    simplejson.dump((scene_error_mean_b0.tolist()), text_file)
                    text_file.write('\nb1:\n')
                    simplejson.dump((scene_error_mean_b1.tolist()), text_file)
                    text_file.write('\nb2:\n')
                    simplejson.dump((scene_error_mean_b2.tolist()), text_file)
                    text_file.write('\nb3:\n')
                    simplejson.dump((scene_error_mean_b3.tolist()), text_file)

    return mean_error_b0, mean_error_b1, mean_error_b2, mean_error_b3


def run_eval_demon(debugpath, image_infos):
    assert FLAGS.view_num==2

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        num_queue = len(image_infos)
        num_images = FLAGS.view_num
        
        ########## buile model ##########
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

                    view_idx = 1
                    _, refined_prob_volume, depth_b2 = TVSNet(images, cams, FLAGS.max_d, depth_start, depth_interval, view_i=view_idx, ref_i=0, output_depth_map=True)
                    depth_refined, depth_refined_up = prob2depth_upsample(refined_prob_volume, FLAGS.max_d, depth_start, depth_interval, out_prob_map=False, normalized=False)                    

        # initialization option
        # init_op = tf.keras.initializers.he_normal(seed=42)#tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, report_tensor_allocations_upon_oom = True)

        with tf.Session(config=config) as sess:
            # load pre-trained model
            if FLAGS.pretrained_model_ckpt_path is not None:
                restorer = tf.train.Saver(tf.global_variables())
                restorer.restore(
                    sess, '-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)]))
                print(Notify.INFO, 'Pre-trained model restored from %s' %
                      ('-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)])), Notify.ENDC)
            else:
                print('FLAGS.pretrained_model_ckpt_path is None !!')
                exit()

            mean_error_b0 = []
            mean_error_b1 = []
            mean_error_b2 = [] 
            mean_error_b3 = [] 
            for datai in range(num_queue):
                scene_error_b0 = []
                scene_error_b1 = []
                scene_error_b2 = []
                scene_error_b3 = []

                # load data
                image_info = image_infos[datai]
                print('loading image format (demon), {:}'.format(image_info[-1]))

                demon_path = image_info[0]
                num_images_list = image_info[1]

                debugpath_current = os.path.join(debugpath, image_info[2])
                if ~os.path.exists(debugpath_current):
                    os.makedirs(debugpath_current)
                sample_index = -1

                for current_i in tqdm(range(len(num_images_list))):
                    if num_images_list[current_i] < 2:
                        continue
                    sample_index += 1
                    images_data, cams_data, depth_data, image_ref, file_list = load_demon_test_data(
                        demon_path, FLAGS.view_num, current_i)
                    if FLAGS.save_depths:
                        with open(os.path.join(debugpath_current, str(sample_index) + '_file.txt'), "w") as text_file:
                            text_file.write("{:}".format(file_list))
                    out_depth_map_gt = depth_data
                    out_ref_image = images_data[:,0,:,:,:]

                    # test TVSNet
                    out_depth_map_b2, out_depth_map_b3, out_depth_map_b3_up = sess.run([depth_b2, depth_refined, depth_refined_up],
                                                                        feed_dict={images: images_data, cams: cams_data}, options=run_options)

                    if FLAGS.inverse_depth:
                        out_disp_map_gt = out_depth_map_gt.copy()
                        # out_disp_map_b0 = out_depth_map_b0.copy()
                        # out_disp_map_b1 = out_depth_map_b1.copy()
                        out_disp_map_b2 = out_depth_map_b2.copy()
                        out_disp_map_b3 = out_depth_map_b3.copy()
                        out_disp_map_b3_up = out_depth_map_b3_up.copy()
                        out_depth_map_gt[out_depth_map_gt<=0] = float("inf")
                        out_depth_map_gt = 1.0 / out_depth_map_gt
                        # out_depth_map_b0[out_depth_map_b0<=0] = float("inf")
                        # out_depth_map_b0 = 1.0 / out_depth_map_b0
                        # out_depth_map_b1[out_depth_map_b1<=0] = float("inf")
                        # out_depth_map_b1 = 1.0 / out_depth_map_b1
                        out_depth_map_b2[out_depth_map_b2<=0] = float("inf")
                        out_depth_map_b2 = 1.0 / out_depth_map_b2
                        out_depth_map_b3[out_depth_map_b3<=0] = float("inf")
                        out_depth_map_b3 = 1.0 / out_depth_map_b3

                    # error_b0, _ = calc_error(out_depth_map_b0, out_depth_map_gt)
                    # error_b1, _ = calc_error(out_depth_map_b1, out_depth_map_gt)
                    error_b0 = [0.0]
                    error_b1 = [0.0]
                    error_b2, _ = calc_error(out_depth_map_b2, out_depth_map_gt)
                    error_b3, info_b3 = calc_error(out_depth_map_b3, out_depth_map_gt)
                    scene_error_b0.append(error_b0)
                    scene_error_b1.append(error_b1)
                    scene_error_b2.append(error_b2)
                    scene_error_b3.append(error_b3)

                    if FLAGS.save_depths:
                        with open(os.path.join(debugpath_current, str(sample_index) + '_loss.txt'), "w") as text_file:
                            text_file.write('0.mae, 1.rmse, 2.inverse_mae, 3.inverse_rmse, 4.log_mae, 5.log_rmse, 6.scale_invariant_log, 7.abs_relative, 8.squared_relative, 9.mae_normalized, 10+.inlier_ratio_normalized(multiple)\n' )
                            # text_file.write('b0:\n')
                            # simplejson.dump((error_b0.tolist()), text_file)
                            # text_file.write('\nb1:\n')
                            # simplejson.dump((error_b1.tolist()), text_file)
                            text_file.write('\nb2:\n')
                            simplejson.dump((error_b2.tolist()), text_file)
                            text_file.write('\nb3:\n')
                            simplejson.dump((error_b3.tolist()), text_file)
                            text_file.write('\ninfo:\ndepth_num = %4.f, depth_interval = %.4f, depth_min = %.4f, depth_max = %.4f\n' % (info_b3[0], info_b3[1], info_b3[2], info_b3[3]))
                            text_file.write('inlier_threshold:\n')
                            simplejson.dump(info_b3[4], text_file)

                        min_depth = np.min(out_ref_image)
                        max_depth = np.max(out_ref_image)
                        # cv2.imwrite(os.path.join(debugpath_current, str(sample_index) + '_out_ref_image.jpg'), 255*(np.squeeze(np.array(out_ref_image))-min_depth)/(max_depth - min_depth))

                        cmap = 'viridis'
                        # save ground truth
                        # np.save(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_gt.npy'), np.array(out_depth_map_gt))
                        min_depth = np.min(out_disp_map_gt)
                        max_depth = np.max(out_disp_map_gt)
                        plt.imsave(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_gt.jpg'), 255*(np.squeeze(np.array(out_disp_map_gt))-min_depth)/(max_depth - min_depth), cmap=cmap)

                        # np.save(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b0.npy'), np.array(out_depth_map_b0))
                        # plt.imsave(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b0.jpg'), 255*(np.squeeze(np.array(out_disp_map_b0))-min_depth)/(max_depth - min_depth), cmap=cmap)

                        # np.save(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b1.npy'), np.array(out_depth_map_b1))
                        # plt.imsave(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b1.jpg'), 255*(np.squeeze(np.array(out_disp_map_b1))-min_depth)/(max_depth - min_depth), cmap=cmap)

                        # np.save(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b2.npy'), np.array(out_depth_map_b2))
                        plt.imsave(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b2.jpg'), 255*(np.squeeze(np.array(out_disp_map_b2))-min_depth)/(max_depth - min_depth), cmap=cmap)

                        # np.save(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b3.npy'), np.array(out_depth_map_b3))
                        plt.imsave(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b3.jpg'), 255*(np.squeeze(np.array(out_disp_map_b3))-min_depth)/(max_depth - min_depth), cmap=cmap)
                        plt.imsave(os.path.join(debugpath_current, str(sample_index) + '_out_depth_map_b3_up.jpg'), 255*(np.squeeze(np.array(out_disp_map_b3_up))-min_depth)/(max_depth - min_depth), cmap=cmap)

                scene_error_mean_b0 = np.mean(scene_error_b0, axis=0)
                scene_error_mean_b1 = np.mean(scene_error_b1, axis=0)
                scene_error_mean_b2 = np.mean(scene_error_b2, axis=0)
                scene_error_mean_b3 = np.mean(scene_error_b3, axis=0)
                mean_error_b0.append(scene_error_mean_b0)
                mean_error_b1.append(scene_error_mean_b1)
                mean_error_b2.append(scene_error_mean_b2)
                mean_error_b3.append(scene_error_mean_b3)

                np.save(os.path.join(debugpath_current, 'zz_mean.npy'), np.array([scene_error_b0, scene_error_b1, scene_error_b2, scene_error_b3]))
                with open(os.path.join(debugpath_current, 'zz_mean.txt'), "w") as text_file:
                    text_file.write('0.mae, 1.rmse, 2.inverse_mae, 3.inverse_rmse, 4.log_mae, 5.log_rmse, 6.scale_invariant_log, 7.abs_relative, 8.squared_relative, 9.mae_normalized, 10+.inlier_ratio_normalized(multiple)\n' )
                    # text_file.write('b0:\n')
                    # simplejson.dump((scene_error_mean_b0.tolist()), text_file)
                    # text_file.write('\nb1:\n')
                    # simplejson.dump((scene_error_mean_b1.tolist()), text_file)
                    text_file.write('\nb2:\n')
                    simplejson.dump((scene_error_mean_b2.tolist()), text_file)
                    text_file.write('\nb3:\n')
                    simplejson.dump((scene_error_mean_b3.tolist()), text_file)

    return mean_error_b0, mean_error_b1, mean_error_b2, mean_error_b3



def main(argv=None):  # pylint: disable=unused-argument    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
    model_name_split = find_substring('/', FLAGS.pretrained_model_ckpt_path)
    model_name = FLAGS.pretrained_model_ckpt_path[model_name_split[-2]:model_name_split[-1]].replace('/', '-') + '-' + str(FLAGS.ckpt_step)
    time_str = time_str + model_name
    if FLAGS.save_path != '':
        debugpath = '../data/test/' + FLAGS.save_path + '/' + time_str
    else:
        debugpath = '../data/test/' + time_str
    if ~os.path.exists(debugpath):
        os.makedirs(debugpath)

    # modify max_h and max_w to 32x
    factor = 32
    FLAGS.max_h = int(FLAGS.max_h/factor) * factor
    FLAGS.max_w = int(FLAGS.max_w/factor) * factor

    assert FLAGS.data_type in ('colmap', 'demon')
    if FLAGS.data_type == 'colmap':
        # ETH3d low-res or high-res
        dataset_types = ('low', 'high')
        dataset_type = dataset_types[1]

        # image info
        image_infos = []

        assert dataset_type in dataset_types
        if dataset_type == dataset_types[0]:
            base_path = 'ETH3D2017/multi_view_training_rig_undistorted/'
            eth3d_scene_list = ['delivery_area', 'electro', 'forest', 'playground', 'terrains']
            sparse_folder_name = 'rig_calibration_undistorted'
        if dataset_type == dataset_types[1]:
            base_path = 'ETH3D2017/multi_view_training_dslr_undistorted/'
            eth3d_scene_list = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker',
                                'meadow', 'office', 'pipes', 'playground', 'relief', 'relief_2', 'terrace', 'terrains']
            sparse_folder_name = 'dslr_calibration_undistorted'

        for scene_i in range(len(eth3d_scene_list)):
            colmap_folder = base_path + eth3d_scene_list[scene_i]
            image_info = [os.path.join(FLAGS.colmap_data_root, colmap_folder, sparse_folder_name),
                        os.path.join(FLAGS.colmap_data_root, colmap_folder, 'images'), eth3d_scene_list[scene_i]]
            image_infos.append([image_info,'colmap'])

        # run tests
        if FLAGS.dual:
            mean_error_b0, mean_error_b1, mean_error_b2, mean_error_b3 = run_eval_split_dualagg(debugpath, image_infos)

            np.save(os.path.join(debugpath, 'xx_scene.npy'), np.array([mean_error_b0, mean_error_b1, mean_error_b2, mean_error_b3]))
            with open(os.path.join(debugpath, 'xx_scene.txt'), "w") as text_file:
                text_file.write("scene_list; mean_error_b0, mean_error_b1, mean_error_b2, mean_error_b3\n")
                text_file.write('0.mae, 1.rmse, 2.inverse_mae, 3.inverse_rmse, 4.log_mae, 5.log_rmse, 6.scale_invariant_log, 7.abs_relative, 8.squared_relative, 9.mae_normalized, 10+.inlier_ratio_normalized(multiple)\n' )
                simplejson.dump(eth3d_scene_list, text_file)
                text_file.write('\nmean_init:\n')
                simplejson.dump(([mean_error_b0[i].tolist() for i in range(len(mean_error_b0))]), text_file)
                text_file.write('\nagg_init:\n')
                simplejson.dump(([mean_error_b1[i].tolist() for i in range(len(mean_error_b1))]), text_file)
                text_file.write('\nmean:\n')
                simplejson.dump(([mean_error_b2[i].tolist() for i in range(len(mean_error_b2))]), text_file)
                text_file.write('\nagg:\n')
                simplejson.dump(([mean_error_b3[i].tolist() for i in range(len(mean_error_b3))]), text_file)
                
            with open(os.path.join(debugpath, 'yy_overall.txt'), "w") as text_file:
                text_file.write('0.mae, 1.rmse, 2.inverse_mae, 3.inverse_rmse, 4.log_mae, 5.log_rmse, 6.scale_invariant_log, 7.abs_relative, 8.squared_relative, 9.mae_normalized, 10+.inlier_ratio_normalized(multiple)\n')
                text_file.write('mean_init:\n')
                simplejson.dump((np.mean(mean_error_b0, axis=0).tolist()), text_file)
                text_file.write('\nagg_init:\n')
                simplejson.dump((np.mean(mean_error_b1, axis=0).tolist()), text_file)
                text_file.write('\nmean:\n')
                simplejson.dump((np.mean(mean_error_b2, axis=0).tolist()), text_file)
                text_file.write('\nagg:\n')
                simplejson.dump((np.mean(mean_error_b3, axis=0).tolist()), text_file)
                text_file.write("\n{:}.{:}\n".format(FLAGS.pretrained_model_ckpt_path, FLAGS.ckpt_step))
                text_file.write("\nmax_d: {:}, view_num: {:}, interval_scale: {:}\n".format(
                        FLAGS.max_d, FLAGS.view_num, FLAGS.interval_scale))
                        
            # write xlsx
            if 'singleW' in FLAGS.pretrained_model_ckpt_path:
                workbook = xlsxwriter.Workbook(os.path.join(debugpath, str(FLAGS.view_num)+'_dualagg_singleW_'+str(FLAGS.ckpt_step)+'.xlsx'))
                worksheet = workbook.add_worksheet(str(FLAGS.view_num)+'_dualagg_singleW')
            else:
                workbook = xlsxwriter.Workbook(os.path.join(debugpath, str(FLAGS.view_num)+'_dualagg_'+str(FLAGS.ckpt_step)+'.xlsx'))
                worksheet = workbook.add_worksheet(str(FLAGS.view_num)+'_dualagg')
            for row_i, name in enumerate(metrics_namelist):
                worksheet.write(row_i + 1, 0, name) 
            overall_error_b0 = np.mean(mean_error_b0, axis=0).tolist()
            overall_error_b1 = np.mean(mean_error_b1, axis=0).tolist()
            overall_error_b2 = np.mean(mean_error_b2, axis=0).tolist()
            overall_error_b3 = np.mean(mean_error_b3, axis=0).tolist()
            worksheet.write(0, 1, 'mean_init') 
            worksheet.write(0, 2, 'agg_init') 
            worksheet.write(0, 3, 'mean') 
            worksheet.write(0, 4, 'agg') 
            for row_i in range(len(overall_error_b3)):
                worksheet.write(row_i + 1, 1, overall_error_b0[row_i])
                worksheet.write(row_i + 1, 2, overall_error_b1[row_i])
                worksheet.write(row_i + 1, 3, overall_error_b2[row_i])
                worksheet.write(row_i + 1, 4, overall_error_b3[row_i])
            workbook.close()            
        else: 
            _, _, mean_error_b0, mean_error_b1 = run_eval_split_singleagg(debugpath, image_infos)

            np.save(os.path.join(debugpath, 'xx_scene.npy'), np.array([mean_error_b0, mean_error_b1]))
            with open(os.path.join(debugpath, 'xx_scene.txt'), "w") as text_file:
                text_file.write("scene_list; mean_error_b0; mean_error_b1\n")
                text_file.write('0.mae, 1.rmse, 2.inverse_mae, 3.inverse_rmse, 4.log_mae, 5.log_rmse, 6.scale_invariant_log, 7.abs_relative, 8.squared_relative, 9.mae_normalized, 10+.inlier_ratio_normalized(multiple)\n' )
                simplejson.dump(eth3d_scene_list, text_file)
                text_file.write('\nmean:\n')
                simplejson.dump(([mean_error_b0[i].tolist() for i in range(len(mean_error_b0))]), text_file)
                text_file.write('\nagg:\n')
                simplejson.dump(([mean_error_b1[i].tolist() for i in range(len(mean_error_b1))]), text_file)

            with open(os.path.join(debugpath, 'yy_overall.txt'), "w") as text_file:
                text_file.write('0.mae, 1.rmse, 2.inverse_mae, 3.inverse_rmse, 4.log_mae, 5.log_rmse, 6.scale_invariant_log, 7.abs_relative, 8.squared_relative, 9.mae_normalized, 10+.inlier_ratio_normalized(multiple)\n')
                text_file.write('mean:\n')
                simplejson.dump((np.mean(mean_error_b0, axis=0).tolist()), text_file)
                text_file.write('\nagg:\n')
                simplejson.dump((np.mean(mean_error_b1, axis=0).tolist()), text_file)
                text_file.write("\n{:}.{:}\n".format(FLAGS.pretrained_model_ckpt_path, FLAGS.ckpt_step))
                text_file.write("\nmax_d: {:}, view_num: {:}, interval_scale: {:}\n".format(
                        FLAGS.max_d, FLAGS.view_num, FLAGS.interval_scale))
                        
            # write xlsx
            if 'singleW' in FLAGS.pretrained_model_ckpt_path:
                workbook = xlsxwriter.Workbook(os.path.join(debugpath, str(FLAGS.view_num)+'_singleagg_singleW_'+str(FLAGS.ckpt_step)+'.xlsx'))
                worksheet = workbook.add_worksheet(str(FLAGS.view_num)+'_singleagg_singleW')
            else:
                workbook = xlsxwriter.Workbook(os.path.join(debugpath, str(FLAGS.view_num)+'_singleagg_'+str(FLAGS.ckpt_step)+'.xlsx'))
                worksheet = workbook.add_worksheet(str(FLAGS.view_num)+'_singleagg')
            for row_i, name in enumerate(metrics_namelist):
                worksheet.write(row_i + 1, 0, name) 
            overall_error_b0 = np.mean(mean_error_b0, axis=0).tolist()
            overall_error_b1 = np.mean(mean_error_b1, axis=0).tolist()
            worksheet.write(0, 1, 'mean') 
            worksheet.write(0, 2, 'agg') 
            for row_i in range(len(overall_error_b1)):
                worksheet.write(row_i + 1, 1, overall_error_b0[row_i])
                worksheet.write(row_i + 1, 2, overall_error_b1[row_i])
            workbook.close()

        with open(os.path.join(debugpath, 'zz_ckpt.txt'), "w") as text_file:
            text_file.write("{:}.{:}\n".format(FLAGS.pretrained_model_ckpt_path, FLAGS.ckpt_step))
        copyfile(os.path.join(debugpath, 'yy_overall.txt'),
                os.path.join('../data/test/results', time_str+'.txt'))

    else: # demon two-view dataset
        # image info
        image_infos = []

        # demon_scene_list = ['mvs', 'rgbd', 'scenes11', 'sun3d', 'nyu2']
        demon_scene_list = ['mvs', 'rgbd', 'scenes11', 'sun3d']

        for scene_i in range(len(demon_scene_list)):
            demon_folder = os.path.join(FLAGS.demon_data_root, demon_scene_list[scene_i] + '_test')
            if os.path.isfile(os.path.join(demon_folder, "num_images.json")):
                with open(os.path.join(demon_folder, "num_images.json")) as f:
                    num_images_list = (json.load(f))
                num_images_list = np.array(num_images_list)
                num_images_list[num_images_list < 2] = 0
            else:
                print('ImageList init error: num_images.json NOT exists at', demon_folder)
                exit()
            image_info = [demon_folder, num_images_list, demon_scene_list[scene_i]]
            image_infos.append(image_info)

        _, _, mean_error_b0, mean_error_b1 = run_eval_demon(debugpath, image_infos)

        np.save(os.path.join(debugpath, 'xx_scene.npy'), np.array([mean_error_b0, mean_error_b1]))
        with open(os.path.join(debugpath, 'xx_scene.txt'), "w") as text_file:
            text_file.write("scene_list; mean_error_b0; mean_error_b1\n")
            text_file.write('0.mae, 1.rmse, 2.inverse_mae, 3.inverse_rmse, 4.log_mae, 5.log_rmse, 6.scale_invariant_log, 7.abs_relative, 8.squared_relative, 9.mae_normalized, 10+.inlier_ratio_normalized(multiple)\n' )
            simplejson.dump(eth3d_scene_list, text_file)
            text_file.write('\nmean:\n')
            simplejson.dump(([mean_error_b0[i].tolist() for i in range(len(mean_error_b0))]), text_file)
            text_file.write('\nagg:\n')
            simplejson.dump(([mean_error_b1[i].tolist() for i in range(len(mean_error_b1))]), text_file)

        with open(os.path.join(debugpath, 'yy_overall.txt'), "w") as text_file:
            text_file.write('0.mae, 1.rmse, 2.inverse_mae, 3.inverse_rmse, 4.log_mae, 5.log_rmse, 6.scale_invariant_log, 7.abs_relative, 8.squared_relative, 9.mae_normalized, 10+.inlier_ratio_normalized(multiple)\n')
            text_file.write('mean:\n')
            simplejson.dump((np.mean(mean_error_b0, axis=0).tolist()), text_file)
            text_file.write('\nagg:\n')
            simplejson.dump((np.mean(mean_error_b1, axis=0).tolist()), text_file)
            text_file.write("\n{:}.{:}\n".format(FLAGS.pretrained_model_ckpt_path, FLAGS.ckpt_step))
            text_file.write("\nmax_d: {:}, view_num: {:}, interval_scale: {:}\n".format(
                    FLAGS.max_d, FLAGS.view_num, FLAGS.interval_scale))
                    
        # write xlsx
        workbook = xlsxwriter.Workbook(os.path.join(debugpath, str(FLAGS.view_num)+'_twoview.xlsx'))
        worksheet = workbook.add_worksheet(str(FLAGS.view_num)+'_twoview')
        for row_i, name in enumerate(metrics_namelist):
            worksheet.write(row_i + 1, 0, name) 
        overall_error_b0 = np.mean(mean_error_b0, axis=0).tolist()
        overall_error_b1 = np.mean(mean_error_b1, axis=0).tolist()
        worksheet.write(0, 1, 'mean') 
        worksheet.write(0, 2, 'agg') 
        for row_i in range(len(overall_error_b1)):
            worksheet.write(row_i + 1, 1, overall_error_b0[row_i])
            worksheet.write(row_i + 1, 2, overall_error_b1[row_i])
        workbook.close()

        with open(os.path.join(debugpath, 'zz_ckpt.txt'), "w") as text_file:
            text_file.write("{:}.{:}\n".format(FLAGS.pretrained_model_ckpt_path, FLAGS.ckpt_step))
        copyfile(os.path.join(debugpath, 'yy_overall.txt'),
                os.path.join('../data/test/results', time_str+'.txt'))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--view_num', type=int, default=FLAGS.view_num)
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--data_type', type=str, default='colmap')
    parser.add_argument('--pretrained_model_ckpt_path', type=str, default=FLAGS.pretrained_model_ckpt_path)
    parser.add_argument('--ckpt_step', type=int, default=FLAGS.ckpt_step)
    parser.add_argument('--save_depths', type=lambda x: (str(x).lower() in ['true','1','yes','y']), default=FLAGS.save_depths)
    parser.add_argument('--save_path', type=str, default=FLAGS.save_path)
    parser.add_argument('--max_w', type=int, default=FLAGS.max_w)
    parser.add_argument('--max_h', type=int, default=FLAGS.max_h)
    parser.add_argument('--dual', type=lambda x: (str(x).lower() in ['true','1','yes','y']), default=FLAGS.dual)
    args = parser.parse_args()

    FLAGS.ckpt_step = args.ckpt_step
    FLAGS.view_num = args.view_num
    FLAGS.gpu_id = args.gpu_id
    FLAGS.data_type = args.data_type
    FLAGS.pretrained_model_ckpt_path = args.pretrained_model_ckpt_path
    FLAGS.ckpt_step = args.ckpt_step
    FLAGS.save_depths = args.save_depths
    FLAGS.save_path = args.save_path
    FLAGS.max_w = args.max_w
    FLAGS.max_h = args.max_h
    FLAGS.dual = args.dual
    print(Notify.INFO, 'data_type ', FLAGS.data_type, Notify.ENDC)
    print(Notify.INFO, 'dual ', FLAGS.dual, Notify.ENDC)
    
    if FLAGS.gpu_id >= 0:
        print ('Testing MVSNet with %d views, on gpu_id %d' % (args.view_num, FLAGS.gpu_id))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_id)
        FLAGS.num_gpus = 1
    else:
        print ('Testing MVSNet with %d views' % args.view_num)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print (FLAGS.pretrained_model_ckpt_path,  FLAGS.ckpt_step, FLAGS.save_depths, FLAGS.save_path)

    # exit()

    tf.app.run()
