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
from model import cost_volume_aggregation, cost_volume_aggregation_refine
from model import output_conv, output_conv_refine, prob2depth, prob2depth_upsample


FLAGS = tf.app.flags.FLAGS


# params for datasets
tf.app.flags.DEFINE_string('data_type', 'demon',
                           """demon or colmap(i.e. eth3d).""")
tf.app.flags.DEFINE_string('demon_data_root', '../../../../../dataset/demon_test/',
                           """Path to demon format dataset.""")
tf.app.flags.DEFINE_string('colmap_data_root', '../../../../../dataset/',
                           """Path to colmap format dataset.""")

# params for training
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """Number of GPUs.""")
tf.app.flags.DEFINE_integer('gpu_id', 0,
                            """GPU index.""")
tf.app.flags.DEFINE_integer('view_num', 2,
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

tf.app.flags.DEFINE_boolean('save_depths', False,
                            """save output results.""")
tf.app.flags.DEFINE_boolean('dual', True,
                            """dual AAM flag.""")
tf.app.flags.DEFINE_boolean('depth_upsample', True,
                            """original size depth map flag.""")

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

    if not FLAGS.depth_upsample:
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
    # return np.expand_dims(centered_images, axis=0), np.expand_dims(scaled_cams, axis=0), np.expand_dims(croped_depth, axis=0), image_ref, file_list
    return np.expand_dims(croped_images, axis=0), np.expand_dims(scaled_cams, axis=0), np.expand_dims(croped_depth, axis=0), image_ref, file_list
    

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

    if not FLAGS.depth_upsample:
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
    # return np.expand_dims(centered_images, axis=0), np.expand_dims(scaled_cams, axis=0), np.expand_dims(croped_depth, axis=0), image_ref, data
    return np.expand_dims(croped_images, axis=0), np.expand_dims(scaled_cams, axis=0), np.expand_dims(croped_depth, axis=0), image_ref, data


def run_eval_split_dualagg(debugpath, image_infos):

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

        # initialization option
        # init_op = tf.keras.initializers.he_normal(seed=42)#tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, report_tensor_allocations_upon_oom = True)

        with tf.Session(config=config) as sess:
            # # load pre-trained model
            # if FLAGS.pretrained_model_ckpt_path is not None:
            #     restorer = tf.train.Saver(tf.global_variables())
            #     restorer.restore(
            #         sess, '-'.join([FLAGS.pretrained_model_ckpt_path]))
            #     print(Notify.INFO, 'Pre-trained model restored from %s' %
            #           ('-'.join([FLAGS.pretrained_model_ckpt_path])), Notify.ENDC)
            # else:
            #     print('FLAGS.pretrained_model_ckpt_path is None !!')
            #     exit()

            mean_error_b0 = []
            mean_error_b1 = []
            mean_error_b2 = [] 
            mean_error_b3 = [] 
            for datai in range(num_queue):
                # load data
                format=image_infos[datai][1]
                assert format == 'colmap'

                image_info = image_infos[datai][0]
                print('loading image format (colmap), {:}'.format(image_info[1]))

                sparse_path = image_info[0]
                image_path = image_info[1]
                mvs_list, colmap_sparse = colmap.gen_pipeline_mvs_list(sparse_path, image_path)

                debugpath_current = os.path.join(debugpath, image_info[2])
                if not os.path.exists(debugpath_current):
                    os.makedirs(debugpath_current)
                sample_index = -1

                for current_i in tqdm(range(len(mvs_list))):
                    sample_index += 1
                    images_data, cams_data, depth_data, image_ref, file_list = load_colmap_data(
                        mvs_list, colmap_sparse, FLAGS.view_num, current_i)
                    with open(os.path.join(debugpath_current, str(sample_index) + '_file.txt'), "w") as text_file:
                        text_file.write("{:}".format(file_list))
                    out_depth_map_gt = depth_data
                    out_ref_image = images_data[:,0,:,:,:]

                    if os.path.exists(os.path.join(debugpath_current, str(sample_index))) is False:
                        os.makedirs(os.path.join(debugpath_current, str(sample_index)))
                    out_disp_map_gt = out_depth_map_gt.copy()
                    out_depth_map_gt[out_depth_map_gt<=0] = float("inf")
                    out_depth_map_gt = 1.0 / out_depth_map_gt
                    min_depth = np.min(out_disp_map_gt)
                    max_depth = np.max(out_disp_map_gt)
                    plt.imsave(os.path.join(debugpath_current, str(sample_index), '0_gt.jpg'), 255*(np.squeeze(np.array(out_disp_map_gt))-min_depth)/(max_depth - min_depth), cmap='viridis')
                    np.save(os.path.join(debugpath_current, str(sample_index), '0_gt.npy'), np.squeeze(np.array(out_depth_map_gt), axis=0))

                    for view_i in range(0, FLAGS.view_num):
                        current_image = images_data[0,view_i,:,:,:] 
                        current_cam = cams_data[0,view_i,:,:,:] 
                        cv2.imwrite(os.path.join(debugpath_current, str(sample_index), str(view_i)+'.jpg'), current_image)
                        np.save(os.path.join(debugpath_current, str(sample_index), str(view_i)+'_cam.npy'), current_cam)
                    continue

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

                    # view_idx = 1
                    # _, refined_prob_volume = TVSNet(images, cams, FLAGS.max_d, depth_start, depth_interval, view_i=view_idx, ref_i=0)
                    # depth_refined, depth_refined_up = prob2depth_upsample(refined_prob_volume, FLAGS.max_d, depth_start, depth_interval, out_prob_map=False, normalized=False)                    

        # initialization option
        # init_op = tf.keras.initializers.he_normal(seed=42)#tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, report_tensor_allocations_upon_oom = True)

        with tf.Session(config=config) as sess:
            # # load pre-trained model
            # if FLAGS.pretrained_model_ckpt_path is not None:
            #     restorer = tf.train.Saver(tf.global_variables())
            #     restorer.restore(
            #         sess, '-'.join([FLAGS.pretrained_model_ckpt_path]))
            #     print(Notify.INFO, 'Pre-trained model restored from %s' %
            #           ('-'.join([FLAGS.pretrained_model_ckpt_path])), Notify.ENDC)
            # else:
            #     print('FLAGS.pretrained_model_ckpt_path is None !!')
            #     exit()

            mean_error_b0 = []
            mean_error_b1 = []
            mean_error_b2 = [] 
            mean_error_b3 = [] 
            for datai in range(num_queue):
                # load data
                image_info = image_infos[datai]
                print('loading image format (demon), {:}'.format(image_info[-1]))

                demon_path = image_info[0]
                num_images_list = image_info[1]

                debugpath_current = os.path.join(debugpath, image_info[2])
                if not os.path.exists(debugpath_current):
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

                    if os.path.exists(os.path.join(debugpath_current, str(sample_index))) is False:
                        os.makedirs(os.path.join(debugpath_current, str(sample_index)))
                    out_disp_map_gt = out_depth_map_gt.copy()
                    out_depth_map_gt[out_depth_map_gt<=0] = float("inf")
                    out_depth_map_gt = 1.0 / out_depth_map_gt
                    min_depth = np.min(out_disp_map_gt)
                    max_depth = np.max(out_disp_map_gt)
                    plt.imsave(os.path.join(debugpath_current, str(sample_index), '0_gt.jpg'), 255*(np.squeeze(np.array(out_disp_map_gt))-min_depth)/(max_depth - min_depth), cmap='viridis')
                    np.save(os.path.join(debugpath_current, str(sample_index), '0_gt.npy'), np.squeeze(np.array(out_depth_map_gt), axis=0))

                    for view_i in range(0, FLAGS.view_num):
                        current_image = images_data[0,view_i,:,:,:] 
                        current_cam = cams_data[0,view_i,:,:,:] 
                        cv2.imwrite(os.path.join(debugpath_current, str(sample_index), str(view_i)+'.jpg'), current_image)
                        np.save(os.path.join(debugpath_current, str(sample_index), str(view_i)+'_cam.npy'), current_cam)
                    continue

    return mean_error_b0, mean_error_b1, mean_error_b2, mean_error_b3



def main(argv=None):  # pylint: disable=unused-argument    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
    debugpath = '../data/test/'
    if not os.path.exists(debugpath):
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
        mean_error_b0, mean_error_b1, mean_error_b2, mean_error_b3 = run_eval_split_dualagg(debugpath, image_infos)    
        return
          
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

        return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--view_num', type=int, default=FLAGS.view_num)
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--data_type', type=str, default=FLAGS.data_type)
    parser.add_argument('--pretrained_model_ckpt_path', type=str, default=FLAGS.pretrained_model_ckpt_path)
    parser.add_argument('--save_depths', type=lambda x: (str(x).lower() in ['true','1','yes','y']), default=FLAGS.save_depths)
    parser.add_argument('--max_w', type=int, default=FLAGS.max_w)
    parser.add_argument('--max_h', type=int, default=FLAGS.max_h)
    parser.add_argument('--dual', type=lambda x: (str(x).lower() in ['true','1','yes','y']), default=FLAGS.dual)
    args = parser.parse_args()

    FLAGS.view_num = args.view_num
    FLAGS.gpu_id = args.gpu_id
    FLAGS.data_type = args.data_type
    FLAGS.pretrained_model_ckpt_path = args.pretrained_model_ckpt_path
    FLAGS.save_depths = args.save_depths
    FLAGS.max_w = args.max_w
    FLAGS.max_h = args.max_h
    FLAGS.dual = args.dual
    print(Notify.INFO, 'data_type ', FLAGS.data_type, Notify.ENDC)
    print(Notify.INFO, 'dual ', FLAGS.dual, Notify.ENDC)
    
    if FLAGS.gpu_id >= 0:
        print ('Testing A-TVSNet with %d views, on gpu_id %d' % (args.view_num, FLAGS.gpu_id))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_id)
        FLAGS.num_gpus = 1
    else:
        print ('Testing A-TVSNet with %d views' % args.view_num)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print (FLAGS.pretrained_model_ckpt_path, FLAGS.save_depths)

    # exit()

    tf.app.run()
