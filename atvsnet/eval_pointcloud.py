#!/usr/bin/env python
from __future__ import print_function

import os
import time
import sys
import math
import argparse
from random import randint
from tqdm import tqdm

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import matplotlib.pyplot as plt
import imageio

sys.path.append("../")
from tools.common import Notify

from preprocess import *
from model import TVSNet, TVSNet_base, TVSNet_base_siamese, TVSNet_refine
from model import cost_volume_aggregation, cost_volume_aggregation_refine
from model import output_conv, output_conv_refine, prob2depth, prob2depth_upsample

FLAGS = tf.app.flags.FLAGS

# params for datasets
tf.app.flags.DEFINE_string('data_root', '../data/',
                           """Path to deepmvs dataset.""")
tf.app.flags.DEFINE_string('savepath', '../eval/pointcloud/',
                           """Path to save depth results.""")
tf.app.flags.DEFINE_string('pretrained_model_ckpt_path', '../model/model.ckpt', 
                           """Path to restore the model.""")

# params for eval
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """Number of GPUs.""")
tf.app.flags.DEFINE_integer('gpu_id', 0, 
                            """GPU index.""")
tf.app.flags.DEFINE_integer('view_num', 8,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 128,
                            """Maximum depth step when training.""")
tf.app.flags.DEFINE_integer('max_w', 896, 
                            """Maximum image width when training. should be multiple of 32""")
tf.app.flags.DEFINE_integer('max_h', 480, 
                            """Maximum image height when training. should be multiple of 32""")
tf.app.flags.DEFINE_float('sample_scale', 0.25,
                          """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """training batch size""")
tf.app.flags.DEFINE_bool('adaptive_scaling', True, 
                            """Let image size to fit the network, including 'scaling', 'cropping'""")
tf.app.flags.DEFINE_boolean('inverse_depth', True,
                            """Use inversed depth.""")


def gen_data_list(dense_folder):
    """ mvs input path list """
    image_folder = os.path.join(dense_folder, 'images')
    cam_folder = os.path.join(dense_folder, 'cams')
    cluster_list_path = os.path.join(dense_folder, 'pair.txt')
    cluster_list = open(cluster_list_path).read().split()

    # for each dataset
    mvs_list = []
    pos = 1
    for i in range(int(cluster_list[0])):
        paths = []
        # ref image
        ref_index = int(cluster_list[pos])
        pos += 1
        ref_image_path = os.path.join(image_folder, ('%08d.jpg' % ref_index))
        ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
        paths.append(ref_image_path)
        paths.append(ref_cam_path)
        # view images
        all_view_num = int(cluster_list[pos])
        pos += 1
        check_view_num = min(FLAGS.view_num - 1, all_view_num)
        for view in range(check_view_num):
            view_index = int(cluster_list[pos + 2 * view])
            view_image_path = os.path.join(image_folder, ('%08d.jpg' % view_index))
            view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
            paths.append(view_image_path)
            paths.append(view_cam_path)
        pos += 2 * all_view_num
        # depth path
        mvs_list.append(paths)
    return mvs_list


def load_data(sample_list, data_index):
    data = sample_list[data_index]

    # read input data
    images = []
    cams = []
    image_index = int(os.path.splitext(os.path.basename(data[0]))[0])
    selected_view_num = int(len(data) / 2)
    interval_scale = 1.0

    for view in range(min(FLAGS.view_num, selected_view_num)):
        image = cv2.imread(data[2 * view])
        cam_file = file_io.FileIO(data[2 * view + 1], mode='r')
        cam = load_cam(cam_file, interval_scale)
        if cam[1][3][2] == 0:
            cam[1][3][2] = FLAGS.max_d
        images.append(image)
        cams.append(cam)

    if selected_view_num < FLAGS.view_num:
        for view in range(selected_view_num, FLAGS.view_num):
            image = cv2.imread(data[0])
            cam_file = file_io.FileIO(data[1], mode='r')
            cam = load_cam(cam_file, interval_scale)
            images.append(image)
            cams.append(cam)
    # print ('range: ', cams[0][1, 3, 0], cams[0][1, 3, 1], cams[0][1, 3, 2], cams[0][1, 3, 3])

    # determine a proper scale to resize input 
    resize_scale = 1
    if FLAGS.adaptive_scaling:
        h_scale = 0
        w_scale = 0
        for view in range(FLAGS.view_num):
            height_scale = float(FLAGS.max_h) / images[view].shape[0]
            width_scale = float(FLAGS.max_w) / images[view].shape[1]
            if height_scale > h_scale:
                h_scale = height_scale
            if width_scale > w_scale:
                w_scale = width_scale
        if h_scale > 1 or w_scale > 1:
            print ("max_h, max_w should < W and H!")
            print (images[view].shape, 'h_scale', h_scale, 'w_scale', w_scale)
            exit(-1)
        resize_scale = h_scale
        if w_scale > h_scale:
            resize_scale = w_scale
    scaled_input_images, scaled_input_cams = scale_mvs_input(images, cams, scale=resize_scale)

    # crop to fit network
    croped_images, croped_cams = crop_mvs_input(scaled_input_images, scaled_input_cams, base_image_size=32)

    # center images
    centered_images = []
    for view in range(FLAGS.view_num):
        centered_images.append(center_image(croped_images[view]))

    if FLAGS.inverse_depth==True:
        for view in range(FLAGS.view_num):
            depth_min = croped_cams[view][1][3][0]
            depth_interval = croped_cams[view][1][3][1]
            if croped_cams[view][1][3][2]>0 and croped_cams[view][1][3][3]>0:
                num_d = croped_cams[view][1][3][2]
                depth_max = croped_cams[view][1][3][3]
            else:
                num_d = FLAGS.max_d
                depth_max = depth_min + float(num_d - 1) * depth_interval
            disp_min = 1.0 / depth_max
            disp_max = 1.0 / depth_min
            disp_interval = (disp_max - disp_min) / FLAGS.max_d
            croped_cams[view][1][3][0] = disp_min
            croped_cams[view][1][3][1] = disp_interval
            croped_cams[view][1][3][2] = FLAGS.max_d
            croped_cams[view][1][3][3] = disp_max

    # use ground truth depth range
    ref_image_path = data[0][0:data[0].rfind('.')+1] + 'txt'
    if os.path.exists(ref_image_path) == True:
        f = open(ref_image_path, "r")
        filename = f.readline()
        ref_image_path = data[0][0:data[0].rfind('/')+1] + filename
        depth_path = ref_image_path.replace('/images/', '/depths/')
        depth_path = depth_path[0:depth_path.rfind('.')+1] + 'exr'
        if os.path.exists(depth_path) == True:
            depth_gt = imageio.imread(depth_path)
            depth_gt = depth_gt[:, :, 0]
            if FLAGS.inverse_depth:
                depth_gt[depth_gt <= 0.0] = float("inf")
                depth_gt = 1.0/(depth_gt)
            disp_max = np.max(depth_gt)
            depth_gt[depth_gt <= 0.0] = float("inf")
            disp_min = np.min(depth_gt)
            disp_interval = (disp_max - disp_min) / FLAGS.max_d
            for view in range(FLAGS.view_num):
                croped_cams[view][1][3][0] = disp_min
                croped_cams[view][1][3][1] = disp_interval
                croped_cams[view][1][3][2] = FLAGS.max_d
                croped_cams[view][1][3][3] = disp_max
        else:
            print(depth_path, 'not exist.')

    scaled_cams = scale_mvs_camera(croped_cams, scale=FLAGS.sample_scale)

    # return mvs input
    scaled_images = []
    for view in range(FLAGS.view_num):
        scaled_images.append(scale_image(croped_images[view], scale=FLAGS.sample_scale))
    scaled_depth = scaled_images[view].copy()
    scaled_depth = scaled_depth[:,:,0:1]
    scaled_images = np.stack(scaled_images, axis=0)
    croped_images = np.stack(croped_images, axis=0)
    scaled_cams = np.stack(scaled_cams, axis=0)

    return np.expand_dims(scaled_images, axis=0), np.expand_dims(centered_images, axis=0), np.expand_dims(scaled_cams, axis=0), np.expand_dims(scaled_depth, axis=0), image_index


def run_eval_pc(savepath, image_infos):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
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
                    depth_agg_b2, confidence_agg_b2 = prob2depth(prob_volume_agg, FLAGS.max_d, depth_start, depth_interval, out_prob_map=True)
                    # prob_volume_mean = tf.reduce_mean(prob_volumes_in, axis=-1, keepdims=False)
                    # depth_mean_b2 = prob2depth(prob_volume_mean, FLAGS.max_d, depth_start, depth_interval, out_prob_map=False)

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

                    refined_cost_volume_agg = cost_volume_aggregation_refine(refined_cost_volumes_in, reuse=False, keepchannel=True)
                    refined_prob_volume_agg = output_conv_refine(refined_cost_volume_agg, reuse=False) 
                    depth_refine_agg, depth_refine_agg_up, confidence_refine_agg, confidence_refine_agg_up = prob2depth_upsample(
                        refined_prob_volume_agg, FLAGS.max_d, depth_start, depth_interval, out_prob_map=True)


        # initialization option
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, report_tensor_allocations_upon_oom = True)

        with tf.Session(config=config) as sess:
            # load pre-trained model
            if FLAGS.pretrained_model_ckpt_path is not None:
                restorer = tf.train.Saver(tf.global_variables())
                restorer.restore(sess, FLAGS.pretrained_model_ckpt_path)
                print(Notify.INFO, 'Pre-trained model restored from %s' % (FLAGS.pretrained_model_ckpt_path), Notify.ENDC)
            else:
                print('FLAGS.pretrained_model_ckpt_path is None !!')
                exit()

            num_queue = len(image_infos)
            for datai in range(num_queue):
                # load data
                format=image_infos[datai][1]

                image_info = image_infos[datai][0]

                dense_path = image_info[0]
                mvs_list = gen_data_list(dense_path)
                # print('dense_path', dense_path, len(mvs_list))

                savepath_current = os.path.join(savepath, image_info[2])
                if os.path.exists(savepath_current) is False:
                    os.makedirs(savepath_current)

                scene_runtime = 0.0
                for current_i in tqdm(range(len(mvs_list))):
                    image_data_raw, images_data, cams_data, depth_data, out_index = load_data(mvs_list, current_i)
                    out_ref_image = image_data_raw[0,0,:,:,:]
                    start_time = time.time()

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
                    out_depth_map_b1, out_prob_map_b1, out_prob_volume_agg, out_cost_volume_agg = sess.run([depth_agg_b2, confidence_agg_b2, prob_volume_agg, cost_volume_agg],
                                                                feed_dict={images: images_data, cams: cams_data,
                                                                filtered_cost_volumes_in: filtered_cost_volumes,
                                                                prob_volumes_in: prob_volumes}, 
                                                                options=run_options)

                    # run refinement
                    refined_cost_volumes = []
                    for view_i in range(1, FLAGS.view_num):
                        out_refined_cost_volume, out_refined_prob_volume = sess.run([refined_cost_volume, refined_prob_volume],
                            feed_dict={images: images_data, cams: cams_data, view_idx: view_i,
                            depth_ref_in: out_depth_map_b1,
                            depth_view_in: depth_views[view_i-1],
                            prob_agg_in: out_prob_volume_agg, 
                            cost_agg_in: out_cost_volume_agg},
                            options=run_options)
                        refined_cost_volumes.append(out_refined_cost_volume)
                    refined_cost_volumes = np.stack(refined_cost_volumes, axis=-1)

                    # run AAM2
                    out_depth_map_b3, out_depth_map_b3_up, out_prob_map_b3, out_prob_map_b3_up = sess.run([depth_refine_agg, depth_refine_agg_up, confidence_refine_agg, confidence_refine_agg_up],
                                                feed_dict={images: images_data, cams: cams_data,
                                                refined_cost_volumes_in: refined_cost_volumes}, 
                                                options=run_options)

                    duration = time.time() - start_time
                    scene_runtime = scene_runtime + duration

                    out_disp_map_b3 = out_depth_map_b3.copy()
                    out_disp_map_b3_up = out_depth_map_b3_up.copy()
                    if FLAGS.inverse_depth:
                        out_depth_map_b3[out_depth_map_b3<=0] = float("inf")
                        out_depth_map_b3 = 1.0 / out_depth_map_b3
                        out_depth_map_b3_up[out_depth_map_b3_up<=0] = float("inf")
                        out_depth_map_b3_up = 1.0 / out_depth_map_b3_up

                    # save_depths
                    output_folder = os.path.join(savepath_current, 'depths_atvsnet')
                    if os.path.exists(output_folder) is False:
                        os.makedirs(output_folder)

                    depth_map_path = output_folder + ('/%08d.pfm' % out_index)
                    prob_map_path = output_folder + ('/%08d_prob.pfm' % out_index)
                    depth_map_up_path = output_folder + ('/%08d_up.pfm' % out_index)
                    prob_map_up_path = output_folder + ('/%08d_prob_up.pfm' % out_index) 
                    out_ref_image_path = output_folder + ('/%08d.jpg' % out_index)
                    out_ref_cam_path = output_folder + ('/%08d.txt' % out_index)

                    out_depth_map_b3 = np.squeeze(out_depth_map_b3)
                    out_prob_map_b3 = np.squeeze(out_prob_map_b3)
                    out_disp_map_b3 = np.squeeze(out_disp_map_b3)
                    out_depth_map_b3_up = np.squeeze(out_depth_map_b3_up)
                    out_prob_map_b3_up = np.squeeze(out_prob_map_b3_up)
                    out_disp_map_b3_up = np.squeeze(out_disp_map_b3_up)
                    ref_cam = cams_data[0, 0, :, :, :]

                    # write to pfm
                    write_pfm(depth_map_path, out_depth_map_b3)
                    write_pfm(prob_map_path, out_prob_map_b3)
                    # write_pfm(depth_map_up_path, out_depth_map_b3_up)
                    # write_pfm(prob_map_up_path, out_prob_map_b3_up)
                    cv2.imwrite(out_ref_image_path, out_ref_image)
                    write_cam(out_ref_cam_path, ref_cam)

                    # write to png
                    cmap = 'viridis'
                    # plt.imsave(output_folder + ('/%08d.png' % out_index), out_disp_map_b3, cmap=cmap)
                    plt.imsave(output_folder + ('/%08d.png' % out_index), out_disp_map_b3_up, cmap=cmap)

                with open(os.path.join(savepath_current, 'zz_runtime.txt'), "w") as text_file:
                    text_file.write('runtime ' + str(scene_runtime))
    return


def main(argv=None):  # pylint: disable=unused-argument
    # ETH3d low-res many-view
    base_path = 'eth3d/'
    eth3d_scene_list = ['lakeside', 'sand_box', 'storage_room', 'storage_room_2', 'tunnel']
    # eth3d_scene_list = ['delivery_area', 'electro', 'forest', 'playground', 'terrains']
    
    savepath = FLAGS.savepath
    if os.path.exists(savepath) is False:
        os.makedirs(savepath)

    # modify max_h and max_w to 32x
    factor = 32
    FLAGS.max_h = int(FLAGS.max_h/factor) * factor
    FLAGS.max_w = int(FLAGS.max_w/factor) * factor

    # image info
    image_infos = []
    for scene_i in range(len(eth3d_scene_list)):
        data_folder = base_path + eth3d_scene_list[scene_i]
        image_info = [os.path.join(FLAGS.data_root, data_folder),
                      os.path.join(FLAGS.data_root, data_folder, 'images'), eth3d_scene_list[scene_i]]
        image_infos.append([image_info,'preprocessed'])

    # run split
    run_eval_pc(savepath, image_infos)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=FLAGS.data_root)
    parser.add_argument('--savepath', type=str, default=FLAGS.savepath)
    parser.add_argument('--pretrained_model_ckpt_path', type=str, default=FLAGS.pretrained_model_ckpt_path)
    parser.add_argument('--view_num', type=int, default=FLAGS.view_num)
    parser.add_argument('--max_w', type=int, default=FLAGS.max_w)
    parser.add_argument('--max_h', type=int, default=FLAGS.max_h)
    args = parser.parse_args()

    FLAGS.data_root = args.data_root
    FLAGS.savepath = args.savepath
    FLAGS.pretrained_model_ckpt_path = args.pretrained_model_ckpt_path
    FLAGS.view_num = args.view_num
    FLAGS.max_w = args.max_w
    FLAGS.max_h = args.max_h
    print ('Evaluate A-TVSNet pointcloud with %d views' % (args.view_num))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_id)

    tf.app.run()
