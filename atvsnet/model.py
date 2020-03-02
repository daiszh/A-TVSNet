import sys
import math
import tensorflow as tf
import numpy as np

sys.path.append("../")
from cnn_wrapper.atvsnet import *
from homography_warping import get_homographies, homography_warping, homography_warping_by_depth, transform_depth, get_visual_hull

FLAGS = tf.app.flags.FLAGS


def get_propability_map(cv, depth_map, depth_start, depth_interval):
    # get probability map from cost volume

    def _repeat_(x, num_repeats):
        # repeat each element num_repeats times
        x = tf.reshape(x, [-1])
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1, 1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    shape = tf.shape(depth_map)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    depth = tf.shape(cv)[1]

    # byx coordinate, batched & flattened
    b_coordinates = tf.range(batch_size)
    y_coordinates = tf.range(height)
    x_coordinates = tf.range(width)
    b_coordinates, y_coordinates, x_coordinates = tf.meshgrid(
        b_coordinates, y_coordinates, x_coordinates)
    b_coordinates = _repeat_(b_coordinates, batch_size)
    y_coordinates = _repeat_(y_coordinates, batch_size)
    x_coordinates = _repeat_(x_coordinates, batch_size)

    # d coordinate (floored and ceiled), batched & flattened
    d_coordinates = tf.reshape((depth_map - depth_start) / depth_interval, [-1])
    d_coordinates_left0 = tf.clip_by_value(tf.cast(tf.floor(d_coordinates), 'int32'), 0, depth - 1)
    d_coordinates_left1 = tf.clip_by_value(d_coordinates_left0 - 1, 0, depth - 1)
    d_coordinates1_right0 = tf.clip_by_value(tf.cast(tf.ceil(d_coordinates), 'int32'), 0, depth - 1)
    d_coordinates1_right1 = tf.clip_by_value(d_coordinates1_right0 + 1, 0, depth - 1)

    # voxel coordinates
    voxel_coordinates_left0 = tf.stack(
        [b_coordinates, d_coordinates_left0, y_coordinates, x_coordinates], axis=1)
    voxel_coordinates_left1 = tf.stack(
        [b_coordinates, d_coordinates_left1, y_coordinates, x_coordinates], axis=1)
    voxel_coordinates_right0 = tf.stack(
        [b_coordinates, d_coordinates1_right0, y_coordinates, x_coordinates], axis=1)
    voxel_coordinates_right1 = tf.stack(
        [b_coordinates, d_coordinates1_right1, y_coordinates, x_coordinates], axis=1)

    # get probability image by gathering and interpolation
    prob_map_left0 = tf.gather_nd(cv, voxel_coordinates_left0)
    prob_map_left1 = tf.gather_nd(cv, voxel_coordinates_left1)
    prob_map_right0 = tf.gather_nd(cv, voxel_coordinates_right0)
    prob_map_right1 = tf.gather_nd(cv, voxel_coordinates_right1)
    prob_map = prob_map_left0 + prob_map_left1 + prob_map_right0 + prob_map_right1
    prob_map = tf.reshape(prob_map, [batch_size, height, width, 1])

    return prob_map


def upsample_prob_vol(prob_vol, up_scale=4):
    prob_shape = tf.shape(prob_vol)
    up_scale = tf.constant(up_scale, dtype=tf.int32)
    prob_vol_reshape = tf.transpose(prob_vol, [0, 2, 3, 1])
    prob_vol_reshape_up = tf.image.resize_images(prob_vol_reshape, size=(
        prob_shape[2]*up_scale, prob_shape[3]*up_scale), 
        method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    prob_vol_upsample = tf.transpose(prob_vol_reshape_up, [0, 3, 1, 2])    
    return prob_vol_upsample


# output moudle
def prob2depth(prob_volume, depth_num, depth_start, depth_interval, out_prob_map=False):
    # input:
    # prob_volume (B, D, H, W)
    # output:
    # estimated_depth_map (B, H, W, 1)
    # *prob_map (B, H, W, 1)

    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    # depth map by softArgmin
    # estimated_depth_map, size of (B, H, W, 1)
    with tf.name_scope('soft_arg_min'):
        # probability volume by soft max
        probability_volume = tf.nn.softmax(tf.scalar_mul(-1, prob_volume), axis=1, name='prob_volume')
        # depth image by soft argmin
        volume_shape = tf.shape(probability_volume)
        soft_2d = []
        for i in range(FLAGS.batch_size):
            soft_1d = tf.linspace(depth_start[i], depth_end[i], tf.cast(depth_num, tf.int32))
            soft_2d.append(soft_1d)
        soft_2d = tf.reshape(tf.stack(soft_2d, axis=0), [volume_shape[0], volume_shape[1], 1, 1])
        soft_4d = tf.tile(soft_2d, [1, 1, volume_shape[2], volume_shape[3]])
        estimated_depth_map = tf.reduce_sum(soft_4d * probability_volume, axis=1)
        estimated_depth_map = tf.expand_dims(estimated_depth_map, axis=3)

    if out_prob_map:
        # probability map
        prob_map = get_propability_map(probability_volume, estimated_depth_map, depth_start, depth_interval)
        return estimated_depth_map, prob_map
    else:
        return estimated_depth_map


# output moudle and upsample prob volume
def prob2depth_upsample(prob_volume, depth_num, depth_start, depth_interval, out_prob_map=False):
    # input:
    # prob_volume (B, D, H, W)
    # output:
    # estimated_depth_map (B, H, W, 1)
    # estimated_depth_map_up (B, 4H, 4W, 1)
    # *prob_map (B, H, W, 1)

    prob_volume_up = upsample_prob_vol(prob_volume)
    if out_prob_map:
        estimated_depth_map_up, prob_map_up = prob2depth(prob_volume_up, depth_num, depth_start, depth_interval, out_prob_map=True)
        estimated_depth_map, prob_map = prob2depth(prob_volume, depth_num, depth_start, depth_interval, out_prob_map=True)
        return estimated_depth_map, estimated_depth_map_up, prob_map, prob_map_up
    else:
        estimated_depth_map_up = prob2depth(prob_volume_up, depth_num, depth_start, depth_interval, out_prob_map=False)
        estimated_depth_map = prob2depth(prob_volume, depth_num, depth_start, depth_interval, out_prob_map=False)
        return estimated_depth_map, estimated_depth_map_up


def output_conv(cost_volume, reuse=tf.AUTO_REUSE):
    # (B, H, W, D, C) ---conv---> (B, H, W, D)
    prob_vol_tower = OutputConv({'data': cost_volume}, is_training=True, reuse=reuse)
    return tf.squeeze(prob_vol_tower.get_output(), axis=-1)

def output_conv_refine(cost_volume, reuse=tf.AUTO_REUSE):
    # (B, H, W, D, C) ---conv---> (B, H, W, D)
    prob_vol_tower = OutputConv_refine({'data': cost_volume}, is_training=True, reuse=reuse)
    return tf.squeeze(prob_vol_tower.get_output(), axis=-1)


# low-level feature extraction
def extract_feature_shallow(images, ref_id=0, view_id=1):
    # low-level image feature extraction
    ref_image = tf.squeeze(tf.slice(images, [0, ref_id, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_tower = ResNetDS2SPP_shallow_f16({'data': ref_image}, is_training=True, reuse=tf.AUTO_REUSE)
    ref_feature = ref_tower.get_output() # size of (B, H, W, F)

    view_image = tf.squeeze(tf.slice(images, [0, view_id, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    view_tower = ResNetDS2SPP_shallow_f16({'data': view_image}, is_training=True, reuse=tf.AUTO_REUSE)
    view_feature = view_tower.get_output()  # size of (B, H, W, F)

    return ref_feature, view_feature


def build_cost_volume(ref_feature, view_feature, cams, depth_num, depth_start, depth_interval, ref_id, view_id, output_homo=False, warp_ref=False):
    # input:
    # ref_feature/view_feature (B, H, W, F)
    # cams (B, N, 2, 4, 4), N=2
    # output:
    # cost_volume (B, D, H, W, 2F)
    # *view_homographies (B, D, 3, 3)

    # reference cam
    ref_cam = tf.squeeze(tf.slice(cams, [0, ref_id, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
    # get view homographies
    view_cam = tf.squeeze(tf.slice(cams, [0, view_id, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
    view_homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                    depth_start=depth_start, depth_interval=depth_interval)

    # build cost volume by differentialble homography
    with tf.name_scope('cost_volume_homography'):
        # ref2ref cost volume
        if warp_ref:
            depth_costs = []
            ref_homography = get_homographies(ref_cam, ref_cam, depth_num=depth_num,
                                            depth_start=depth_start, depth_interval=depth_interval)
            for d in range(depth_num): 
                homography = tf.slice(ref_homography, begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
                warped_ref_feature = homography_warping(ref_feature, homography)
                depth_costs.append(warped_ref_feature)
            cost_volume = tf.stack(depth_costs, axis=1) # size of (B, D, H, W, F)
        else:
            cost_volume = tf.tile(tf.expand_dims(ref_feature, axis=1), [1, depth_num, 1, 1, 1])

        # view2ref cost volume
        depth_costs = []
        for d in range(depth_num): 
            homography = tf.slice(view_homographies, begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
            homography = tf.squeeze(homography, axis=1)
            warped_view_feature = homography_warping(view_feature, homography)                
            depth_costs.append(warped_view_feature)
        cost_volume = tf.concat([cost_volume, tf.stack(depth_costs, axis=1)], axis=-1) # size of (B, D, H, W, 2F)

    if output_homo:
        return cost_volume, view_homographies
    else:
        return cost_volume


# cost volume regularization module (CRM)
def cost_volume_reasoning(cost_volume, output_prob=True, output_filtered_cost=False, reuse=tf.AUTO_REUSE):
    # input:
    # cost_volume (B, D, H, W, 2F)
    # output:
    # filtered_cost_volume_b2 (B, D, H, W) if output_prob is True
    # filtered_cost_volume_b2 (B, D, H, W, C) if output_prob is False
    # output both if output_prob&&output_filtered_cost is True

    # build cost volume by differentialble homography
    with tf.name_scope('cost_volume_resoning'):
        if output_prob:
            filtered_cost_volume_tower = StackedUNet_prob({'data': cost_volume}, is_training=True, reuse=reuse)       
            filtered_cost_volume_b2 = tf.squeeze(filtered_cost_volume_tower.get_output(), axis=-1) # size of (B, D, H, W)
            if output_filtered_cost:
                return filtered_cost_volume_b2, filtered_cost_volume_tower.get_output_by_name('conv_b2_6_1')
        else:
            filtered_cost_volume_tower = StackedUNet({'data': cost_volume}, is_training=True, reuse=reuse)       
            filtered_cost_volume_b2 = filtered_cost_volume_tower.get_output_by_name('conv_b2_6_1') # size of (B, D, H, W, C)

    return filtered_cost_volume_b2
    

# refinement network
def refinement(init_depth_images, cams, 
    depth_num, depth_start, depth_interval,
    images, prob_vol,
    ref_id, view_id, 
    view_homographies=None,
    num_depths = None,
    depth_ref_id=None, depth_view_id=None):
    # input: N=2
    # init_depth_images (B, N, H, W, 1)
    # cams (B, N, 2, 4, 4)
    # view_homographies (B, D, 3, 3)
    # images (B, N, H, W, 3), N=2
    # prob_vol (B, D, H, W)
    # output:
    # refined_prob_vol (B, D, H, W)

    if depth_ref_id == None:
        depth_ref_id = ref_id
    if depth_view_id == None:
        depth_view_id = view_id
    if num_depths == None:
        num_depths = FLAGS.view_num

    prob_vol = tf.expand_dims(prob_vol, axis=-1)

    init_depth_image = tf.squeeze(tf.slice(init_depth_images, [0, depth_ref_id, 0, 0, 0], [-1, 1, -1, -1, 1]), axis=1) # (B, H, W, 1)
    init_depth_image_view = tf.squeeze(tf.slice(init_depth_images, [0, depth_view_id, 0, 0, 0], [-1, 1, -1, -1, 1]), axis=1) # (B, H, W, 1)

    ref_cam = tf.squeeze(tf.slice(cams, [0, ref_id, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
    view_cam = tf.squeeze(tf.slice(cams, [0, view_id, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
    init_depth_image_view_trans = transform_depth(init_depth_image_view, view_cam, ref_cam) # depth value to ref coor (B, H, W, 1)
    
    if view_homographies is None:
        view_homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                        depth_start=depth_start, depth_interval=depth_interval)

    # image feature extraction
    ref_feature, view_feature = extract_feature_shallow(images, ref_id, view_id)
    chan_num = ref_feature.get_shape()[3]

    #####
    # photometric cost volume (L1 norm) with low level feature
    #####
    with tf.name_scope('global_refine_photo_cost_volume'):
        # view2ref cost volume
        photo_costs = []
        for d in range(depth_num):
            homography = tf.slice(view_homographies, begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
            homography = tf.squeeze(homography, axis=1)
            warped_view_feature, valid_mask = homography_warping(view_feature, homography, output_mask=True)   
            err_photo_L1 = tf.multiply(tf.abs(warped_view_feature - ref_feature), 
                                                    tf.cast(tf.tile(valid_mask, [1,1,1,chan_num]), ref_feature.dtype))                         
            photo_costs.append(err_photo_L1)
        cost_vol_photo = tf.stack(photo_costs, axis=1) # size of (B, D, H, W, F)

    #####
    # geometric cost volume (L1 norm)
    #####
    with tf.name_scope('global_refine_geo_cost_volume'):
        cost_volume_geo_view = []
        cost_volume_geo_ref = []
        for d in range(depth_num):
            ref_depth_value = depth_start + tf.cast(d, tf.float32) * depth_interval
            cost_volume_geo_ref.append(tf.abs(init_depth_image - ref_depth_value) / depth_interval / tf.cast(depth_num, tf.float32))
                        
            homography = tf.slice(view_homographies, begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
            homography = tf.squeeze(homography, axis=1)
            warped_view_init_depth_inRef, view_valid_mask = homography_warping(init_depth_image_view_trans, homography, output_mask = True)
            geo_errors = tf.multiply(tf.abs(warped_view_init_depth_inRef - ref_depth_value) / depth_interval / tf.cast(depth_num, tf.float32), 
                                     tf.cast(tf.tile(view_valid_mask, [1, 1, 1, chan_num]), init_depth_image.dtype))
            cost_volume_geo_view.append(geo_errors)
            
        cost_volume_geo_ref = tf.stack(cost_volume_geo_ref, axis=1) # size of (B, D, H, W, 1)
        cost_volume_geo = tf.concat([cost_volume_geo_ref, tf.stack(cost_volume_geo_view, axis=1)], axis=-1) # size of (B, D, H, W, 2)


    #####
    # photo and geo error
    #####
    with tf.name_scope('global_refine_photo_geo_error'): 
        # prob_vol_normalized = tf.nn.softmax(tf.scalar_mul(-1, prob_vol), axis=1, name='prob_volume')
        # photo error
        warped_feature, valid_mask_photo = homography_warping_by_depth(view_feature, ref_cam, view_cam, init_depth_image, output_mask=True)
        photo_err = tf.multiply(tf.abs(warped_feature - ref_feature), tf.cast(tf.tile(valid_mask_photo, [1,1,1,ref_feature.get_shape()[3]]), ref_feature.dtype))
        photo_err = tf.tile(tf.expand_dims(photo_err, axis=1), [1, depth_num, 1, 1, 1])
        # geo error
        view_depth_warped2ref, valid_mask_geo = homography_warping_by_depth(init_depth_image_view_trans, ref_cam, view_cam, init_depth_image, 
                                                                            output_mask=True, method='nearest')
        geo_err = tf.multiply(tf.abs(view_depth_warped2ref - init_depth_image), tf.cast(valid_mask_geo, init_depth_image.dtype))
        geo_err = tf.tile(tf.expand_dims(geo_err, axis=1), [1, depth_num, 1, 1, 1])

    #####
    # visual hull
    #####
    with tf.name_scope('global_refine_visual_hull'):
        # visual_hull in(B, N, H, W), out(B, D, H, W, 1)
        vis_hull = get_visual_hull(tf.squeeze(init_depth_images, axis=-1), cams, depth_num,
                                   depth_start, depth_interval, ref_id=ref_id, view_num=num_depths)

    #####
    # refinement network
    #####
    ref_cost_volume = tf.tile(tf.expand_dims(ref_feature, axis=1), [1, depth_num, 1, 1, 1])
    ref_geo_volume = tf.tile(tf.expand_dims(init_depth_image, axis=1), [1, depth_num, 1, 1, 1])

    prob_vol_residual_tower = CostVolRefineNet({
        'photo_group': tf.concat([cost_vol_photo, photo_err, ref_cost_volume], axis=-1),
        'geo_group': tf.concat([cost_volume_geo, geo_err, ref_geo_volume], axis=-1),
        'prob_vol': prob_vol,
        'vis_hull': vis_hull
        }, is_training=True, reuse=tf.AUTO_REUSE)

    return prob_vol_residual_tower.get_output_by_name('global_refine_3dconv6_1'), tf.squeeze(prob_vol_residual_tower.get_output(),axis=-1)


#################################################################
######################## A-TVSNet ###############################
#################################################################

def TVSNet(images, cams, depth_num, depth_start, depth_interval, view_i, ref_i=0):
    # image feature extraction (FEM)
    ref_image = tf.squeeze(tf.slice(images, [0, ref_i, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_tower = ResNetDS2SPP({'data': ref_image}, is_training=True, reuse=tf.AUTO_REUSE)
    ref_feature = ref_tower.get_output() # size of (B, H, W, F)

    view_image = tf.squeeze(tf.slice(images, [0, view_i, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    view_tower = ResNetDS2SPP({'data': view_image}, is_training=True, reuse=tf.AUTO_REUSE)
    view_feature = view_tower.get_output()

    # cost volume regularization (CRM)
    cost_vol_view = build_cost_volume(view_feature, ref_feature, cams, depth_num, depth_start, depth_interval, ref_id=view_i, view_id=0)
    prob_vol_view  = cost_volume_reasoning(cost_vol_view, output_filtered_cost=False)
    depth_view = prob2depth(prob_vol_view, depth_num, depth_start, depth_interval, out_prob_map=False)

    cost_vol = build_cost_volume(ref_feature, view_feature, cams, depth_num, depth_start, depth_interval, ref_id=0, view_id=view_i) # cost_volume (B, D, H, W, 2F)
    prob_vol_b2, filtered_cost_volume = cost_volume_reasoning(cost_vol, output_filtered_cost=True)
    depth_b2 = prob2depth(prob_vol_b2, depth_num, depth_start, depth_interval, out_prob_map=False)

    # refinement
    init_depth_images = [depth_b2, depth_view]
    init_depth_images = tf.stack(init_depth_images, axis=1) # (B, N, H, W, 1)
    _, prob_residual = refinement(init_depth_images, cams,
        depth_num, depth_start, depth_interval,
        images, prob_vol_b2,
        ref_id=ref_i, view_id=view_i, 
        view_homographies=None,
        num_depths=2,
        depth_ref_id=0, depth_view_id=1)

    refined_prob_vol = prob_vol_b2 + prob_residual
    return refined_prob_vol


def TVSNet_base(images, cams, depth_num, depth_start, depth_interval, view_i, ref_i=0):
    # image feature extraction (FEM)
    ref_image = tf.squeeze(tf.slice(images, [0, ref_i, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_tower = ResNetDS2SPP({'data': ref_image}, is_training=True, reuse=tf.AUTO_REUSE)
    ref_feature = ref_tower.get_output() # size of (B, H, W, F)

    view_image = tf.squeeze(tf.slice(images, [0, view_i, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    view_tower = ResNetDS2SPP({'data': view_image}, is_training=True, reuse=tf.AUTO_REUSE)
    view_feature = view_tower.get_output()

    # cost volume regularization (CRM)
    cost_vol = build_cost_volume(ref_feature, view_feature, cams, depth_num, depth_start, depth_interval, ref_id=0, view_id=view_i) # (B, D, H, W, 2F)
    prob_vol_b2, filtered_cost_volume = cost_volume_reasoning(cost_vol, output_filtered_cost=True) # (B, D, H, W)
    depth_b2 = prob2depth(prob_vol_b2, depth_num, depth_start, depth_interval, out_prob_map=False)

    return depth_b2, prob_vol_b2, filtered_cost_volume


def TVSNet_base_siamese(images, cams, depth_num, depth_start, depth_interval, view_i, ref_i=0):
    # image feature extraction (FEM)
    ref_image = tf.squeeze(tf.slice(images, [0, ref_i, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_tower = ResNetDS2SPP({'data': ref_image}, is_training=True, reuse=tf.AUTO_REUSE)
    ref_feature = ref_tower.get_output() # size of (B, H, W, F)

    view_image = tf.squeeze(tf.slice(images, [0, view_i, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    view_tower = ResNetDS2SPP({'data': view_image}, is_training=True, reuse=tf.AUTO_REUSE)
    view_feature = view_tower.get_output()

    # cost volume regularization (CRM)
    cost_vol = build_cost_volume(ref_feature, view_feature, cams, depth_num, depth_start, depth_interval, ref_id=0, view_id=view_i) # (B, D, H, W, 2F)
    prob_vol_b2, filtered_cost_volume = cost_volume_reasoning(cost_vol, output_filtered_cost=True) # (B, D, H, W)
    depth_b2 = prob2depth(prob_vol_b2, depth_num, depth_start, depth_interval, out_prob_map=False)

    cost_vol_view = build_cost_volume(view_feature, ref_feature, cams, depth_num, depth_start, depth_interval, ref_id=view_i, view_id=0)
    prob_vol_view = cost_volume_reasoning(cost_vol_view, output_filtered_cost=False, reuse=tf.AUTO_REUSE)
    depth_view = prob2depth(prob_vol_view, depth_num, depth_start, depth_interval, out_prob_map=False)

    return depth_b2, prob_vol_b2, filtered_cost_volume, depth_view


def TVSNet_feature_extraction(images, view_i):
    # image feature extraction (FEM) for single image
    image = tf.squeeze(tf.slice(images, [0, view_i, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    feature_tower = ResNetDS2SPP({'data': image}, is_training=True, reuse=tf.AUTO_REUSE)
    feature = feature_tower.get_output() # size of (B, H, W, F)
    return feature


def TVSNet_refine(depth_b2, depth_view, prob_vol_b2, filtered_cost_volume, images, cams, depth_num, depth_start, depth_interval, view_i, ref_i=0):
    init_depth_images = [depth_b2, depth_view]
    init_depth_images = tf.stack(init_depth_images, axis=1) # (B, N, H, W, 1)
    cost_residual, prob_residual = refinement(init_depth_images, cams,
        depth_num, depth_start, depth_interval,
        images, prob_vol_b2,
        ref_id=ref_i, view_id=view_i, 
        view_homographies=None,
        num_depths=2,
        depth_ref_id=0, depth_view_id=1)

    refined_cost_volume = filtered_cost_volume + cost_residual
    refined_prob_vol = prob_vol_b2 + prob_residual
    return refined_prob_vol, refined_cost_volume


# AAM1
def cost_volume_aggregation(cost_volumes, reuse=tf.AUTO_REUSE, keepchannel=False):
    # aggregation for base net (AAM1)
    # 'data' size of  (B, D, H, W, C, N-1)
    # output: prob_volume_b2 (B, D, H, W) if keepchannel=False
    #         cost_volume_b2 (B, D, H, W, C) if keepchannel=True
    if keepchannel:
        aggregated_cost_volume_tower = AttAggregation_keepchannel({'data': cost_volumes}, is_training=True, reuse=reuse)
        aggregated_prob_volume = aggregated_cost_volume_tower.get_output()
    else:
        aggregated_cost_volume_tower = AttAggregation({'data': cost_volumes}, is_training=True, reuse=reuse)
        aggregated_prob_volume = tf.squeeze(aggregated_cost_volume_tower.get_output(), axis=-1)
    return aggregated_prob_volume


# AAM2
def cost_volume_aggregation_refine(cost_volumes, reuse=tf.AUTO_REUSE, keepchannel=False):
    # aggregation for refinement (AAM2)
    if keepchannel:
        aggregated_cost_volume_tower = AttAggregation_refine_keepchannel({'data': cost_volumes}, is_training=True, reuse=reuse)
        aggregated_prob_volume = aggregated_cost_volume_tower.get_output()
    else:
        aggregated_cost_volume_tower = AttAggregation_refine({'data': cost_volumes}, is_training=True, reuse=reuse)
        aggregated_prob_volume = tf.squeeze(aggregated_cost_volume_tower.get_output(), axis=-1)
    return aggregated_prob_volume

    