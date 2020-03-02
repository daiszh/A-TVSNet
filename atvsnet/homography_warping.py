#!/usr/bin/env python

import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def get_pixel_grids(height, width):
    # texture coordinate
    x_linspace = tf.linspace(0.5, tf.cast(width, 'float32') - 0.5, width)
    y_linspace = tf.linspace(0.5, tf.cast(height, 'float32') - 0.5, height)
    x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
    x_coordinates = tf.reshape(x_coordinates, [-1])
    y_coordinates = tf.reshape(y_coordinates, [-1])
    ones = tf.ones_like(x_coordinates)
    indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
    return indices_grid

def repeat_int(x, num_repeats):
    ones = tf.ones((1, num_repeats), dtype='int32')
    x = tf.reshape(x, shape=(-1, 1))
    x = tf.matmul(x, ones)
    return tf.reshape(x, [-1])

def repeat_float(x, num_repeats):
    ones = tf.ones((1, num_repeats), dtype='float')
    x = tf.reshape(x, shape=(-1, 1))
    x = tf.matmul(x, ones)
    return tf.reshape(x, [-1])

def interpolate(image, x, y, output_mask=False, method='bilinear'):
    image_shape = tf.shape(image)
    batch_size = image_shape[0]
    height =image_shape[1]
    width = image_shape[2]

    x = x - 0.5
    y = y - 0.5
    valid_mask = tf.logical_and(tf.logical_and(tf.greater_equal(x, 0), tf.greater_equal(y, 0)), 
                                tf.logical_and(tf.less(x, tf.cast(width - 1,  dtype=x.dtype)), tf.less(y, tf.cast(height - 1, dtype=y.dtype))))

    nan_mask = tf.logical_and(tf.logical_not(tf.is_nan(x)), tf.logical_not(tf.is_nan(y)))
    valid_mask = tf.logical_and(nan_mask, valid_mask)

    if method == 'nearest':
        x0 = tf.cast(tf.round(x), 'int32')
        y0 = tf.cast(tf.round(y), 'int32')
        x0 = tf.multiply(x0, tf.cast(valid_mask, x0.dtype))
        y0 = tf.multiply(y0, tf.cast(valid_mask, y0.dtype))
        b = repeat_int(tf.range(batch_size), height * width)
        indices = tf.stack([b, y0, x0], axis=1)
        output = tf.gather_nd(image, indices)
        if output_mask:
            return output, valid_mask
        else:
            return output

    # image coordinate to pixel coordinate
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    x = tf.multiply(x, tf.cast(valid_mask, x.dtype))
    y = tf.multiply(y, tf.cast(valid_mask, y.dtype))
    x0 = tf.multiply(x0, tf.cast(valid_mask, x0.dtype))
    x1 = tf.multiply(x1, tf.cast(valid_mask, x1.dtype))
    y0 = tf.multiply(y0, tf.cast(valid_mask, y0.dtype))
    y1 = tf.multiply(y1, tf.cast(valid_mask, y1.dtype))
    max_y = tf.cast(height - 1, dtype='int32')
    max_x = tf.cast(width - 1,  dtype='int32')
    x0 = tf.clip_by_value(x0, 0, max_x)
    x1 = tf.clip_by_value(x1, 0, max_x)
    y0 = tf.clip_by_value(y0, 0, max_y)
    y1 = tf.clip_by_value(y1, 0, max_y)

    b = repeat_int(tf.range(batch_size), height * width)
    indices_a = tf.stack([b, y0, x0], axis=1)
    indices_b = tf.stack([b, y0, x1], axis=1)
    indices_c = tf.stack([b, y1, x0], axis=1)
    indices_d = tf.stack([b, y1, x1], axis=1)

    pixel_values_a = tf.gather_nd(image, indices_a)
    pixel_values_b = tf.gather_nd(image, indices_b)
    pixel_values_c = tf.gather_nd(image, indices_c)
    pixel_values_d = tf.gather_nd(image, indices_d)

    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')
    area_a = tf.expand_dims(((y1 - y) * (x1 - x)), 1)
    area_b = tf.expand_dims(((y1 - y) * (x - x0)), 1)
    area_c = tf.expand_dims(((y - y0) * (x1 - x)), 1)
    area_d = tf.expand_dims(((y - y0) * (x - x0)), 1)
    output = tf.add_n([area_a * pixel_values_a,
                        area_b * pixel_values_b,
                        area_c * pixel_values_c,
                        area_d * pixel_values_d])

    if output_mask:
        return output, valid_mask
    else:
        return output

# warping right image(input_image) to left image(reference) with given depth_image(size (B, H, W, 1))
# return warpped image
def homography_warping_by_depth(input_image, left_cam, right_cam, depth_image, output_mask=False, method='bilinear'):
    with tf.name_scope('homography_warping_by_depth'):
        image_shape = tf.shape(input_image)
        batch_size = image_shape[0]
        height = image_shape[1]
        width = image_shape[2]

        # cameras (K, R, t)
        R_left = tf.squeeze(tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3]), axis=1)
        R_right = tf.squeeze(tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3]), axis=1)
        t_left = tf.squeeze(tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1]), axis=1)
        t_right = tf.squeeze(tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1]), axis=1)
        K_left = tf.squeeze(tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3]), axis=1)
        K_right = tf.squeeze(tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3]), axis=1)
        # preparation
        K_left_inv = tf.matrix_inverse(K_left)
        R_left_trans = tf.transpose(R_left, perm=[0, 2, 1])
        c_left = -tf.matmul(R_left_trans, t_left)

        # generate pixel grids of size (B, 3, (W) x (H)), each column is (u,v,1)
        pixel_grids = get_pixel_grids(height, width)
        pixel_grids = tf.expand_dims(pixel_grids, 0)
        pixel_grids = tf.tile(pixel_grids, [batch_size, 1])
        pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, -1))

        # (u,v,1)*depth
        # if FLAGS.inverse_depth:
        #     valid_mask_depth = tf.greater(depth_image, 1e-10)
        #     depth_image = tf.clip_by_value(depth_image, 1e-10, tf.reduce_max(depth_image))
        #     depth_image = tf.reciprocal(depth_image)
        #     depth_image = tf.multiply(depth_image, tf.cast(valid_mask_depth, depth_image.dtype))
        
        depth_image_reshape = tf.reshape(depth_image, [batch_size, 1, height*width])
        depth_image_reshape = tf.tile(depth_image_reshape, [1, 3, 1])
        # pixel_grids = tf.multiply(pixel_grids, depth_image_reshape)

        # compute
        mat_tmp = tf.matmul(K_right, tf.matmul(R_right, tf.matmul(R_left_trans, K_left_inv)))
        vec_tmp = tf.add(tf.matmul(K_right, tf.matmul(R_right, c_left)), tf.matmul(K_right, t_right))
        vec_tmp = tf.tile(vec_tmp, [1,1,height*width])

        if FLAGS.inverse_depth:
            vec_tmp = vec_tmp * depth_image_reshape
        else:
            vec_tmp = vec_tmp / depth_image_reshape

        xy_warped = tf.add(tf.matmul(mat_tmp, pixel_grids), vec_tmp)
        # normalize
        d_warped = tf.slice(xy_warped, [0, 2, 0], [-1, 1, -1])
        # d_warped = tf.clip_by_value(d_warped, 1e-10, tf.reduce_max(d_warped))
        xy_warped = tf.div(xy_warped, tf.tile(d_warped, [1, 3, 1]))
        # flatten
        x_warped = tf.slice(xy_warped, [0, 0, 0], [-1, 1, -1])
        y_warped = tf.slice(xy_warped, [0, 1, 0], [-1, 1, -1])
        x_warped_flatten = tf.reshape(x_warped, [-1])
        y_warped_flatten = tf.reshape(y_warped, [-1])

        # interpolation
        if output_mask:
            warped_image, valid_mask = interpolate(input_image, x_warped_flatten, y_warped_flatten, output_mask=True, method=method)
            warped_image = tf.reshape(warped_image, shape=image_shape, name='warped_feature')

            valid_mask = tf.reshape(valid_mask, shape=(image_shape[0], image_shape[1], image_shape[2], 1), name='warp_valid_mask')
            return warped_image, valid_mask
        else:
            warped_image = interpolate(
                input_image, x_warped_flatten, y_warped_flatten, output_mask=False, method=method)
            warped_image = tf.reshape(warped_image, shape=image_shape, name='warped_feature')        
            return warped_image


def get_homographies(left_cam, right_cam, depth_num, depth_start, depth_interval):
    with tf.name_scope('get_homographies'):
        # cameras (K, R, t)
        R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
        t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
        K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
        batch_size = tf.shape(R_left)[0]

        # depth 
        # depth = depth_start + tf.cast(tf.range(depth_num), tf.float32) * depth_interval
        # num_depth = tf.shape(depth)[0]
        depth = tf.tile(tf.expand_dims(depth_start, axis=-1), [1, depth_num]) + \
                tf.multiply(tf.tile(tf.expand_dims(tf.cast(tf.range(depth_num), tf.float32), axis=0), [batch_size, 1]),
                            tf.tile(tf.expand_dims(depth_interval, axis=-1), [1, depth_num]))
        num_depth = tf.shape(depth)[1]

        # preparation
        K_left_inv = tf.matrix_inverse(tf.squeeze(K_left, axis=1))
        R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
        R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])

        fronto_direction = tf.slice(tf.squeeze(R_left, axis=1), [0, 2, 0], [-1, 1, 3])          # (B, D, 1, 3)

        c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
        c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))                        # (B, D, 3, 1)
        c_relative = tf.subtract(c_right, c_left)        

        # compute   
        temp_vec = tf.matmul(c_relative, fronto_direction)
        depth_mat = tf.tile(tf.reshape(depth, [batch_size, num_depth, 1, 1]), [1, 1, 3, 3])

        temp_vec = tf.tile(tf.expand_dims(temp_vec, axis=1), [1, num_depth, 1, 1])

        if FLAGS.inverse_depth:
            middle_mat0 = tf.eye(3, batch_shape=[batch_size, num_depth]) - temp_vec * depth_mat
        else:
            middle_mat0 = tf.eye(3, batch_shape=[batch_size, num_depth]) - temp_vec / depth_mat

        middle_mat1 = tf.tile(tf.expand_dims(tf.matmul(R_left_trans, K_left_inv), axis=1), [1, num_depth, 1, 1])
        middle_mat2 = tf.matmul(middle_mat0, middle_mat1)

        homographies = tf.matmul(tf.tile(K_right, [1, num_depth, 1, 1])
                     , tf.matmul(tf.tile(R_right, [1, num_depth, 1, 1])
                     , middle_mat2))

    return homographies


def homography_warping(input_image, homography, method='bilinear', output_mask=False):
    with tf.name_scope('warping_by_homography'):
        image_shape = tf.shape(input_image)
        batch_size = image_shape[0]
        height = image_shape[1]
        width = image_shape[2]

        # turn homography to affine_mat of size (B, 2, 3) and div_mat of size (B, 1, 3)
        affine_mat = tf.slice(homography, [0, 0, 0], [-1, 2, 3])
        div_mat = tf.slice(homography, [0, 2, 0], [-1, 1, 3])

        # generate pixel grids of size (B, 3, (W+1) x (H+1))
        pixel_grids = get_pixel_grids(height, width)
        pixel_grids = tf.expand_dims(pixel_grids, 0)
        pixel_grids = tf.tile(pixel_grids, [batch_size, 1])
        pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, -1))
        # return pixel_grids

        # affine + divide tranform, output (B, 2, (W+1) x (H+1))
        grids_affine = tf.matmul(affine_mat, pixel_grids)
        grids_div = tf.matmul(div_mat, pixel_grids)
        grids_zero_add = tf.cast(tf.equal(grids_div, 0.0), dtype='float32') * 1e-7 # handle div 0
        grids_div = grids_div + grids_zero_add
        grids_div = tf.tile(grids_div, [1, 2, 1])
        grids_inv_warped = tf.div(grids_affine, grids_div)
        x_warped, y_warped = tf.unstack(grids_inv_warped, axis=1)
        x_warped_flatten = tf.reshape(x_warped, [-1])
        y_warped_flatten = tf.reshape(y_warped, [-1])

        # interpolation
        if output_mask:
            warped_image, valid_mask = interpolate(
                input_image, x_warped_flatten, y_warped_flatten, method=method, output_mask = True)
            warped_image = tf.reshape(warped_image, shape=image_shape, name='warped_feature')
            valid_mask = tf.reshape(valid_mask, shape=(image_shape[0], image_shape[1], image_shape[2], 1), name='warp_valid_mask')
            return warped_image, valid_mask
        else:
            warped_image = interpolate(
                input_image, x_warped_flatten, y_warped_flatten, method=method, output_mask = False)
            warped_image = tf.reshape(warped_image, shape=image_shape, name='warped_feature')

            return warped_image


# transform left pixel depth value to right coor
def transform_depth(left_depth, left_cam, right_cam):
    with tf.name_scope('transform_depth'):
        image_shape = tf.shape(left_depth)
        batch_size = image_shape[0]
        height = image_shape[1]
        width = image_shape[2]

        # cameras (K, R, t)
        R_left = tf.squeeze(tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3]), axis=1)
        R_right = tf.squeeze(tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3]), axis=1)
        t_left = tf.squeeze(tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1]), axis=1)
        t_right = tf.squeeze(tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1]), axis=1)
        K_left = tf.squeeze(tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3]), axis=1)
        K_right = tf.squeeze(tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3]), axis=1)
        # preparation
        K_left_inv = tf.matrix_inverse(K_left)
        R_left_trans = tf.transpose(R_left, perm=[0, 2, 1])
        c_left = -tf.matmul(R_left_trans, t_left)

        # generate pixel grids of size (B, 3, (W+1) x (H+1)), each column is (u,v,1)
        pixel_grids = get_pixel_grids(height, width)
        pixel_grids = tf.expand_dims(pixel_grids, 0)
        pixel_grids = tf.tile(pixel_grids, [batch_size, 1])
        pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, -1))

        # (u,v,1)*depth
        if FLAGS.inverse_depth:
            valid_mask_depth = tf.greater(left_depth, 1e-10)
            left_depth = tf.clip_by_value(left_depth, 1e-10, tf.reduce_max(left_depth))
            left_depth = tf.reciprocal(left_depth)
            left_depth = tf.multiply(left_depth, tf.cast(valid_mask_depth, left_depth.dtype))

        depth_image_reshape = tf.reshape(left_depth, (batch_size, 1, height*width))
        depth_image_reshape = tf.tile(depth_image_reshape, [1, 3, 1])
        pixel_grids = np.multiply(pixel_grids, depth_image_reshape)

        # compute
        mat_tmp = tf.matmul(K_right, tf.matmul(R_right, tf.matmul(R_left_trans, K_left_inv)))
        vec_tmp = tf.matmul(K_right, tf.matmul(R_right, c_left)) + tf.matmul(K_right, t_right)
        vec_tmp = tf.tile(vec_tmp, [1,1,height*width])
        xy_warped = tf.matmul(mat_tmp, pixel_grids) + vec_tmp
        # get transformed depth
        d_warped = tf.slice(xy_warped, [0, 2, 0], [-1, 1, -1])
        d_warped_flatten = tf.reshape(d_warped, [-1])
        warped_depth = tf.reshape(d_warped_flatten, image_shape)

        if FLAGS.inverse_depth:
            warped_depth = tf.clip_by_value(warped_depth, 1e-10, tf.reduce_max(warped_depth))
            warped_depth = tf.reciprocal(warped_depth)
            warped_depth = tf.multiply(warped_depth, tf.cast(valid_mask_depth, warped_depth.dtype))

        return warped_depth


def get_visual_hull(depth_images, cams, depth_num, depth_start, depth_interval, ref_id=0, view_num=None):
    # depth_images (B, N, H, W)
    # cams (B, N, 2, 4, 4)
    # output: visual_hull (B, D, H, W, 1)

    with tf.name_scope('get_visual_hull'):
        image_shape = tf.shape(depth_images)
        batch_size = image_shape[0]
        height = image_shape[2]
        width = image_shape[3]

        if view_num is None:
            # view_num = image_shape[1]
            view_num = FLAGS.view_num

        id_reorder = [i for i in range(view_num)]
        id_reorder[0] = ref_id
        id_reorder[ref_id] = 0

        # visual_hull = tf.zeros((batch_size, depth_num, height, width))
        visual_hull = []
        ref_cam = tf.squeeze(tf.slice(cams, [0, ref_id, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1) # (B, 2, 4, 4)
        ref_depth = tf.squeeze(tf.slice(depth_images, [0, ref_id, 0, 0], [-1, 1, -1, -1]), axis=1) # (B, H, W)

        homographies_list = []
        view_depth_trans2ref_list = []
        for view_i in id_reorder[1:]:
            view_cam = tf.squeeze(tf.slice(cams, [0, view_i, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1) # (B, 2, 4, 4)
            homographies = get_homographies(ref_cam, view_cam, depth_num, depth_start, depth_interval)
            view_depth = tf.squeeze(tf.slice(depth_images, [0, view_i, 0, 0], [-1, 1, -1, -1]), axis=1) # (B, H, W)
            view_depth_trans2ref = transform_depth(view_depth, view_cam, ref_cam) # depth in view, trans value to ref coor
            homographies_list.append(homographies)
            view_depth_trans2ref_list.append(view_depth_trans2ref)

        for di in range(depth_num):
            depth_current = depth_start + depth_interval * di
            ref_depth_slice = depth_current * tf.ones((batch_size, height, width))

            # visual hull slice for reference depth image
            valid_mask = tf.cast(tf.greater(ref_depth, 0), tf.float32)
            if FLAGS.inverse_depth:
                visual_hull_slice = valid_mask * tf.cast(tf.greater(ref_depth, ref_depth_slice), tf.float32) # (B, H, W)
            else:
                visual_hull_slice = valid_mask * tf.cast(tf.greater(ref_depth_slice, ref_depth), tf.float32) # (B, H, W)

            for view_i in range(view_num-1):
                homo = tf.squeeze(tf.slice(homographies_list[view_i], begin=[0, di, 0, 0], size=[-1, 1, 3, 3]), axis=1)
                warped_depth = homography_warping(view_depth_trans2ref_list[view_i], homo, method='nearest')
                valid_mask = tf.cast(tf.greater(warped_depth, 0), tf.float32)
                if FLAGS.inverse_depth:
                    visual_hull_slice_tmp = valid_mask * tf.cast(tf.greater(warped_depth, ref_depth_slice), tf.float32) # (B, H, W)
                else:
                    visual_hull_slice_tmp = valid_mask * tf.cast(tf.greater(ref_depth_slice, warped_depth), tf.float32) # (B, H, W)
                visual_hull_slice = visual_hull_slice + visual_hull_slice_tmp
            visual_hull.append(visual_hull_slice)
        visual_hull = tf.stack(visual_hull, axis=1)  # (B, D, H, W)
        visual_hull = tf.divide(visual_hull, tf.cast(view_num, tf.float32))

    return tf.expand_dims(visual_hull, axis=-1)

