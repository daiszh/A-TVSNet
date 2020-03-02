import numpy as np

'''
  errors:
  0. mae
  1. rmse
  2. inverse mae
  3. inverse rmse
  4. log mae
  5. log rmse
  6. scale invariant log
  7. abs relative
  8. squared relative
  -----------------------------
  9. mae for normalized depth range
  10+. inlier ratio: percentage that mae<threshold for normalized depth range
'''
inlier_thres = [1,3,5,10]

err_metrics_namelist = ['mae', 'rmse', 'inverse_mae', 'inverse_rmse', 'log_mae', 'log_rmse',
                        'scale_invariant_log', 'abs_relative', 'squared_relative', 'mae_normalized']

acc_metrics_namelist = ['inlier_ratios_' + str(i) for i in inlier_thres]

def calc_error(depth_predict_in, depth_gt_in, num_depths = 100, inlier_threshold = inlier_thres):
    assert depth_predict_in.shape == depth_gt_in.shape

    depth_predict = depth_predict_in.copy()
    depth_gt = depth_gt_in.copy()
    depth_gt[np.isnan(depth_gt)] = 0.0
    depth_predict[np.isnan(depth_predict)] = 0.0

    # get depth_gt range
    depthvec = depth_gt.copy()
    depthvec = depthvec.flatten()
    depthvec = depthvec[(depthvec < 1e10) * (depthvec > 0.0)] # handle invalid values
    depthvec = np.sort(depthvec)
    depth_interval = float(depthvec[-1] - depthvec[0]) / float(num_depths)

    # handle invalid pixels(nans, infs and non-positive values)
    valid_mask = (depth_gt > 0.0) * (depth_gt < 1e10) * (depth_predict > 0.0) * (depth_predict < 1e10)
    valid_num = float(np.sum(valid_mask))
    assert valid_num > 0
    
    depth_gt[~valid_mask] = 1.0
    depth_predict[~valid_mask] = 1.0

    # init errors
    errors = np.zeros(10 + len(inlier_threshold), dtype=np.float32)

    d_err = valid_mask * np.abs(depth_gt - depth_predict)
    d_err_squared = d_err * d_err
    d_err_inv = valid_mask * np.abs(1.0 / (depth_gt) - 1.0 / (depth_predict))
    d_err_inv_squared = d_err_inv * d_err_inv
    d_err_log = valid_mask * np.abs(np.log(depth_gt) - np.log(depth_predict))
    d_err_log_squared = d_err_log * d_err_log

    # mae (l1)
    errors[0] = np.sum(d_err) / valid_num
    # rmse
    errors[1] = np.sum(d_err_squared) / valid_num
    errors[1] = np.sqrt(errors[1])
    # inverse mae (l1_inverse)
    errors[2] = np.sum(d_err_inv) / valid_num
    # inverse rmse
    errors[3] = np.sum(d_err_inv_squared) / valid_num
    errors[3] = np.sqrt(errors[3])
    # log mae
    errors[4] = np.sum(d_err_log) / valid_num
    # log rmse
    normalizedSquaredLog = np.sum(d_err_log_squared) / valid_num
    errors[5] = np.sqrt(normalizedSquaredLog)
    # log diff for scale invariant metric
    logSum = valid_mask * (np.log(depth_gt) - np.log(depth_predict))
    logSum = np.sum(logSum)
    errors[6] = np.sqrt(normalizedSquaredLog - (logSum*logSum / (valid_num*valid_num)))
    # abs relative
    errors[7] = np.sum((d_err/(depth_gt))) / valid_num
    # squared relative
    errors[8] = np.sum(d_err_squared/(depth_gt*depth_gt)) / valid_num
    # --------------------
    # mae for normalized depth range
    errors[9] = np.sum(d_err) / depth_interval / valid_num
    # inlier ratio: percentage that mae<threshold for normalized depth range
    diff_image = d_err[valid_mask] / depth_interval
    for i in range(len(inlier_threshold)):
        th = inlier_threshold[i]
        errors[10+i] = float(np.sum(diff_image < th)) / valid_num

    # other infos (num_depths, depth_interval, depth_min, depth_max, inlier_threshold)
    infos = [num_depths, depth_interval, depthvec[0], depthvec[-1], inlier_threshold]

    return errors, infos
