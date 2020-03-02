import os
import re
import json

import numpy as np
from numpy import linalg as la
import cv2
import imageio
from pyquaternion import Quaternion

from tqdm import tqdm

# original unit in dataset is in 'mm'
unit_scale = 1.0

class ImageList:

    class Image:
        def __init__(self, id, seq_id, img_id, extrinsic, filename, filepath, fx, fy ,cx, cy, depthpath=None):
            self.id = id
            self.seq_id = seq_id
            self.img_id = img_id
            self.extrinsic = extrinsic
            self.filename = filename
            self.filepath = filepath
            self.depthpath = depthpath
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy

            K_inv = la.inv(np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]))
            R_c2w = np.transpose(extrinsic[0:3, 0:3])
            self.M_inv = R_c2w.dot(K_inv)
            self.t_c2w = -R_c2w.dot(extrinsic[0:3, 3])
            self.view_vec = self.get_view_vector(cx, cy)

        def get_view_vector(self, x, y):
            """ get the viewing ray for a pixel position of the camera """
            # get some point on the line (the other point on the line is the camera center)
            pt = [x, y, 1.0]
            depth = 1.0
            ptX = self.M_inv.dot(pt) * depth + self.t_c2w
            # get vector between camera center and other point on the line
            v = ptX - self.t_c2w
            return v / (la.norm(v) + 1e-10)

    def __init__(self, path):
        
        self.basepath = path
        self.num_images = []
        if os.path.isfile(os.path.join(self.basepath, "num_images.json")):
            with open(os.path.join(self.basepath, "num_images.json")) as f:
                self.num_images = (json.load(f))
            self.num_images = np.array(self.num_images)   
            # print('self.num_images', self.num_images, min(self.num_images), max(self.num_images))
        else:
            print('ImageList init error: num_images.json NOT exists at', self.basepath)
        # Fetch image dimensions
        img = imageio.imread(os.path.join(
            self.basepath, "{:04d}".format(0), "images", "{:04d}.png".format(0)))
        self.width = img.shape[1]
        self.height = img.shape[0]

        self.images = []
        self.seq_length = len(self.num_images)
        self.length = np.sum(self.num_images)
        count = 0
        self.map_id_idx = {}
        self.map_seq_idx = {}
        self.map_name_idx = {}
        for seq_idx in tqdm(range(self.seq_length)):
            if self.num_images[seq_idx] < 2:
                print('WARNING: num_images_seq<2', os.path.join(self.basepath, "{:04d}".format(seq_idx)))
                self.length = self.length - self.num_images[seq_idx]
                continue
            for img_idx in range(0, self.num_images[seq_idx]):
                filename = os.path.join("{:04d}".format(seq_idx), "images", "{:04d}.png".format(img_idx))
                filepath = os.path.join(self.basepath, filename)
                posepath = os.path.join(self.basepath, "{:04d}".format(seq_idx), "poses", "{:04d}.json".format(img_idx))
                depthpath = os.path.join(self.basepath, "{:04d}".format(seq_idx), "depths", "{:04d}.exr".format(img_idx)) ## only training
                with open(posepath) as f:
                    r_info = json.load(f)
                c_x = r_info["c_x"]
                c_y = r_info["c_y"]
                f_x = r_info["f_x"]
                f_y = r_info["f_y"]
                # extrinsic is world to cam
                extrinsic = np.array(r_info["extrinsic"])
                self.images.append(self.Image(
                    count, seq_idx, img_idx, extrinsic, filename, filepath, f_x, f_y, c_x, c_y, depthpath))
                # update dict for searching
                self.map_id_idx[count] = count
                self.map_seq_idx[(seq_idx, img_idx)] = count
                self.map_name_idx[filename] = count
                count = count + 1
        self.num_images[self.num_images < 2] = 0

    def get_by_filename_with_idx(self, filename):
        if filename in self.map_name_idx:
            idx = self.map_name_idx[filename]
            assert self.images[idx].filename == filename
            return self.images[idx], idx
        else:
            return None, None

    def get_id_by_seq(self, seq_id, img_id):
        if (seq_id, img_id) in self.map_seq_idx:
            idx = self.map_seq_idx[(seq_id, img_id)]
            assert self.images[idx].seq_id == seq_id
            assert self.images[idx].img_id == img_id
            return self.images[idx].id
        else:
            return None

    def get_by_seq_with_idx(self, seq_id, img_id):
        if (seq_id, img_id) in self.map_seq_idx:
            idx = self.map_seq_idx[(seq_id, img_id)]
            assert self.images[idx].seq_id == seq_id
            assert self.images[idx].img_id == img_id
            return self.images[idx], idx
        else:
            return None, None

    def get_by_id(self, id):
        if id in self.map_id_idx:
            idx = self.map_id_idx[id]
            assert self.images[idx].id == id
            return self.images[idx]
        else:
            return None

    def get_by_id_with_idx(self, id):
        if id in self.map_id_idx:
            idx = self.map_id_idx[id]
            assert self.images[idx].id == id
            return self.images[idx], idx
        else:
            return None, None

    def get_filename_by_id(self, id):
        if id in self.map_id_idx:
            idx = self.map_id_idx[id]
            assert self.images[idx].id == id
            return self.images[idx].filename
        else:
            return None

class MVS_Syn:
    def __init__(self, datapath, num_neighbors=9, get_neighs_neigh_list=False):
        self.image_list = ImageList(datapath)
        self.estimate_max_disparities()
        self.generate_neighbor_list(num_neighbors)
        if get_neighs_neigh_list:
            self.get_sub_neighbor_list(num_neighbors)

    def estimate_max_disparities(self, percentile=0.99, stretch=1.33333):
        for (img_idx, image) in enumerate(self.image_list.images):
            self.image_list.images[img_idx].estimated_max_disparity = None
            self.image_list.images[img_idx].estimated_min_disparity = None

    def get_angle(self, v1, v2):
        """ get angle between two vectors in 3D """
        # result in radius
        angle = np.arccos(v1.dot(v2))
        return angle

    def generate_neighbor_list(self, num_neighbors, max_neighbor_range = 5):
        self.max_neigh_range = max_neighbor_range
        self.num_permute = []
        for ref_idx in tqdm(range(self.image_list.length)):
            neighbor_list = []

            ref_image = self.image_list.images[ref_idx]
            ref_seq_id = ref_image.seq_id
            ref_img_id = ref_image.img_id
            view_vec_ref = ref_image.view_vec

            # preprocessed dataset and load neighbors according to depth reprojection
            neighbor_list_from_file = []
            neighpath = os.path.join(self.image_list.basepath, "{:04d}".format(ref_seq_id), "neighbors", "{:04d}.json".format(ref_img_id))            
            if os.path.isfile(neighpath):
                with open(neighpath) as f:
                    neighbor_list_from_file = np.array(json.load(f))
                if len(neighbor_list_from_file) == 0:
                    # print('WARNING: no valid neighbor found for (seq_id, img_id)', (ref_seq_id, ref_img_id))
                    self.num_permute.append(1)
                    self.image_list.images[ref_idx].neighbor_list = [[ref_image.id for ni in range(num_neighbors)]]
                    self.image_list.images[ref_idx].isValid = False
                    continue
            neighscorepath = os.path.join(self.image_list.basepath, "{:04d}".format(ref_seq_id), "neighbors", "{:04d}_score.json".format(ref_img_id))
            if os.path.isfile(neighscorepath):
                with open(neighscorepath) as f:
                    neighbor_score_list_from_file = json.load(f)
                    valid_depth_ratio = neighbor_score_list_from_file[1]
                    neighbor_score_list_from_file = np.array(neighbor_score_list_from_file[0])
                if valid_depth_ratio < 0.2:
                    # print('WARNING: no valid(valid depth insufficient) neighbor for (seq_id, img_id)', (ref_seq_id, ref_img_id))
                    self.num_permute.append(1)
                    self.image_list.images[ref_idx].neighbor_list = [[ref_image.id for ni in range(num_neighbors)]]
                    self.image_list.images[ref_idx].isValid = False
                    continue
                if len(neighbor_list_from_file)>0:
                    score_thres = 0.6
                    min_score_thres = 0.2
                    min_neigh_left = min(len(neighbor_list_from_file), num_neighbors)
                    while np.sum(neighbor_score_list_from_file > score_thres) < min_neigh_left and score_thres > min_score_thres:
                        score_thres = score_thres * 0.8
                    neighbor_list_from_file = neighbor_list_from_file[neighbor_score_list_from_file>score_thres]
                    if len(neighbor_list_from_file) == 0:
                        # print('WARNING: no valid(score too low) neighbor for (seq_id, img_id)', (ref_seq_id, ref_img_id))
                        self.num_permute.append(1)
                        self.image_list.images[ref_idx].neighbor_list = [[ref_image.id for ni in range(num_neighbors)]]
                        self.image_list.images[ref_idx].isValid = False
                        continue
            neighbor_list_from_file = neighbor_list_from_file[0:min(len(neighbor_list_from_file), self.max_neigh_range)]

            ###### random neighbor index
            num_images_seq = self.image_list.num_images[ref_seq_id]
            valid_neigh_idx = []
            ref_pos = -1
            if len(neighbor_list_from_file) < 1:
                minimum_angle_radians = 0.0*np.pi/180.0
                maximum_angle_radians = 60.0*np.pi/180.0
                # for each image in the same sequence, filter by view angle
                for img_id in range(num_images_seq):
                    if img_id == ref_img_id:
                        ref_pos = img_id
                        continue
                    neigh_image, neigh_idx = self.image_list.get_by_seq_with_idx(ref_seq_id, img_id)
                    if neigh_idx is None:
                        continue
                    view_angle = self.get_angle(view_vec_ref, neigh_image.view_vec)
                    if (view_angle > minimum_angle_radians and view_angle < maximum_angle_radians):
                        valid_neigh_idx.append(neigh_idx)
                    # else:
                    #     print('INFO: view_angle inappropriate',
                    #           ref_image.filename, neigh_image.filename, minimum_angle_radians, maximum_angle_radians, view_angle, view_angle*180/np.pi)
            else:
                for img_id in neighbor_list_from_file:
                    _, neigh_idx = self.image_list.get_by_seq_with_idx(ref_seq_id, img_id)
                    if neigh_idx is None:
                        continue
                    valid_neigh_idx.append(neigh_idx)

            num_images_seq_valid = len(valid_neigh_idx)
            valid_neigh_idx = np.array(valid_neigh_idx)
            num_neigh_images = min(self.max_neigh_range, num_images_seq_valid)
            self.num_permute.append(int(np.ceil(float(num_neigh_images) / float(num_neighbors))))
            if num_images_seq < 2 or num_images_seq_valid < 1:
                print('WARNING: num_images_seq<2,  or num_images_seq_valid < 2', num_images_seq, num_images_seq_valid, ref_image.filename)
                self.image_list.images[ref_idx].neighbor_list = [[ref_image.id for ni in range(num_neighbors)]]
                self.image_list.images[ref_idx].isValid = False
                continue

            # skip current image if insufficient number of images in seq
            if num_images_seq_valid < num_neighbors:
                self.image_list.images[ref_idx].neighbor_list = None
                self.image_list.images[ref_idx].isValid = False
                continue

            if num_images_seq_valid > self.max_neigh_range:
                ref_pos_valid = np.argmin(abs(np.array(valid_neigh_idx)-ref_image.id-1))
                if ref_pos_valid < self.max_neigh_range/2:
                    nidx_start = 0
                    nidx_end = self.max_neigh_range
                else:
                    if ref_pos_valid > (num_images_seq_valid - self.max_neigh_range/2):
                        nidx_start = num_images_seq_valid - self.max_neigh_range
                        nidx_end = num_images_seq_valid
                    else:    
                        nidx_start = max(0, ref_pos_valid - int(self.max_neigh_range/2))
                        nidx_end = min(num_images_seq_valid, self.max_neigh_range + nidx_start)
                neigh_id_list = valid_neigh_idx[np.arange(nidx_start, nidx_end)]
            else:
                neigh_id_list = valid_neigh_idx
            neigh_id_list_extend = np.array(neigh_id_list)
            neigh_id_list = np.append(ref_image.id, neigh_id_list)
            # while len(neigh_id_list_extend) < num_neighbors:
            #     neigh_id_list_extend = np.append(neigh_id_list_extend, np.random.choice(neigh_id_list))
            for _ in range(self.num_permute[-1]):
                np.random.shuffle(neigh_id_list_extend)
                tmp_list = []
                for ni in range(num_neighbors):
                    nidx = neigh_id_list_extend[ni]
                    tmp_list.append(self.image_list.images[nidx].id)
                neighbor_list.append(tmp_list)
            ###### 

            # ###### sequential neighbor index
            # self.num_permute.append(1)
            # tmp_list = []
            # for ni in range(num_neighbors):
            #     nidx0 = ref_img_id + 1 + ni
            #     nidx1 = ref_img_id - 1 - ni
            #     nimg0,_ = self.image_list.get_by_filename_with_idx(os.path.join("{:04d}".format(ref_seq_id), "images", "{:04d}.png".format(nidx0)))
            #     nimg1,_ = self.image_list.get_by_filename_with_idx(os.path.join("{:04d}".format(ref_seq_id), "images", "{:04d}.png".format(nidx1)))
            #     if (nimg0 is not None):
            #         neighbor_list.append(nimg0.id)
            #     elif (nimg1 is not None):
            #         neighbor_list.append(nimg1.id)
            #     else:
            #         neighbor_list.append(ref_image.id)
            # neighbor_list.append(tmp_list)
            # ######

            if neighbor_list:
                # print(len(neighbor_list), neighbor_list)
                self.image_list.images[ref_idx].neighbor_list = neighbor_list
                self.image_list.images[ref_idx].isValid = True
            else:
                self.image_list.images[ref_idx].neighbor_list = None
                self.image_list.images[ref_idx].isValid = False

    def get_sub_neighbor_list(self, num_neighbors):
        for ref_idx in tqdm(range(self.image_list.length)):
            if self.image_list.images[ref_idx].isValid == False:
                continue
            sub_neighbor_list = []
            sub_neighbor_filename_list = []
            neighbor_list = self.image_list.images[ref_idx].neighbor_list[0]
            for neigh_idx in neighbor_list:
                # print('neigh_idx', neighbor_list, neigh_idx)
                neigh_image = self.image_list.get_by_id(neigh_idx)
                if neigh_image.isValid == False:
                    sub_neighbor = np.array(neighbor_list).tolist()
                    sub_neighbor.append(self.image_list.images[ref_idx].id)
                    sub_neighbor.remove(neigh_idx)
                else:
                    sub_neighbor = neigh_image.neighbor_list[0]
                sub_neighbor_list.append(sub_neighbor)
                sub_neighbor_filename = []
                for sub_id in sub_neighbor_list[-1]:
                    sub_neighbor_filename.append(
                        self.image_list.get_filename_by_id(sub_id)[-8:-4])
                sub_neighbor_filename_list.append(sub_neighbor_filename)
            self.image_list.images[ref_idx].sub_neighbor_list = sub_neighbor_list
            self.image_list.images[ref_idx].sub_neighbor_filename_list = sub_neighbor_filename_list
            

class MVS_Syn_flexible:
    def __init__(self, datapath, max_num_neighbors=9, get_neighs_neigh_list=False):
        self.image_list = ImageList(datapath)
        self.estimate_max_disparities()
        self.generate_neighbor_list(max_num_neighbors)
        if get_neighs_neigh_list:
            self.get_sub_neighbor_list(max_num_neighbors)

    def estimate_max_disparities(self, percentile=0.99, stretch=1.33333):
        for (img_idx, image) in enumerate(self.image_list.images):
            self.image_list.images[img_idx].estimated_max_disparity = None
            self.image_list.images[img_idx].estimated_min_disparity = None

    def get_angle(self, v1, v2):
        """ get angle between two vectors in 3D """
        # result in radius
        angle = np.arccos(v1.dot(v2))
        return angle

    def generate_neighbor_list(self, num_neighbors, max_neighbor_range = 10):
        self.max_neigh_range = max_neighbor_range
        self.num_permute = []
        # for each image
        for ref_idx in tqdm(range(self.image_list.length)):
            neighbor_list = []

            ref_image = self.image_list.images[ref_idx]
            ref_seq_id = ref_image.seq_id
            ref_img_id = ref_image.img_id
            view_vec_ref = ref_image.view_vec
            num_image_seq = self.image_list.num_images[ref_seq_id]
            if num_image_seq < num_neighbors+1:
                self.num_permute.append(1)
                neighbor_image_id_list = [ni for ni in range(num_image_seq)]
                neighbor_image_id_list.remove(ref_img_id)
                self.image_list.images[ref_idx].neighbor_list = [
                    [self.image_list.get_id_by_seq(ref_seq_id, img_i) for img_i in neighbor_image_id_list]]
                self.image_list.images[ref_idx].isValid = True
                continue

            # preprocessed dataset and load neighbors according to depth reprojection
            neighbor_list_from_file = []
            neighpath = os.path.join(self.image_list.basepath, "{:04d}".format(ref_seq_id), "neighbors", "{:04d}.json".format(ref_img_id))
            if os.path.isfile(neighpath):
                with open(neighpath) as f:
                    neighbor_list_from_file = np.array(json.load(f))
                if len(neighbor_list_from_file) == 0:
                    # print('WARNING: no valid neighbor found for (seq_id, img_id)', (ref_seq_id, ref_img_id))
                    self.num_permute.append(1)
                    self.image_list.images[ref_idx].neighbor_list = [[ref_image.id for ni in range(num_neighbors)]]
                    self.image_list.images[ref_idx].isValid = False
                    continue
            neighscorepath = os.path.join(self.image_list.basepath, "{:04d}".format(ref_seq_id), "neighbors", "{:04d}_score.json".format(ref_img_id))
            if os.path.isfile(neighscorepath):
                with open(neighscorepath) as f:
                    neighbor_score_list_from_file = json.load(f)
                    valid_depth_ratio = neighbor_score_list_from_file[1]
                    neighbor_score_list_from_file = np.array(neighbor_score_list_from_file[0])
                if valid_depth_ratio < 0.2:
                    # print('WARNING: no valid(valid depth insufficient) neighbor for (seq_id, img_id)', (ref_seq_id, ref_img_id))
                    self.num_permute.append(1)
                    self.image_list.images[ref_idx].neighbor_list = [[ref_image.id for ni in range(num_neighbors)]]
                    self.image_list.images[ref_idx].isValid = False
                    continue
                if len(neighbor_list_from_file)>0:
                    score_thres = 0.6
                    min_score_thres = 0.2
                    min_neigh_left = min(len(neighbor_list_from_file), num_neighbors)
                    while np.sum(neighbor_score_list_from_file > score_thres) < min_neigh_left and score_thres > min_score_thres:
                        score_thres = score_thres * 0.8
                    neighbor_list_from_file = neighbor_list_from_file[neighbor_score_list_from_file>score_thres]
                    if len(neighbor_list_from_file) == 0:
                        # print('WARNING: no valid(score too low) neighbor for (seq_id, img_id)', (ref_seq_id, ref_img_id))
                        self.num_permute.append(1)
                        self.image_list.images[ref_idx].neighbor_list = [[ref_image.id for ni in range(num_neighbors)]]
                        self.image_list.images[ref_idx].isValid = False
                        continue
            neighbor_list_from_file = neighbor_list_from_file[0:min(len(neighbor_list_from_file), self.max_neigh_range)]

            ###### random neighbor index
            num_images_seq = self.image_list.num_images[ref_seq_id]
            valid_neigh_idx = []
            ref_pos = -1
            if len(neighbor_list_from_file) < 1:
                minimum_angle_radians = 0.0*np.pi/180.0
                maximum_angle_radians = 60.0*np.pi/180.0
                # for each image in the same sequence, filter by view angle
                for img_id in range(num_images_seq):
                    if img_id == ref_img_id:
                        ref_pos = img_id
                        continue
                    neigh_image, neigh_idx = self.image_list.get_by_seq_with_idx(ref_seq_id, img_id)
                    if neigh_idx is None:
                        continue
                    view_angle = self.get_angle(view_vec_ref, neigh_image.view_vec)
                    if (view_angle > minimum_angle_radians and view_angle < maximum_angle_radians):
                        valid_neigh_idx.append(neigh_idx)
                    # else:
                    #     print('INFO: view_angle inappropriate',
                    #           ref_image.filename, neigh_image.filename, minimum_angle_radians, maximum_angle_radians, view_angle, view_angle*180/np.pi)
            else:
                for img_id in neighbor_list_from_file:
                    _, neigh_idx = self.image_list.get_by_seq_with_idx(ref_seq_id, img_id)
                    if neigh_idx is None:
                        continue
                    valid_neigh_idx.append(neigh_idx)

            num_images_seq_valid = len(valid_neigh_idx)
            valid_neigh_idx = np.array(valid_neigh_idx)
            num_neigh_images = min(self.max_neigh_range, num_images_seq_valid)
            self.num_permute.append(int(np.ceil(float(num_neigh_images) / float(num_neighbors))))
            if num_images_seq < 2 or num_images_seq_valid < 1:
                print('WARNING: num_images_seq<2,  or num_images_seq_valid < 2', num_images_seq, num_images_seq_valid, ref_image.filename)
                self.image_list.images[ref_idx].neighbor_list = [[ref_image.id for ni in range(num_neighbors)]]
                self.image_list.images[ref_idx].isValid = False
                continue

            # skip current image if insufficient number of images in seq
            if num_images_seq_valid < num_neighbors:
                self.image_list.images[ref_idx].neighbor_list = None
                self.image_list.images[ref_idx].isValid = False
                continue

            if num_images_seq_valid > self.max_neigh_range:
                ref_pos_valid = np.argmin(abs(np.array(valid_neigh_idx)-ref_image.id-1))
                if ref_pos_valid < self.max_neigh_range/2:
                    nidx_start = 0
                    nidx_end = self.max_neigh_range
                else:
                    if ref_pos_valid > (num_images_seq_valid - self.max_neigh_range/2):
                        nidx_start = num_images_seq_valid - self.max_neigh_range
                        nidx_end = num_images_seq_valid
                    else:    
                        nidx_start = max(0, ref_pos_valid - int(self.max_neigh_range/2))
                        nidx_end = min(num_images_seq_valid, self.max_neigh_range + nidx_start)
                neigh_id_list = valid_neigh_idx[np.arange(nidx_start, nidx_end)]
            else:
                neigh_id_list = valid_neigh_idx
            neigh_id_list_extend = np.array(neigh_id_list)
            neigh_id_list = np.append(ref_image.id, neigh_id_list)
            # while len(neigh_id_list_extend) < num_neighbors:
            #     neigh_id_list_extend = np.append(neigh_id_list_extend, np.random.choice(neigh_id_list))
            for _ in range(self.num_permute[-1]):
                np.random.shuffle(neigh_id_list_extend)
                tmp_list = []
                for ni in range(num_neighbors):
                    nidx = neigh_id_list_extend[ni]
                    tmp_list.append(self.image_list.images[nidx].id)
                neighbor_list.append(tmp_list)
            ###### 

            if neighbor_list:
                # print(len(neighbor_list), neighbor_list)
                self.image_list.images[ref_idx].neighbor_list = neighbor_list
                self.image_list.images[ref_idx].isValid = True
            else:
                self.image_list.images[ref_idx].neighbor_list = None
                self.image_list.images[ref_idx].isValid = False

    def get_sub_neighbor_list(self, num_neighbors):
        for ref_idx in tqdm(range(self.image_list.length)):
            if self.image_list.images[ref_idx].isValid == False:
                continue
            sub_neighbor_list = []
            sub_neighbor_filename_list = []
            neighbor_list = self.image_list.images[ref_idx].neighbor_list[0]
            for neigh_idx in neighbor_list:
                # print('neigh_idx', neighbor_list, neigh_idx)
                neigh_image = self.image_list.get_by_id(neigh_idx)
                if neigh_image.isValid == False:
                    sub_neighbor = np.array(neighbor_list).tolist()
                    sub_neighbor.append(self.image_list.images[ref_idx].id)
                    sub_neighbor.remove(neigh_idx)
                else:
                    sub_neighbor = neigh_image.neighbor_list[0]
                sub_neighbor_list.append(sub_neighbor)
                sub_neighbor_filename = []
                for sub_id in sub_neighbor_list[-1]:
                    sub_neighbor_filename.append(
                        self.image_list.get_filename_by_id(sub_id)[-8:-4])
                sub_neighbor_filename_list.append(sub_neighbor_filename)
            self.image_list.images[ref_idx].sub_neighbor_list = sub_neighbor_list
            self.image_list.images[ref_idx].sub_neighbor_filename_list = sub_neighbor_filename_list
            

