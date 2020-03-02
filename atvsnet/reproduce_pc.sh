
# # gen ours point cloud on low-res ETH3D
RECON_PATH='../eval/pointcloud/' 

# gen A-TVSNet depth
INPUT_PATH='../data/' 
MODEL_PATH='../model/model.ckpt' 
python eval_pointcloud.py \
--pretrained_model_ckpt_path=${MODEL_PATH} \
--data_root=${INPUT_PATH} \
--savepath=${RECON_PATH} 

# fuse depths
scene_list=('lakeside' 'sand_box' 'storage_room' 'storage_room_2' 'tunnel')
fusibile_exe_path='../fusibile/build/fusibile'
prob_thres=0.8
consist_num=2
disp_thres=0.01
for scene_name in ${scene_list[@]}
do
    dense_path=${RECON_PATH}${scene_name}
    python depth_fusion.py \
    --fusibile_exe_path=${fusibile_exe_path} \
    --dense_folder=${dense_path} \
    --prob_threshold=${prob_thres} \
    --num_consistent=${consist_num} \
    --disp_threshold=${disp_thres} 
done
