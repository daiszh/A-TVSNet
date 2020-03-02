

device_id=1

# python eval_end2end_eth3d_split_agg_refine.py --gpu_id=$device_id \
# --pretrained_model_ckpt_path='../tf_model_mvs_evo_end2end/bkp_evo_end2end_refine_singleagg_singleW_demon_batch16/model.ckpt' \
# --save_path='ablation_agg_refined' \
# --ckpt_step=100000 \
# --view_num=10 \
# --save_depths=True \
# --dual=False

# python eval_end2end_eth3d_split_agg_refine.py --gpu_id=$device_id \
# --pretrained_model_ckpt_path='../tf_model_mvs_evo_end2end/bkp_evo_end2end_refine_singleagg_demon_batch16/model.ckpt' \
# --save_path='ablation_agg_refined' \
# --ckpt_step=100000 \
# --view_num=10 \
# --save_depths=True \
# --dual=False

# python eval_end2end_eth3d_split_agg_refine.py --gpu_id=$device_id \
# --pretrained_model_ckpt_path='../tf_model_mvs_evo_end2end/bkp_evo_end2end_refine_singleagg_demon_batch16/model.ckpt' \
# --ckpt_step=150000 \
# --save_path='ablation_agg_refined' \
# --view_num=10 \
# --save_depths=True \
# --dual=False

# # ablation AAM2 / Attset, newly trained 2019/10
# for views in 3 5 10
# do
#     python eval_end2end_eth3d_split_agg_refine.py --gpu_id=$device_id \
#     --pretrained_model_ckpt_path='../tf_model_mvs_evo_end2end/bkp_evo_end2end_refine_singleagg_singleW_demon_batch16/model.ckpt' \
#     --save_path='ablation_agg_refined' \
#     --ckpt_step=100000 \
#     --view_num=$views \
#     --save_depths=True \
#     --dual=False
#     python eval_end2end_eth3d_split_agg_refine.py --gpu_id=$device_id \
#     --pretrained_model_ckpt_path='../tf_model_mvs_evo_end2end/bkp_evo_end2end_refine_singleagg_demon_batch16/model.ckpt' \
#     --save_path='ablation_agg_refined' \
#     --ckpt_step=150000 \
#     --view_num=$views \
#     --save_depths=True \
#     --dual=False
# done

for views in 3 4 5 6 7 8 9 10 11 12 13 14 15
do
    python eval_end2end_eth3d_split_agg_refine.py --gpu_id=$device_id --view_num=$views \
    --pretrained_model_ckpt_path='../tf_model_mvs_evo_end2end/bkp_evo_end2end_refine_singleagg_singleW_demon_batch16/model.ckpt' \
    --ckpt_step=100000 \
    --save_path='ablation_agg_refined' \
    --save_depths=False \
    --dual=False
    python eval_end2end_eth3d_split_agg_refine.py --gpu_id=$device_id --view_num=$views \
    --pretrained_model_ckpt_path='../tf_model_mvs_evo_end2end/bkp_evo_end2end_refine_singleagg_singleW_demon_batch16/model.ckpt' \
    --ckpt_step=50000 \
    --save_path='ablation_agg_refined' \
    --save_depths=False \
    --dual=False
    python eval_end2end_eth3d_split_agg_refine.py --gpu_id=$device_id --view_num=$views \
    --pretrained_model_ckpt_path='../tf_model_mvs_evo_end2end/bkp_evo_end2end_refine_singleagg_demon_batch16/model.ckpt' \
    --save_path='ablation_agg_refined' \
    --ckpt_step=100000 \
    --save_depths=False \
    --dual=False
done


# python eval_end2end_eth3d_split_agg_refine.py --gpu_id=1 \
# --pretrained_model_ckpt_path='../tf_model_mvs_evo_end2end/bkp_evo_end2end_refine_multiagg_singleconv_plus_demon_batch16/model.ckpt' \
# --save_path='ablation_agg_refined' \
# --ckpt_step=150000 \
# --view_num=5 \
# --save_depths=True \
# --dual=True

# python eval_end2end_eth3d_split_agg_refine.py --gpu_id=1 \
# --pretrained_model_ckpt_path='../tf_model_mvs_evo_end2end/bkp_evo_end2end_refine_multiagg_singleconv_plus_demon_batch16/model.ckpt' \
# --save_path='ablation_agg_refined' \
# --ckpt_step=200000 \
# --view_num=5 \
# --save_depths=True \
# --dual=True

# A-TVSNet ETH3D (ablation AAM1 & AAM2)
for checkpoint in 200000 150000 
do 
    for views in 3 5 10
    do
        python eval_end2end_eth3d_split_agg_refine.py --gpu_id=$device_id \
        --pretrained_model_ckpt_path='../tf_model_mvs_evo_end2end/bkp_evo_end2end_refine_multiagg_singleconv_plus_demon_batch16/model.ckpt' \
        --save_path='ablation_dualagg_refined' \
        --ckpt_step=$checkpoint \
        --view_num=$views \
        --save_depths=True \
        --dual=True --max_w=960 --max_h=640
    done

    for views in 3 4 5 6 7 8 9 10 11 12 13 14 15
    do
        python eval_end2end_eth3d_split_agg_refine.py --gpu_id=$device_id \
        --pretrained_model_ckpt_path='../tf_model_mvs_evo_end2end/bkp_evo_end2end_refine_multiagg_singleconv_plus_demon_batch16/model.ckpt' \
        --save_path='ablation_dualagg_refined' \
        --ckpt_step=$checkpoint \
        --view_num=$views \
        --save_depths=False \
        --dual=True
    done
done



# # TVSNet demon
# python eval_end2end_eth3d_split_agg_refine.py --gpu_id=$device_id \
# --pretrained_model_ckpt_path='../tf_model_mvs_evo_end2end/bkp_evo_end2end_refine_multiagg_singleconv_plus_demon_batch16/model.ckpt' \
# --save_path='ablation_agg_refined' \
# --data_type='demon' \
# --ckpt_step=200000 \
# --view_num=2 \
# --save_depths=True \
# --dual=True
