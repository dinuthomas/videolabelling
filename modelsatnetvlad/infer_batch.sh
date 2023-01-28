# please do not change the values of the arguments if you really know what will happen

RECORDPATVAL="../data/yt8m/frame/val"
#SAVEPATH="./ckpt" # the pre-trained model path, please keep the default value
SAVEPATH="/opt/pri/HuaweiVideoTagging/tagging_model/ckpt" # the pre-trained model path, please keep the default value
#FEAT_PATH="./data/gpu8_feats" # the extracted feature path in the previous feature extraction step
FEAT_PATH="/opt/capstone/video_features"
export CUDA_VISIBLE_DEVICES=7 # the gpu id that will be used to do the inference

python infer_batch.py \
      --model=NetVLADModelLF \
      --train_dir="$SAVEPATH/NetVLADV3_distill/" \
      --eval_data_pattern="$FEAT_PATH" \
      --frame_features=True --feature_names="rgb,audio" \
        --feature_sizes="1024,128" \
        --batch_size=64 --base_learning_rate=0.0002 \
        --netvlad_cluster_size=108 \
        --netvlad_hidden_size=800 \
        --moe_l2=1e-6 --iterations=300 \
        --learning_rate_decay=0.8 \
        --netvlad_relu=False \
        --gating=True \
        --moe_prob_gating=True \
        --lightvlad=False \
        --num_gpu=1 \
        --num_epochs=10 \
        --run_once \
        --build_only=True \
        --sample_all
