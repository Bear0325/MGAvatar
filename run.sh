# conda activate gsavatar

export CUDA_VISIBLE_DEVICES=0
subjects=(074)
for SUBJECT in "${subjects[@]}"; do
    python train.py \
    -s ../${SUBJECT}/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION10_${SUBJECT} \
    -m output/UNION10EMOEXP_${SUBJECT}_eval_600k\
    --port 6005 --eval --white_background --bind_to_mesh
done
