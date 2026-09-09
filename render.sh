export CUDA_VISIBLE_DEVICES=0
subjects=(074)
for SUBJECT in "${subjects[@]}"; do
    python render.py \
    --iteration 240000\
    -m output/UNION10EMOEXP_${SUBJECT}_eval_600k\
    --skip_train
done
