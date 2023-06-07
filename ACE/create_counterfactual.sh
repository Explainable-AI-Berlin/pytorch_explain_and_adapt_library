SAMPLE_FLAGS="--batch_size 50 --timestep_respacing 200"
DATAPATH=/home/space/datasets/celeba
MODELPATH=../ACE/ddpm-celeba.pt
CLASSIFIERPATH=../ACE/classifier.pth
ORACLEPATH=../ACE/oracle.pth
OUTPUT_PATH=outputs
EXPNAME=example_name
NUMBATCHES=1

# parameters of the sampling
GPU=0
S=60
SEED=4
USE_LOGITS=True
CLASS_SCALES='8,10,15'
LAYER=18
PERC=30
L1=0.05
QUERYLABEL=31
TARGETLABEL=-1
IMAGESIZE=128  # dataset shape
python3 -W ignore main.py \
    --query_label $QUERYLABEL --target_label $TARGETLABEL \
    --output_path $OUTPUT_PATH --num_batches $NUMBATCHES \
    --start_step $S --dataset 'CelebAMV' \
    --exp_name $EXPNAME --gpu $GPU \
    --model_path $MODELPATH --classifier_scales $CLASS_SCALES \
    --classifier_path $CLASSIFIERPATH --seed $SEED \
    --oracle_path $ORACLEPATH \
    --l1_loss $L1 --use_logits $USE_LOGITS \
    --l_perc $PERC --l_perc_layer $LAYER \
    --save_x_t True --save_z_t True \
    --use_sampling_on_x_t True \
    --save_images True --image_size $IMAGESIZE \
    --data_dir $DATAPATH \
    $SAMPLE_FLAGS