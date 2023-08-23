MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 500 --learn_sigma True
--noise_schedule linear --num_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True
--use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 50 --timestep_respacing 200"
DATAPATH=datasets/celeba_dime
#MODELPATH=peal_runs/dime_test/model.pt
MODELPATH=peal_runs/ddpm_CopyrightTag_celeba_new/final.pt
CLASSIFIERPATH=peal_runs/ace_test/classifier.pth
#CLASSIFIERPATH=peal_runs/Smiling_confounding_CopyrightTag_celeba_classifier_unpoisened/model.cpl
ORACLEPATH=peal_runs/dime_test/oracle.pth
#OUTPUT_PATH=peal_runs/dime_test/outputs_own_classifier
OUTPUT_PATH=peal_runs/dime_test/outputs_own_generator
EXPNAME=experiment1

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
NUMBATCHES=1 #100

python -W ignore run_dime.py $MODEL_FLAGS $SAMPLE_FLAGS \
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
  --data_dir $DATAPATH