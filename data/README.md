run create_tf.ipynb to create train.tf and val.tf first
then run

PIPELINE_CONFIG_PATH=\~/data/inception_aws.config  \
MODEL_DIR=~/modeloutput/ \
NUM_TRAIN_STEPS=50000   \
SAMPLE_1_OF_N_EVAL_EXAMPLES=1    \
python \~/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
--alsologtostderr




