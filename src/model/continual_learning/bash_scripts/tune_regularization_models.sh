#!/usr/bin/env bash
PROJ_ROOT=/media/tyler/Data/codes/Continual-Analogical-Reasoning/src/model
export PYTHONPATH=${PROJ_ROOT}
source activate continual_analogical_reasoning
cd ${PROJ_ROOT}/continual_learning

EPOCHS=50
IMG_DIR=/media/tyler/Data/datasets/RAVEN-10000-small
TASK_ORDERS=(['cs' 'io' 'lr' 'ud' 'd4' 'd9' '4c'])
TASK_ORDER="${TASK_ORDERS[0]}"
BASE_TASK=cs

SAVE_DIR=${PROJ_ROOT}/continual_learning/analogical_reasoning_results/
CKPT_DIR=${PROJ_ROOT}/checkpoints
CKPT=${CKPT_DIR}/single_task_expert_base_init_${BASE_TASK}_final.pth


for REG in 1 10 100
do
    for MODEL in distillation ewc
    do
	    EXPT_NAME=incremental_${MODEL}_grid_search_lambda_${REG}_raven_base_task_${BASE_TASK}
	    echo "Experiment: ${EXPT_NAME}"
	    
	    python -u main_continual_raven.py \
	    --path ${IMG_DIR} \
	    --task_order ${TASK_ORDER} \
            --model Rel-Base \
            --epochs ${EPOCHS} \
            --save_dir ${SAVE_DIR}${EXPT_NAME} \
            --classifier_ckpt ${CKPT} \
            --val_every 1 \
            --test_every 1 \
	    --model_type ${MODEL} \
	    --reg_lambda ${REG} \
	    --expt_name ${EXPT_NAME} > analogical_reasoning_logs/${EXPT_NAME}.log
    done
done
