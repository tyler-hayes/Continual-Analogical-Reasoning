#!/usr/bin/env bash
PROJ_ROOT=/media/tyler/Data/codes/Continual-Analogical-Reasoning/src/model
export PYTHONPATH=${PROJ_ROOT}
source activate base
cd ${PROJ_ROOT}/continual_learning

EPOCHS=50
IMG_DIR=/media/tyler/Data/datasets/RAVEN-10000-small

SAVE_DIR=${PROJ_ROOT}/continual_learning/analogical_reasoning_results/
CKPT_DIR=${PROJ_ROOT}/checkpoints

TASK_ORDERS=(
['cs' 'io' 'lr' 'ud' 'd4' 'd9' '4c']
['ud' 'cs' 'io' '4c' 'd9' 'd4' 'lr']
['d4' 'lr' '4c' 'ud' 'd9' 'cs' 'io']
)

BASE_TASKS=('cs' 'ud' 'd4')


for ((i=0;i<${#TASK_ORDERS[@]};++i))
do
    TASK_ORDER="${TASK_ORDERS[i]}"
    BASE_TASK="${BASE_TASKS[i]}"
    CKPT=${CKPT_DIR}/single_task_expert_base_init_${BASE_TASK}_final.pth
    
    for MODEL in fine_tune fine_tune_batch distillation ewc cumulative_replay
    do
    
    	if [ "${MODEL}" == "ewc" ]; then
        	REG=10
    	else
        	REG=1
    	fi
    	
	    EXPT_NAME=incremental_${MODEL}_lambda_${REG}_raven_base_task_${BASE_TASK}
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
