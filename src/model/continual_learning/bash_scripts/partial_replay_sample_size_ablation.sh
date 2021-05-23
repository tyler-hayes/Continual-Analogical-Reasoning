#!/usr/bin/env bash
PROJ_ROOT=/media/tyler/Data/codes/Continual-Analogical-Reasoning/src/model
export PYTHONPATH=${PROJ_ROOT}
source activate continual_analogical_reasoning
cd ${PROJ_ROOT}/continual_learning

IMG_DIR=/media/tyler/Data/datasets/RAVEN-10000-small
SAVE_DIR=${PROJ_ROOT}/continual_learning/analogical_reasoning_results/
CKPT_DIR=${PROJ_ROOT}/checkpoints


REPLAY_TYPES=('random' 'logit_dist_proba_shift_min' 'confidence_proba_shift_min' 'margin_proba_shift_min' 'time_proba_shift_min' 'loss_proba_shift_min' 'replay_count_proba_shift_min')

TASK_ORDERS=(
['cs' 'io' 'lr' 'ud' 'd4' 'd9' '4c']
['ud' 'cs' 'io' '4c' 'd9' 'd4' 'lr']
['d4' 'lr' '4c' 'ud' 'd9' 'cs' 'io']
)

BASE_TASKS=('cs' 'ud' 'd4')

MODEL=partial_replay


# unbalanced
for ((i=0;i<${#TASK_ORDERS[@]};++i))
do
    TASK_ORDER="${TASK_ORDERS[i]}"
    BASE_TASK="${BASE_TASKS[i]}"
    CKPT=${CKPT_DIR}/single_task_expert_base_init_${BASE_TASK}_final.pth
	for ((j=0;j<${#REPLAY_TYPES[@]};++j))
	do
	    REPLAY_STRATEGY="${REPLAY_TYPES[j]}"
	    
	    for SAMPLES in 8 16 64
	    do
		    EXPT_NAME=incremental_${MODEL}_strategy_${REPLAY_STRATEGY}_raven_samples_${SAMPLES}_base_task_${BASE_TASK}
		    echo "Experiment: ${EXPT_NAME}"
		    
		    python -u main_continual_raven.py \
		    --path ${IMG_DIR} \
		    --task_order ${TASK_ORDER} \
		    --classifier_ckpt ${CKPT} \
		    --save_dir ${SAVE_DIR}${EXPT_NAME} \
		    --model Rel-Base \
		    --val_every 1 \
		    --test_every 1 \
		    --replay_samples ${SAMPLES} \
		    --model_type ${MODEL} \
		    --replay_strategy ${REPLAY_STRATEGY} \
		    --expt_name ${EXPT_NAME} > analogical_reasoning_logs/${EXPT_NAME}.log
	    done
	done
done



