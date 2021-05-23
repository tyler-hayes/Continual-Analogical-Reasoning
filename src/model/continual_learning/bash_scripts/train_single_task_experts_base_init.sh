#!/usr/bin/env bash
PROJ_ROOT=/media/tyler/Data/codes/Continual-Analogical-Reasoning/src/model
export PYTHONPATH=${PROJ_ROOT}
source activate continual_analogical_reasoning
cd ${PROJ_ROOT}/continual_learning

IMG_DIR=/media/tyler/Data/datasets/RAVEN-10000-small
CKPT_PATH=${PROJ_ROOT}/continual_learning/analogical_reasoning_results/offline_ckpts
LOG_PATH=${PROJ_ROOT}/continual_learning/analogical_reasoning_results/offline_tb_logs

TASK_TYPES=('cs' 'ud' 'd4')

for ((i=0;i<${#TASK_TYPES[@]};++i))
do
    TASK="${TASK_TYPES[i]}"

	EXPT_NAME=single_task_expert_base_init_${TASK}
	echo "Experiment: ${EXPT_NAME}"

	python -u main_offline_model.py \
	--path ${IMG_DIR} \
	--epochs 50 \
	--trn_configs ${TASK} \
	--tst_configs ${TASK} \
	--model Rel-Base \
	--ckpt_path ${CKPT_PATH} \
	--ckpt_name ${EXPT_NAME} \
	--log ${LOG_PATH}tensorboard_logs_${EXPT_NAME}/ > analogical_reasoning_logs/${EXPT_NAME}.log
	
done
