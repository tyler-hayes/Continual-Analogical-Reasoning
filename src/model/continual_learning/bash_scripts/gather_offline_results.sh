#!/usr/bin/env bash
PROJ_ROOT=/media/tyler/Data/codes/Continual-Analogical-Reasoning/src/model
export PYTHONPATH=${PROJ_ROOT}
source activate continual_analogical_reasoning
cd ${PROJ_ROOT}/continual_learning

IMG_DIR=/media/tyler/Data/datasets/RAVEN-10000-small
CKPT_PATH=${PROJ_ROOT}/continual_learning/analogical_reasoning_results/offline_ckpts

for BASE in cs ud d4
do
	for TASK in 1 2 3 4 5 6
	do
		NAME=${BASE}_${TASK}
		EXPT_NAME=offline_expert_tasks_${NAME}
		echo "Experiment: ${EXPT_NAME}"

		python -u gather_offline_expert_results.py \
		--path ${IMG_DIR} \
		--epochs 250 \
		--model Rel-Base \
		--ckpt_path ${CKPT_PATH} \
		--load_ckpt_name ${EXPT_NAME} > analogical_reasoning_logs/${EXPT_NAME}_evaluate.log
	done
done

