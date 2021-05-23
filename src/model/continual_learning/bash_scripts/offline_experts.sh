#!/usr/bin/env bash
PROJ_ROOT=/media/tyler/Data/codes/Continual-Analogical-Reasoning/src/model
export PYTHONPATH=${PROJ_ROOT}
source activate continual_analogical_reasoning
cd ${PROJ_ROOT}/continual_learning

IMG_DIR=/media/tyler/Data/datasets/RAVEN-10000-small
CKPT_PATH=${PROJ_ROOT}/continual_learning/analogical_reasoning_results/offline_ckpts
LOG_PATH=${PROJ_ROOT}/continual_learning/analogical_reasoning_results/offline_tb_logs

EXPT_NAME=offline_expert_tasks_${NAME}
echo "Experiment: ${EXPT_NAME}"
echo "Task: ${TASK}"

python -u main_offline_model.py \
--path ${IMG_DIR} \
--epochs 250 \
--trn_configs ${TASK} \
--tst_configs ${TASK} \
--model Rel-Base \
--early_stop \
--ckpt_path ${CKPT_PATH} \
--ckpt_name ${EXPT_NAME} \
--log ${LOG_PATH}tensorboard_logs_${EXPT_NAME}/ > analogical_reasoning_logs/${EXPT_NAME}.log

