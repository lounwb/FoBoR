#!/bin/bash
# Usage: CUDA_VISIBLE_DEVICES=1 bash eval.sh <trainer1> [trainer2 ...] <dataset> <cfg> <ctp> <nctx> <shots> <csc> <alpha_value> <beta_value> <eta_value> <n_class> <n_patch> <lambda_value> <top_k>
# Example: CUDA_VISIBLE_DEVICES=1 bash eval.sh LoCoOp SCT Mambo imagenet vit_b16_ep30 end 16 16 False 0.2 3.0 5.0 1 1 0.2 200

# Basic configuration
ROOT="[YOUR_DATASET_PATH]"
T=1
SEEDS=(1 2 3)  # List of random seeds

# ===================== Core Modification: Argument Parsing =====================
# Validate argument count (At least 1 trainer + 13 parameters)
if [ $# -lt 14 ]; then
    echo "❌ Insufficient parameters!"
    echo "✅ Correct usage: $0 <trainer1> [trainer2 ...] <dataset> <cfg> <ctp> <nctx> <shots> <csc> <alpha_value> <beta_value> <eta_value> <n_class> <n_patch> <lambda_value> <top_k>"
    exit 1
fi

top_k=${@: -1}
lambda_value=${@: -2:1}
n_patch=${@: -3:1}
n_class=${@: -4:1}
eta_value=${@: -5:1}    
beta_value=${@: -6:1}
alpha_value=${@: -7:1}
csc=${@: -8:1}
shots=${@: -9:1}
nctx=${@: -10:1}
ctp=${@: -11:1}
cfg=${@: -12:1}
dataset=${@: -13:1}

TRAINERS=("${@:1: $#-13}")
# ============================================================

# Extract epoch number from CFG string
epoch=$(echo $cfg | grep -oE '[0-9]+' | tail -1)

for SEED in "${SEEDS[@]}";
do
    for TRAINER in "${TRAINERS[@]}";
    do
        # Process trainer name (remove suffix)
        TRAINER_NAME=$(echo "$TRAINER" | sed 's/-.*//')
        TRAINER_UPPER=$(echo $TRAINER_NAME | tr '[:lower:]' '[:upper:]')
        
        # Build output directory
        DIR=output/${dataset}/${TRAINER}/${cfg}_${shots}shots/nctx${nctx}_csc${csc}_ctp${ctp}_alpha${alpha_value}_beta${beta_value}_eta${eta_value}_nclass${n_class}_npatch${n_patch}_lambda${lambda_value}_topk${top_k}/seed${SEED}
        
        echo "📌 Evaluating model in: $DIR"
        # Execute evaluation script
        python eval_ood_detection.py \
            --root ${ROOT} \
            --seed ${SEED} \
            --trainer ${TRAINER_NAME} \
            --dataset-config-file configs/datasets/${dataset}.yaml \
            --config-file configs/trainers/${TRAINER_NAME}/${cfg}.yaml \
            --output-dir ${DIR} \
            --lambda_value ${lambda_value} \
            --topk ${top_k} \
            --model-dir ${DIR} \
            --load-epoch ${epoch} \
            --alpha_value ${alpha_value} \
            --beta_value ${beta_value} \
            --eta_value ${eta_value} \
            --num_classes ${n_class} \
            --num_patches ${n_patch} \
            --T ${T} \
            DATASET.NUM_SHOTS ${shots} \
            TRAINER.${TRAINER_UPPER}.N_CTX ${nctx} \
            TRAINER.${TRAINER_UPPER}.CSC ${csc} \
            TRAINER.${TRAINER_UPPER}.CLASS_TOKEN_POSITION ${ctp}
    done
done