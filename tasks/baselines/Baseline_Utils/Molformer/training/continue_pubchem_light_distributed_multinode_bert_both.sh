: "${MOLFORMER_RESTART_PATH:?Set MOLFORMER_RESTART_PATH to a checkpoint inside this repository}"
source "$(dirname "${BASH_SOURCE[0]}")/environ.sh" || exit 1
python train_pubchem_light.py \
        --device cuda \
        --n_batch 800  \
        --n_head 12 \
        --n_layer 12 \
        --n_embd 768 \
        --max_len 202 \
        --d_dropout 0.2 \
        --lr_start 3e-5 \
        --lr_multiplier 8 \
        --n_workers 16 \
        --max_epochs 4 \
        --gpu 8 \
        --num_nodes 2 \
        --accelerator ddp \
        --num_feats 32 \
        --root_dir . \
        --checkpoint_every 1000 \
        --grad_acc 1\
        --train_load 'both' \
        --eval_every 2500 \
        --rotate \
        --debug \
        --model_arch 'BERT_16GPU_Long_Run_with_Rotate_Continued' \
        --restart_path "$MOLFORMER_RESTART_PATH" \
        | tee $HOSTNAME.$LSF_PM_XPROCID.$(date +%F_%R).log
