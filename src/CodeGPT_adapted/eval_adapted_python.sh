export CUDA_VISIBLE_DEVICES=0
LANG=python                       # set python for py150
DATADIR=../dataset/py150/token_completion
LITFILE=../dataset/py150/literals.json
OUTPUTDIR=../save/pyAdapted2
PRETRAINDIR=../save/pyAdapted2/checkpoint-last       # directory of your saved model
LOGFILE=py_adapted_eval.log

python -u run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --do_eval \
        --per_gpu_eval_batch_size=16 \
        --logging_steps=5 \
        --seed=42