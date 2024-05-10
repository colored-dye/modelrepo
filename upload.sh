#!/usr/bin/sh

export HF_ENDPOINT="https://hf-mirror.com"

ROOT_DIR="store/"
CSV_FILE="test.csv"
BACKEND="torch"
CACHE_DIR="cache/"
LOG_DIR="logs/"
HF_USERNAME="colored-dye"
HF_TOKEN="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"


python upload.py \
    --root_dir $ROOT_DIR \
    --csv_file $CSV_FILE \
    --backend $BACKEND \
    --cache_dir $CACHE_DIR \
    --log_dir $LOG_DIR \
    --hf_username $HF_USERNAME \
    --hf_token $HF_TOKEN \
    --max_connections 16
