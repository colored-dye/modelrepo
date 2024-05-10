$env:HF_ENDPOINT="https://hf-mirror.com"

$ROOT_DIR="store/"
$CSV_FILE="job.csv"
$BACKEND="torch"
$CACHE_DIR="cache/"
$HF_USERNAME="colored-dye"
$HF_TOKEN="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"


python upload.py --root_dir $ROOT_DIR --csv_file $CSV_FILE --backend $BACKEND --cache_dir $CACHE_DIR --hf_username $HF_USERNAME --hf_token $HF_TOKEN --max_connections 16
