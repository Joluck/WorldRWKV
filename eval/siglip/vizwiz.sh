answer_file=/mnt/program/WorldRWKV/Visual/jl_data/eval/vizwiz/answers/rwkv7-1.5b-siglip
mod="/home/rwkvos/models/siglip2"
type=siglip
base_dir=/mnt/program/WorldRWKV/Visual/jl_data/eval/vizwiz



python -m eval.vqa2 \
    --model-path /mnt/program/WorldRWKV/Visual/rwkv7-1.5b-siglip/rwkv-0  \
    --question-file $base_dir/llava_test.jsonl \
    --image-folder $base_dir/images/test \
    --answers-file $answer_file/merge.jsonl \
    --temperature 0 \
    --conv_mode  $mod \
    --type $type

python -m eval.convert_vizwiz_for_submission \
    --annotation-file $base_dir/llava_test.jsonl \
    --result-file $answer_file/merge.jsonl \
    --result-upload-file /mnt/program/WorldRWKV/Visual/jl_data/eval/vizwiz/answers_upload/rwkv7-1.5b-siglip.json