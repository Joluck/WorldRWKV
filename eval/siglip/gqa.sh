answer_file=/mnt/program/WorldRWKV/Visual/jl_data/eval/gqa/answers/rwkv7-3b-conv
mod="/home/rwkvos/models/siglip2"
type=siglip
base_dir=/mnt/program/WorldRWKV/Visual/jl_data/eval/gqa 


python -m eval.vqa2 \
    --model-path /home/rwkvos/world/out_model/rwkv7-3b-conv/step2/rwkv-0  \
    --question-file $base_dir/llava_gqa_testdev_balanced.jsonl \
    --image-folder $base_dir/images \
    --answers-file $answer_file/merge.jsonl \
    --conv_mode $mod \
    --type $type \
    --temperature 0

python -m eval.convert_gqa \
    --src $answer_file/merge.jsonl \
    --dst $base_dir/testdev_balanced_predictions.json

python $base_dir/eval.py --tier $base_dir/testdev_balanced
