answer_file=/mnt/program/WorldRWKV/Visual/jl_data/eval/pope/answers/rwkv7-0.4b-g1-siglip
mod="/home/rwkvos/models/siglip2"
type=siglip
base_dir=/mnt/program/WorldRWKV/Visual/jl_data/eval/pope



python -m eval.vqa2 \
    --model-path /home/rwkvos/WorldRWKV/out_model/rwkv7-g1-0.4b/step2/rwkv-0  \
    --question-file $base_dir/llava_pope_test.jsonl \
    --image-folder $base_dir/images/val2014 \
    --answers-file $answer_file/merge.jsonl \
    --temperature 0 \
    --conv_mode  $mod \
    --type $type

python -m eval.eval_pope \
    --annotation-dir $base_dir/coco \
    --question-file $base_dir/llava_pope_test.jsonl \
    --result-file $answer_file/merge.jsonl