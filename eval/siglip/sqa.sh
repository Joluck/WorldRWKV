answer=/mnt/program/WorldRWKV/Visual/jl_data/eval/gqa/answers/rwkv7-0.4b-g1-siglip
mod="/home/rwkvos/models/siglip2"
type=siglip
base_dir=/mnt/program/WorldRWKV/Visual/jl_data/eval/scienceqa 

python -m eval.model_vqa_science \
    --model-path /home/rwkvos/WorldRWKV/out_model/rwkv7-g1-0.4b/step2/rwkv-0 \
    --question-file $base_dir/llava_test_CQM-A.json \
    --image-folder $base_dir/images/test \
    --answers-file $answer.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $mod \
    --type $type

python -m eval.eval_scienceqa \
    --base-dir $base_dir \
    --result-file $answer.jsonl \
    --output-file $answer-output.jsonl \
    --output-result $answer-result.json
