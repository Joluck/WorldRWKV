answer_file=/mnt/program/WorldRWKV/Visual/jl_data/eval/vqav2/answers/rwkv7-3b-siglip
mod="/home/rwkvos/models/siglip2"
type=siglip
base_dir=/mnt/program/WorldRWKV/Visual/jl_data/eval/vqav2 


python -m eval.vqa2 \
    --model-path /home/rwkvos/world/out_model/rwkv7-3b-conv/step2/rwkv-0 \
    --question-file $base_dir/llava_vqav2_mscoco_test-dev2015.jsonl \
    --image-folder $base_dir/test2015 \
    --answers-file $answer_file/merge.jsonl \
    --conv_mode $mod \
    --type $type \
    --temperature 0

python -m eval.eval_vqav2 \
    --dir $base_dir \
    --split llava_vqav2_mscoco_test-dev2015 \
    --ckpt $answer_file
