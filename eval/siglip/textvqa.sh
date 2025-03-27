answer=/mnt/program/WorldRWKV/Visual/jl_data/eval/textvqa/answers/rwkv7-0.4b-g1-siglip
mod=/home/rwkvos/models/siglip2
type=siglip
base_dir=/mnt/program/WorldRWKV/Visual/jl_data/eval/textvqa
python -m eval.textvqa \
    --model-path /home/rwkvos/WorldRWKV/out_model/rwkv7-g1-0.4b/step2/rwkv-0 \
    --question-file $base_dir/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $base_dir/images/train_images \
    --answers-file $answer.jsonl \
    --conv_mode $mod \
    --type $type \
    --temperature 0

python -m eval.eval_textvqa \
    --annotation-file $base_dir/TextVQA_0.5.1_val.json \
    --result-file $answer.jsonl
