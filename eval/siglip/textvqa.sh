answer=/home/rwkvos/data/eval/textvqa/answers/rwkv7-3b-siglip
mod="google/siglip2-base-patch16-384"
type=siglip
python -m eval.textvqa \
    --model-path /home/rwkvos/JL/out_model/rwkv7-3b-siglip/rwkv-0 \
    --question-file /home/rwkvos/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /home/rwkvos/data/eval/textvqa/images/train_images \
    --answers-file $answer.jsonl \
    --conv_mode $mod \
    --type $type \
    --temperature 0

python -m eval.eval_textvqa \
    --annotation-file /home/rwkvos/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $answer.jsonl