answer=/home/rwkv/data/eval/textvqa/answers/rwkv7-1.5b-siglip
mod=/home/rwkv/model/siglip2basep16s384
type=siglip
python -m eval.textvqa \
    --model-path /home/rwkv/model/rwkv7-1.5b-siglip/rwkv-0 \
    --question-file /home/rwkv/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /home/rwkv/data/eval/textvqa/train_images \
    --answers-file $answer.jsonl \
    --conv_mode $mod \
    --type $type \
    --temperature 0

python -m eval.eval_textvqa \
    --annotation-file /home/rwkv/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $answer.jsonl