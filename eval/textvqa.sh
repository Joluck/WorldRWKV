python -m eval.textvqa \
    --model-path /home/rwkvos/JL/out_model/rwkv7-3b-inst-mlp/rwkv-0 \
    --question-file /home/rwkvos/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /home/rwkvos/data/eval/textvqa/images/train_images \
    --answers-file /home/rwkvos/data/eval/textvqa/answers/rwkv7-3-mlp.jsonl \
    --conv_mode /home/rwkvos/model/clip \
    --temperature 0