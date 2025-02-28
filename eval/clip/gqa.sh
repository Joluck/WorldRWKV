answer_file=/home/rwkvos/data/eval/gqa/answers/rwkv7-3b-visual
mod=/home/rwkvos/model/clip
type=clip
python -m eval.vqa2 \
    --model-path /home/rwkvos/JL/out_model/rwkv7-3b-visual/rwkv-0 \
    --question-file /home/rwkvos/data/eval/gqa/llava_gqa_testdev_balanced.jsonl \
    --image-folder /home/rwkvos/data/eval/gqa/images \
    --answers-file $answer_file/merge.jsonl \
    --conv_mode $mod \
    --type $type \
    --temperature 0

python -m eval.convert_gqa \
    --src $answer_file/merge.jsonl \
    --dst /home/rwkvos/data/eval/gqa/testdev_balanced_predictions.json

python /home/rwkvos/data/eval/gqa/eval.py --tier /home/rwkvos/data/eval/gqa/testdev_balanced