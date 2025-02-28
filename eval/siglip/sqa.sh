answer=/home/rwkvos/data/eval/gqa/answers/rwkv7-3b-siglip
mod="google/siglip2-base-patch16-384"
type=siglip
python -m eval.model_vqa_science \
    --model-path /home/rwkvos/JL/out_model/rwkv7-3b-siglip/rwkv-0 \
    --question-file /home/rwkvos/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /home/rwkvos/data/eval/scienceqa/images/test \
    --answers-file $answer.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $mod \
    --type $type

python -m eval.eval_scienceqa \
    --base-dir /home/rwkvos/data/eval/scienceqa \
    --result-file $answer.jsonl \
    --output-file $answer-output.jsonl \
    --output-result $answer-result.json