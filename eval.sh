export HF_TOKEN="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"
export HF_HOME="~/.cache/huggingface"
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://192.168.0.18:8811/v1"
# export AZURE_OPENAI_API_VERSION="qwen2.5vl"

# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
python3 -m lmms_eval \
    --model openai_compatible \
    --tasks mme \
    --batch_size 1