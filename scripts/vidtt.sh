load_model=/home/rwkv/alic-li/WorldRWKV/rwkv-0.pth
proj_dir=rwkv7-0.4b-video-siglip-ocr-base
data_file=/home/rwkv/alic-li/video_datasets/LLaVA-Video-178K/1_2_m_nextqa

n_layer=24
n_embd=1024

encoder_path="google/siglip2-base-patch16-384"
encoder_type=siglip
data_type=video

micro_bsz=12
epoch_save=1
epoch_steps=570
ctx_len=2048


HF_ENDPOINT="https://hf-mirror.com" python world_train.py \
--load_model $load_model  \
--proj_dir $proj_dir --data_file $data_file \
--data_type $data_type \
--vocab_size 65536 \
--n_layer $n_layer --n_embd $n_embd \
--ctx_len $ctx_len --micro_bsz $micro_bsz \
--epoch_steps $epoch_steps --epoch_count 1 --epoch_begin 0 --epoch_save $epoch_save \
--lr_init 1e-3 --lr_final 0 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 4 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
--encoder_path $encoder_path --encoder_type $encoder_type \
--my_testing "x070" --train_step adapter rwkv