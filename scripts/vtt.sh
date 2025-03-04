load_model=RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth
proj_dir=rwkv7-3b-pretrain-siglip
data_file=llava_pretrain595

n_layer=32
n_embd=2560

encoder_path="google/siglip2-base-patch16-384"
encoder_type=siglip
data_type=hf_img

micro_bsz=8
epoch_save=1
epoch_steps=10000
ctx_len=2048


HF_ENDPOINT="https://hf-mirror.com" python world_train.py \
--load_model $load_model \
--proj_dir $proj_dir --data_file $data_file \
--data_type $data_type \
--vocab_size 65536 \
--n_layer $n_layer --n_embd $n_embd \
--ctx_len $ctx_len --micro_bsz $micro_bsz \
--epoch_steps $epoch_steps --epoch_count 1 --epoch_begin 0 --epoch_save $epoch_save \
--lr_init 1e-3 --lr_final 0 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 8 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
--encoder_path $encoder_path --encoder_type $encoder_type \
--my_testing "x070" --train_step adapter rwkv