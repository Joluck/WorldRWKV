load_model=/home/rwkvos/model/rwkv/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth
proj_dir=/home/rwkvos/peter/out_model/rwkv7-0.4b-clip-pretrain
data_file=/home/rwkvos/data/hf-imgs/pretrain595

n_layer=24
n_embd=1024

encoder_path=/home/rwkvos/model/clip
encoder_type=clip
data_type=hf_img

micro_bsz=32
epoch_save=1
epoch_steps=18605 #6171
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
--my_testing "x070" --train_step adapter --wandb visual