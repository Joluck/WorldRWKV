load_model=rwkv-x070-450m-world-v2.9-83%trained-20250101-ctx4k.pth
proj_dir=world-stt
data_file='/home/rwkv/JL/audio-data/new'

n_layer=24
n_embd=1024

micro_bsz=16
epoch_save=1
epoch_steps=1000
ctx_len=1024

encoder_path='/home/rwkv/JL/audio'
encoder_type='speech'


HF_ENDPOINT="https://hf-mirror.com" python world_train.py \
--load_model $load_model \
--proj_dir $proj_dir --data_file $data_file \
--data_type qa \
--vocab_size 65536 \
--n_layer $n_layer --n_embd $n_embd \
--ctx_len $ctx_len --micro_bsz $micro_bsz \
--epoch_steps $epoch_steps --epoch_count 1 --epoch_begin 0 --epoch_save $epoch_save \
--lr_init 1e-4 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 4 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
--encoder_path $encoder_path --encoder_type $encoder_type \
--my_testing "x070" --train_step adapter rwkv