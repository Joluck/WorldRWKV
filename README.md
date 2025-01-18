
<h1 align="center">
  <p>RWKV-World</p>
</h1>
world
world_load.py  加载不同模态的encoder
world_encoder.py 打算作为不同模态的统一框架

sh中的参数
train_steps moda adapter rwkv  3部分控制整个world model的训练。一般情况下不训练moda他只作为encoder 

目前encoder并没有直接batch而是先计算 其他模态+语言模态总长 batch内进行填充
world.py 中的pad_mod 主要目前将batch长度填充为16倍数

gen.py 目前是推理结构后续还要优化的更分离
