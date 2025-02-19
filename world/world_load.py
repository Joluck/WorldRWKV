from world.model import RWKV
from world.world_encoder import WorldEncoder
def WorldLoading(args):
    config = {
        'encoder_type': args.encoder_type,
        'encoder_path': args.encoder_path,
        'project_dim' : args.n_embd
        }
    modality = WorldEncoder(**config)
    
    model = RWKV(args, modality=modality)
    #model = RWKV(args)
    print(model)

    if 'moda' not in args.train_step:
        for param in model.modality.world_encoder.model.parameters():
            param.requires_grad = False
    if 'adapter' not in args.train_step:
        for param in model.modality.world_encoder.adapter.parameters():
            param.requires_grad = False
    if 'rwkv' not in args.train_step:
        for param in model.emb.parameters():
            param.requires_grad = False
        for param in model.blocks.parameters():
            param.requires_grad = False
        for param in model.ln_out.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = False
    return model