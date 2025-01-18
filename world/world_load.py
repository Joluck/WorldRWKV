from world.speech_encoder import SpeechEncoder
from world.model import RWKV

def WorldLoading(args):
    speech_encoder = SpeechEncoder(
            args.load_moda,
            args.n_embd,
            downsample_K=5,
            hidden_dim=2048,
            device='cuda'
        )
    model = RWKV(args, modality=speech_encoder)
    #model = RWKV(args)
    print(model)

    if 'moda' not in args.train_step:
        for param in model.modality.model.parameters():
            param.requires_grad = False
    if 'adapter' not in args.train_step:
        for param in model.modality.adapter.parameters():
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