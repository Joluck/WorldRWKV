from world.speech_encoder import SpeechEncoder
from world.model import RWKV
from world.visual_encoder import VisualEncoder
def WorldLoading(args):
    modality = SpeechEncoder(
            args.load_moda,
            args.n_embd,
            device='cuda'
        )
    # modality = VisualEncoder(
    #     args.load_moda,
    #     args.n_embd,
    # )
    model = RWKV(args, modality=modality)
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