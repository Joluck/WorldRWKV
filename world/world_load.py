from world.model import ModRWKV
import torch
from collections import OrderedDict

def WorldLoading(args):

    model = ModRWKV(args)
    model._set_trainable()
    #model = RWKV(args)
    print(model)
    print(f"########## Loading {args.load_model}... ##########")
    state_dict = torch.load(args.load_model, map_location="cpu", weights_only=True)
    new_state_dict = {
        f"llm.{k}" if not k.startswith('llm.') else k: v
        for k, v in state_dict.items()
        if not (k.startswith('proj.') or k.startswith('encoder.'))
    }
    model.load_state_dict(new_state_dict, strict=False)

    return model