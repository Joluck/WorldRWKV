
import torch
from torch.nn import functional as F
def pad_mod(self, tensor_list, signal_list):
    """
    对一个包含不同长度张量的列表进行填充，使所有张量的长度相同且为16的整数倍，并生成掩码。
    参数:
        tensor_list (list of torch.Tensor): 输入的张量列表，每个张量形状为 [seq_len]。
        pad_value (int, optional): 填充值，默认值为 0。
    返回:
        padded_tensor (torch.Tensor): 填充后的张量，形状为 [batch_size, target_len]。
        mask (torch.Tensor): 填充掩码，1 表示有效数据，0 表示填充部分。
    """

    modality_list = []
    #max_len = max((token.size(0) + signal.size(1)) for token, signal in zip(tensor_list, modality_list))
    max_len = 0
    for token, signal in zip(tensor_list, signal_list):

        modality = self.modality(signal)
        if modality is False:
            modality_list.append(False)
            continue
        modality_list.append(modality)
        max_len = max(token.size(0) + modality.size(1), max_len)

    # 计算目标长度（向上取整到16的整数倍）
    target_len = ((max_len + 15) // 16 * 16)+1

    if self.args.ctx_len is not None:
        target_len = min(target_len, self.args.ctx_len+1)

    masks = torch.zeros((len(tensor_list), target_len-1), dtype=torch.int32)
    x = []
    y = []
    s = []
    m = []
    for token, signal, mask in zip(tensor_list, modality_list, masks):
        if signal is False:
            continue
        pad_len = target_len-(token.size(0) + signal.size(1))
        
        padded_token = F.pad(token, (0, pad_len), value=0)

        x_token = padded_token[:-1]
        y_token = F.pad(padded_token, (signal.size(1)-1, 0), value=0)

        mask[signal.size(1) : -pad_len] = 1 
        
        s.append(signal)
        x.append(x_token)
        y.append(y_token)
        m.append(mask)

    y = torch.stack(y, dim=0)
    m = torch.stack(m, dim=0).cuda()
    
    return s, x, y, m



def mod_pad_text(self, signal_list, text_inputs, text_labels):
    """
    对一个包含不同长度张量的列表进行填充，使所有张量的长度相同且为16的整数倍，并生成掩码。
    参数:
        tensor_list (list of torch.Tensor): 输入的张量列表，每个张量形状为 [seq_len]。
        pad_value (int, optional): 填充值，默认值为 0。
    返回:
        padded_tensor (torch.Tensor): 填充后的张量，形状为 [batch_size, target_len]。
        mask (torch.Tensor): 填充掩码，1 表示有效数据，0 表示填充部分。
    """

    modality_list = []
    #max_len = max((token.size(0) + signal.size(1)) for token, signal in zip(tensor_list, modality_list))
    max_len = 0
    for i, (signal, token, label) in enumerate(zip(signal_list, text_inputs, text_labels)):

        modality = self.modality(signal)
        modality_list.append(modality)
        mod_label = torch.full((modality.size(1),), -100, device='cuda')
        text_labels[i] = torch.cat([mod_label, label])
        max_len = max(token.size(0) + modality.size(1), max_len)

    # 计算目标长度（向上取整到16的整数倍）
    target_len = ((max_len + 15) // 16 * 16)+1

    if self.args.ctx_len is not None:
        target_len = min(target_len, self.args.ctx_len+1)

 
    for i, (signal, token, label) in enumerate(zip(modality_list , text_inputs, text_labels)):
        pad_len = target_len-(token.size(0) + signal.size(1))
        
        text_inputs[i] = F.pad(token, (0, pad_len), value=0)[:-1]
        text_labels[i] = F.pad(label, (0, pad_len), value=-100)[1:]


    targets = torch.stack(text_labels, dim=0).cuda()
    
    return modality_list, text_inputs, targets



def cat_tts(self, tensor_list, signal_list):
    """
    对一个包含不同长度张量的列表进行填充，使所有张量的长度相同且为16的整数倍，并生成掩码。
    参数:
        tensor_list (list of torch.Tensor): 输入的张量列表，每个张量形状为 [seq_len]。
        pad_value (int, optional): 填充值，默认值为 0。
    返回:
        padded_tensor (torch.Tensor): 填充后的张量，形状为 [batch_size, target_len]。
        mask (torch.Tensor): 填充掩码，1 表示有效数据，0 表示填充部分。
    """

    modality_list = []
    atokens = []
    labels_list = [] #多模态拼接标签
    #max_len = max((token.size(0) + signal.size(1)) for token, signal in zip(tensor_list, modality_list))
    max_len = 0
    for token, signal in zip(tensor_list, signal_list):
        global_tokens, semantic_tokens = self.modality.world_encoder.encoder(signal)
        # print(global_tokens.squeeze(0).squeeze(0), global_tokens.squeeze(0).squeeze(0)+8192)
        global_tokens = global_tokens.squeeze(0).squeeze(0)+8194
        # global_tokens = F.pad(global_tokens.squeeze(0).squeeze(0)+8194, (0, 1), value=8193)
        semantic_tokens = F.pad(semantic_tokens.squeeze(0), (0, 1), value=8192)
        audio_token = torch.cat([global_tokens,semantic_tokens]) 
        mask_gt = torch.full_like(global_tokens, -100)
        label = torch.cat([global_tokens-1,semantic_tokens-1]) 
        # modality = self.modality.encoder(audio_token)

        # if modality is False:
        #     modality_list.append(False)
        #     continue
        
        mask_t = torch.full_like(token, -100)
        label = torch.cat([mask_t,label])
        atokens.append(audio_token)
        # modality_list.append(modality)
        labels_list.append(label)
        max_len = max(label.size(0), max_len)

    # 计算目标长度（向上取整到16的整数倍）
    target_len = ((max_len + 15) // 16 * 16)+1

    if self.args.ctx_len is not None:
        target_len = min(target_len, self.args.ctx_len+1)

    text_token = []
    labels = []
    mod_token = []

    for token, atoken, mask in zip(tensor_list, atokens, labels_list):

        pad_len = target_len-(token.size(0) + atoken.size(0))
        
        padded_atoken = F.pad(atoken, (0, pad_len), value=8192)

        atoken = padded_atoken[:-1]
        mod = self.modality(atoken)
        # padded_token = F.pad(signal, (0, 0, 0, pad_len), value=0)

        # pad_mod = padded_token[:,:-1,:]

        pad_mask = F.pad(mask, (0, pad_len), value=-100)[1:]
        
        mod_token.append(mod)

        labels.append(pad_mask)

    labels = torch.stack(labels, dim=0)

    
    return mod_token, tensor_list, labels