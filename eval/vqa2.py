import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import re


from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder):
        self.questions = questions
        self.image_folder = image_folder

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        # if self.model_config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')

        # input_ids = f'\x16User: {qs}\x17Assistant:'
        input_ids = f'\x16User: {qs}\x17\x16Assistant:'
        # input_ids = f'\x16<|user|>:{qs}\x17<|assistant|>:'
        return input_ids, image

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors = zip(*batch)
    input_ids = list(input_ids)
    image_tensors = list(image_tensors)
    return input_ids, image_tensors

# DataLoader
def create_data_loader(questions, image_folder, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

from infer.worldmodel import Worldinfer


def eval_model(args):
    # Model
    model_name = 'RWKV-7'
    model_path = os.path.expanduser(args.model_path)
    model = Worldinfer(model_path=args.model_path, encoder_type=args.type, encoder_path=args.conv_mode)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")



    data_loader = create_data_loader(questions, args.image_folder)

    for (text, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        with torch.inference_mode():
            output_ids,_ = model.generate(text[0], image_tensor)


        outputs = output_ids[1:] #remove ' '
        if args.model_type == 'g1':
            answer_matches = re.findall(r'<answer>(.*?)</answer>', outputs, re.DOTALL)
            outputs = answer_matches[0].strip() if answer_matches else ""
        else:
            outputs = outputs   
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--type", type=str, default="clip")
    parser.add_argument("--model_type", type=str, default='world')
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
