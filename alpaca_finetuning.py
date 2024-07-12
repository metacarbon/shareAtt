import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DummyOptim
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import logging
import json
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

IGNORE_TOKEN_ID = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def safe_ids(ids, max_value, pad_id):
    return [i if i < max_value else pad_id for i in ids]


def tokenize(messages, tokenizer):
    input_ids = []
    labels = []
    
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

    if messages.get("input", "") != "":
        prompt = prompt_input.format_map(messages)
    else:
        prompt = prompt_no_input.format_map(messages)
    
    response = f"{messages['output']}{DEFAULT_EOS_TOKEN}"
    
    
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    # Append all the sections to the input_ids
    input_ids += prompt_ids + response_ids
    
    # The labels should ignore the prompt (set to IGNORE_TOKEN_ID),
    # and should match the response_ids for the response part
    labels += [IGNORE_TOKEN_ID] * len(prompt_ids) + response_ids
    
    # Ensure lengths do not exceed model's max length
    input_ids = input_ids[:tokenizer.model_max_length]
    labels = labels[:tokenizer.model_max_length]
    
    input_ids = safe_ids(input_ids, tokenizer.vocab_size, tokenizer.eos_token_id)
    labels = safe_ids(labels, tokenizer.vocab_size, IGNORE_TOKEN_ID)
    return input_ids, labels


class AlpacaData(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item = self.data[item]
        input_ids, labels = tokenize(item, self.tokenizer)
        return torch.tensor(input_ids), torch.tensor(labels)

    def collate_fn(self, data):
        input_ids, labels = zip(*data)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.eos_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
        attention_mask = input_ids.ne(self.tokenizer.eos_token_id)
        features = {
            'input_ids': input_ids.long(),
            'labels': labels.long(),
            'attention_mask': attention_mask.long(),
        }
        return features


def main():
    parser = argparse.ArgumentParser(description='Fine-tuning LLM')
    parser.add_argument('--model_path', type=str, default='./models/Llama-3-8b/', help='Path to the pre-trained model')
    parser.add_argument('--save_path', type=str, default='./out/llama3_8b_alpaca/', help='Path to save the fine-tuned model')
    args = parser.parse_args()
    model_path = args.model_path
    save_path = args.save_path
    
    set_seed(42)
    accelerator = Accelerator()
    batch_size = 16

    logger.info('Initializing tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="left", model_max_length=4096, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token

    logger.info('Initializing model...')
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    

    dataset = AlpacaData(json.load(open('data/alpaca_data_cleaned.json')), tokenizer)

    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn,
                                              batch_size=batch_size, num_workers=0, shuffle=True)

    dummy_optimizer = DummyOptim(model.parameters())

    logger.info('accelerator preparing...')
    model, optimizer, data_loader = accelerator.prepare(model, dummy_optimizer, data_loader)

    for epoch in range(2):
        logger.info('=' * 10 + f'Start training {save_path} epoch {epoch + 1}' + '=' * 10)
        accelerator.wait_for_everyone()
        model.train()
        pbar = tqdm(enumerate(data_loader), total=len(data_loader), disable=(not accelerator.is_local_main_process))
        loss_report = []
        with accelerator.accumulate(model):
            for i, batch in pbar:
                out = model(**batch)
                loss = out.loss

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()

                loss_report.append(accelerator.gather(loss).mean().item())
                pbar.set_description(f"epoch {epoch + 1} step {i}: train loss {sum(loss_report[-100:]) / len(loss_report[-100:]):.5f}.")
                
        accelerator.wait_for_everyone()
        # save model states
        model.save_checkpoint(f'{save_path}/{epoch}')
        logger.info(f'model for epoch {epoch + 1} is saved...')
        

if __name__ == '__main__':
    main()
