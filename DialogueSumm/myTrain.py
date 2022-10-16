import os
import nltk
import json
import torch
import numpy as np
import random
import warnings
import importlib
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BartTokenizerFast
from transformers import BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from rouge import Rouge
from utils.config import *
from utils.utils import get_dataset, collate_fn


SummarizationModel = importlib.import_module('models.' + args.model_file).SummarizationModel
nltk.download('punkt')


MODEL_CLASSES = {
    'bart': (BartConfig, BartTokenizerFast)
}


warnings.filterwarnings("ignore")
device = torch.device("cuda:"+str(args.cuda)) if USE_CUDA else torch.device("cpu")
train_path = os.path.join(args.data_path, args.dataset, "train.json")
eval_path = os.path.join(args.data_path, args.dataset, "dev.json")
test_path = os.path.join(args.data_path, args.dataset, "test.json")
config_class, tokenizer_class = MODEL_CLASSES[args.model_type]


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


def cur_larger(cur_result, cur_best_result):
    if cur_result['r1'] != cur_best_result['r1']:
        return cur_result['r1'] > cur_best_result['r1']
    if cur_result['r2'] != cur_best_result['r2']:
        return cur_result['r2'] > cur_best_result['r2']
    return cur_result['rL'] > cur_best_result['rL']


def train(model, train_loader, eval_dataloader, test_dataloader, tokenizer):
    print("Traning arguments:")
    print(args)
    
    best_result = {'r1': 0.0, 'r2': 0.0, 'rL': 0.0}
    model.train()
    model.zero_grad()

    if args.fp16:
        scaler = GradScaler()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    ]
    all_optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_loader) * args.epochs
    num_warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(all_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
    logging_step = len(train_loader)
    steps = 0

    # eval_result = evaluate(model, eval_dataloader, tokenizer, best_result, is_test=False, file_name='eval_result.json')
    # evaluate(model, test_dataloader, tokenizer, cur_best_result=None, is_test=True, file_name='test_result.json')

    for epoch in range(args.epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
        for _, batch in pbar:
            inputs = {'input_ids': batch['input_ids'],
                      'attention_mask': batch['attention_mask'],
                      'decoder_attention_mask': batch['decoder_attention_mask'],
                      'labels': batch['labels']
                     }
            if args.num_decouple_layers > 0:
                inputs.update({'decoder_masks_dict': batch['masks_dict']})

            if args.fp16:
                with autocast():
                    outputs = model(**inputs)
                    loss = outputs['loss']
                    print_loss = loss.item()
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.unscale_(all_optimizer)
                    scaler.step(all_optimizer)
                    scaler.update()
            else:
                outputs = model(**inputs)
                loss = outputs['loss']
                print_loss = loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                all_optimizer.step()

            scheduler.step()
            model.zero_grad()

            pbar.set_description("Loss:%.3f,CL:%.3f" %(print_loss, print_loss))
            if steps != 0 and steps % logging_step == 0 or steps == t_total-1:
                print("\nEpoch {}, Step {}".format(epoch, steps))
                eval_result = evaluate(model, eval_dataloader, tokenizer, best_result, is_test=False, file_name='eval_result.json')
                if eval_result['r1'] < 0.2:
                    print("Become untrainable! Training canceled!!!")
                    return
                if cur_larger(eval_result, best_result):
                    best_result = eval_result
                evaluate(model, test_dataloader, tokenizer, cur_best_result=None, is_test=True, file_name='test_result.json')
            steps += 1


def evaluate(model, eval_loader, tokenizer, cur_best_result=None, is_test=False, file_name=None):
    def _cal_metrics(golden_dict, pred_dict):
        r1_num, r2_num, rL_num = 0.0, 0.0, 0.0
        total = len(golden_dict)
        evaluator = Rouge(metrics=['rouge-n', 'rouge-l'],
                           max_n=2,
                           limit_length=True,
                           length_limit=128,
                           length_limit_type='words',
                           apply_avg=args.avg,
                           apply_best=not args.avg,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)
        for qid in golden_dict.keys():
            golden = golden_dict[qid]
            pred = pred_dict[qid]
            rouge_scores = evaluator.get_scores(pred, golden)
            r1_num += rouge_scores['rouge-1']['f']
            r2_num += rouge_scores['rouge-2']['f']
            rL_num += rouge_scores['rouge-l']['f']
        return {"r1": r1_num / total, "r2": r2_num / total, "rL": rL_num / total}

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), ncols=100)
        pred_dict = {}

        for _, batch in pbar:
            inputs = {'input_ids': batch['input_ids'],
                      'attention_mask': batch['attention_mask']
                    }
            if args.num_decouple_layers > 0:
                inputs.update({'decoder_masks_dict': batch['masks_dict']})
                if hasattr(model.model, 'load_mha_params'):
                    inputs.update({'output_hidden_states': True})

            summary_ids_list = model.generate(**inputs, num_beams=args.num_beams,\
                    max_length=90 if args.dataset=='samsum' else 100, min_length=10, no_repeat_ngram_size=5, early_stopping=True)
            for summary_ids, qid in zip(summary_ids_list, batch['qid']):
                decoded_summary = tokenizer.decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                pred_dict[qid] = decoded_summary

    with open(test_path if is_test else eval_path, "r", encoding='utf-8') as reader:
        eval_data = json.load(reader)

    golden_dict = {d['qid']: d['summary'] for d in eval_data}
    result_dict = _cal_metrics(golden_dict, pred_dict)
    print("Test Result:" if is_test else "Eval result:", result_dict)

    if cur_best_result is not None:
        if cur_larger(result_dict, cur_best_result):
            print("model and arguments saved to {}...".format(args.save_path))
            save_path = os.path.join(args.save_path, "best_model.pth")
            args_save_path = os.path.join(args.save_path, "args.pth")
            torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
            torch.save(args, args_save_path, _use_new_zipfile_serialization=False)
    model.train()

    if file_name is not None:
        with open(os.path.join(args.save_path, file_name), "w", encoding='utf-8') as f:
            json.dump(pred_dict, f, indent=2)

    return result_dict


if __name__ == "__main__":
    is_eval = False
    # is_eval = True
    set_seed()

    if is_eval:
        model_path = "dialogsum_saves/bart_BiDeNL1lr2e-05_save_512_1919810/"
        # args = torch.load(model_path + 'args.pth')
        args.num_beams = 4
        args.batch_size = 8
        args.cuda = 3

    tokenizer = tokenizer_class.from_pretrained(args.model_name, cache_dir=args.cache_path)
    config = config_class.from_pretrained(args.model_name, cache_dir=args.cache_path)

    if not is_eval:
        train_dataset = get_dataset(train_path, args.cache_path,\
                tokenizer, args.max_length, training=True)
        eval_dataset = get_dataset(eval_path, args.cache_path,\
                tokenizer, args.max_length, training=False)
        test_dataset = get_dataset(test_path, args.cache_path,\
                tokenizer, args.max_length, training=False)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=collate_fn)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=collate_fn)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    else:
        eval_dataset = get_dataset(eval_path, args.cache_path, tokenizer, args.max_length, training=False)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=collate_fn)
        test_dataset = get_dataset(test_path, args.cache_path, tokenizer, args.max_length, training=False)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, collate_fn=collate_fn)

    model = SummarizationModel.from_pretrained(args.model_name, config=config, cache_dir=args.cache_path)
    if hasattr(model.model, 'load_mha_params'):
        print("Loading multi-head attention parameters from pretrained model...")
        model.model.load_mha_params()
    model = model.to(device)

    if not is_eval:
        train(model, train_dataloader, eval_dataloader, test_dataloader, tokenizer)
    else:
        model_saved = torch.load(model_path + "best_model.pth")
        # model_saved = torch.load(model_path + "best_model.pth", map_location={'cuda:0':'cuda:1'})
        model.load_state_dict(model_saved)
        # model.model.n_beams = args.num_beams
        evaluate(model, test_dataloader, tokenizer, cur_best_result=None, is_test=True,\
             file_name="result_retest.json")
        # evaluate(model, eval_dataloader, tokenizer)
