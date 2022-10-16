import os
import json
import torch
import numpy as np
import random
import warnings
import importlib
from math import sqrt
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizerFast, ElectraTokenizerFast, AlbertTokenizerFast
from transformers import BertConfig, ElectraConfig, AlbertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.config import *
from utils.utils import get_dataset, collate_fn


MultipleChoiceModel = importlib.import_module('models.' + args.model_file).MultipleChoiceModel


MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizerFast),
    'electra': (ElectraConfig, ElectraTokenizerFast),
    'albert': (AlbertConfig, AlbertTokenizerFast)
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
    return cur_result['mrr'] > cur_best_result['mrr']


def train(model, train_loader, eval_dataloader, test_dataloader, tokenizer):
    print("Traning arguments:")
    print(args)
    
    best_result = {'r1': 0.0, 'r2': 0.0, 'mrr': 0.0}
    model.train()
    model.zero_grad()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    ]
    all_optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_loader) * args.epochs
    num_warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(all_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
    logging_step = len(train_loader) // 4
    steps = 0

    # eval_result = evaluate(model, eval_dataloader, tokenizer, best_result, is_test=False)
    # evaluate(model, test_dataloader, tokenizer, best_result, is_test=True)

    for epoch in range(args.epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
        for _, batch in pbar:
            inputs = {'ids_dict': batch['ids_dict'],
                      'qid': batch['qid'],
                      'sep_poses': batch['sep_poses'],
                      'answer_ids': batch['answer_ids']
                     }
            if args.num_decouple_layers > 0:
                inputs.update({'masks_dict': batch['masks_dict']})

            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            all_optimizer.step()
            scheduler.step()

            if len(outputs) == 1:
                pbar.set_description("Loss:%.3f,CL:%.3f" %(loss.item(), loss.item()))
            elif len(outputs) == 4:
                token_level_loss, utter_level_loss, ori_out_loss = outputs[1], outputs[2], outputs[3]
                pbar.set_description("Loss:%.3f,TCL:%.3f,UCL:%.3f,OCL:%.3f" %(\
                    loss.item(), token_level_loss, utter_level_loss, ori_out_loss))
            else: # TODO
                pass
            model.zero_grad()
            if steps != 0 and steps % logging_step == 0:
                print("\nEpoch {}, Step {}".format(epoch, steps))
                eval_result = evaluate(model, eval_dataloader, tokenizer, best_result, is_test=False)
                if eval_result['r1'] < 0.3:
                    print("Become untrainable! Training canceled!!!")
                    return
                if cur_larger(eval_result, best_result):
                    best_result = eval_result
                evaluate(model, test_dataloader, tokenizer, best_result, is_test=True)
            steps += 1

    eval_result = evaluate(model, eval_dataloader, tokenizer, best_result, is_test=False)


def evaluate(model, eval_loader, tokenizer, cur_best_result=None, is_test=False, file_name=None):
    def _cal_metrics(golden_dict, pred_dict):
        r1_num, r2_num, mrr = 0, 0, 0
        for qid in golden_dict.keys():
            golden = golden_dict[qid]
            prob_dict = pred_dict[qid]
            pred_choices = list(dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)).keys())
            index = pred_choices.index(golden)
            if index == 0:
                r1_num += 1
            elif index == 1:
                r2_num += 1
            mrr += 1 / (index + 1)
        total_num = len(golden_dict)
        r1 = r1_num / total_num
        r2 = (r1_num + r2_num) / total_num
        mrr = mrr / total_num
        return r1, r2, mrr

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), ncols=100)
        pred_dict = {}

        for _, batch in pbar:
            inputs = {'ids_dict': batch['ids_dict'],
                      'sep_poses': batch['sep_poses'],
                      'qid': batch['qid']
                    }
            if args.num_decouple_layers > 0:
                inputs.update({'masks_dict': batch['masks_dict']})

            outputs = model(**inputs)
            pred_dict.update(outputs[0])
    
    if file_name is None and not is_test:
        with open(eval_path, "r", encoding='utf-8') as reader:
            eval_data = json.load(reader)

        golden_dict = {d['id']: d['answers'] for d in eval_data}
        r1, r2, mrr = _cal_metrics(golden_dict, pred_dict)
        result_dict = {'r1': r1, 'r2': r2, 'mrr': mrr}
        print("Eval result:", result_dict)

        if cur_best_result is not None:
            if cur_larger(result_dict, cur_best_result):
                print("model and arguments saved to {}...".format(args.save_path))
                save_path = os.path.join(args.save_path, "best_model.pth")
                args_save_path = os.path.join(args.save_path, "args.pth")
                torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
                torch.save(args, args_save_path, _use_new_zipfile_serialization=False)
        model.train()
        return result_dict
    else:
        lines = []
        for qid in pred_dict.keys():
            prob_dict = pred_dict[qid]
            pred_choices = list(dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)).keys())
            line = qid + '\t' + pred_choices[0] + '\t' + pred_choices[1] + '\t' + pred_choices[2] + '\t' + pred_choices[3] + '\n'
            lines.append(line)
        if file_name is not None:
            outf = open(file_name, "w")
            outf.writelines(lines)
        model.train()


if __name__ == "__main__":
    is_eval = False
    # is_eval = True
    set_seed()

    if is_eval:
        model_path = "mutual_saves/electra_best_model/"
        # args = torch.load(model_path + 'args.pth')
        args.batch_size = 8
        args.cuda = 1

    tokenizer = tokenizer_class.from_pretrained(args.model_name, cache_dir=args.cache_path)
    config = config_class.from_pretrained(args.model_name, cache_dir=args.cache_path)

    if not is_eval:
        train_dataset = get_dataset(train_path, args.cache_path,\
                tokenizer, args.max_length, training=True)
        eval_dataset = get_dataset(eval_path, args.cache_path,\
                tokenizer, args.max_length, training=True)
        test_dataset = get_dataset(test_path, args.cache_path,\
                tokenizer, args.max_length, training=False)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=collate_fn)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=collate_fn)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    else:
        eval_dataset = get_dataset(eval_path, args.cache_path, tokenizer, args.max_length, training=True)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=collate_fn)
        test_dataset = get_dataset(test_path, args.cache_path, tokenizer, args.max_length, training=False)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, collate_fn=collate_fn)

    model = MultipleChoiceModel.from_pretrained(args.model_name, config=config, cache_dir=args.cache_path)
    if hasattr(model, 'load_mha_params'):
        print("Loading multi-head attention parameters from pretrained model...")
        model.load_mha_params()
    model = model.to(device)

    if not is_eval:
        train(model, train_dataloader, eval_dataloader, test_dataloader, tokenizer)
    else:
        model_saved = torch.load(model_path + "best_model.pth")
        model.load_state_dict(model_saved)
        # evaluate(model, test_dataloader, tokenizer, cur_best_result=None, is_test=True,\
        #      file_name="result_{}.txt".format(model_path.split('_')[2 if args.dataset=='mutual' else 3]))
        # evaluate(model, eval_dataloader, tokenizer, cur_best_result=None, is_test=False,\
        #      file_name="result_best.txt")
        evaluate(model, eval_dataloader, tokenizer)