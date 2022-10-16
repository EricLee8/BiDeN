import os
import json
import torch
import numpy as np
import random
import warnings
import importlib
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizerFast, XLNetTokenizerFast, ElectraTokenizerFast
from transformers import BertConfig, XLNetConfig, ElectraConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.evaluate_v2 import main as evaluate_on_squad, EVAL_OPTS
from utils.config import *


MRCModel = importlib.import_module('models.' + args.model_file).MRCModel

if "SUP" in args.model_file:
    from utils.utils_sup import get_dataset, collate_fn
else:
    from utils.utils import get_dataset, collate_fn

MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizerFast),
    'xlnet': (XLNetConfig, XLNetTokenizerFast),
    'electra': (ElectraConfig, ElectraTokenizerFast)
}


warnings.filterwarnings("ignore")
device = torch.device("cuda:"+str(args.cuda)) if USE_CUDA else torch.device("cpu")
train_path = os.path.join(args.data_path, "train.json")
eval_path = os.path.join(args.data_path, "dev.json")
test_path = os.path.join(args.data_path, "test.json")
config_class, tokenizer_class = MODEL_CLASSES[args.model_type]


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


def train(model, train_loader, eval_dataloader, test_dataloader, tokenizer):
    print("Traning arguments:")
    print(args)
    
    best_em, best_f1 = 0.0, 0.0
    model.train()
    model.zero_grad()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    all_optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_loader) * args.epochs
    num_warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(all_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
    logging_step = t_total // (args.epochs*2)
    steps = 0

    # eval_result = evaluate(model, eval_dataloader, tokenizer, ('f1', best_f1), is_test=False)
    # print("Eval Result:", eval_result)

    for epidx, epoch in enumerate(range(args.epochs)):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
        for _, batch in pbar:
            inputs = {'ids_dict': batch['ids_dict'],
                      'start_pos': batch['start_pos'],
                      'end_pos': batch['end_pos'],
                      'is_impossible': batch['is_impossible']
                     }
            if args.num_decouple_layers > 0:
                inputs.update({'masks_dict': batch['masks_dict']})
            if 'SUP' in args.model_file:
                inputs.update({'speaker_ids_dict': batch['speaker_ids_dict']})
                inputs.update({'utterance_ids_dict': batch['utterance_ids_dict']})
            else:
                inputs.update({'sep_poses': batch['sep_poses']})
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            all_optimizer.step()
            if t_total is not None:
                scheduler.step()
            if len(outputs)==3:
                gate_loss, span_loss = outputs[1].item(), outputs[2].item()
                pbar.set_description("Loss:%.3f,GL:%.3f,SL:%.3f" \
                    %(loss.item(), gate_loss, span_loss))
            elif len(outputs)==4:
                gate_loss, span_loss, utter_loss = outputs[1].item(),\
                     outputs[2].item(), outputs[3].item()
                pbar.set_description("Loss:%.3f,GL:%.3f,SL:%.3f,UL:%.3f" \
                    %(loss.item(), gate_loss, span_loss, utter_loss))
            elif len(outputs)==5:
                gate_loss, span_loss, utter_loss, speaker_loss = outputs[1].item(),\
                     outputs[2].item(), outputs[3].item(), outputs[4].item()
                pbar.set_description("Loss:%.3f,GL:%.3f,SL:%.3f,UL:%.3f,SpkL:%.3f" \
                    %(loss.item(), gate_loss, span_loss, utter_loss, speaker_loss))
            else: # TODO
                pass
            model.zero_grad()
            if steps != 0 and steps % logging_step == 0 or steps == t_total-1:
                eval_result = evaluate(model, eval_dataloader, tokenizer, ('f1', best_f1), is_test=False)
                if eval_result['F1'] < 30:
                    print("Become untrainable, stoped!")
                    return
                test_result = evaluate(model, test_dataloader, tokenizer, ('f1', best_f1), is_test=True)
                print("Epoch {}, Step {}".format(epidx, steps))
                if best_f1 < eval_result["F1"]:
                    print("F1: Test True Result:", test_result)
                if best_em < eval_result['EM']:
                    print("EM: Test True Result", test_result)
                if steps > logging_step * 2: # in case some early extreme high case
                    best_em = max(best_em, eval_result["EM"])
                    best_f1 = max(best_f1, eval_result['F1'])
            steps += 1


def evaluate(model, eval_loader, tokenizer, cur_best_metric=None, is_test=False, with_na=True):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), ncols=100)
    answer_dict, na_dict = {}, {}

    with torch.no_grad():
        model.eval()
        for _, batch in pbar:
            inputs = {'ids_dict': batch['ids_dict'],
                        'context': batch['context'],
                        'offset_mapping': batch['offset_mapping'],
                        'qid': batch['qid']
                    }
            if args.num_decouple_layers > 0:
                inputs.update({'masks_dict': batch['masks_dict']})
            if 'SUP' in args.model_file:
                inputs.update({'speaker_ids_dict': batch['speaker_ids_dict']})
                inputs.update({'utterance_ids_dict': batch['utterance_ids_dict']})
            else:
                inputs.update({'sep_poses': batch['sep_poses']})
            outputs = model(**inputs)
            answer_list, na_list = outputs[0], outputs[1]
            for qid, ans_text in answer_list:
                answer_dict[qid] = ans_text
            for qid, na_prob in na_list:
                na_dict[qid] = na_prob

    with open(args.pred_file, "w") as f:
        json.dump(answer_dict, f, indent=2)
    with open(args.na_file, "w") as f:
        json.dump(na_dict, f, indent=2)
    evaluate_options = EVAL_OPTS(data_file=test_path if is_test else eval_path,
                                 pred_file=args.pred_file,
                                 na_prob_file=args.na_file if with_na else None)
    res = evaluate_on_squad(evaluate_options)
    em = res['best_exact']
    f1 = res['best_f1']
    rtv_dict = {'EM': em, 'F1': f1}
    print("Eval result:" if not is_test else "Test result:", rtv_dict)

    if cur_best_metric is not None:
        now_metric = em if cur_best_metric[0]=='em' else f1
        if not is_test and now_metric > cur_best_metric[1]:
            print("model and arguments saved to {}...".format(args.save_path))
            save_path = os.path.join(args.save_path, "best_model.pth")
            args_save_path = os.path.join(args.save_path, "args.pth")
            torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
            torch.save(args, args_save_path, _use_new_zipfile_serialization=False)
    
    model.train()
    return rtv_dict


if __name__ == "__main__":
    is_eval = False
    # is_eval = True

    if is_eval:
        model_path = "saves/electra_baseline_save_384_1919810_large/"
        args = torch.load(model_path + 'args.pth')
        args.batch_size = 8
        args.cuda = 0

    set_seed()
    tokenizer = tokenizer_class.from_pretrained(args.model_name, cache_dir=args.cache_path)
    config = config_class.from_pretrained(args.model_name, cache_dir=args.cache_path)
    if args.model_type != 'xlnet':
        config.start_n_top = 5
        config.end_n_top = 5

    # for training or evaluation
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
        true_test_path = test_path
        test_dataset = get_dataset(true_test_path, args.cache_path, tokenizer, args.max_length, training=False)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, collate_fn=collate_fn)

    model = MRCModel.from_pretrained(args.model_name, config=config, cache_dir=args.cache_path)
    if hasattr(model, 'load_mha_params'):
        print("Loading multi-head attention parameters from pretrained model...")
        model.load_mha_params()
    model = model.to(device)

    if not is_eval:
        train(model, train_dataloader, eval_dataloader, test_dataloader, tokenizer)
    else:
        model_saved = torch.load(model_path + "best_model.pth")
        model.load_state_dict(model_saved)
        result = evaluate(model, test_dataloader, tokenizer, is_test=True if 'test' in true_test_path else False, with_na=True)
        print(result)
