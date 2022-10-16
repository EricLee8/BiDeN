from math import sin
import os
import re
import torch
import json
import string
import collections
import numpy as np
from torch._C import dtype
import torch.utils.data as data
from tqdm import tqdm
from collections import deque

from transformers.pipelines import token_classification
from .config import *


class Example(object):
    def __init__(self, utterances, options, qid, answer=None):
        self.utterances = utterances
        self.options = options
        self.qid = qid
        self.answer = answer

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "utterances: " + self.utterances + '\n'
        s += "options: " + self.options
        s += "qid: " + self.qid + '\n'
        s += "answer: " + self.answer
        return s


class InputFeature(object):
    def __init__(self, qid, ids_dict, sep_poses, piece_idxes_list, answer_ids=None):
        self.qid = qid
        self.ids_dict = ids_dict
        self.sep_poses = sep_poses
        self.piece_idxes_list = piece_idxes_list
        self.answer_ids = answer_ids


class Dataset(data.Dataset):
    def __init__(self, features):
        self.features = features
    
    def __getitem__(self, index):
        data_info = {}
        data_info['qid'] = self.features[index].qid
        data_info['ids_dict'] = self.features[index].ids_dict
        data_info['sep_poses'] = self.features[index].sep_poses
        piece_indxes_list = self.features[index].piece_idxes_list
        front_masks = []
        tail_masks = []
        self_masks = []
        fusion_masks = []
        for piece_idxes, sep_poses in zip(piece_indxes_list, data_info['sep_poses']):
            if not args.three_channels:
                front_mask, tail_mask = self._get_masks(args.max_length, piece_idxes, sep_poses)
                front_masks.append(front_mask)
                tail_masks.append(tail_mask)
            else:
                front_mask, tail_mask, self_mask, fusion_mask = self._get_masks(args.max_length, piece_idxes, sep_poses)
                front_masks.append(front_mask)
                tail_masks.append(tail_mask)
                self_masks.append(self_mask)
                fusion_masks.append(fusion_mask)
        front_masks = np.array(front_masks)
        tail_masks = np.array(tail_masks)
        masks_dict = {'front_masks': front_masks, 'tail_masks': tail_masks}
        if args.three_channels:
            self_masks = np.array(self_masks)
            fusion_masks = np.array(fusion_masks)
            masks_dict.update({'self_masks': self_masks, 'fusion_masks': fusion_masks})
        data_info['masks_dict'] = masks_dict
        data_info['answer_ids'] = torch.tensor(self.features[index].answer_ids, dtype=torch.long) if\
            self.features[index].answer_ids is not None else None
        return data_info

    def __len__(self):
        return len(self.features)

    def _get_masks(self, slen, piece_idxes, sep_poses):
        self_mask = None
        if not args.three_channels:
            front_mask = np.zeros([slen, slen], dtype=int)
            tail_mask = np.zeros([slen, slen], dtype=int)
            for i in range(slen):
                if piece_idxes[i] == -1:
                    continue
                for j, (s, e) in enumerate(sep_poses):
                    if j >= piece_idxes[i]:
                        front_mask[i][s: e] = 1 # mask the front utterances and make tail ones visible
                    if j <= piece_idxes[i]:
                        tail_mask[i][s: e] = 1
        else:
            front_mask = np.zeros([slen, slen], dtype=int)
            tail_mask = np.zeros([slen, slen], dtype=int)
            self_mask = np.zeros([slen, slen], dtype=int)
            for i in range(slen):
                if piece_idxes[i] == -1:
                    continue
                for j, (s, e) in enumerate(sep_poses):
                    if j > piece_idxes[i]:
                        tail_mask[i][s: e] = 1 # make tail ones visible
                    if j < piece_idxes[i]:
                        front_mask[i][s: e] = 1
                    if j == piece_idxes[i]:
                        self_mask[i][s: e] = 1
            fusion_mask = np.ones([slen, 3], dtype=int)
            for i in range(slen):
                if front_mask[i].sum() == 0:
                    fusion_mask[i][0] = 0
                if tail_mask[i].sum() == 0:
                    fusion_mask[i][1] = 0
                if self_mask[i].sum() == 0:
                    fusion_mask[i][2] = 0

        if self_mask is not None:    
            return front_mask, tail_mask, self_mask, fusion_mask
        else:
            return front_mask, tail_mask


def _cuda(x):
    if USE_CUDA:
        return x.cuda(device="cuda:"+str(args.cuda))
    else:
        return x


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def collate_fn(data):
    data_info = {}
    float_type_keys = []
    for k in data[0].keys():
        data_info[k] = [d[k] for d in data]
    for k in data_info.keys():
        if isinstance(data_info[k][0], torch.Tensor):
            data_info[k] = _cuda(torch.stack(data_info[k]))
        elif isinstance(data_info[k][0], dict):
            new_dict = {}
            for id_key in data_info[k][0].keys():
                if data_info[k][0][id_key] is None:
                    new_dict[id_key] = None
                    continue
                id_key_list = [torch.tensor(sub_dict[id_key], dtype=torch.long if id_key not in float_type_keys else torch.float) for sub_dict in data_info[k]] # (bsz, seqlen)
                id_key_tensor = torch.stack(id_key_list)
                new_dict[id_key] = _cuda(id_key_tensor)
            data_info[k] = new_dict
    return data_info


def read_examples(input_file, training=True):
    examples = []
    max_utter_num = 0
    print("Reading examples from {}...".format(input_file))
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    answer_to_id = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    for dialogue in tqdm(input_data, ncols=100):
        utterances = dialogue['utterances']
        qid = dialogue['id']
        options = dialogue['options']
        max_utter_num = max(max_utter_num, len(utterances))
        if not training: # during inference
            exp = Example(utterances, options, qid, None)
            examples.append(exp)
            continue
        answer = answer_to_id[dialogue['answers']]
        exp = Example(utterances, options, qid, answer)
        examples.append(exp)

    print("Max utterance num: {}".format(max_utter_num))
    if args.small:
        examples = examples[:100] if training else examples[:100]
    return examples


def convert_examples_to_features(examples, tokenizer, max_length, training=True):
    # tokenizer should be a tokenizer that is inherent from PreTrainedTokenizerFast
    def _clean_pad_attention_mask(ids_dict):
        for idx in range(len(ids_dict['attention_mask'])):
            if ids_dict['input_ids'][idx] == tokenizer.pad_token_id:
                ids_dict['attention_mask'][idx] = 0
        return ids_dict

    def _get_sep_poses(input_ids, piece_idxes):
        sep_poses = []
        last_idx = 0 # except the [CLS]
        for idx, inpidx in enumerate(input_ids):
            if inpidx == tokenizer.sep_token_id:
                if piece_idxes[idx] != -1:
                    sep_poses.append((last_idx+1, idx+1))
                last_idx = idx
        return sep_poses

    print("Converting examples to features...")
    max_tokens, max_option_tokens = 0, 0

    total_num, truncation_num = len(examples), 0
    features = []
    for exp in tqdm(examples, ncols=100):
        piece_idxes = []
        start_idx = 0
        context = ''
        context_max_length = args.max_length - args.option_max_length
        for idx in list(range(len(exp.utterances)))[::-1]:
            utterance = exp.utterances[idx]
            tmp_context = tokenizer.sep_token + ' ' + utterance + ' ' + context
            if len(tokenizer.encode(tmp_context)) > context_max_length:
                start_idx = idx + 1
                truncation_num += 1
                break
            context = tmp_context
            cur_utter_len = len(tokenizer.tokenize(tokenizer.sep_token + ' ' + utterance))
            piece_idxes = [idx] * cur_utter_len + piece_idxes

        context_utterances = exp.utterances[start_idx:]
        utter_num = len(context_utterances)

        context = context.strip()[len(tokenizer.sep_token)+1:] # remove the first sep token and ' '
        piece_idxes = [-1] + piece_idxes[:-1] # [CLS] belongs to no utterances, idxes shift right
        context += ' ' + tokenizer.sep_token # add [SEP] before padding
        piece_idxes.append(piece_idxes[-1]) # [SEP] for the last utterance

        context_length = len(tokenizer.encode(context)) - 1 # including [CLS] and the manually added [SEP]
        # print(exp.qid)
        assert len(piece_idxes) == context_length, "{} vs. {}".format(len(piece_idxes), context_length) 
        remain_length = context_max_length - context_length - 1 # leave a position for the [SEP] added by encode_plus()
        context += ' ' + ' '.join([tokenizer.pad_token] * remain_length)
        piece_idxes += [-1] * (remain_length + 1) # [PAD] and the [SEP] after [PAD] belongs to no utterances
        max_tokens = max(max_tokens, len(tokenizer.encode(context)))
        assert len(tokenizer.encode(context)) == context_max_length == len(piece_idxes)
        while piece_idxes[1] != 0: # the first utterance index should be 0, dealing with truncation cases
            for i in range(1, len(piece_idxes)):
                if piece_idxes[i] != -1: piece_idxes[i] -= 1

        ids_dict = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
        sep_poses, piece_idxes_list = [], []
        for option in exp.options:
            single_piece_idxes = piece_idxes.copy()
            max_option_tokens = max(max_option_tokens, len(tokenizer.encode(option)))
            option_length = len(tokenizer.encode(option)) - 1 # except the [CLS] and including the [SEP]
            if option_length > args.option_max_length:
                while len(tokenizer.encode(option)) - 1 > args.option_max_length:
                    option = option[:-1]
            option_length = len(tokenizer.encode(option)) - 1
            single_piece_idxes += [utter_num] * option_length # including [SEP]

            remain_length = args.option_max_length - option_length # here we DO NOT add [PAD] manually, only deal with piece_idxes
            single_piece_idxes += [-1] * remain_length # [PAD] belongs to no utterances

            single_ids_dict = tokenizer.encode_plus(context, option, truncation=True, max_length=max_length, padding='max_length')
            single_ids_dict = _clean_pad_attention_mask(single_ids_dict)
            assert len(single_ids_dict['input_ids']) == len(single_piece_idxes) == max_length

            single_sep_poses = _get_sep_poses(single_ids_dict['input_ids'], single_piece_idxes)

            ids_dict['input_ids'].append(single_ids_dict['input_ids'])
            ids_dict['token_type_ids'].append(single_ids_dict['token_type_ids'])
            ids_dict['attention_mask'].append(single_ids_dict['attention_mask'])
            sep_poses.append(single_sep_poses)
            piece_idxes_list.append(single_piece_idxes)

        # inference
        if not training:
            f_tmp = InputFeature(exp.qid, ids_dict, sep_poses, piece_idxes_list)
            features.append(f_tmp)
            continue
        # training
        f_tmp = InputFeature(exp.qid, ids_dict, sep_poses, piece_idxes_list, exp.answer)
        features.append(f_tmp)

    print("max token length, max_option_length: ", max_tokens, max_option_tokens)
    print("truncation num: %d, truncation rate: %.3f%%" %(truncation_num, truncation_num/total_num*100))
    return features


def get_dataset(input_file, save_path, tokenizer, max_length, training=True):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    postfix = ""
    for type_ in ["train", "dev", "test", "diff"]:
        if type_ in input_file:
            postfix = type_
            break
    example_path = os.path.join(save_path, "example_{}.cache".format(postfix))
    if not os.path.exists(example_path):
        examples = read_examples(input_file, training=training)
        if not args.colab:
            print("Examples saved to " + example_path)
            torch.save(examples, example_path)
    else:
        print("Read {}_examples from cache...".format(postfix))
        examples = torch.load(example_path)
    feature_path = os.path.join(save_path, "feature_{}.cache".format(postfix))
    if not os.path.exists(feature_path):
        features = convert_examples_to_features(examples, tokenizer, max_length, training=training)
        if not args.colab:
            print("Features saved to " + feature_path)
            torch.save(features, feature_path)
    else:
        print("Read {}_features from cache...".format(postfix))
        features = torch.load(feature_path)
    dataset = Dataset(features)
    return dataset

    
if __name__ == "__main__":
    input_file = "data/dream/train.json"

    from transformers import ElectraTokenizerFast
    tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-large-discriminator')
    all_examples = read_examples(input_file, training=True if 'test' not in input_file else False)
    all_features = convert_examples_to_features(all_examples, tokenizer, max_length=args.max_length,\
     training=True if 'test' not in input_file else False)

    # from transformers import ElectraTokenizerFast
    # from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    # tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-large-discriminator')
    # dataset = get_dataset(input_file, "tmp", tokenizer, args.max_length,\
    #      training=True if 'test' not in input_file else False)
    # sampler = RandomSampler(dataset)
    # dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    # for batch in tqdm(dataloader, ncols=100):
    #     pass
        # print(batch['front_masks'].shape)
        # print(batch['tail_masks'].shape)
        # print(batch['ids_dict']['input_ids'].shape)
        # print(batch['answer_ids'])
