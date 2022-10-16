import os
import torch
import json
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
from .config import *


class Example(object):
    def __init__(self, context, utterances, relations, question, qid, ori_start_pos=None, ori_end_pos=None, answer=None):
        self.context = context
        self.utterances = utterances
        self.relations = relations
        self.question = question
        self.qid = qid
        self.ori_start_pos = ori_start_pos
        self.ori_end_pos = ori_end_pos
        self.answer = answer

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "context: " + self.context + '\n'
        s += "utterances: " + self.utterances + '\n'
        s += "relations: " + self.relations + '\n'
        s += "question: " + self.question + '\n'
        s += "qid: " + self.qid + '\n'
        s += "answer: " + self.answer
        return s


class InputFeature(object):
    def __init__(self, qid, ids_dict, offset_mapping, context, piece_idxes, sep_poses,\
         relations, start_pos=None, end_pos=None, is_impossible=None):
        self.qid = qid
        self.ids_dict = ids_dict
        self.offset_mapping = offset_mapping
        self.context = context
        self.piece_idxes = piece_idxes
        self.sep_poses = sep_poses
        self.relations = relations
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.is_impossible = is_impossible


class Dataset(data.Dataset):
    def __init__(self, features):
        self.features = features
    
    def __getitem__(self, index):
        data_info = {}
        data_info['qid'] = self.features[index].qid
        data_info['ids_dict'] = self.features[index].ids_dict
        data_info['offset_mapping'] = self.features[index].offset_mapping
        data_info['context'] = self.features[index].context
        data_info['sep_poses'] = self.features[index].sep_poses
        data_info['start_pos'] = torch.tensor(self.features[index].start_pos, dtype=torch.long) if\
            self.features[index].start_pos is not None else None
        data_info['end_pos'] = torch.tensor(self.features[index].end_pos, dtype=torch.long) if\
            self.features[index].end_pos is not None else None
        data_info['is_impossible'] = torch.tensor(self.features[index].is_impossible, dtype=torch.float) if\
            self.features[index].is_impossible is not None else None
        slen = len(data_info['ids_dict']['input_ids'])
        data_info['masks_dict'] = self._get_masks(slen, self.features[index].piece_idxes,\
            self.features[index].sep_poses)
        return data_info
    
    def __len__(self):
        return len(self.features)

    def _get_masks(self, slen, piece_idxes, sep_poses):
        front_masks = np.zeros([slen, slen], dtype=int)
        tail_masks = np.zeros([slen, slen], dtype=int)
        self_masks = np.zeros([slen, slen], dtype=int)
        for i in range(slen):
            if piece_idxes[i] == -1:
                continue
            for j, (s, e) in enumerate(sep_poses):
                if j > piece_idxes[i]:
                    tail_masks[i][s: e] = 1 # make tail ones visible
                if j < piece_idxes[i]:
                    front_masks[i][s: e] = 1
                if j == piece_idxes[i]:
                    self_masks[i][s: e] = 1
        
        fusion_masks = np.ones([slen, 3], dtype=int)
        for i in range(slen):
            if front_masks[i].sum() == 0:
                fusion_masks[i][0] = 0
            if tail_masks[i].sum() == 0:
                fusion_masks[i][1] = 0
            if self_masks[i].sum() == 0:
                fusion_masks[i][2] = 0
        
        masks_dict = {}
        masks_dict['front_masks'] = front_masks
        masks_dict['tail_masks'] = tail_masks
        masks_dict['self_masks'] = self_masks
        masks_dict['fusion_masks'] = fusion_masks
            
        return masks_dict


def _cuda(x):
    if USE_CUDA:
        return x.cuda(device="cuda:"+str(args.cuda))
    else:
        return x


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def convert_index_to_text(offset_mapping, orig_text, start_index, end_index):
    orig_start_idx = offset_mapping[start_index][0]
    orig_end_idx = offset_mapping[end_index][1]
    return orig_text[orig_start_idx : orig_end_idx]


# in some cases the model will extract long sentence whose first tokens equals to the last tokens
def clean_answer(s):
    def _get_max_matched_str(tlist):
        for length in range(1, len(tlist)):
            if s[:length] == s[-length:]:
                return length
        return -1

    token_list = s.split(' ')
    if len(token_list) > 20:
        max_length = _get_max_matched_str(token_list)
        if max_length == -1:
            rtv = s
        else:
            rtv = " ".join(token_list[:max_length])
        return rtv
    return s


def collate_fn(data):
    data_info = {}
    float_type_keys = []
    for k in data[0].keys():
        data_info[k] = [d[k] for d in data]
    for k in data_info.keys():
        if isinstance(data_info[k][0], torch.Tensor):
            data_info[k] = _cuda(torch.stack(data_info[k]))
        if isinstance(data_info[k][0], dict):
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
    print("Reading examples from {}...".format(input_file))
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)['data']['dialogues']

    for dialogue in tqdm(input_data, ncols=100):
        context = dialogue['context']
        utterances = dialogue['edus']
        relations = dialogue['relations']
        for qa in dialogue['qas']:
            question = qa['question']
            qid = qa['id']
            if not training: # during inference
                exp = Example(context, utterances, relations, question, qid)
                examples.append(exp)
                continue
            if qa['is_impossible'] or len(qa['answers']) == 0:
                exp = Example(context, utterances, relations, question, qid, -1, -1, '')
                examples.append(exp)
                continue

            for answer in qa['answers']: # during training
                ans_text = answer['text']
                ori_start_pos = answer['answer_start']
                ori_end_pos = ori_start_pos + len(ans_text)
                exp = Example(context, utterances, relations, question, qid, ori_start_pos, ori_end_pos, ans_text)
                examples.append(exp)
    if args.debug:
        examples = examples[:2000] if training else examples[:400]
    if args.small:
        examples = examples[:100] if training else examples[:100]
    return examples


def convert_examples_to_features(examples, tokenizer, max_length, training=True, max_utterance_num=args.max_utter_num):
    # tokenizer should be a tokenizer that is inherent from PreTrainedTokenizerFast
    def _get_target_span(target_ids:list, input_ids:list, id_list=None, use_rfind=False):
        if id_list is None:
            id_list = [i for i in range(len(input_ids))]
        span_start_index, span_end_index = args.max_length-1, args.max_length-1
        for idx in range(len(id_list)): # sometimes id_list will exceed the length of input_ids
            id_list[idx] = min(len(input_ids)-1, id_list[idx])
            id_list[idx] = max(0, id_list[idx])
        id_list = list(set(id_list)) # get rid of redundent ids
        for idx in id_list if not use_rfind else id_list[::-1]:
            is_found = False
            if input_ids[idx] == target_ids[0]:
                is_found = True
                for offset in range(1, len(target_ids)):
                    if idx+offset > len(input_ids)-1: # out of range
                        is_found = False
                        break
                    if input_ids[idx+offset] != target_ids[offset]:
                        is_found = False
                        break
                if is_found:
                    span_start_index, span_end_index = idx, idx+len(target_ids)-1
                    break
        span = (span_start_index, span_end_index)
        return span

    def _get_key_utterance_target(start_pos, end_pos, input_ids):
        utterance_gather_ids = []
        for idx, token_id in enumerate(input_ids):
            if token_id == tokenizer.sep_token_id:
                utterance_gather_ids.append(idx)
        for idx, cur_utter_id in enumerate(utterance_gather_ids):
            if start_pos < cur_utter_id and end_pos < cur_utter_id:
                return idx
        return -1
    
    def _get_sep_poses(input_ids):
        sep_poses = []
        last_idx = 1 # except the [CLS] and [PAD]
        for idx, inpidx in enumerate(input_ids):
            if inpidx == tokenizer.sep_token_id:
                sep_poses.append((last_idx+1, idx+1))
                last_idx = idx
        return sep_poses
    
    def _get_piece_idxes(input_ids):
        cur_utter_idx = 0
        piece_idxes = []
        for inpid in input_ids:
            if inpid == tokenizer.pad_token_id or inpid == tokenizer.cls_token_id:
                piece_idxes.append(-1) # [PAD] and [CLS] belong to no utterances
            elif inpid == tokenizer.sep_token_id:
                piece_idxes.append(cur_utter_idx)
                cur_utter_idx += 1
            else:
                piece_idxes.append(cur_utter_idx)
        return piece_idxes

    print("Converting examples to features...")
    max_tokens = 0

    p_mask_ids = [tokenizer.sep_token_id, tokenizer.eos_token_id,\
         tokenizer.bos_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id]

    total_num, unptr_num, too_long_num = len(examples), 0, 0
    features = []
    for exp in tqdm(examples, ncols=100):
        question = exp.question
        answer_text = exp.answer
        context = ''

        for utterance_dict in exp.utterances:
            text = utterance_dict['text']
            speaker = utterance_dict['speaker']
            context += tokenizer.sep_token + ' ' + speaker + ': ' + text + ' '

        context = context.strip()[len(tokenizer.sep_token)+1:] # remove the first sep token and ' '
        if args.model_type == 'xlnet': context = context.lower()
        context = tokenizer.pad_token + ' ' + context # add [PAD] for padding utterances

        ids_dict = tokenizer.encode_plus(context, question, padding='max_length',\
             truncation=True, max_length=max_length, return_offsets_mapping=True)
        offset_mapping = ids_dict['offset_mapping']
        input_ids = ids_dict['input_ids']
        token_type_ids = ids_dict['token_type_ids']
        attention_mask = ids_dict['attention_mask']
        for i in range(len(attention_mask)):
            if input_ids[i] == tokenizer.pad_token_id:
                attention_mask[i] = 0
        p_mask = [1] * len(input_ids)
        for i in range(len(input_ids)):
            if input_ids[i] in p_mask_ids or token_type_ids[i] == 1:
                p_mask[i] = 0
        text_len = len(tokenizer.encode(context + ' ' + tokenizer.sep_token + ' ' + question))
        if text_len > max_length: too_long_num += 1

        ids_dict = {}
        ids_dict['input_ids'] = input_ids
        ids_dict['token_type_ids'] = token_type_ids
        ids_dict['attention_mask'] = attention_mask
        ids_dict['p_mask'] = p_mask
        piece_idxes = _get_piece_idxes(input_ids)
        sep_poses = _get_sep_poses(input_ids)

        # inference
        if not training:
            ids_dict.update({'key_utterance_target': None})
            f_tmp = InputFeature(exp.qid, ids_dict, offset_mapping, context, piece_idxes, sep_poses, exp.relations)
            features.append(f_tmp)
            continue
        # training
        is_impossible = 1 if exp.answer == '' else 0
        start_pos, end_pos = _get_target_span(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(answer_text)),\
             input_ids) if not is_impossible else (args.max_length-1, args.max_length-1)
        if not is_impossible and (start_pos==max_length-1 or end_pos==max_length-1):
            unptr_num += 1
            # print(exp.qid)
            continue

        key_utterance_target = _get_key_utterance_target(start_pos, end_pos, input_ids) if not is_impossible else max_utterance_num
        assert key_utterance_target != -1, "qid: {}, start: {}, end: {}, utter_gather_ids: {}".format(exp.qid, start_pos, end_pos)
        ids_dict.update({'key_utterance_target': key_utterance_target})
        
        f_tmp = InputFeature(exp.qid, ids_dict, offset_mapping, context, piece_idxes, sep_poses, exp.relations, start_pos, end_pos, is_impossible)
        features.append(f_tmp)
        max_tokens = max(max_tokens, text_len)

    print("Max token length:", max_tokens)
    print("Unpointable num: %d, unpointable rate %.2f%%" %(unptr_num, unptr_num/total_num * 100))
    print("Truncation num: %d, truncation rate %.2f%%" %(too_long_num, too_long_num/total_num * 100))

    return features


def get_dataset(input_file, save_path, tokenizer, max_length, training=True):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    postfix = ""
    for type_ in ["train", "dev", "test"]:
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
    input_file = "data/train.json"

    # from transformers import ElectraTokenizerFast
    # all_examples = read_examples(input_file, training=True)
    # tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-large-discriminator')
    # all_features = convert_examples_to_features(all_examples,\
    #      tokenizer, max_length=args.max_length, training=True)

    from transformers import ElectraTokenizerFast
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-large-discriminator')
    dataset = get_dataset(input_file, "tmp", tokenizer, args.max_length, training=True)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    for batch in tqdm(dataloader, ncols=100):
        pass