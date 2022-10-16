import os
import torch
import json
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
from .config import *


class Example(object):
    def __init__(self, utterances, qid, summary=None):
        self.utterances = utterances
        self.qid = qid
        self.summary = summary

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "utterances: " + self.utterances + '\n'
        s += "qid: " + self.qid + '\n'
        s += "summary: "  + self.summary
        return s


class InputFeature(object):
    def __init__(self, qid, input_ids, attention_mask, piece_idxes, sep_poses,\
         decoder_attention_mask=None, labels=None):
        self.qid = qid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.piece_idxes = piece_idxes
        self.sep_poses = sep_poses
        self.decoder_attention_mask = decoder_attention_mask
        self.labels = labels


class Dataset(data.Dataset):
    def __init__(self, features):
        self.features = features
    
    def __getitem__(self, index):
        data_info = {}
        data_info['qid'] = self.features[index].qid
        data_info['input_ids'] = torch.tensor(self.features[index].input_ids, dtype=torch.long)
        data_info['attention_mask'] = torch.tensor(self.features[index].attention_mask, dtype=torch.long)
        data_info['sep_poses'] = self.features[index].sep_poses
        data_info['decoder_attention_mask'] = torch.tensor(self.features[index].decoder_attention_mask, dtype=torch.long) if\
            self.features[index].decoder_attention_mask is not None else None
        data_info['labels'] = torch.tensor(self.features[index].labels, dtype=torch.long) if\
            self.features[index].labels is not None else None
        slen = len(data_info['input_ids'])
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
    max_utter_num = 0
    print("Reading examples from {}...".format(input_file))
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    for dialogue in tqdm(input_data, ncols=100):
        utterances = dialogue['utterances']
        qid = dialogue['qid']
        summary = dialogue['summary']
        max_utter_num = max(max_utter_num, len(utterances))
        if not training: # during inference
            exp = Example(utterances, qid, None)
            examples.append(exp)
            continue
        exp = Example(utterances, qid, summary)
        examples.append(exp)

    print("Max utterance num: {}".format(max_utter_num))
    if args.small:
        examples = examples[:100] if training else examples[:100]
    return examples


def convert_examples_to_features(examples, tokenizer, max_length, training=True):
    # tokenizer should be a tokenizer that is inherent from PreTrainedTokenizerFast
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
    max_tokens, summary_max_tokens = 0, 0

    total_num, too_long_num, summary_too_long_num = len(examples), 0, 0
    features = []
    for exp in tqdm(examples, ncols=100):
        context = ''
        for utterance in exp.utterances:
            context += tokenizer.sep_token + ' ' + utterance + ' '
        context = context.strip()[len(tokenizer.sep_token)+1:] # remove the first sep token and ' '

        ids_dict = tokenizer.encode_plus(context, padding='max_length',\
             truncation=True, max_length=max_length)
        input_ids = ids_dict['input_ids']
        attention_mask = ids_dict['attention_mask']

        text_len = len(tokenizer.encode(context))
        max_tokens = max(max_tokens, text_len)
        if text_len > max_length: too_long_num += 1

        piece_idxes = _get_piece_idxes(input_ids)
        sep_poses = _get_sep_poses(input_ids)

        # inference
        if not training:
            f_tmp = InputFeature(exp.qid, input_ids, attention_mask, piece_idxes, sep_poses)
            features.append(f_tmp)
            continue
        # training
        summary = exp.summary
        summary_len = len(tokenizer.encode(summary))
        summary_max_tokens = max(summary_max_tokens, summary_len)
        if summary_len > args.summary_max_length: summary_too_long_num += 1

        decoder_ids_dict = tokenizer.encode_plus(summary, padding='max_length', truncation=True, max_length=args.summary_max_length)
        labels = decoder_ids_dict['input_ids']
        decoder_attention_mask = decoder_ids_dict['attention_mask']
        f_tmp = InputFeature(exp.qid, input_ids, attention_mask, piece_idxes, sep_poses, decoder_attention_mask, labels)
        features.append(f_tmp)

    print("Max token length:", max_tokens)
    print("Truncation num: %d, truncation rate %.2f%%" %(too_long_num, too_long_num/total_num * 100))
    if training:
        print("Max summary token length:", summary_max_tokens)
        print("Summary truncation num: %d, truncation rate %.2f%%" %(summary_too_long_num, summary_too_long_num/total_num * 100))

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
    input_file = "data/test.json"

    # from transformers import BartTokenizerFast
    # all_examples = read_examples(input_file, training=True)
    # tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')
    # all_features = convert_examples_to_features(all_examples,\
    #      tokenizer, max_length=args.max_length, training=True if 'test' not in input_file else False)

    from transformers import BartTokenizerFast
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')
    dataset = get_dataset(input_file, "tmp", tokenizer, args.max_length, training=True if 'test' not in input_file else False)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    for batch in tqdm(dataloader, ncols=100):
        pass