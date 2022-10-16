import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizerFast
from transformers import ElectraModel, ElectraConfig, ElectraPreTrainedModel, ElectraTokenizerFast
from transformers import AlbertModel, AlbertConfig, AlbertPreTrainedModel, AlbertTokenizerFast
from utils.config import *
from utils.utils import to_list

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertPreTrainedModel, BertTokenizerFast),
    'electra': (ElectraConfig, ElectraModel, ElectraPreTrainedModel, ElectraTokenizerFast),
    'albert': (AlbertConfig, AlbertModel, AlbertPreTrainedModel, AlbertTokenizerFast)
}
TRANSFORMER_CLASS = {'bert': 'bert', 'electra': 'electra', 'albert': 'albert'}
CLS_INDEXES = {'bert': 0, 'electra': 0, 'albert': 0}

model_class, config_class, pretrained_model_class, tokenizer_class = MODEL_CLASSES[args.model_type]


class MultipleChoiceModel(pretrained_model_class):
    def __init__(self, config):
        super().__init__(config)
        self.transformer_name = TRANSFORMER_CLASS[args.model_type]
        self.tokenizer = tokenizer_class.from_pretrained(args.model_name)
        self.cls_index = CLS_INDEXES[args.model_type]
        self.hidden_size = config.hidden_size
        self.n_choices = 4

        if args.model_type == 'bert':
            self.bert = BertModel(config)
        elif args.model_type == 'electra':
            self.electra = ElectraModel(config)
        elif args.model_type == 'albert':
            self.albert = AlbertModel(config)

        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        ids_dict=None,
        qid=None,
        sep_poses=None,
        answer_ids=None,
        output_attentions=False
    ):
        training = answer_ids is not None
        transformer = getattr(self, self.transformer_name)
        assert ids_dict is not None
        input_ids =ids_dict['input_ids'].view(-1, args.max_length) # (bsz*n_choices, slen)
        token_type_ids =ids_dict['token_type_ids'].view(-1, args.max_length)
        attention_mask =ids_dict['attention_mask'].view(-1, args.max_length)
        
        single_output = transformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        hidden_states = single_output[0].view(-1, self.n_choices, args.max_length, self.hidden_size) # (bsz, n_choices, slen, hsz)
        bsz, _, slen, hsz = hidden_states.shape
        pooler_out = hidden_states[:, :, self.cls_index, :] # (bsz, n_choices, hsz)
        logits = self.classifier(pooler_out).squeeze(-1) # (bsz, n_choices)

        if training:
            choice_loss_fct = CrossEntropyLoss()
            choice_loss = choice_loss_fct(logits, answer_ids)
        else:
            choice_probs = to_list(F.softmax(logits)) # (bsz, n_choices)
            choice_dicts = []
            for bidx in range(bsz):
                prob_list = choice_probs[bidx]
                prob_dict = {'A': prob_list[0], 'B': prob_list[1], 'C': prob_list[2]}
                if self.n_choices == 4: prob_dict.update({'D': prob_list[3]})
                choice_dicts.append(prob_dict)
            pred_dict = {ID: prob_dcit for ID, prob_dcit in zip(qid, choice_dicts)}

        outputs = (choice_loss,) if training else (pred_dict,)
        return outputs

