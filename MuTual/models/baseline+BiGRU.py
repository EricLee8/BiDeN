import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizerFast
from transformers import ElectraModel, ElectraConfig, ElectraPreTrainedModel, ElectraTokenizerFast
from transformers import AlbertModel, AlbertConfig, AlbertPreTrainedModel, AlbertTokenizerFast
from transformers import BertLayer
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
        self.config = config
        self.cls_index = CLS_INDEXES[args.model_type]
        self.hidden_size = config.hidden_size
        self.n_decouple_layers = args.num_decouple_layers
        self.n_choices = 4

        if args.model_type == 'bert':
            self.bert = BertModel(config)
        elif args.model_type == 'electra':
            self.electra = ElectraModel(config)
        elif args.model_type == 'albert':
            self.albert = AlbertModel(config)
        
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.gru = GRUWithPadding(config)
        self.context_gru = nn.GRU(config.hidden_size, config.hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Tanh()
        )
        self.context_pooler = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Tanh()
        )

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
        bsz = ids_dict['input_ids'].shape[0]
        input_ids = ids_dict['input_ids'].view(-1, args.max_length) # (bsz*n_choices, slen)
        token_type_ids = ids_dict['token_type_ids'].view(-1, args.max_length)
        attention_mask = ids_dict['attention_mask'].view(-1, args.max_length)

        single_output = transformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        ori_hidden_states = single_output[0] # (bsz*n_choices, seqlen, hsz)
        out, _ = self.context_gru(ori_hidden_states) # (bsz*n_choices, slen, hsz*2)
        hidden_states = self.context_pooler(out).view(-1, args.max_length, self.hidden_size) # (bsz*n_choices, slen, hsz)

        # utterance-level summarization
        batch_utter_emb_list = []
        for i in range(bsz):
            for j in range(self.n_choices):
                utter_emb_list = []
                offset = i * self.n_choices + j
                for s, e in sep_poses[i][j]:
                    cur_utter_emb = torch.max(hidden_states[offset][s: e], dim=0).values # (hsz)
                    utter_emb_list.append(cur_utter_emb)
                utters_tensor = torch.stack(utter_emb_list, dim=0) # (utter_num, hsz)
                batch_utter_emb_list.append(utters_tensor)
        gru_summary = self.gru(batch_utter_emb_list).view(-1, self.n_choices, self.hidden_size * 2) # (bsz, n_choices, 2*hsz)

        # fuse the representation of the two level summarizations and compute logits
        summary_vec = self.pooler(gru_summary)
        logits = self.classifier(summary_vec).squeeze(-1) # (bsz, n_choices)

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


class GRUWithPadding(nn.Module):
    def __init__(self, config, num_rnn = 1):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = num_rnn
        self.biGRU = nn.GRU(config.hidden_size, config.hidden_size, self.num_layers, batch_first = True, bidirectional = True)

    def forward(self, inputs):
        batch_size = len(inputs)
        sorted_inputs = sorted(enumerate(inputs), key=lambda x: x[1].size(0), reverse = True)
        idx_inputs = [i[0] for i in sorted_inputs]
        inputs = [i[1] for i in sorted_inputs]
        inputs_lengths = [len(i[1]) for i in sorted_inputs]

        inputs = rnn_utils.pad_sequence(inputs, batch_first = True)
        inputs = rnn_utils.pack_padded_sequence(inputs, inputs_lengths, batch_first = True) #(batch_size, seq_len, hidden_size)

        # h0 = torch.rand(2 * self.num_layers, batch_size, self.hidden_size).to(inputs.data.device) # (2, batch_size, hidden_size)
        self.biGRU.flatten_parameters()
        out, _ = self.biGRU(inputs) # (batch_size, 2, hidden_size )
        out_pad, out_len = rnn_utils.pad_packed_sequence(out, batch_first = True) # (batch_size, seq_len, 2 * hidden_size)

        _, idx2 = torch.sort(torch.tensor(idx_inputs))
        idx2 = idx2.to(out_pad.device)
        output = torch.index_select(out_pad, 0, idx2)
        out_len = out_len.to(out_pad.device)
        out_len = torch.index_select(out_len, 0, idx2)

        out_idx = (out_len - 1).unsqueeze(1).unsqueeze(2).repeat([1,1,self.hidden_size * 2])
        output = torch.gather(output, 1, out_idx).squeeze(1)

        return output

