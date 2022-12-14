import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizerFast
from transformers import XLNetModel, XLNetConfig, XLNetPreTrainedModel, XLNetTokenizerFast
from transformers import ElectraModel, ElectraConfig, ElectraPreTrainedModel, ElectraTokenizerFast
from transformers import BertLayer
from typing import Optional

from utils.config import *
from utils.utils import convert_index_to_text, to_list


_PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["start_index", "end_index", "start_log_prob", "end_log_prob"])


MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertPreTrainedModel, BertTokenizerFast),
    'xlnet': (XLNetConfig, XLNetModel, XLNetPreTrainedModel, XLNetTokenizerFast),
    'electra': (ElectraConfig, ElectraModel, ElectraPreTrainedModel, ElectraTokenizerFast)
}
TRANSFORMER_CLASS = {'bert': 'bert', 'xlnet': 'transformer', 'electra': 'electra'}
CLS_INDEXES = {'bert': 0, 'xlnet': -1, 'electra': 0}

model_class, config_class, pretrained_model_class, tokenizer_class = MODEL_CLASSES[args.model_type]


class MRCModel(pretrained_model_class):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top
        self.impossible_threshold = 1.0
        self.transformer_name = TRANSFORMER_CLASS[args.model_type]
        self.cls_index = CLS_INDEXES[args.model_type]
        self.question_start = args.max_length - args.question_max_length
        self.speaker_mha_layers = 3
        self.n_decouple_layers = args.num_decouple_layers
        if args.model_type == 'xlnet': self.question_start -= 1

        if args.model_type == 'bert':
            self.bert = BertModel(config)
        elif args.model_type == 'xlnet':
            self.transformer = XLNetModel(config)
        elif args.model_type == 'electra':
            self.electra = ElectraModel(config)

        self.sigmoid = nn.Sigmoid()
        self.start_predictor = PoolerStartLogits(config)
        self.end_predictor = PoolerEndLogits(config)
        self.verifier = nn.Linear(config.hidden_size, 1)
        self.attn_fct = nn.Linear(config.hidden_size * 3, 1)
        self.utter_filter = nn.Linear(config.hidden_size*4, 1)
        self.speaker_detector = nn.Linear(config.hidden_size*4, 1)
        for i in range(self.speaker_mha_layers):
            mha = BertLayer(config)
            self.add_module("MHA_{}".format(str(i)), mha)
        self.fusion_fct = nn.Sequential(
            nn.Linear(config.hidden_size*4, config.hidden_size),
            nn.Tanh()
        )

        self.fuse_layer = FuseLayer(config)
        for i in range(self.n_decouple_layers):
            mha = BertLayer(config)
            self.add_module("front_mha_{}".format(str(i)), mha)
        for i in range(self.n_decouple_layers):
            mha = BertLayer(config)
            self.add_module("tail_mha_{}".format(str(i)), mha)
        for i in range(self.n_decouple_layers):
            mha = BertLayer(config)
            self.add_module("self_mha_{}".format(str(i)), mha)

        self.init_weights()

    def forward(
        self,
        ids_dict=None,
        context=None,
        utterance_ids_dict=None,
        speaker_ids_dict=None,
        masks_dict=None,
        offset_mapping=None,
        qid=None,
        start_pos=None,
        end_pos=None,
        is_impossible=None,
        output_attentions=False
    ):
        input_ids = ids_dict['input_ids']
        attention_mask = ids_dict['attention_mask']
        token_type_ids = ids_dict['token_type_ids']
        p_mask = ids_dict['p_mask']
        front_masks = self._transform_masks(masks_dict['front_masks'])
        tail_masks = self._transform_masks(masks_dict['tail_masks'])
        self_masks = self._transform_masks(masks_dict['self_masks'])
        fusion_masks = self._transform_masks(masks_dict['fusion_masks'], is_fusion_mask=True)

        utterance_gather_ids = utterance_ids_dict['utterance_gather_ids']
        utterance_p_mask = utterance_ids_dict['utterance_p_mask']
        utterance_repeat_num = utterance_ids_dict['utterance_repeat_num']
        key_utterance_target = utterance_ids_dict['key_utterance_target']
        speaker_attn_mask = speaker_ids_dict['speaker_attn_mask']
        speaker_gather_ids = utterance_ids_dict['utterance_gather_ids']
        target_speaker_gather_id = speaker_ids_dict['target_speaker_gather_id']
        speaker_target = speaker_ids_dict['speaker_target']
        speaker_target_mask =speaker_ids_dict['speaker_target_mask']

        training = start_pos is not None and end_pos is not None and is_impossible is not None
        transformer = getattr(self, self.transformer_name)
        transformer_outputs = transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )

        hidden_states = transformer_outputs[0] # (bsz, seqlen, hsz)
        speaker_hidden_states = transformer_outputs[1 if args.model_type=='electra' else 2][-(self.speaker_mha_layers+1)] # (bsz, seqlen, hsz)
        bsz, slen, hsz = hidden_states.size()

        gate_loss_fct = nn.BCEWithLogitsLoss()
        span_loss_fct = CrossEntropyLoss(ignore_index=hidden_states.shape[1]-1)
        utter_loss_fct = CrossEntropyLoss(ignore_index=14)

        # deal with speaker information
        hidden_states_detached = speaker_hidden_states.detach() # (bsz, slen, hsz)
        speaker_attn_mask[:, self.question_start:] = 0 # (bsz, slen)
        speaker_attn_mask = (1 - speaker_attn_mask) * -1e30
        speaker_attn_mask = speaker_attn_mask.unsqueeze(1).unsqueeze(1).expand(-1, self.config.num_attention_heads, slen, -1) # (bsz, n_heads, slen, slen)

        speaker_mha_out = getattr(self, "MHA_0")(hidden_states_detached, attention_mask=speaker_attn_mask)[0] # (bsz, slen, hsz)
        for i in range(1, self.speaker_mha_layers):
            speaker_mha_out = getattr(self, "MHA_{}".format(str(i)))(speaker_mha_out, speaker_attn_mask)[0]
        speaker_embs = speaker_mha_out.gather(dim=1, index=speaker_gather_ids.unsqueeze(-1).expand(-1, -1, hsz)) # (bsz, max_utterm hsz)
        masked_speaker_embs = speaker_embs.gather(dim=1, index=target_speaker_gather_id.unsqueeze(-1).expand(-1, -1, hsz)) # (bsz, 1, hsz)
        masked_speaker_embs_expand = masked_speaker_embs.expand_as(speaker_embs) # (bsz, max_utter, hsz)
        speaker_logits = self.speaker_detector(
            torch.cat([speaker_embs, masked_speaker_embs_expand,\
                        speaker_embs*masked_speaker_embs_expand, speaker_embs-masked_speaker_embs_expand], dim=-1)
            ).squeeze(-1) # (bsz, max_utter)
        speaker_logits = speaker_logits * speaker_target_mask - 1e30 * (1-speaker_target_mask)

        # dealing with inforamtion decoupling
        front_mha_states = self._get_mha_states('front', hidden_states, front_masks, output_attentions)
        tail_mha_states = self._get_mha_states('tail', hidden_states, tail_masks, output_attentions)
        self_mha_states = self._get_mha_states('self', hidden_states, self_masks, output_attentions)

        # fuse information of the three channels
        fused_hidden_states = self.fuse_layer(hidden_states, front_mha_states, tail_mha_states, self_mha_states, fusion_masks) # (bsz*n_choices, seqlen, hsz)

        # fuse information
        fused_hidden_states = self.fusion_fct(
            torch.cat([hidden_states, speaker_mha_out, hidden_states*speaker_mha_out, hidden_states-speaker_mha_out], dim=-1)
        )
        question_emb = torch.mean(fused_hidden_states[:, self.question_start:, :], dim=1) # (bsz, hsz)

        # deal with gate and start position
        _, context_vec = self._compute_attention(fused_hidden_states, question_emb, p_mask=attention_mask)
        gate_logits = self.verifier(context_vec).squeeze(-1) # (bsz)
        start_logits = self.start_predictor(fused_hidden_states, question_emb, p_mask=p_mask) # (bsz, seqlen)

        # deal with utterance prediction
        utter_embs = fused_hidden_states.gather(dim=1, index=utterance_gather_ids.unsqueeze(-1).expand(-1, -1, hsz)) # (bsz, max_utter, hsz)
        question_emb_expand = question_emb.unsqueeze(1).expand_as(utter_embs)
        utter_logits = self.utter_filter(torch.cat(
            [utter_embs, question_emb_expand, utter_embs*question_emb_expand, utter_embs-question_emb_expand], dim=-1)
        ).squeeze(-1) # (bsz, max_utter)
        utter_logits = utter_logits * utterance_p_mask - 1e30 * (1-utterance_p_mask)
        utter_weights = torch.softmax(utter_logits, dim=-1)
        utter_weights_repeated = utter_weights.view(-1).repeat_interleave(utterance_repeat_num.view(-1)).view(bsz, -1) # (bsz, slen)

        if training:
            end_logits = self.end_predictor(fused_hidden_states, start_positions=start_pos, p_mask=p_mask)
            start_loss = span_loss_fct(start_logits, start_pos)
            end_loss = span_loss_fct(end_logits, end_pos)
            gate_loss = gate_loss_fct(gate_logits, is_impossible)
            span_loss = (start_loss + end_loss) / 2
            utter_loss = utter_loss_fct(utter_logits, key_utterance_target)
            speaker_loss = gate_loss_fct(speaker_logits, speaker_target)
            total_loss = span_loss + gate_loss + utter_loss + speaker_loss

        else:
            # during inference, compute the end logits based on beam search
            assert context is not None and offset_mapping is not None
            gate_log_probs = self.sigmoid(gate_logits) # (bsz)
            gate_index = gate_log_probs > self.impossible_threshold # (bsz)
            gate_log_probs_list = to_list(gate_log_probs)
            gate_index = to_list(gate_index)

            speaker_index = self.sigmoid(speaker_logits) > 0.5 # (bsz)
            correct_num = ((speaker_index==speaker_target.long())==speaker_target_mask).sum().item()
            all_num = speaker_target_mask.sum().item()

            start_log_probs = F.softmax(start_logits, dim=-1) * utter_weights_repeated  # shape (bsz, slen)

            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(fused_hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            fused_hidden_states_expanded = fused_hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_predictor(fused_hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = F.softmax(end_logits, dim=1) * utter_weights_repeated.unsqueeze(-1).expand_as(end_logits) # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)
            
            start_top_index = to_list(start_top_index)
            start_top_log_probs = to_list(start_top_log_probs)
            end_top_index = to_list(end_top_index)
            end_top_log_probs = to_list(end_top_log_probs)
            
            answer_list = []
            na_list = []
            for bidx in range(bsz):
                na_list.append((qid[bidx], gate_log_probs_list[bidx]))
                if self.impossible_threshold != -1 and gate_index[bidx] == 1:
                    answer_list.append((qid[bidx], ''))
                    continue
                b_offset_mapping = offset_mapping[bidx]
                b_orig_text = context[bidx]
                prelim_predictions = []
                for i in range(self.start_n_top):
                    for j in range(self.end_n_top):
                        start_log_prob = start_top_log_probs[bidx][i]
                        start_index = start_top_index[bidx][i]
                        j_index = i * self.end_n_top + j
                        end_log_prob = end_top_log_probs[bidx][j_index]
                        end_index = end_top_index[bidx][j_index]

                        if end_index < start_index:
                            continue

                        prelim_predictions.append(
                            _PrelimPrediction(
                                start_index=start_index,
                                end_index=end_index,
                                start_log_prob=start_log_prob,
                                end_log_prob=end_log_prob))

                prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_log_prob + x.end_log_prob),
                    reverse=True)
                best_text = ''
                if len(prelim_predictions) > 0:
                    best_one = prelim_predictions[0]
                    best_start_index = best_one.start_index
                    best_end_index = best_one.end_index
                    best_text = convert_index_to_text(b_offset_mapping, b_orig_text, best_start_index, best_end_index)
                answer_list.append((qid[bidx], best_text))

        outputs = (total_loss, gate_loss, span_loss, utter_loss, speaker_loss) if training else (answer_list, na_list, (correct_num, all_num))
        return outputs
    
    def _compute_attention(self, sentence, query, p_mask=None): # (bsz, slen, hsz) and (bsz, hsz)
        query = query.unsqueeze(1).expand_as(sentence)
        scores = self.attn_fct(torch.cat([sentence, query, sentence*query], dim=-1)).squeeze(-1) # (bsz, slen)
        if p_mask is not None:
            scores = scores * p_mask - 1e30 * (1-p_mask)
        weights = torch.softmax(scores, dim=-1)
        context_vec = sentence.mul(weights.unsqueeze(-1).expand_as(sentence)).sum(1) # (bsz, hsz)
        return weights, context_vec

    def _transform_masks(self, masks, is_fusion_mask=False): # to make sure that the attention weight on masked positions be 0
        masks = (1 - masks) * -1e30
        if is_fusion_mask:
            masks = masks.view(-1, args.max_length, 3) # (bsz, slen, 3)
        else:
            masks = masks.view(-1, args.max_length, args.max_length).unsqueeze(1).expand(-1, self.config.num_attention_heads, -1, -1) # (bsz, n_heads, slen, slen)
        return masks

    def _get_mha_states(self, mha_name, ori_hidden_states, attention_masks, output_attentions=False):
        mha_outs = getattr(self, "{}_mha_0".format(mha_name))(ori_hidden_states, attention_mask=attention_masks, output_attentions=output_attentions)
        mha_states = mha_outs[0] # (bsz*n_choices, slen, hsz)
        for i in range(1, self.n_decouple_layers):
            mha_outs = getattr(self, "{}_mha_{}".format(mha_name, str(i)))(mha_states, attention_masks, output_attentions=output_attentions)
            mha_states = mha_outs[0]
        return mha_states
    
    def load_mha_params(self):
        for i in range(self.speaker_mha_layers):
            mha = getattr(self, "MHA_{}".format(str(i)))
            rtv = mha.load_state_dict(getattr(self, self.transformer_name).encoder.layer[i-self.speaker_mha_layers].state_dict().copy())
            print(rtv)


class PoolerStartLogits(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.ReLU())
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(
        self, hidden_states: torch.FloatTensor,
        question_emb: torch.FloatTensor, # (bsz, hsz)
        p_mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        question_emb = question_emb.unsqueeze(1).expand_as(hidden_states)
        x = self.fusion(torch.cat([hidden_states, question_emb, hidden_states*question_emb], dim=-1))
        x = self.dense(x).squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * p_mask - 65500 * (1-p_mask)
            else:
                x = x * p_mask - 1e30 * (1-p_mask)
        return x


class PoolerEndLogits(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        p_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * p_mask - 65500 * (1-p_mask)
            else:
                x = x * p_mask - 1e30 * (1-p_mask)

        return x


class FuseLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear1 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.linear3 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.expert = nn.Linear(3 * config.hidden_size, 3)
        self.activation = nn.ReLU()

    def forward(self, orig, input1, input2, input3, fusion_masks):
        out1 = self.activation(self.linear1(torch.cat([orig, input1, orig - input1, orig * input1], dim=-1)))
        out2 = self.activation(self.linear2(torch.cat([orig, input2, orig - input2, orig * input2], dim=-1)))
        out3 = self.activation(self.linear3(torch.cat([orig, input3, orig - input3, orig * input3], dim=-1)))
        expert_score = self.expert(torch.cat([out1, out2, out3], dim=-1)) # (bsz, slen, 3)
        expert_score = expert_score + fusion_masks
        input_stack = torch.stack([input1, input2, input3], dim=-1)
        fuse_prob = F.softmax(expert_score, dim=-1).unsqueeze(2).expand_as(input_stack) # (bsz, slen, hsz, 3)
        out_emb = input_stack.mul(fuse_prob).sum(-1) # (bsz, slen, hsz)
        return out_emb
