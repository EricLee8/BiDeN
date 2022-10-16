import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder,\
     BartPretrainedModel, BartEncoderLayer, BartDecoderLayer, shift_tokens_right
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput
from transformers import BartModel, BartConfig, BartTokenizerFast
from utils.config import *

MODEL_CLASSES = {
    'bart': (BartConfig, BartModel, BartPretrainedModel, BartTokenizerFast)
}

model_class, config_class, pretrained_model_class, tokenizer_class = MODEL_CLASSES[args.model_type]


class BiDeNBartModel(pretrained_model_class):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.n_decouple_layers = args.num_decouple_layers

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        for i in range(self.n_decouple_layers):
            mha = BartEncoderLayer(config)
            self.add_module("front_mha_{}".format(str(i)), mha)
        for i in range(self.n_decouple_layers):
            mha = BartEncoderLayer(config)
            self.add_module("tail_mha_{}".format(str(i)), mha)
        for i in range(self.n_decouple_layers):
            mha = BartEncoderLayer(config)
            self.add_module("self_mha_{}".format(str(i)), mha)
        self.fuse_layer = FuseLayer(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_masks_dict=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        front_masks = self._transform_masks(decoder_masks_dict['front_masks'])
        tail_masks = self._transform_masks(decoder_masks_dict['tail_masks'])
        self_masks = self._transform_masks(decoder_masks_dict['self_masks'])
        fusion_masks = decoder_masks_dict['fusion_masks']

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = True
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # dealing with inforamtion decoupling
        ori_hidden_states = encoder_outputs['hidden_states'][-(self.n_decouple_layers+1)]
        front_mha_states = self._get_mha_states('front', ori_hidden_states, front_masks, output_attentions)
        tail_mha_states = self._get_mha_states('tail', ori_hidden_states, tail_masks, output_attentions)
        self_mha_states = self._get_mha_states('self', ori_hidden_states, self_masks, output_attentions)

        # fuse information of the three channels
        hidden_states = self.fuse_layer(ori_hidden_states, front_mha_states, tail_mha_states, self_mha_states, fusion_masks=fusion_masks) # (bsz, seqlen, hsz)
        if not self.training:
            hidden_states = hidden_states.repeat_interleave(args.num_beams, dim=0)

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states, # use BiDeN hiden states
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    def _transform_masks(self, masks): # to make sure that the attention weight on masked positions be 0
        masks = (1 - masks) * -1e30
        masks = masks.view(-1, args.max_length, args.max_length).unsqueeze(1) # (bsz, 1, slen, slen)
        return masks

    def _get_mha_states(self, mha_name, ori_hidden_states, attention_masks, output_attentions=False):
        mha_outs = getattr(self, "{}_mha_0".format(mha_name))(ori_hidden_states, attention_mask=attention_masks, layer_head_mask=None, output_attentions=output_attentions)
        mha_states = mha_outs[0] # (bsz*n_choices, slen, hsz)
        for i in range(1, self.n_decouple_layers):
            mha_outs = getattr(self, "{}_mha_{}".format(mha_name, str(i)))(mha_states, attention_masks, layer_head_mask=None, output_attentions=output_attentions)
            mha_states = mha_outs[0]
        return mha_states

    def load_mha_params(self):
        for prefix in ['front', 'tail', 'self']:
            print("Loading weights for {} information decoupling layers from pretrained model...".format(prefix))
            for i in range(self.n_decouple_layers):
                mha = getattr(self, "{}_mha_{}".format(prefix, str(i)))
                rtv = mha.load_state_dict(self.encoder.layers[i-self.n_decouple_layers].state_dict().copy())
                print(rtv)
            print("Successfully loaded!")


class FuseLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear1 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.linear3 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.expert = nn.Linear(3 * config.hidden_size, 3)
        self.activation = nn.ReLU()

    def forward(self, orig, input1, input2, input3, fusion_masks=None):
        out1 = self.activation(self.linear1(torch.cat([orig, input1, orig - input1, orig * input1], dim=-1)))
        out2 = self.activation(self.linear2(torch.cat([orig, input2, orig - input2, orig * input2], dim=-1)))
        out3 = self.activation(self.linear3(torch.cat([orig, input3, orig - input3, orig * input3], dim=-1)))
        expert_score = self.expert(torch.cat([out1, out2, out3], dim=-1)) # (bsz, slen, 3)
        if fusion_masks is not None:
            fusion_masks = (1 - fusion_masks) * -1e30
            fusion_masks = fusion_masks.view(-1, args.max_length, 3) # (bsz, slen, 3)
            expert_score = expert_score + fusion_masks
        input_stack = torch.stack([input1, input2, input3], dim=-1)
        fuse_prob = F.softmax(expert_score, dim=-1).unsqueeze(2).expand_as(input_stack) # (bsz, slen, hsz, 3)
        out_emb = input_stack.mul(fuse_prob).sum(-1) # (bsz, slen, hsz)
        return out_emb


class BartModel(pretrained_model_class):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
