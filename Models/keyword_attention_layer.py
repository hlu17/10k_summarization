"""
This code implements a custom version of the BART model for conditional generation with an additional keyword self-attention mechanism. The main components of the code are:

1.KeywordSelfAttention: A custom attention mechanism that computes attention scores based on a set of predefined keywords. It projects the hidden states to the keyword space, calculates attention scores using keyword scores and weights, and applies attention scores to the hidden states.

2.shift_tokens_right: A utility function that shifts the input token IDs one position to the right, which is used for preparing the decoder input when ground truth labels are provided.

3.MyBartForConditionalGeneration: A custom BART model for conditional generation, which inherits from the BartForConditionalGeneration class from the transformers library. It adds the KeywordSelfAttention mechanism to the original BART model and modifies the forward method to incorporate the keyword attention mechanism.
"""


import torch
import torch.nn as nn
from transformers import (
    BartTokenizer,
    BartModel,
    BartConfig,
    BartForConditionalGeneration,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import logging

logger = logging.getLogger(__name__)


class KeywordSelfAttention(nn.Module):
    def __init__(self, hidden_size, key_words):
        super().__init__()
        self.key_words = key_words
        self.weights = nn.Parameter(torch.ones(len(key_words)))
        self.softmax = nn.Softmax(dim=-1)
        self.projection = nn.Linear(hidden_size, len(key_words))
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # Project hidden_states to key_words space
        key_word_scores = self.activation(self.projection(hidden_states))
        # Calculate attention scores using key_word_scores and weights
        attention_scores = torch.matmul(
            key_word_scores, self.softmax(self.weights)
        )
        # Apply attention scores to hidden_states
        attention_scores1 = attention_scores.unsqueeze(2)
        attention_output = torch.matmul(
            hidden_states.unsqueeze(3), attention_scores1.unsqueeze(3)
        ).squeeze(3)

        return attention_output


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    # Replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class MyBartForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config:BartConfig,key_words):
        super().__init__(config)
        self.config = config
        self.model = BartModel(config)
        self.keyword_attention = KeywordSelfAttention(config.d_model, key_words)
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

    def forward(
        self,
        input_ids= None,
        attention_mask= None,
        decoder_input_ids= None,
        decoder_attention_mask= None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values= None,
        inputs_embeds= None,
        decoder_inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions= None,
        output_hidden_states= None,
        return_dict= None,
    ) :

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        
        keyword_attention_output = self.keyword_attention(outputs[0])
        # keyword_attention_output = keyword_attention_output.unsqueeze(1)

        lm_logits = self.lm_head(keyword_attention_output)+ self.final_logits_bias

        # lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias#原始的

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past



