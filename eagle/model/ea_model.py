# [Multiple GPU Support] are places with weird dtype transitions, may cause additional latency.

import torch
import torch.nn as nn

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM, LlamaModel
# import transformers

# # monkey patch
# transformers.models.llama.modeling_llama.LlamaForCausalLM = KVLlamaForCausalLM
# transformers.models.mistral.modeling_mistral.MistralForCausalLM = KVMistralForCausalLM

from transformers import PreTrainedModel, PretrainedConfig
from .utils import *
from .kv_cache import initialize_past_key_values
from transformers import AutoTokenizer, AutoConfig
import os
from huggingface_hub import hf_hub_download
import warnings
from copy import deepcopy

from .configuration_eagle import EagleConfig
from safetensors.torch import load_model

from typing import List, Optional, Tuple, Union


def debug_print(tensor_dict, headings="DEBUG", exit_code=False):
    pass

    # print(f"------------------ {headings}")
    # for key, value in tensor_dict.items():
    #     print(f"{key}:\t {value.device}, {value.shape}")
    # print("------------------")

    # if exit_code is True:
    #     print("Exited.")
    #     exit(1)


class DraftModel(nn.Module):
    def __init__(self, config, model: LlamaModel = None):
        super().__init__()
        if hasattr(model, "embed_tokens"):
            del model.embed_tokens

        self.fc = nn.Linear(config.hidden_size*2, config.hidden_size, bias=True)
        self.model = model
        self.lm_head = None
        self.embed_tokens = None

        self.total_tokens = 60 -1
        self.depth = 5
        self.top_k = 10
        self.threshold = 1.0
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def set_head_and_embed(self, lm_head, embed_tokens):
        self.lm_head = lm_head
        self.embed_tokens = embed_tokens
        for param in self.lm_head.parameters():
            param.requires_grad = False
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def discard_head_and_embed(self):
        lm_head = self.lm_head
        embed_tokens = self.embed_tokens
        self.lm_head = None
        self.embed_tokens = None

        return lm_head, embed_tokens

    def forward(self, hidden_states, input_ids, embed_tokens=None, past_key_values=None, **kwargs):
        if embed_tokens is None:
            if self.embed_tokens is None:
                raise ValueError("embed_tokens is not provided")
            embed_tokens = self.embed_tokens

        with torch.no_grad():
            debug_print(
                {
                    "input_ids": input_ids,
                    "embed_tokens": embed_tokens.weight
                }, 
                headings="Draft Forward", exit_code=False
            )
            inputs_embeds = embed_tokens(input_ids)  # [Multiple GPU Support]

        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))
        # returns hidden_states only
        return self.model(inputs_embeds=hidden_states, past_key_values=past_key_values, **kwargs)

    def reset_kv(self):
        self.stable_kv = None

    def reset(self):
        self.tree_mask = None

    def init_tree(self):
        self.tree_mask_init = torch.eye(
            self.top_k, device=self.embed_tokens.weight.device)[None, None]
        self.position_ids = torch.zeros(
            self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long)
        self.tree_mask_init = self.tree_mask_init.to(
            self.embed_tokens.weight.device)

    @torch.no_grad()
    def topK_generate(self, hidden_states, input_ids, head, logits_processor=None):
        debug_print(
            {
                "hidden_states": hidden_states,
                "input_ids": input_ids
            }, 
            headings="topK_generate", exit_code=False
        )
        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_ids[:, -1]
        input_ids = input_ids.to(hidden_states.device)

        scores_list = []
        parents_list = []
        ss_token = []

        input_ids = input_ids[:, 1:]
        # input_ids = input_ids.to(hidden_states.device)

        len_posi = input_ids.shape[1]
        self.reset()

        # * forward once, use cache if possible
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            outputs = self(hidden_states, input_ids[:, kv_len:], past_key_values=self.stable_kv)
            out_hidden = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

        else:
            outputs = self(hidden_states, input_ids)
            out_hidden = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]

        # * pass through lm_head
        last_headout = self.lm_head(last_hidden)
        # last_headout = head(last_hidden)

        # * sample top_k
        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values

        # * append scores and corresponding tokens to lists
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        ss_token.append(topk_index)
        input_ids = topk_index
        input_hidden = last_hidden[None].repeat(1, top_k, 1)  # ! What is this???
        tree_mask = self.tree_mask_init
        # [0, 1, 2, 3, ..., top_k-1]
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
        # 4

        # * start generation
        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            # print(f"depth: {i}")
            outputs = self(
                input_hidden, input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids
            )
            out_hidden = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
            len_posi += 1

            # with Timer("sort1"):
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)

            last_headout = self.lm_head(out_hidden[0])
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            input_ids = topk_index.view(-1)[topk_cs_index][None]

            ss_token.append(topk_index)
            scores_list.append(cu_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)

        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)
        #[Modified] multigpu problem, check why original eagle doesn't do this
        # draft_tokens = torch.cat((sample_token.to(draft_tokens.device), draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        # with Timer("mask"):
        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        # with Timer("mask1"):
        #     tree_mask0 = [[False for _ in range(total_tokens + 1)] for _ in range(total_tokens + 1)]
        #     tree_mask0[0][0] = True
        #     for i in range(total_tokens):
        #         #tree_mask0[i + 1][0]=True
        #         tree_mask0[i + 1][i + 1] = True
        #         p=mask_index_list[i]
        #         tree_mask0[i + 1][p] = True
        #         while p:
        #             p=mask_index_list[p-1]
        #             tree_mask0[i + 1][p] = True
        #     tree_mask0 = torch.tensor(tree_mask0, dtype=torch.bool)
        #
        # print(tree_mask0.equal(tree_mask))
        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        # with Timer("retrieve"):

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                # sort_keys=[len(list)]
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(hidden_states.device)

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

class EagleModelABC(nn.Module):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    def __init__(
        self,
        config,
    ):
        """
        Args:
            config (PretrainedConfig): The configuration of the MedusaModel.
        """
        super().__init__(config)  # base model initializations
        # init draft model here


    def get_tiny_model(self, config):
        # draft_config = config.copy()
        # draft_config.num_hidden_layers = 1
        # return LlamaModel(config)
        raise NotImplementedError

    # Add a link named base_model to self
    # @property
    # def base_model(self):
    #     return self
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        load_draft_weight = True,
        draft_only=False,
        config=None,
        total_tokens=-1,
        *args,
        **kwargs,
    ):
        # Assume pretrained_model_name_or_path only loads the draft model,
        # base model's weight is loaded from the config

        # load config
        print("load eagle config...")
        config = EagleConfig.from_pretrained(pretrained_model_name_or_path)

        # base model
        print("load base model...")
        base_model_name_or_path = config.base_model_name_or_path
        model = super().from_pretrained(
            base_model_name_or_path,
            *args,
            **kwargs,
        )

        print("Load draft model...")
        draft_model_path = os.path.join(
            pretrained_model_name_or_path, "model.safetensors")
        model.setup_draft_model(draft_model_path, config=config, load_draft_weight=load_draft_weight)

        # tokenizer
        print("Load tokenizer...")
        model.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path, use_fast=False)

        # Calibrate tree
        if total_tokens == -1:
            print("Calibrating total tokens...")
            device = model.model.layers[0].self_attn.q_proj.weight.device
            candidate_tokens = [40, 48, 50, 56, 60]
            base_times = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for t_cnt, base_t in zip(candidate_tokens, base_times):
                input_ids = torch.randint(
                    0, model.config.vocab_size - 200, (1, t_cnt)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        _ = model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / base_t)
            total_token = candidate_tokens[times.index(min(times))]
            model.draft_model.total_tokens = total_token-1
            print(f"total_tokens set to: {total_token}")

        if draft_only:
            draft_model = model.draft_model
            del model
            return draft_model
        else:
            return model

    def setup_draft_model(self, draft_model_path, config, load_draft_weight=True):
        tiny_model = self.get_tiny_model(config)
        self.draft_model = DraftModel(config, tiny_model)

        if load_draft_weight:
            load_model(self.draft_model, draft_model_path, strict=True)

        # set lm_-head, embed_tokens
        lm_head = self.lm_head
        embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size, self.config.pad_token_id)
        embed_tokens.weights = self.model.embed_tokens.weight.clone()
        self.draft_model.set_head_and_embed(lm_head, embed_tokens)

        # Assign drafft model's device and dtype
        device = self.model.layers[-1].self_attn.q_proj.weight.device
        self.draft_model.to(self.dtype).to(device)

        # Init tree mask and position id
        self.draft_model.init_tree()

    # Calling self() causes weird problems.
    # Hence rename forward function, call self.eagle_forward() instead.
    def eagle_forward( 
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.lm_head(outputs[0])
            hidden_states = outputs[0]

            debug_print(
                {
                    "self.lm_head": self.lm_head.weight,
                    "hidden_states": hidden_states,
                    "input_ids": input_ids
                }, 
                headings="In EAModel", exit_code=False
            )
            
        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    def eagle_generate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
    ):

        # * not sure, total_tokens are found from calibration, why max_length is reduced by (total_tokens - 10)?
        max_length = max_length-self.draft_model.total_tokens-10

        # * prepare the logits processor
        logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)

        # * initialize the padding, input_ids
        padding = (torch.zeros(1, 1, dtype=torch.long)-1).to(input_ids.device)
        input_ids = input_ids.clone()
        prev_input_len = input_ids.shape[1]

        # * reset draft model kv cache
        self.draft_model.reset_kv()

        # * Initialize kv cache if none
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        debug_print(
            {
                # "hidden_states": hidden_states,
                "input_ids": input_ids
            }, 
            headings="Before Prefill", exit_code=False
        )
        # * target model first inference
        self.model.tree_mask = None
        outputs, orig, hidden_states = self.eagle_forward(
            input_ids,
            past_key_values=past_key_values,
            output_orig=True
        )
        # orig = outputs.logits
        # hidden_states = outputs.hidden_states
        debug_print(
            {
                "hidden_states": hidden_states,
                "input_ids": input_ids
            }, 
            headings="After Prefill", exit_code=False
        )

        # * sample token
        token = sampling_logit(logits=orig[:, -1], logits_processor=logits_processor)

        print(f"token: {token.shape} | input_ids: {input_ids.shape} | hidden_state: {hidden_states.shape}")
        print("Entering Loop...")
        new_token = 0
        for idx in range(max_length):
            # * Run draft model
            # print("Draft phase...")
            # *      1. concatenate the token temporaily to input_ids (This should be permanent, not temporary, find a way to replace this)
            temp_input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.draft_model.topK_generate(hidden_states, temp_input_ids, logits_processor)
            draft_tokens = draft_tokens.to(input_ids.device)  # [Multiple GPU Support]

            # * obtains the logits of predictions from target model, by considering the tree_mask (task: validation)
            # print("Verification Phase...")
            self.model.tree_mask = tree_mask
            logits, hidden_state_new = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )

            # * pad draft_tokens (learn why need this)
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)

            # TODO: understand how to select the best candidate, probably check specinfer's code?
            # * speculative decoding (tree version), select the best candidate from the draft tokens
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            # *
            input_ids, token, hidden_states, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                hidden_state_new,
                sample_p
            )

            if self.tokenizer.eos_token_id in input_ids[0, prev_input_len:]:
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        return input_ids

    def eagle_generate_generator(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
    ):

        # * not sure, total_tokens are found from calibration, why max_length is reduced by (total_tokens - 10)?
        max_length = max_length-self.draft_model.total_tokens-10

        # * prepare the logits processor
        logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)

        # * initialize the padding, input_ids
        padding = (torch.zeros(1, 1, dtype=torch.long)-1).to(input_ids.device)
        input_ids = input_ids.clone()
        prev_input_len = input_ids.shape[1]

        # * reset draft model kv cache
        self.draft_model.reset_kv()

        # * Initialize kv cache if none
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        # * target model first inference
        self.model.tree_mask = None
        outputs = self(
            input_ids,
            past_key_values=past_key_values,
            output_hidden_states=True
        )
        orig = outputs.logits
        hidden_states = outputs.hidden_states
        # past_key_values = outputs.past_key_values
        # print("outputs.past_key_values before entering loop")
        # print(past_key_values) # cache exists!

        # * sample token
        token = sampling_logit(logits=orig[:, -1], logits_processor=logits_processor)

        print(f"token: {token.shape} | input_ids: {input_ids.shape} | hidden_state: {hidden_states.shape}")
        print("Entering Loop...")
        new_token = 0
        for idx in range(max_length):
            # * Run draft model
            # print("Draft phase...")
            # *      1. concatenate the token temporaily to input_ids (This should be permanent, not temporary, find a way to replace this)
            temp_input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.draft_model.topK_generate(hidden_states, temp_input_ids, logits_processor)
            draft_tokens = draft_tokens.to(input_ids.device)  # [Multiple GPU Support]

            # * obtains the logits of predictions from target model, by considering the tree_mask (task: validation)
            # print("Verification Phase...")
            self.model.tree_mask = tree_mask
            logits, hidden_state_new = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )

            # * pad draft_tokens (learn why need this)
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)

            # TODO: understand how to select the best candidate, probably check specinfer's code?
            # * speculative decoding (tree version), select the best candidate from the draft tokens
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            # *
            input_ids, token, hidden_states, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                hidden_state_new,
                sample_p
            )

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, prev_input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            if self.tokenizer.eos_token_id in input_ids[0, prev_input_len:]:
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break


class EagleModelLlama(EagleModelABC, KVLlamaForCausalLM):

    def get_tiny_model(self, config):
        draft_config = deepcopy(config)
        draft_config.num_hidden_layers = 1
        return LlamaModel(draft_config)

# class MedusaModelMistral(MedusaModelABC, KVMistralForCausalLM):
#     pass


class EagleModel():

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except:
            # MEDUSA-v0.1 load
            config = EagleConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(
                config.base_model_name_or_path)
            config.model_type = base_model_config.model_type

        if config.model_type == "llama":
            return EagleModelLlama.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        # elif config.model_type == "mistral":
        #     return MedusaModelMistral.from_pretrained(
        #         pretrained_model_name_or_path,
        #         *args,
        #         **kwargs,
        #     )
        else:
            raise ValueError("Only support llama and mistral for now!!")
