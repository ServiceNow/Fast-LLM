import torch
import transformers
from fast_llm.data.stardoc_data_utils import conversation as conversation_lib
from fast_llm.data.stardoc_data_utils.mm_utils import tokenizer_image_token
from fast_llm.data.stardoc_data_utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from typing import Dict


def docowl_text_preprocess_v1(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    split: str = "train",
) -> Dict:
    """
    source: list of {'role':'user'/'assistant', 'content':xxxx}
    """
    conv = conversation_lib.conv_mplug_owl2.copy()
    # conv.roles: ("USER", "ASSISTANT")
    roles = {"user": conv.roles[0], "assistant": conv.roles[1]}

    if split == "train" or split == "val" or split == "test":

        # Apply prompt templates
        conversations = []

        # Skip the first one if it is not from human
        if roles[source[0]["role"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["role"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["content"])
        
        # conv.get_prompt(): USER: {content} ASSISTANT: {content}</s>USER: {content} ASSISTANT: {content}</s>...
        conversations.append(conv.get_prompt()) 

        # Tokenize conversations
        if has_image:
            input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        else:
            input_ids = tokenizer.tokenize(
                conversations,
                return_tensors="pt",
                padding="longest",
                truncation=True,
            ).input_ids

        targets = input_ids.clone()

        assert conv.sep_style == conversation_lib.SeparatorStyle.TWO or conv.sep_style == conversation_lib.SeparatorStyle.TWO_NO_SYS

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": " # ' ASSISTANT: '
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2) # split by </s>
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep) # split each round by ' ASSISTANT: '
                if len(parts) != 2:
                    break
                parts[0] += sep # input query, ignore for loss

                if has_image:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.max_seq_length: # ignore padding
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )
    else:
        text = source[0]["content"]
        roles = conv.roles # ("USER", "ASSISTANT")
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        stop_str = conv.sep2
        keywords = [stop_str]
        return dict(    
            input_ids=input_ids,
            labels=input_ids,
            stop_str=stop_str,
            keywords=keywords,
        )