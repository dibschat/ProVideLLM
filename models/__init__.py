from transformers import HfArgumentParser

from .arguments_live import get_args_class, ProVideLLMBaseTrainingArguments
from .live_llama import (
    build_live_llama as build_model_and_tokenizer,
    LiveLlamaForCausalLM,
)
from .utils import count_parameters


def parse_args() -> ProVideLLMBaseTrainingArguments:
    (args,) = HfArgumentParser(ProVideLLMBaseTrainingArguments).parse_args_into_dataclasses()
    (args,) = HfArgumentParser(get_args_class(args)).parse_args_into_dataclasses()
    return args
