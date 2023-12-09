import os,sys,torch,math
import torch.nn as nn 
import transformers
from transformers import AutoConfig,AutoTokenizer,LlamaForCausalLM,LlamaTokenizer,Trainer,DataCollatorWithPadding,AutoModelForCausalLM,BitsAndBytesConfig
import logging 

sys.path.append('..')
from utils.parser_args import parser_arguments
from peft import LoraConfig,PeftModel,TaskType,get_peft_model,get_peft_model_state_dict
from pathlib import Path 
from datasets import load_dataset,concatenate_datasets
from itertools import chain
from utils.trainer import PeftTrainer
from utils.data_collator import DataCollatorForSupervisedDataset
from utils.utils import PROMPT_TEMPLATE

# os.environ["WANDB_DISABLED"] = "true"
logger = logging.getLogger(__name__)
IGNORE_INDEX = -100
MODEL_CLASSES = {
    "llama": (AutoConfig, LlamaTokenizer, LlamaForCausalLM),
    "auto": (AutoConfig, AutoTokenizer, AutoModelForCausalLM),
}


def create_model(model_args, data_args, training_args):
    ## load model 
    config_class, tokenizer_class, model_class = MODEL_CLASSES[model_args.model_type]
    if model_args.tokenizer_name_or_path is None:
        tokenizer = tokenizer_class.from_pretrained(model_args.model_name_or_path, use_fast=model_args.use_fast_tokenizer)
    else:
        tokenizer = tokenizer_class.from_pretrained(model_args.tokenizer_name_or_path, use_fast=model_args.use_fast_tokenizer)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id # set as the <unk> token

    config_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype),
        "low_cpu_mem_usage": True
    }
    if model_args.load_in_4bit:
        config_kwargs["load_in_4bit"] = True
        config_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        **config_kwargs
    )
    model.resize_token_embeddings(len(tokenizer))
    
    if model_args.peft_path is not None:
        logger.info(f"Load pre-trained model: {model_args.peft_path}" )
        model = PeftModel.from_pretrained(model, model_args.peft_path, is_trainable=True)
    else:
        logger.info("Init new peft model")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=training_args.lora_target.split(','),
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            modules_to_save=training_args.modules_to_save.split(',') if training_args.modules_to_save is not None else None
        )

        model = get_peft_model(model, peft_config=lora_config)
    model.print_trainable_parameters()


    return model, tokenizer

def main():
    model_args, data_args, training_args = parser_arguments(logger)
    transformers.set_seed(training_args.seed)

    model, tokenizer = create_model(model_args, data_args, training_args)
    model_origin, tokenizer_origin = create_model(model_args, data_args, training_args)

    # Compare the model weights
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not p1.data.equal(p2.data):
            print("Model weights are different.")
            return False

    print("Model weights are identical.")
    return True


if __name__ == "__main__":
    main()
