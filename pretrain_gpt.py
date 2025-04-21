# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

import os
import sys

# Add mup package to sys.path
sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'mup/'))

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args

from datasets import load_dataset
from itertools import chain

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(get_args())
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    # Extra items
    if args.return_doc_ids and data is not None:
        _key = 'doc_ids'
        if _key in data.keys():
            keys.append(_key)
        _key = 'dataset_idx'
        if _key in data.keys():
            data[_key] = data[_key].long()
            keys.append(_key)

    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    if args.data_searching_save is not None and args.return_doc_ids:
        doc_ids = None
        _key = 'doc_ids'
        if _key in keys:
            doc_ids = data_b['doc_ids'].long()
        dataset_idx = None
        _key = 'dataset_idx'
        if _key in keys:
            dataset_idx = data_b['dataset_idx'].long()
        return tokens, labels, loss_mask, attention_mask, position_ids, dataset_idx
    else:
        return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()
    # print('tokens: ', tokens.size())

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        return_doc_ids=args.return_doc_ids,
        data_cache_path=args.data_cache_path)
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds

def train_valid_test_datasets_provider_hg(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    # train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
    #     data_prefix=args.data_path,
    #     data_impl=args.data_impl,
    #     splits_string=args.split,
    #     train_valid_test_num_samples=train_val_test_num_samples,
    #     seq_length=args.seq_length,
    #     seed=args.seed,
    #     skip_warmup=(not args.mmap_warmup),
    #     train_data_prefix=args.train_data_path,
    #     valid_data_prefix=args.valid_data_path,
    #     test_data_prefix=args.test_data_path,
    #     data_cache_path=args.data_cache_path)
    data_files = {}

    data_files["train"] = args.data_path[0] + '/wiki.train.raw'
    data_files["validation"] = args.data_path[0] + '/wiki.valid.raw'
    data_files["test"] = args.data_path[0] + '/wiki.test.raw'

    raw_datasets = load_dataset('text', data_files=data_files)
    print_rank_0("> finished creating GPT datasets ...")

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    tokenizer = get_tokenizer()

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=2,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    block_size = args.seq_length + 1

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=2,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_ds = lm_datasets["train"]
    valid_ds = lm_datasets["validation"]
    test_ds = lm_datasets["test"]
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
             get_batch_fn=get_batch)
