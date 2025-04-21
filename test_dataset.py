from megatron.data.gpt_dataset import build_train_valid_test_datasets

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""

    print('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=['/home/dist/dataset/pile/pile_wikipedia_demo'],
        data_impl='mmap',
        splits_string='1',
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=2048,
        seed=1234,
        skip_warmup=False,
        train_data_prefix=None,
        valid_data_prefix=None,
        test_data_prefix=None,
        return_doc_ids=False,
        data_cache_path=None)
    print("> finished creating GPT datasets ...")

train_samples = 200000
eval_samples = 0
test_samples = 0
train_val_test_num_samples = [train_samples,
                              eval_samples,
                              test_samples]

train_valid_test_datasets_provider(train_val_test_num_samples=train_val_test_num_samples)