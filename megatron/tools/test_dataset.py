import subprocess

cmd = "python preprocess_data_single.py --input /data4/dataplatform_datasets/TeleChat-PTD/data/ --json-keys data --output-prefix /data4/haoran.huang/train_data_tele/train_tele --dataset-impl mmap --tokenizer-type SentencePieceTokenizer --tokenizer-model /data1/kechun.wu/LLMTokenizer/output/mt_gpt_500w_add.model --workers 15".split()
