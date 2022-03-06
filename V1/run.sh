# find fastNLP lib path
python
import fastNLP
fastNLP.__file__

# modify predictor.py
cd fastnlp/core/

# replace the predictor.py file in core
cd {$fastnlp_path}/core

# preprocess
python preprocess.py
cd data/corpus/NovelNER
python data_utils.py

# use V1 (with Bert)
cd V1

# train
CUDA_VISIBLE_DEVICES=6 python flat_main.py --dataset novel --status train --optim adam --lr 5e-4 --bert_lr_rate 0.1 --weight_decay 0.01 --warmup 0

# test
CUDA_VISIBLE_DEVICES=6 python flat_main.py --dataset novel --status test

# predict
CUDA_VISIBLE_DEVICES=7 python flat_main.py --dataset novel --status predict

# sgd best perform on dev: f=0.968419, pre=0.971356, rec=0.9655
# adam best perform on dev: 