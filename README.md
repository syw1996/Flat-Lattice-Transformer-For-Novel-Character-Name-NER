[English](#Requirement)

# Flat Lattice Transformer For Novel Character Name NER
Based on ACL 2020 paper [FLAT: Chinese NER Using Flat-Lattice Transformer](https://arxiv.org/pdf/2004.11795.pdf).
Original code for [Flat-Lattice-Transformer](https://github.com/LeeSureman/Flat-Lattice-Transformer).


# Requirement:

```
Python: 3.7.3
PyTorch: 1.2.0
FastNLP: 0.5.0
Numpy: 1.16.4
```
you can go [here](https://fastnlp.readthedocs.io/zh/latest/) to know more about FastNLP.


How to run the code?
====
1. Download the character embeddings and word embeddings.

      Character and Bigram embeddings (gigaword_chn.all.a2b.{'uni' or 'bi'}.ite50.vec) : [Google Drive](https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)

      Word(Lattice) embeddings: 
      
      yj, (ctb.50d.vec) : [Google Drive](https://drive.google.com/file/d/1K_lG3FlXTgOOf8aQ4brR9g3R40qi1Chv/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)
      
      ls, (sgns.merge.word.bz2) : [Baidu Pan](https://pan.baidu.com/s/1luy-GlTdqqvJ3j-A4FcIOw)

2. Modify the `paths.py` to add the pretrained embedding and the dataset

3. find fastNLP lib path
```
python
import fastNLP
print(fastNLP.__file__）
```

4. modify predictor.py
```
cd fastnlp/core/
```

5. replace the predictor.py file in fastNLP lib's core
```
cp fastnlp/core/predictor.py {$fastNLP_LIB_PATH}/core
```

6. preprocess novel corpus (use [weiboNER](https://github.com/hltcoe/golden-horse/tree/master/data) data format)
```
cd data/corpus/NovelNER
python data_utils.py
```

7. train/test/predict
```
cd V1 (with Bert)
CUDA_VISIBLE_DEVICES=0 python flat_main.py --dataset novel --status train
CUDA_VISIBLE_DEVICES=0 python flat_main.py --dataset novel --status test
CUDA_VISIBLE_DEVICES=0 python flat_main.py --dataset novel --status predict
```

8. you can download the ner model and predict directly.
      
      Model finetuned by three novels (with sgd) : [Google Drive](https://drive.google.com/file/d/1sWZWy7uhZ2vsdb-29Er9zeCjPqJK631-/view?usp=sharing)

If you want to record experiment result, you can use fitlog:
```
pip install fitlog
fitlog init V1
cd V1
fitlog log logs
```
then set use_fitlog = True in flat_main.py.

you can go [here](https://fitlog.readthedocs.io/zh/latest/) to know more about Fitlog.


Cite: 
========
[bibtex](https://www.aclweb.org/anthology/2020.acl-main.611.bib)

