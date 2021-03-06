import collections
from fastNLP import cache_results
def get_skip_path(chars,w_trie):
    sentence = ''.join(chars)
    result = w_trie.get_lexicon(sentence)

    return result

# @cache_results(_cache_fp='cache/get_skip_path_trivial',_refresh=True)
def get_skip_path_trivial(chars,w_list):
    chars = ''.join(chars)
    w_set = set(w_list)
    result = []
    # for i in range(len(chars)):
    #     result.append([])
    for i in range(len(chars)-1):
        for j in range(i+2,len(chars)+1):
            if chars[i:j] in w_set:
                result.append([i,j-1,chars[i:j]])

    return result


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_w = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self,w):

        current = self.root
        for c in w:
            current = current.children[c]

        current.is_w = True

    def search(self,w):
        '''

        :param w:
        :return:
        -1:not w route
        0:subroute but not word
        1:subroute and word
        '''
        current = self.root

        for c in w:
            current = current.children.get(c)

            if current is None:
                return -1

        if current.is_w:
            return 1
        else:
            return 0

    def get_lexicon(self,sentence):
        result = []
        for i in range(len(sentence)):
            current = self.root
            for j in range(i, len(sentence)):
                current = current.children.get(sentence[j])
                if current is None:
                    break

                if current.is_w:
                    result.append([i,j,sentence[i:j+1]])

        return result

from fastNLP.core.field import Padder
import numpy as np
import torch
from collections import defaultdict
class LatticeLexiconPadder(Padder):

    def __init__(self, pad_val=0, pad_val_dynamic=False,dynamic_offset=0, **kwargs):
        '''

        :param pad_val:
        :param pad_val_dynamic: if True, pad_val is the seq_len
        :param kwargs:
        '''
        self.pad_val = pad_val
        self.pad_val_dynamic = pad_val_dynamic
        self.dynamic_offset = dynamic_offset

    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        # ???autoPadder??? dim=2 ???????????????
        max_len = max(map(len, contents))

        max_len = max(max_len,1)#avoid 0 size dim which causes cuda wrong

        max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
                            content_i in contents])

        max_word_len = max(max_word_len,1)
        if self.pad_val_dynamic:
            # print('pad_val_dynamic:{}'.format(max_len-1))

            array = np.full((len(contents), max_len, max_word_len), max_len-1+self.dynamic_offset,
                            dtype=field_ele_dtype)

        else:
            array = np.full((len(contents), max_len, max_word_len), self.pad_val, dtype=field_ele_dtype)
        for i, content_i in enumerate(contents):
            for j, content_ii in enumerate(content_i):
                array[i, j, :len(content_ii)] = content_ii
        array = torch.tensor(array)

        return array

from fastNLP.core.metrics import MetricBase

def get_yangjie_bmeso(label_list,ignore_labels=None):
    def get_ner_BMESO_yj(label_list):
        def reverse_style(input_string):
            target_position = input_string.index('[')
            input_len = len(input_string)
            output_string = input_string[target_position:input_len] + input_string[0:target_position]
            # print('in:{}.out:{}'.format(input_string, output_string))
            return output_string

        # list_len = len(word_list)
        # assert(list_len == len(label_list)), "word list size unmatch with label list"
        list_len = len(label_list)
        begin_label = 'b-'
        end_label = 'e-'
        single_label = 's-'
        whole_tag = ''
        index_tag = ''
        tag_list = []
        stand_matrix = []
        for i in range(0, list_len):
            # wordlabel = word_list[i]
            current_label = label_list[i].lower()
            if begin_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)

            elif single_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(single_label, "", 1) + '[' + str(i)
                tag_list.append(whole_tag)
                whole_tag = ""
                index_tag = ""
            elif end_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i))
                whole_tag = ''
                index_tag = ''
            else:
                continue
        if (whole_tag != '') & (index_tag != ''):
            tag_list.append(whole_tag)
        tag_list_len = len(tag_list)

        for i in range(0, tag_list_len):
            if len(tag_list[i]) > 0:
                tag_list[i] = tag_list[i] + ']'
                insert_list = reverse_style(tag_list[i])
                stand_matrix.append(insert_list)
        # print stand_matrix
        return stand_matrix

    def transform_YJ_to_fastNLP(span):
        span = span[1:]
        span_split = span.split(']')
        # print('span_list:{}'.format(span_split))
        span_type = span_split[1]
        # print('span_split[0].split(','):{}'.format(span_split[0].split(',')))
        if ',' in span_split[0]:
            b, e = span_split[0].split(',')
        else:
            b = span_split[0]
            e = b

        b = int(b)
        e = int(e)

        e += 1

        return (span_type, (b, e))
    yj_form = get_ner_BMESO_yj(label_list)
    # print('label_list:{}'.format(label_list))
    # print('yj_from:{}'.format(yj_form))
    fastNLP_form = list(map(transform_YJ_to_fastNLP,yj_form))
    return fastNLP_form

class SpanFPreRecMetric_YJ(MetricBase):
    r"""
    ?????????:class:`fastNLP.SpanFPreRecMetric` :class:`fastNLP.core.metrics.SpanFPreRecMetric`

    ??????????????????????????????span???????????????F, pre, rec.
    ????????????Part of speech????????????character?????????????????????????????? `???????????????` ?????????POS?????????(???BMES??????)
    ['B-NN', 'E-NN', 'S-DET', 'B-NN', 'E-NN']??????metric???????????????????????????F1?????????
    ???????????????metric?????????::

        {
            'f': xxx, # ????????????f????????????????????????f_beta???
            'pre': xxx,
            'rec':xxx
        }

    ???only_gross=False, ?????????????????????label???metric?????????::

        {
            'f': xxx,
            'pre': xxx,
            'rec':xxx,
            'f-label': xxx,
            'pre-label': xxx,
            'rec-label':xxx,
            ...
        }

    :param tag_vocab: ????????? :class:`~fastNLP.Vocabulary` ?????????????????????"B"(??????label)??????"B-xxx"(xxx?????????label?????????POS??????NN)???
        ???????????????????????????xxx?????????????????????label?????????['B-NN', 'E-NN']?????????????????????'NN'.
    :param str pred: ??????key???evaluate()????????????dict?????????prediction????????? ???None???????????? `pred` ?????????
    :param str target: ??????key???evaluate()????????????dict?????????target????????? ???None???????????? `target` ?????????
    :param str seq_len: ??????key???evaluate()????????????dict?????????sequence length????????????None???????????? `seq_len` ????????????
    :param str encoding_type: ????????????bio, bmes, bmeso, bioes
    :param list ignore_labels: str ?????????list. ??????list??????class?????????????????????????????????POS tagging?????????['NN']??????????????????'NN'???
        ???label
    :param bool only_gross: ?????????????????????f1, precision, recall??????????????????False?????????????????????f1, pre, rec, ??????????????????
        label???f1, pre, rec
    :param str f_type: `micro` ??? `macro` . `micro` :????????????????????????TP???FN???FP?????????????????????f, precision, recall; `macro` :
        ???????????????????????????f, precision, recall??????????????????????????????f??????????????????
    :param float beta: f_beta????????? :math:`f_{beta} = \frac{(1 + {beta}^{2})*(pre*rec)}{({beta}^{2}*pre + rec)}` .
        ?????????beta=0.5, 1, 2. ??????0.5?????????????????????????????????????????????1???????????????????????????2???????????????????????????????????????
    """
    def __init__(self, tag_vocab, pred=None, target=None, seq_len=None, encoding_type='bio', ignore_labels=None,
                 only_gross=True, f_type='micro', beta=1):
        from fastNLP.core import Vocabulary
        from fastNLP.core.metrics import _bmes_tag_to_spans,_bio_tag_to_spans,\
            _bioes_tag_to_spans,_bmeso_tag_to_spans
        from collections import defaultdict

        encoding_type = encoding_type.lower()

        if not isinstance(tag_vocab, Vocabulary):
            raise TypeError("tag_vocab can only be fastNLP.Vocabulary, not {}.".format(type(tag_vocab)))
        if f_type not in ('micro', 'macro'):
            raise ValueError("f_type only supports `micro` or `macro`', got {}.".format(f_type))

        self.encoding_type = encoding_type
        # print('encoding_type:{}'self.encoding_type)
        if self.encoding_type == 'bmes':
            self.tag_to_span_func = _bmes_tag_to_spans
        elif self.encoding_type == 'bio':
            self.tag_to_span_func = _bio_tag_to_spans
        elif self.encoding_type == 'bmeso':
            self.tag_to_span_func = _bmeso_tag_to_spans
        elif self.encoding_type == 'bioes':
            self.tag_to_span_func = _bioes_tag_to_spans
        elif self.encoding_type == 'bmesoyj':
            self.tag_to_span_func = get_yangjie_bmeso
            # self.tag_to_span_func =
        else:
            raise ValueError("Only support 'bio', 'bmes', 'bmeso' type.")

        self.ignore_labels = ignore_labels
        self.f_type = f_type
        self.beta = beta
        self.beta_square = self.beta ** 2
        self.only_gross = only_gross

        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)

        self.tag_vocab = tag_vocab

        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)

    def evaluate(self, pred, target, seq_len):
        from fastNLP.core.utils import _get_func_signature
        """evaluate??????????????????????????????????????????????????????????????????

        :param pred: [batch, seq_len] ?????? [batch, seq_len, len(tag_vocab)], ???????????????
        :param target: [batch, seq_len], ?????????
        :param seq_len: [batch] ??????????????????
        :return:
        """
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if not isinstance(seq_len, torch.Tensor):
            raise TypeError(f"`seq_lens` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(seq_len)}.")

        if pred.size() == target.size() and len(target.size()) == 2:
            pass
        elif len(pred.size()) == len(target.size()) + 1 and len(target.size()) == 2:
            num_classes = pred.size(-1)
            pred = pred.argmax(dim=-1)
            if (target >= num_classes).any():
                raise ValueError("A gold label passed to SpanBasedF1Metric contains an "
                                 "id >= {}, the number of classes.".format(num_classes))
        else:
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        batch_size = pred.size(0)
        pred = pred.tolist()
        target = target.tolist()
        for i in range(batch_size):
            pred_tags = pred[i][:int(seq_len[i])]
            gold_tags = target[i][:int(seq_len[i])]

            pred_str_tags = [self.tag_vocab.to_word(tag) for tag in pred_tags]
            gold_str_tags = [self.tag_vocab.to_word(tag) for tag in gold_tags]

            pred_spans = self.tag_to_span_func(pred_str_tags, ignore_labels=self.ignore_labels)
            gold_spans = self.tag_to_span_func(gold_str_tags, ignore_labels=self.ignore_labels)

            for span in pred_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1
            for span in gold_spans:
                self._false_negatives[span[0]] += 1

    def get_metric(self, reset=True):
        """get_metric???????????????evaluate??????????????????????????????????????????????????????????????????."""
        evaluate_result = {}
        if not self.only_gross or self.f_type == 'macro':
            tags = set(self._false_negatives.keys())
            tags.update(set(self._false_positives.keys()))
            tags.update(set(self._true_positives.keys()))
            f_sum = 0
            pre_sum = 0
            rec_sum = 0
            for tag in tags:
                tp = self._true_positives[tag]
                fn = self._false_negatives[tag]
                fp = self._false_positives[tag]
                f, pre, rec = self._compute_f_pre_rec(tp, fn, fp)
                f_sum += f
                pre_sum += pre
                rec_sum += rec
                if not self.only_gross and tag != '':  # tag!=''?????????tag?????????
                    f_key = 'f-{}'.format(tag)
                    pre_key = 'pre-{}'.format(tag)
                    rec_key = 'rec-{}'.format(tag)
                    evaluate_result[f_key] = f
                    evaluate_result[pre_key] = pre
                    evaluate_result[rec_key] = rec

            if self.f_type == 'macro':
                evaluate_result['f'] = f_sum / len(tags)
                evaluate_result['pre'] = pre_sum / len(tags)
                evaluate_result['rec'] = rec_sum / len(tags)

        if self.f_type == 'micro':
            f, pre, rec = self._compute_f_pre_rec(sum(self._true_positives.values()),
                                                  sum(self._false_negatives.values()),
                                                  sum(self._false_positives.values()))
            evaluate_result['f'] = f
            evaluate_result['pre'] = pre
            evaluate_result['rec'] = rec

        if reset:
            self._true_positives = defaultdict(int)
            self._false_positives = defaultdict(int)
            self._false_negatives = defaultdict(int)

        for key, value in evaluate_result.items():
            evaluate_result[key] = round(value, 6)

        return evaluate_result

    def _compute_f_pre_rec(self, tp, fn, fp):
        """

        :param tp: int, true positive
        :param fn: int, false negative
        :param fp: int, false positive
        :return: (f, pre, rec)
        """
        pre = tp / (fp + tp + 1e-13)
        rec = tp / (fn + tp + 1e-13)
        f = (1 + self.beta_square) * pre * rec / (self.beta_square * pre + rec + 1e-13)

        return f, pre, rec




