import re
import os
import json
from ltp import LTP
from tqdm import tqdm
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def judge_quote(sent):
    return (sent[0] == "“" and sent[-1] == "”") or "--------QUOTE" in sent


def preprocess_novel(ltp, novel, min_len=5, max_len=200):
    with open(f"{novel}/all_name_list.txt", "r", encoding="utf-8") as fr:
        name_list = []
        for name_line in fr.readlines():
            split_res = name_line.strip().split(' ')
            name_list.extend([re.escape(n) for n in split_res[1:] if len(n) > 1])
    name_list = sorted(name_list, key=lambda x: len(x), reverse=True)
    name_pattern = re.compile("|".join(name_list))

    with open(f"{novel}/{novel}_quote.txt", "r", encoding="utf-8") as fr:
        lines = [line.strip() for line in fr.readlines()]
    n = len(lines)
    context_line_idx_set = set([])
    for i, line in enumerate(lines):
        if "--------QUOTE" not in line:
            continue
        pre_idx = i - 1
        while pre_idx >= 0 and i - pre_idx <= 3:
            if judge_quote(lines[pre_idx]):
                break
            if min_len <= len(lines[pre_idx]) <= max_len and len(name_pattern.findall(lines[pre_idx])) > 0:
                context_line_idx_set.add(pre_idx)
            pre_idx -= 1
        next_idx = i + 1
        while next_idx < n and next_idx - i <= 3:
            if judge_quote(lines[next_idx]):
                break
            if min_len <= len(lines[next_idx]) <= max_len and len(name_pattern.findall(lines[next_idx])) > 0:
                context_line_idx_set.add(next_idx)
            next_idx += 1
    context_line_list = [lines[idx].replace("--------QUOTE", "") for idx in list(context_line_idx_set)]
    print(f"context sent num: {len(context_line_list)}")

    start_tag = "B-PER.NAM"
    end_tag = "I-PER.NAM"
    none_tag = "O"
    context_ner_sent_list = []
    for line in tqdm(context_line_list):
        seg, _ = ltp.seg([line])
        context_ner_sent = []
        # 分词
        for w in seg[0]:
            for i, c in enumerate(w):
                context_ner_sent.append(f"{c}{i}")
        # 名字标签
        tag_idxs = set([])
        for match in name_pattern.finditer(line):
            start, end = match.span()
            context_ner_sent[start] = context_ner_sent[start] + "\t" + start_tag
            tag_idxs.add(start)
            for i in range(start + 1, end):
                context_ner_sent[i] = context_ner_sent[i] + "\t" + end_tag
                tag_idxs.add(i)
        for i in range(len(context_ner_sent)):
            if i not in tag_idxs:
                context_ner_sent[i] = context_ner_sent[i] + "\t" + none_tag
        context_ner_sent.append("")
        context_ner_sent_list.extend(context_ner_sent)
    
    with open(f"{novel}/{novel}NER_2nd_deseg", "w", encoding="utf-8") as fw:
        fw.write("\n".join(context_ner_sent_list))


def build_train_dev(train_novels, dev_novel, dev_num):
    merge_lines = []
    for novel in train_novels:
        with open(f"{novel}/{novel}NER_2nd_deseg", "r", encoding="utf-8") as fr:
            lines = fr.readlines()
            merge_lines.extend(lines + ["\n"])
    with open("novelNER.train_deseg", "w", encoding="utf-8") as fw:
        fw.write("".join(merge_lines))

    with open(f"{dev_novel}/{dev_novel}NER_2nd_deseg", "r", encoding="utf-8") as fr:
        cnt = 0
        dev_lines = []
        for line in fr.readlines():
            if line == "\n":
                cnt += 1
                if cnt == dev_num:
                    break
            dev_lines.append(line)
    with open("novelNER.dev_deseg", "w", encoding="utf-8") as fw:
        fw.write("".join(dev_lines))


def extract_name(novel):
    with open(f"{novel}/ner_result.json", "r") as fr:
        tag_lst = json.load(fr)
    with open(f"{novel}/{novel}NER_2nd_deseg", "r") as fr:
        sents = []
        ws = []
        for line in fr.readlines():
            line = line.strip()
            if line != "":
                ws.append(line[0])
            else:
                sents.append("".join(ws))
                ws = []
        if ws:
            sents.append("".join(ws))
    
    name2freq = defaultdict(int)
    for sent, tags in zip(sents, tag_lst):
        name = ""
        for w, t in zip(sent, tags):
            if t != 3 and name != "":
                name2freq[name] += 1
                name = ""
            if t == 4 or (t == 3 and name != ""):
                name += w
        if name != "":
            name2freq[name] += 1
    name2freq = sorted(name2freq.items(), key=lambda x: x[1], reverse=True)

    with open(f"{novel}/ner_name_list.txt", "w", encoding="utf-8") as fw:
        fw.write("\n".join([f"{freq} {name}" for name, freq in name2freq]))


def evaluate_ner(novel):
    with open(f"{novel}/all_name_list.txt", "r", encoding="utf-8") as fr:
        gold_name_lst = []
        for line in fr.readlines():
            gold_name_lst.append(line.strip().split()[1])
    with open(f"{novel}/span_ner_name_list.txt", "r", encoding="utf-8") as fr:
        ner_name_lst = [line.strip().split()[0] for line in fr.readlines()]
    
    top_30_recall = len(set(gold_name_lst[:30]) & set(ner_name_lst[:500])) / 30
    top_500_recall = len(set(gold_name_lst) & set(ner_name_lst[:500])) / len(gold_name_lst)
    all_recall = len(set(gold_name_lst) & set(ner_name_lst)) / len(gold_name_lst)
    print(f"top30 recall: {top_30_recall}")
    print(f"top500 recall: {top_500_recall}")
    print(f"all recall: {all_recall}")


if __name__ == "__main__":
    # 预处理小说 生成标准格式文件
    ltp = LTP(path="/data/home/yaweisun/download/ltp_base")
    for novel in ["万族之劫", "斗破苍穹", "饲养全人类"]:
        print(f"start preprocess {novel}")
        preprocess_novel(ltp, novel)
    # 合并小说 构造训练集 验证集
    build_train_dev(["万族之劫", "斗破苍穹", "饲养全人类"], "白袍总管", 5000)
    
    # 从结果中抽取姓名
    extract_name("白袍总管")
    # 评估结果
    evaluate_ner("白袍总管")
