# from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
#
# from pycocoevalcap.bleu.bleu import Bleu
# from pycocoevalcap.meteor import Meteor
# from pycocoevalcap.rouge import Rouge
#
#
# def compute_scores(gts, res):
#     """
#     Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)
#
#     :param gts: Dictionary with the image ids and their gold captions,
#     :param res: Dictionary with the image ids ant their generated captions
#     :print: Evaluation score (the mean of the scores of all the instances) for each measure
#     """
#
#     # Set up scorers
#     scorers = [
#         (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
#         (Meteor(), "METEOR"),
#         (Rouge(), "ROUGE_L")
#     ]
#     eval_res = {}
#     # Compute score for each metric
#     for scorer, method in scorers:
#         try:
#             score, scores = scorer.compute_score(gts, res, verbose=0)
#         except TypeError:
#             score, scores = scorer.compute_score(gts, res)
#         if type(method) == list:
#             for sc, m in zip(score, method):
#                 eval_res[m] = sc
#         else:
#             eval_res[method] = score
#     return eval_res
#
#
# def compute_mlc(gt, pred, label_set):
#     res_mlc = {}
#     avg_aucroc = 0
#     for i, label in enumerate(label_set):
#         res_mlc['AUCROC_' + label] = roc_auc_score(gt[:, i], pred[:, i])
#         avg_aucroc += res_mlc['AUCROC_' + label]
#     res_mlc['AVG_AUCROC'] = avg_aucroc / len(label_set)
#
#     res_mlc['F1_MACRO'] = f1_score(gt, pred, average="macro")
#     res_mlc['F1_MICRO'] = f1_score(gt, pred, average="micro")
#     res_mlc['RECALL_MACRO'] = recall_score(gt, pred, average="macro")
#     res_mlc['RECALL_MICRO'] = recall_score(gt, pred, average="micro")
#     res_mlc['PRECISION_MACRO'] = precision_score(gt, pred, average="macro")
#     res_mlc['PRECISION_MICRO'] = precision_score(gt, pred, average="micro")
#
#     return res_mlc
#
#
# class MetricWrapper(object):
#     def __init__(self, label_set):
#         self.label_set = label_set
#
#     def __call__(self, gts, res, gts_mlc, res_mlc):
#         eval_res = compute_scores(gts, res)
#         eval_res_mlc = compute_mlc(gts_mlc, res_mlc, self.label_set)
#
#         eval_res.update(**eval_res_mlc)
#         return eval_res
from __future__ import annotations
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
# from .Radgraph import F1RadGraph
from .f1chexbert import F1CheXbert
from typing import Dict, Tuple, Optional
import pandas as pd
import re

DEFAULT_ARGS = {
    'chexbert_path': "/data/yz/hf_cache/checkpoints/chexbert.pth",
    'bert_path': "/data/yz/hf_cache/checkpoints/bert-base-uncased",
    'radgraph_path': "/data/yz/hf_cache/checkpoints/radgraph.tar.gz",
}

def compute_nlg_scores(gts, res, args=None):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    # 如果 gts/res 是字典，需要提取值；如果是列表，需要转换为字典
    if isinstance(gts, dict):
        # 字典格式：{id: [text]} 或 {id: text}
        gts = {k: [re.sub(' +', ' ', (v[0] if isinstance(v, list) else v).replace(".", " ."))] 
               for k, v in gts.items()}
    else:
        # 列表格式，转换为字典
        gts = {i: [re.sub(' +', ' ', gt.replace(".", " ."))] for i, gt in enumerate(gts)}
    
    if isinstance(res, dict):
        res = {k: [re.sub(' +', ' ', (v[0] if isinstance(v, list) else v).replace(".", " ."))] 
               for k, v in res.items()}
    else:
        res = {i: [re.sub(' +', ' ', hpy.replace(".", " ."))] for i, hpy in enumerate(res)}
    scorers = [
        (Bleu(4), ["BLeu_1", "BLeu_2", "BLeu_3", "Bleu_4"]),
        # (Meteor(), "METEOR"),
        # (Rouge(), "ROUGE_L"),
        (Cider(), 'CIDEr'),
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res


def compute_ce_scores(gts, res, args):
    # gts and res is list, e.g., [str1, str2]
    # roberta-large
    # model_type = 'distilbert-base-uncased',
    # P, R, F1 = score(res, gts, model_type=args['bertscore_checkpoint'],
    #                  num_layers=5, batch_size=64, nthreads=4, all_layers=False, idf=False, baseline_path=None,
    #                  device='cuda' if torch.cuda.is_available() else 'cpu', lang='en', rescale_with_baseline=True)
    # bertscore = F1.mean().cpu().item()

    f1chexbert = F1CheXbert(chexbert_checkpoint=args['chexbert_path'], model_checkpoint=args['bert_path'],
                            tokenizer_checkpoint=args['bert_path'])

    # print("开始计算Chexbert")
    chexbert_all = f1chexbert(hyps=res, refs=gts)
    chexbert_all_micro_f1 = chexbert_all["micro avg"]
    chexbert_all_macro_f1 = chexbert_all["macro avg"]
    # chexbertscore = class_report_5["micro avg"]["f1-score"]
    # print("开始计算RadGraph")
    # f1radgraph_partial = F1RadGraph(reward_level='all', model_path=args['radgraph_path'])
    # partial_mean_reward, reward_list, hypothesis_ann_lists, reference_ann_lists = f1radgraph_partial(hyps=res, refs=gts)

    # f1radgraph_all = F1RadGraph(reward_level='all', model_path=args['radgraph_checkpoint'])
    metrics = {
        # "BERTScore": bertscore,
        # "F1-Radgraph-simple-partial-complete": partial_mean_reward,
        # "F1-Radgraph-all": all_mean_reward,
        "micro_p": chexbert_all_micro_f1['precision'],
        "micro_r": chexbert_all_micro_f1['recall'],
        "micro_f1": chexbert_all_micro_f1["f1-score"],
        "macro_p": chexbert_all_macro_f1['precision'],
        "macro_r": chexbert_all_macro_f1['recall'],
        "macro_f1": chexbert_all_macro_f1["f1-score"],
    }
    # all_mean_reward, reward_list, hypothesis_ann_lists, reference_ann_lists = f1radgraph_all(hyps=res, refs=gts)
    return metrics


def compute_all_scores(gts, res, args):
    # compute_ce_scores 需要列表格式，compute_nlg_scores 需要字典格式
    # 如果输入是字典，提取值列表传给 compute_ce_scores
    if isinstance(gts, dict):
        gts_list = [v[0] if isinstance(v, list) else v for v in gts.values()]
        res_list = [v[0] if isinstance(v, list) else v for v in res.values()]
    else:
        gts_list = gts
        res_list = res
    
    # compute clinical efficacy metrics (需要列表)
    ce_metrics = compute_ce_scores(gts_list, res_list, args)

    # compute natural language generation (NLG) metrics (需要字典)
    nlg_metrics = compute_nlg_scores(gts, res)
    ce_metrics.update(nlg_metrics)
    return ce_metrics


def compute_chexbert_scores(gts, res, args):
    # compute clinical efficacy metrics
    ce_metrics = compute_ce_scores(gts, res, args)
    return ce_metrics


def compute_chexbert_details_scores(gts, res, args):
    f1chexbert = F1CheXbert(chexbert_checkpoint=args['chexbert_path'], model_checkpoint=args['bert_path'],
                            tokenizer_checkpoint=args['bert_path'])
    accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = f1chexbert(hyps=res, refs=gts)
    # default is chexbert_5_micro_f1
    # micro: each sample has the same weight; macro: each class has the same weight
    del chexbert_all['weighted avg']
    del chexbert_all['samples avg']
    sample_num = chexbert_all['micro avg']['support']
    new_results = {}
    for key, value in chexbert_all.items():
        if 'avg' in key:
            new_results[key] = ['-', round(value['precision'], 3), round(value['recall'], 3), round(value['f1-score'], 3)]
        else:
            new_results[key] = [f"{round(value['support'] * 100 / sample_num, 1)} ({int(value['support'])})",
                                round(value['precision'], 3), round(value['recall'], 3), round(value['f1-score'], 3)]
    return new_results
