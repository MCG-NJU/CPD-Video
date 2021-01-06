import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
                    default='./ucf_list/ucf101_01.json',
                    type=str)
parser.add_argument('--pred_results',
                    default='results_ft_0.0001_sf_dp0.0_kin_bert_230/',
                    type=str)

from eval_ucf101 import UCFclassification

args = parser.parse_args()
uc = UCFclassification(args.dataset, 
                       '{}/val.json'.format(args.pred_results), 
                       verbose=True, 
                       top_k=1)
uc.evaluate()