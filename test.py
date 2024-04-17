''' Testing '''

import evaluation
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#model_path = "./model_results/checkpoint/t2i/model_best.pth.tar"
model_path = "./model_results/checkpoint/i2t/model_best.pth.tar"
data_path = "./data/"
evaluation.evalrank(model_path, data_path=data_path, split="test")
