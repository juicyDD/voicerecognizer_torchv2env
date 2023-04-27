import numpy as np
import torch
import torch.nn as nn
import time
import os
import matplotlib.pyplot as plt
import torch.optim as optim

from inference import my_inference
import features_extraction
import multiprocessing
import nhi_config
import data_prep
from training import TripletScoreFetcher
from my_neural_network import get_speaker_encoder

def FRR(labels, scores, thres):
    #fn/(tp+fn)
    fn=0
    tp = 0
    for i in range(len(labels)):
        if scores[i] >= thres:
            if labels[i] == 0:
                fn+=1
            else:
                tp+=1
    if (tp+fn) == 0:
        return 0
    return fn/(tp+fn)

def FAR(labels, scores, thres):
    #fp/(fp+tn)
    tn = 0
    fp = 0
    for i in range(len(labels)):
        if scores[i] < thres:
            if labels[i] == 0:
                tn+=1
            else:
                fp+=1
    if (tn+fp) == 0:
        return 0
    return fp/(fp+tn)

def compute_equal_error_rate2(labels, scores):
    min_delta = 1
    eer= 1
    thres = 0
    while thres<1:
        FAR(labels, scores, thres)
        far_ = FAR(labels, scores, thres)
        frr_ = FRR(labels, scores, thres)
        delta_ = abs(far_ - frr_)
        if delta_ < min_delta:
            min_delta = delta_
            eer = (far_+frr_)/2
            thres += nhi_config.EER_THRESHOLD_STEP
    return eer, thres

def compute_equal_error_rate(labels, scores):
    """Compute the Equal Error Rate (EER)."""
    if len(labels) != len(scores):
        raise ValueError("Length of labels and scored must match")
    eer_threshold = None
    eer = None
    min_delta = 1
    threshold = 0.0
    while threshold < 1.0:
        accept = [score >= threshold for score in scores]
        fa = [a and (1-l) for a, l in zip(accept, labels)]
        fr = [(1-a) and l for a, l in zip(accept, labels)]
        far = sum(fa) / (len(labels) - sum(labels))
        frr = sum(fr) / sum(labels)
        delta = abs(far - frr)
        if delta < min_delta:
            min_delta = delta
            eer = (far + frr) / 2
            eer_threshold = threshold
        threshold += nhi_config.EER_THRESHOLD_STEP

    return eer, eer_threshold

#--- cần nhi_config.NUM_EVAL_TRIPLETS triplets để đánh giá mô hình, score của 1 triplet là cosine similarity của (anchor, positive) (ground truth = 1) và (anchor, negative) (ground truth = 0)
def compute_scores(encoder, dict_speaker, num_eval_triplets=nhi_config.NUM_EVAL_TRIPLETS):
    labels = []
    scores = []
    fetcher = TripletScoreFetcher(dict_speaker, encoder, num_eval_triplets)
    #---multi threading instead of multi processing while running the inference
    with multiprocessing.pool.ThreadPool(nhi_config.NUM_PROCESSES) as pool:
        while num_eval_triplets > len(labels)//2: 
            label_score_pairs = pool.map(fetcher, range(len(labels)//2, num_eval_triplets))
            for triplet_labels, triplet_scores in label_score_pairs:
                labels.extend(triplet_labels)
                scores.extend(triplet_scores)
        pool.close()
                
    print("Evaluated", len(labels)//2, "triplets in total")
    return (labels, scores)

def run_eval():
    """Run evaluation of the saved model on test data."""
    start_time = time.time()
    
    
    spk_to_utts = data_prep.get_utterances_by_speakers(nhi_config.TEST_DATASET_DIR)
    print("Evaluation data:", nhi_config.TEST_DATASET_DIR)
    
    encoder = get_speaker_encoder(nhi_config.SAVED_MODEL_PATH)
    # encoder = MyEncoder().encoder
    
    labels, scores = compute_scores(encoder, spk_to_utts, nhi_config.NUM_EVAL_TRIPLETS)
    
    eer, eer_threshold = compute_equal_error_rate(labels, scores)
    eval_time = time.time() - start_time
    print("Finished evaluation in", eval_time, "seconds")
    print("eer_threshold =", eer_threshold, "eer =", eer)
    
if __name__ == "__main__":
    run_eval()




# Evaluated 1000 triplets in total
# Finished evaluation in 186.06178784370422 seconds
# eer_threshold = 0.6210000000000004 eer = 0.0945

# Evaluated 10000 triplets in total
# Finished evaluation in 2028.7074007987976 seconds
# eer_threshold = 0.6300000000000004 eer = 0.08030000000000001