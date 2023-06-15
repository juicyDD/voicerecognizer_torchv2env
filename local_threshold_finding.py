import os
import glob
import random
import multiprocessing
import multiprocessing.pool 
import nhi_config
import time
import json
from data_prep import get_utterances_by_speakers
from my_neural_network import get_speaker_encoder
from features_extraction import extract_mfcc
from inference import my_inference
from training import cosine_similarity
from evaluation import compute_equal_error_rate
'''15 Jun 2023'''
# def get_neg_pos_dict(dict_speakers,speaker_id='8468'):
#     pos_utts = {}
#     neg_utts ={}
#     for key, val in dict_speakers.items():
#         if key!= speaker_id:
#             neg_utts[key]=val
#         else:
#             pos_utts[key]=val
#     pos_speaker = list(pos_utts.keys())
#     neg_speakers = list(neg_utts.keys())
#     print(pos_speaker,neg_speakers)
#     return  pos_speaker, pos_utts, neg_speakers, neg_utts


def get_neg_pos_dict(speaker_id='8468'):
    pos_speaker = speaker_id
    test_speakers = get_utterances_by_speakers(nhi_config.TEST_DATASET_DIR)
    # print(test_speakers)
    pos_utts = test_speakers[pos_speaker]
    
    neg_utts = get_utterances_by_speakers(nhi_config.TRAIN_DATASET_DIR)
    neg_speakers = neg_utts.keys()
    return pos_speaker, pos_utts, neg_speakers, neg_utts
    # pos_utts = {}
    # neg_utts ={}
    # for key, val in dict_speakers.items():
    #     if key!= speaker_id:
    #         neg_utts[key]=val
    #     else:
    #         pos_utts[key]=val
    # pos_speaker = list(pos_utts.keys())
    # neg_speakers = list(neg_utts.keys())
    # print(pos_speaker,neg_speakers)
    # return  pos_speaker, pos_utts, neg_speakers, neg_utts
def get_triplet_with_fixed_anchor(pos_utts, neg_utts, speaker_id):
    # print(pos_utt)
    pos_utt, anchor_utt = random.sample(pos_utts, 2)
    neg_speaker = random.sample(list(neg_utts.keys()), 1)[0]
    
    neg_utt = random.sample(neg_utts[neg_speaker],1)[0]
    # print('\nanchor:',anchor_utt,'\npos:', pos_utt,'\n neg:', neg_utt)
    return(anchor_utt, pos_utt, neg_utt)
def get_triplet_mfcc(pos_utts, neg_utts, speaker_id):
    
    anchor_utt, pos_utt, neg_utt = get_triplet_with_fixed_anchor(pos_utts, neg_utts, speaker_id)
    
    return (extract_mfcc(anchor_utt), extract_mfcc(pos_utt), extract_mfcc(neg_utt))


class TripletScoreFetcher:
    """Class for computing triplet scores with multi-processing."""

    def __init__(self, pos_utts, neg_utts, encoder, num_eval_triplets, speaker_id):#spk_to_utts,
        self.speaker_id = speaker_id
        self.pos_utts = pos_utts
        self.neg_utts = neg_utts
        self.encoder = encoder
        self.num_eval_triplets = num_eval_triplets
        

    def __call__(self, i):
        """Get the labels and scores from a triplet."""
        anchor, pos, neg = get_triplet_mfcc(self.pos_utts, self.neg_utts,self.speaker_id)
        
        anchor_embedding = my_inference(anchor, self.encoder)
        pos_embedding = my_inference(pos, self.encoder)
        neg_embedding = my_inference(neg, self.encoder)
        
        if ((anchor_embedding is None) or (pos_embedding is None) or (neg_embedding is None)): # ----Some utterances might be smaller than a single sliding window.
            return ([], [])
        _labels = [1, 0]
        _scores = [cosine_similarity(anchor_embedding, pos_embedding),cosine_similarity(anchor_embedding, neg_embedding)]
        print("triplets evaluated:", i, "/", self.num_eval_triplets)
        return (_labels, _scores)


def compute_scores(encoder, pos_utts, neg_utts, speaker_id, num_eval_triplets=nhi_config.NUM_EVAL_TRIPLETS):
    labels = []
    scores = []
    fetcher = TripletScoreFetcher(pos_utts, neg_utts, encoder, num_eval_triplets, speaker_id)
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
def find_local_threshold(speaker_id='8555'):

    start_time = time.time()
    # # spk_to_utts = get_utterances_by_speakers(nhi_config.TEST_DATASET_DIR)
    
    pos_speaker, pos_utts, neg_speakers, neg_utts = get_neg_pos_dict(speaker_id)
    
    encoder = get_speaker_encoder(nhi_config.SAVED_MODEL_PATH)
    labels, scores = compute_scores(encoder, pos_utts, neg_utts, speaker_id, nhi_config.NUM_EVAL_TRIPLETS)#spk_to_utts,
    
    eer, eer_threshold = compute_equal_error_rate(labels, scores)
    eval_time = time.time() - start_time
    print("Finished evaluation in", eval_time, "seconds")
    print("eer_threshold =", eer_threshold, "eer =", eer)
    return eer, eer_threshold
    
def find_local_threshold_for_all_test_speakers():
    test_utts = get_utterances_by_speakers(nhi_config.TEST_DATASET_DIR)
    result = {}
    for k in list(test_utts.keys()):
        eer, eer_threshold=find_local_threshold(speaker_id=k)
        result[k] = {"err":eer, "thres": eer_threshold}
    res = json.dumps(result)
    jsonFile = open("eer_thres.json", "w")
    jsonFile.write(res)
    jsonFile.close()
if __name__ == "__main__":
    find_local_threshold_for_all_test_speakers()