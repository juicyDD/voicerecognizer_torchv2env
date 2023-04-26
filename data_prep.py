import os
import glob
import random
import nhi_config

'''Get all utterances by each speaker :"3'''
def get_utterances_by_speakers(mydir):
    path_pattern = os.path.join(mydir,"*","*","*.flac") 
    all_flac_files = glob.glob(path_pattern)
    dict_speakers = dict()
    for myfile in all_flac_files:
        file_name_only = os.path.basename(myfile)
        file_name_only = file_name_only.split("-") 
        speaker = file_name_only[0]         #e.g: 911-130578-0014.flac (991 (file_name_only[0]) is speaker ID)
        
        #each speaker have a couple of different utterances
        if speaker not in dict_speakers:
            dict_speakers[speaker] =[myfile]
        else:
            dict_speakers[speaker].append(myfile)
    return dict_speakers

'''get anchor, positive, negative utterances for triplet loss'''
def get_triplet(dict_speakers):
    speakers = dict_speakers.keys()
    pos_speaker, neg_speaker = random.sample(list(speakers),2) #randomly pick 2 speakers, no replacement
    
    pos_utterance, anchor_utterance = random.sample(dict_speakers[pos_speaker],2)
    neg_utterance = random.sample(dict_speakers[neg_speaker],1)[0]
    
    return (anchor_utterance, pos_utterance, neg_utterance)
    

if __name__ == '__main__':
    result = get_utterances_by_speakers(nhi_config.TRAIN_DATASET_DIR)
    hihi = get_triplet(result)
    print(hihi)
    # for key in result.keys():
    #     print(key,": ",len(result[key]))
    print()