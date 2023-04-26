import soundfile as sf
import librosa
import random
import numpy as np 
import torch

import nhi_config
import data_prep
import specaug

'''read an arbitrary *.flac file, return waveform and sample rate'''
def read_audio(audio_dir):
    waveform, sample_rate = sf.read(audio_dir)
    
    #if number of channels of the target audio file is greater than 1
    if len(waveform.shape)>1: 
        waveform = librosa.to_mono(waveform.transpose())
        
    # Convert to 16kHz.
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, sample_rate, 16000)
        sample_rate = 16000
    return waveform, sample_rate
    
'''extract mfcc feature from an audio file, a frame = 512 samples'''
def extract_mfcc(audio_dir):
    waveform, sample_rate = read_audio(audio_dir)
    
    #mfcc standard frame length = 512 samples
    features = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=nhi_config.N_MFCC)
    return features.transpose()

'''Extract sliding windows from features, dùng cho sliding window inference'''
def extract_sliding_windows(features):
    windows = []
    start_idx = 0
    while (start_idx + nhi_config.SEQ_LENGTH) <= features.shape[0]:
        _ = features[start_idx: (start_idx + nhi_config.SEQ_LENGTH), :]
        windows.append(_)
        start_idx += nhi_config.SLIDING_WINDOW_STEP
    return windows

'''from a triplet of anchor, neg, pos utterances => extract mfcc of 3 utterances'''
def get_triplet_mfcc(dict_speakers):
    anchor_utt, pos_utt, neg_utt = data_prep.get_triplet(dict_speakers)
    return (extract_mfcc(anchor_utt),
            extract_mfcc(pos_utt),
            extract_mfcc(neg_utt))

'''trim mfcc features to nhi_config.SEQ_LENGTH'''
def trim_features(features, apply_specaug):
    full_length = features.shape[0]
    start = random.randint(0, full_length - nhi_config.SEQ_LENGTH)
    trimmed_features = features[start: start + nhi_config.SEQ_LENGTH, :]
    if apply_specaug:
        trimmed_features = specaug.apply_specaug(trimmed_features)
    return trimmed_features

'''The fetcher of trimmed features for multi-processing.'''
class TrimmedTripletFeaturesFetcher:
    def __init__(self, spk_to_utts):
        self.spk_to_utts = spk_to_utts

    def __call__(self, _):
        """Get a triplet of trimmed anchor/pos/neg features."""
        anchor, pos, neg = get_triplet_mfcc(self.spk_to_utts)
        
        #-------anchor, pos, meg have to be greater then seq length
        while (anchor.shape[0] < nhi_config.SEQ_LENGTH or pos.shape[0] < nhi_config.SEQ_LENGTH or neg.shape[0] < nhi_config.SEQ_LENGTH):
            anchor, pos, neg = get_triplet_mfcc(self.spk_to_utts)
        return np.stack([trim_features(anchor, nhi_config.SPECAUG_TRAINING),
                         trim_features(pos, nhi_config.SPECAUG_TRAINING),
                         trim_features(neg, nhi_config.SPECAUG_TRAINING)])
    
def get_batched_triplet_input(spk_to_utts, batch_size, pool=None):
    """Get batched triplet input for PyTorch."""
    fetcher = TrimmedTripletFeaturesFetcher(spk_to_utts)
    if pool is None:
        input_arrays = list(map(fetcher, range(batch_size)))
    else:
        input_arrays = pool.map(fetcher, range(batch_size))
    batch_input = torch.from_numpy(np.concatenate(input_arrays)).float()
    return batch_input

if __name__ == '__main__':
    #test read file
    temp = read_audio(r"D:\SpeechDataset\train\train-clean-100\39\121914\39-121914-0000.flac")
    print(temp[0].shape)

    temp = extract_mfcc(r"D:\SpeechDataset\train\train-clean-100\39\121914\39-121914-0000.flac")
    print(temp.shape)
    print(librosa.__version__)