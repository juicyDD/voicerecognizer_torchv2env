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
from my_neural_network import get_speaker_encoder, MyEncoder

def cosine_similarity(x1,x2):
    return np.dot(x1,x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))


class TripletScoreFetcher:
    """Class for computing triplet scores with multi-processing."""

    def __init__(self, spk_to_utts, encoder, num_eval_triplets):
        self.spk_to_utts = spk_to_utts
        self.encoder = encoder
        self.num_eval_triplets = num_eval_triplets

    def __call__(self, i):
        """Get the labels and scores from a triplet."""
        anchor, pos, neg = features_extraction.get_triplet_mfcc(self.spk_to_utts)
        
        anchor_embedding = my_inference(anchor, self.encoder)
        pos_embedding = my_inference(pos, self.encoder)
        neg_embedding = my_inference(neg, self.encoder)
        
        if ((anchor_embedding is None) or (pos_embedding is None) or (neg_embedding is None)): # ----Some utterances might be smaller than a single sliding window.
            return ([], [])
        triplet_labels = [1, 0]
        triplet_scores = [
            cosine_similarity(anchor_embedding, pos_embedding),
            cosine_similarity(anchor_embedding, neg_embedding)]
        print("triplets evaluated:", i, "/", self.num_eval_triplets)
        return (triplet_labels, triplet_scores)



"""Triplet loss from 1*(a|p|n) triplet"""
def get_triplet_loss_from_single_triplet(anchor, pos, neg): 
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    return torch.max(cos(anchor, neg) - cos(anchor, pos) + nhi_config.TRIPLET_ALPHA, torch.tensor(0.0).to('cpu'))


'''Triplet loss from N*(a|p|n) triplets'''
def get_triplet_loss_from_batch_output(batch_output, batch_size):
    
    batch_output_reshaped = torch.reshape(batch_output, (batch_size, 3, batch_output.shape[1]))
    
    batch_loss = get_triplet_loss_from_single_triplet(batch_output_reshaped[:, 0, :], batch_output_reshaped[:, 1, :], batch_output_reshaped[:, 2, :])
    loss = torch.mean(batch_loss)
    return loss

def save_model(saved_model_path, encoder, losses, start_time):
    t_training = time.time() - start_time
    os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
    if not saved_model_path.endswith(".pt"):
        saved_model_path += ".pt"
    torch.save({"encoder_state_dict": encoder.state_dict(), "losses": losses, "training_time": t_training},saved_model_path)
    

def train_network(spk_to_utts, num_steps, saved_model=None, pool=None):
    start_time = time.time()
    losses = []
    encoder = get_speaker_encoder()
    # encoder = MyEncoder().encoder

    # Train
    optimizer = optim.Adam(encoder.parameters(), lr=nhi_config.LEARNING_RATE)
    print("Start training")
    for step in range(num_steps):
        optimizer.zero_grad()

        # Build batched input.
        batch_input = features_extraction.get_batched_triplet_input(
            spk_to_utts, nhi_config.BATCH_SIZE, pool).to(nhi_config.DEVICE)

        # Compute loss.
        batch_output = encoder(batch_input)
        loss = get_triplet_loss_from_batch_output(
            batch_output, nhi_config.BATCH_SIZE)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print("step:", step, "/", num_steps, "loss:", loss.item())

        if (saved_model is not None and
                (step + 1) % nhi_config.SAVE_MODEL_FREQUENCY == 0):
            checkpoint = saved_model
            if checkpoint.endswith(".pt"):
                checkpoint = checkpoint[:-3]
            checkpoint += ".ckpt-" + str(step + 1) + ".pt"
            save_model(checkpoint,
                       encoder, losses, start_time)

    training_time = time.time() - start_time
    print("Finished training in", training_time, "seconds")
    if saved_model is not None:
        save_model(saved_model, encoder, losses, start_time)
    return losses


def run_training():
    spk_to_utts = data_prep.get_utterances_by_speakers(nhi_config.TRAIN_DATASET_DIR) 
    
    print("Training data:", nhi_config.TRAIN_DATASET_DIR)
    with multiprocessing.Pool(nhi_config.NUM_PROCESSES) as pool:
        losses = train_network(spk_to_utts, nhi_config.TRAINING_STEPS, nhi_config.SAVED_MODEL_PATH, pool)
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()
if __name__ == "__main__":
    # run_training()
    tempdir = r"D:\SpeechDataset\test\LibriSpeech\test-clean\672\122797\672-122797-0004.flac"
    temp = features_extraction.extract_mfcc(tempdir)
    # encoder = get_speaker_encoder(nhi_config.SAVED_MODEL_PATH)
    encoder = MyEncoder().encoder
    embedding_temp = my_inference(temp, encoder)
    
    tempdir2 = r"D:\SpeechDataset\test\LibriSpeech\test-clean\2830\3980\2830-3980-0008.flac"
    temp2 = features_extraction.extract_mfcc(tempdir2)
    embedding_temp2 = my_inference(temp2, encoder)
    
    
    print(embedding_temp)
    print("shape: ", embedding_temp.shape)
    print("cos similarity: ", cosine_similarity(embedding_temp, embedding_temp2))
    
