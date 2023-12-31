import os
import torch
import torch.nn as nn 
#GeForce MX250 => Compute capability = 6.1 => CUDA ver 8.0 => downgrade python xuống 3.6 nhaa
#virtualenv --python="C:/Users/DELL/AppData/Local/Programs/Python/Python36/python.exe" "C:/Users/DELL/Desktop/DATN/currentcode/virtualenv/"
# pip install "C:\Users\DELL\Desktop\DATN\currentcode\torch-0.4.0-cp36-cp36m-win_amd64.whl"
import nhi_config
from torch.nn import functional as F

class PretrainedEncoder(nn.Module):
    def load_pretrained(self, saved_model):
        my_state_dict = torch.load(saved_model, map_location=nhi_config.DEVICE) #, map_location=nhi_config.DEVICE
        self.load_state_dict(my_state_dict["encoder_state_dict"])
      
#LONG SHORT-TERM MEMORY MODEL
class LstmSpeakerEncoder(PretrainedEncoder):
    def __init__(self, saved_model=""):
        super(LstmSpeakerEncoder, self).__init__()
        # Define the LSTM network.
        #--------------------input_size, hidden_size, num_layers
        self.lstm = nn.LSTM( input_size=nhi_config.N_MFCC, hidden_size=nhi_config.LSTM_HIDDEN_SIZE, num_layers=nhi_config.LSTM_NUM_LAYERS, batch_first=True, bidirectional=nhi_config.BI_LSTM)
        #--------- if there is a pretrained model, load it
        if saved_model:
            self.load_pretrained(saved_model)

    '''join output frames'''
    def join_frames(self, batch_output):
        if nhi_config.FRAME_AGGREGATION_MEAN:
            return torch.mean(
                batch_output, dim=1, keepdim=False)
        else:
            return batch_output[:, -1, :]
    def forward(self, x):
        D = 2 if nhi_config.BI_LSTM else 1
        #-----------h0: hidden state; c0: cell state
        #x shape = (batch_size 8x3, seq_len, input_size)
        h0 = torch.zeros(D * nhi_config.LSTM_NUM_LAYERS, x.shape[0], nhi_config.LSTM_HIDDEN_SIZE)
        c0 = torch.zeros(D * nhi_config.LSTM_NUM_LAYERS, x.shape[0], nhi_config.LSTM_HIDDEN_SIZE)
        y, (hn, cn) = self.lstm(x.to(nhi_config.DEVICE), (h0.to(nhi_config.DEVICE), c0.to(nhi_config.DEVICE)))
        return self.join_frames(y)

#LONG SHORT-TERM MEMORY MODEL 2222
class LstmSpeakerEncoder2(PretrainedEncoder):
    def __init__(self, saved_model=""):
        super(LstmSpeakerEncoder2, self).__init__()
        # Define the LSTM network.
        #--------------------input_size, hidden_size, num_layers
        self.lstm1 = nn.LSTM( input_size=nhi_config.N_MFCC, hidden_size=nhi_config.LSTM_HIDDEN_SIZE, num_layers=nhi_config.LSTM_NUM_LAYERS, batch_first=True, bidirectional=nhi_config.BI_LSTM)
        self.lstm2= nn.LSTM( input_size=nhi_config.LSTM_HIDDEN_SIZE * 2, hidden_size=nhi_config.LSTM_HIDDEN_SIZE, num_layers=nhi_config.LSTM_NUM_LAYERS, batch_first=True, bidirectional=nhi_config.BI_LSTM)
        # self.linear = nn.Linear(128, 1)
        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, 128)

        self.linear_global = nn.Linear(128, 128)
        
        #--------- if there is a pretrained model, load it
        if saved_model:
            self.load_pretrained(saved_model)

    '''join output frames'''
    def join_frames(self, batch_output):
        if nhi_config.FRAME_AGGREGATION_MEAN:
            return torch.mean(
                batch_output, dim=1, keepdim=False)
        else:
            return batch_output[:, -1, :]
    def forward(self, x):
        D = 2 if nhi_config.BI_LSTM else 1
        #-----------h0: hidden state; c0: cell state
        #x shape = (batch_size 8x3, seq_len, input_size)
        h0 = torch.zeros(D * nhi_config.LSTM_NUM_LAYERS, x.shape[0], nhi_config.LSTM_HIDDEN_SIZE)
        c0 = torch.zeros(D * nhi_config.LSTM_NUM_LAYERS, x.shape[0], nhi_config.LSTM_HIDDEN_SIZE)
        y1, (hn, cn) = self.lstm1(x.to(nhi_config.DEVICE), (h0.to(nhi_config.DEVICE), c0.to(nhi_config.DEVICE)))
        y2, _ = self.lstm2(y1)
        h_conc_linear1  = F.relu(self.linear1(y1))
        h_conc_linear2  = F.relu(self.linear2(y2))
        y =  y1 + y2 + h_conc_linear1 + h_conc_linear2
        output = self.linear_global(y)
        # output_global = self.linear_global(output.mean(1))
        # y = self.linear(y)
        return self.join_frames(output)
        
class TransformerSpeakerEncoder(PretrainedEncoder):
    
    def __init__(self, saved_model=""):
        super(TransformerSpeakerEncoder, self).__init__()
        # Define the Transformer network.
        self.linear_layer = nn.Linear(nhi_config.N_MFCC, nhi_config.TRANSFORMER_DIM)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=nhi_config.TRANSFORMER_DIM, nhead=nhi_config.TRANSFORMER_HEADS, batch_first=True),
            num_layers=nhi_config.TRANSFORMER_ENCODER_LAYERS)
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=nhi_config.TRANSFORMER_DIM, nhead=nhi_config.TRANSFORMER_HEADS, batch_first=True),
            num_layers=1)

        if saved_model:
            self.load_pretrained(saved_model)

    def forward(self, x):
        encoder_input = torch.sigmoid(self.linear_layer(x))
        encoder_output = self.encoder(encoder_input)
        tgt = torch.zeros(x.shape[0], 1, nhi_config.TRANSFORMER_DIM).to(nhi_config.DEVICE)
        output = self.decoder(tgt, encoder_output)
        return output[:, 0, :]




def get_speaker_encoder(saved_model=""): #function to get encoder(model)
    if nhi_config.USE_TRANSFORMER:
        return TransformerSpeakerEncoder(saved_model).to(nhi_config.DEVICE)
    else:
        return LstmSpeakerEncoder2(saved_model).to(nhi_config.DEVICE)
    
class Singleton(type): #use singleton to get encoder(model) ONCE
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
class MyEncoder(metaclass=Singleton):
    def __init__(self, saved_model=nhi_config.SAVED_MODEL_PATH):
        if nhi_config.USE_TRANSFORMER == True:
            self.encoder = TransformerSpeakerEncoder(saved_model).to(nhi_config.DEVICE)
        else:
            self.encoder = LstmSpeakerEncoder2(saved_model).to(nhi_config.DEVICE)
    

if __name__ == "__main__":
    print(torch.cuda.is_available())
    
#     Finished evaluation in 1561.0991175174713 seconds
# eer_threshold = 0.7010000000000005 eer = 0.0906