import torch
import numpy as np

import features_extraction 
import nhi_config


#---get embedding of a single utterance
def my_inference(features, encoder, is_full_sequence_inference = nhi_config.FULL_SEQUENCE_INFERENCE):
    if is_full_sequence_inference:
        batch_input = torch.unsqueeze(torch.from_numpy(features), dim=0).float().to(nhi_config.DEVICE)
        batch_output = encoder(batch_input)
        return batch_output[0, :].cpu().data.numpy()
    
    else:
        windows = features_extraction.extract_sliding_windows(features)
        if not windows:
            return None
        batch_input = torch.from_numpy(
            np.stack(windows)).float().to(nhi_config.DEVICE)
        batch_output = encoder(batch_input)

        #------Aggregate the inference outputs from sliding windows
        aggregated_output = torch.mean(batch_output, dim=0, keepdim=False).cpu()
        return aggregated_output.data.numpy()



# if __name__ == '__main__':
    # x = torch.tensor([1,2,3,4])
    # print(torch.unsqueeze(x,dim=0))
