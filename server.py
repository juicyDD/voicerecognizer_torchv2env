from flask import Flask, request, jsonify
import datetime
import os
import random
import pandas as pd

import features_extraction 
import my_neural_network
import nhi_config
import inference

app = Flask(__name__)

@app.route('/embedding', methods=["POST"])
def embedding():
    #get audio file
    audio_file = request.files["file"]
    file_name = str(random.randint(0,10000))
    audio_file.save(file_name)
    
    features = features_extraction.extract_mfcc(file_name)
    encoder = my_neural_network.get_speaker_encoder(nhi_config.SAVED_MODEL_PATH)
    # encoder = my_neural_network.MyEncoder().encoder
    embedding = inference.my_inference(features, encoder)
    result = pd.Series(embedding).to_json(orient='values')
    os.remove(file_name)
    data = {"embeddings":result}
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=False)
    