import requests

# URL = "https://juicydd-voice-embedding-api.herokuapp.com/embedding"
URL ="http://127.0.0.1:8000/api/speaker-embedding/"
TEST_AUDIO_FILE_PATH = r"D:\SpeechDataset\test\LibriSpeech\test-clean\1089\134691\1089-134691-0005.flac"

if __name__ == "__main__":
    audio_file = open(TEST_AUDIO_FILE_PATH, 'rb')
    values = {"file":(TEST_AUDIO_FILE_PATH, audio_file, "audio/flac")}
    response = requests.post(URL, files=values)
    print(response)
    data = response.json()
    print('embedding = ',data['embeddings'])
    temp = data['embeddings'].replace("[","]")
    temp = temp.replace("]","")
    temp = temp.split(",")
    print('shape = ', len(temp))
    