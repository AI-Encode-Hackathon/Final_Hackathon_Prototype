import torchaudio

def create_audio_embedding(file, wav2mel, dvector):
    wav_tensor, sample_rate = torchaudio.load(file)
    mel_tensor = wav2mel(wav_tensor, sample_rate)
    emb = dvector.embed_utterance(mel_tensor)

    return emb
