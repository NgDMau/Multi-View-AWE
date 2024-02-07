import librosa
import numpy as np
import matplotlib.pyplot as plt



def extract_mfcc_features(audio_path, start_time, end_time, sr=None):
    """
    Extract 39 MFCC features (13 MFCCs + 13 Delta + 13 Delta-Delta) from a specific segment of an audio file.
    
    Parameters:
    - audio_path: Path to the audio file.
    - start_time: Start time of the word segment in seconds.
    - end_time: End time of the word segment in seconds.
    - sr: Sample rate to use. If None, librosa's default will be used.
    
    Returns:
    - mfcc_features: A numpy array containing 39 MFCC features for the segment.
    """
    
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Extract the segment
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    word_segment = y[start_sample:end_sample]
    
    # Compute 13 MFCCs
    mfccs = librosa.feature.mfcc(y=word_segment, sr=sr, n_mfcc=13)
    
    # Compute Delta and Delta-Delta features
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    
    # Concatenate to get 39 features
    mfcc_features = np.concatenate((mfccs, mfcc_delta, mfcc_delta2), axis=0)
    
    return mfcc_features