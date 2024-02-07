import json
import librosa
import numpy as np

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

# Load the updated_processed_data.json file
with open('Temp_data/new_updated_processed_data.json', 'r') as file:
    data = json.load(file)

items_to_remove = []
items_processed = 0

output_data = {}

# Process each word and its audio segments
for word, sub_json in data.items():
    for audio_path, intervals in sub_json.items():
        # print(f"Audio path: {audio_path}")
        # print(f"Intervals: {intervals}")
        remove_sub_json = False
        # Process each interval
        for i, interval in enumerate(intervals): 
            if isinstance(interval, list):
                start_time, end_time = interval
                if end_time - start_time >= 0.5:
                    # print(f"Start: {start_time}, End: {end_time}")
                    # Extract the MFCC features for this segment
                    mfcc_features = extract_mfcc_features(audio_path, start_time, end_time)
                    mfcc_features_list = mfcc_features.tolist() if mfcc_features is not None else None
                    # Replace the interval with the extracted MFCC features
                    print(type(mfcc_features))
                    output_data[word] = mfcc_features_list
                    print(type(output_data))
                    items_processed += 1
                    print(f"Processed: {(items_processed)}")
                    break
                else: 
                    remove_sub_json = True
                    break

        if remove_sub_json:
            items_to_remove.append((word, audio_path))

    # if items_processed == 10:
    #     break

# Remove the marked items from data
for word, audio_path in items_to_remove:
    # Check if word is still in data and has the specific audio_path, then remove it
    if word in data and audio_path in data[word]:
        del data[word][audio_path]
        # If this was the last audio_path for the word, remove the word entry as well
        if data[word]:  # Check if sub_json is empty
            del data[word]

# Save the updated data back to a new JSON file
with open('Temp_data/mfcc_processed_data.json', 'w') as file:
    json.dump(output_data, file, indent=2)
