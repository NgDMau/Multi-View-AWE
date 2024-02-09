import json
import librosa
import numpy as np

def extract_mfcc_features_from_chunk(audio_path, start_time, end_time, sr=None):
    """
    Extract a feature matrix from a specific segment of an audio file using MFCC.
    Each feature vector in the matrix is 39-dimensional, composed of:
    - 13 MFCCs
    - 13 Delta (first derivative) of MFCCs
    - 13 Delta-Delta (second derivative) of MFCCs

    Parameters:
    - audio_path: Path to the audio file.
    - start_time: Start time of the segment in seconds.
    - end_time: End time of the segment in seconds.
    - sr: Sample rate to use. If None, librosa's default (22050 Hz) will be used.

    Returns:
    - A feature matrix where each row is a 39-dimensional MFCC feature vector.
    """
    
    # Calculate the duration of the segment
    duration = end_time - start_time
    
    # Load only the specified segment of the audio file
    y, sr = librosa.load(audio_path, sr=sr, offset=start_time, duration=duration)
    
    # Compute 13 MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Compute the first and second derivatives (Deltas) of MFCCs
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    
    # Concatenate MFCCs, Deltas, and Delta-Deltas to form 39-dimensional features
    mfcc_features = np.concatenate((mfccs, mfcc_delta, mfcc_delta2), axis=0)
    
    # Transpose the result to have feature vectors as rows
    feature_matrix = mfcc_features.T
    
    return feature_matrix

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
                if end_time - start_time >= 0.3:
                    # print(f"Start: {start_time}, End: {end_time}")
                    # Extract the MFCC features for this segment
                    mfcc_features = extract_mfcc_features_from_chunk(audio_path, start_time, end_time)
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
with open('Temp_data/mfcc_features.json', 'w') as file:
    json.dump(output_data, file, indent=2)
