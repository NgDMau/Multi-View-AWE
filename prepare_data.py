import sys
from pathlib import Path
import json

parent_dir = "/home/ldap-users/s2210403"
sys.path.append(f'{parent_dir}/VG-HuBERT')

import torch
import soundfile as sf
import os
import pickle
from models import audio_encoder
from itertools import groupby
from operator import itemgetter

model_path = "/home/ldap-users/s2210403/VG-HuBERT/vg-hubert_3"
tgt_layer = 9
threshold = 0.7


def cls_attn_seg(cls_attn_weights, threshold, spf, audio_len_in_sec):

    threshold_value = torch.quantile(cls_attn_weights, threshold, dim=-1, keepdim=True) # [n_h, T]
    boundary_idx = torch.where((cls_attn_weights >= threshold_value).float().sum(0) > 0)[0].cpu().numpy()

    word_boundaries_list = []
    word_boundary_intervals = []
    attn_boundary_intervals = []

    for k, g in groupby(enumerate(boundary_idx), lambda ix : ix[0] - ix[1]):
        seg = list(map(itemgetter(1), g))
        t_s, t_e = seg[0], min(seg[-1]+1, cls_attn_weights.shape[-1])
        if len(seg) > 1:
            attn_boundary_intervals.append([spf*t_s, spf*t_e])

    for left, right in zip(attn_boundary_intervals[:-1], attn_boundary_intervals[1:]):
        word_boundaries_list.append((left[1]+right[0])/2.)
    
    for i in range(len(word_boundaries_list)-1):
        word_boundary_intervals.append([word_boundaries_list[i], word_boundaries_list[i+1]])
    return {"attn_boundary_intervals": attn_boundary_intervals, "word_boundary_intervals": word_boundary_intervals}

# setup model
with open(os.path.join(model_path, "args.pkl"), "rb") as f:
    model_args = pickle.load(f)
model = audio_encoder.AudioEncoder(model_args)
bundle = torch.load(os.path.join(model_path, "best_bundle.pth"))
model.carefully_load_state_dict(bundle['dual_encoder'], load_all=True)
model.eval()
model = model.cuda()


def get_segmented(audio_file):
    # load waveform (do not layer normalize the waveform!)
    audio, sr = sf.read(audio_file, dtype = 'float32')
    assert sr == 16000
    audio_len_in_sec = len(audio) / sr
    audio = torch.from_numpy(audio).unsqueeze(0).cuda() # [T] -> [1, T]

    # model forward
    with torch.no_grad():
        model_out = model(audio, padding_mask=None, mask=False, need_attention_weights=True, tgt_layer=tgt_layer)
    feats = model_out['features'].squeeze(0)[1:] # [1, T+1, D] -> [T, D]
    spf = audio.shape[-1]/sr/feats.shape[-2]
    attn_weights = model_out['attn_weights'].squeeze(0) # [1, num_heads, T+1, T+1] -> [num_heads, T+1, T+1] (for the two T+1, first is target length then the source)
    cls_attn_weights = attn_weights[:, 0, 1:] # [num_heads, T+1, T+1] -> [num_heads, T]
    out = cls_attn_seg(cls_attn_weights, threshold, spf, audio_len_in_sec) # out contains attn boundaries and word boundaries in intervals
    return out

# Read the processed data JSON file
with open('updated_processed_data.json', 'r') as file:
    data = json.load(file)

success = 0

# Process each word and its sub-jsons
for word, sub_json in data.items():
    for audio_key, value_indices in sub_json.items():
        # Call the get_segmented function with the audio path
        result = get_segmented(audio_key)
        segmented_list = result["attn_boundary_intervals"]
        # Ensure the list has enough elements
        if not segmented_list or max(value_indices) >= len(segmented_list):
            print(f"Error: The list returned by get_segmented for {audio_key} does not have an index {value_indices}.Length is {len(segmented_list)}")
            # continue
        # Replace the value of the sub-json with the value-th element from the list
        # If the value is a list of indices, this will take all corresponding elements.
        else:
            success += 1
            new_values = [segmented_list[index] for index in value_indices]
            sub_json[audio_key] = new_values
            print(f"Success: {success}")
    # Update the data dictionary with new sub-json
    data[word] = sub_json

# Write the updated data back to a new JSON file
with open('new_updated_processed_data.json', 'w') as file:
    json.dump(data, file, indent=2)