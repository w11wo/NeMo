import json
import os
import re
from tqdm.auto import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--train_manifest", type=str, required=True, help="Path to train manifest."
)

parser.add_argument(
    "--test_manifest", type=str, default=None, help="Path to test manifest."
)

parser.add_argument(
    "--valid_manifest", type=str, default=None, help="Path to valid manifest."
)

args = parser.parse_args()

def read_manifest(path):
    manifest = []
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Reading manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest


def write_processed_manifest(data, original_path):
    original_manifest_name = os.path.basename(original_path)
    new_manifest_name = original_manifest_name.replace(".json", "_processed.json")

    manifest_dir = os.path.split(original_path)[0]
    filepath = os.path.join(manifest_dir, new_manifest_name)
    with open(filepath, 'w') as f:
        for datum in tqdm(data, desc="Writing manifest data"):
            datum = json.dumps(datum)
            f.write(f"{datum}\n")
    print(f"Finished writing manifest: {filepath}")
    return filepath

# Preprocessing steps
def remove_special_characters(data):
    chars_to_ignore_regex = "[\.\,\?\:\-!;“\"”″‟„�‘ˈˌ]"
    data["text"] = re.sub(chars_to_ignore_regex, " ", data["text"])  # replace punctuation by space
    data["text"] = re.sub(r" +", " ", data["text"])  # merge multiple spaces
    return data

# Processing pipeline
def apply_preprocessors(manifest, preprocessors):
    for processor in preprocessors:
        for idx in tqdm(range(len(manifest)), desc=f"Applying {processor.__name__}"):
            manifest[idx] = processor(manifest[idx])

    print("Finished processing manifest !")
    return manifest


# List of pre-processing functions
PREPROCESSORS = [
    remove_special_characters,
]

train_data = read_manifest(args.train_manifest)
# Apply preprocessing
train_data_processed = apply_preprocessors(train_data, PREPROCESSORS)
# Write new manifests
train_manifest_cleaned = write_processed_manifest(train_data_processed, args.train_manifest)

if args.valid_manifest is not None:
    valid_data = read_manifest(args.valid_manifest)
    valid_data_processed = apply_preprocessors(valid_data, PREPROCESSORS)
    dev_manifest_cleaned = write_processed_manifest(valid_data_processed, args.valid_manifest)

if args.test_manifest is not None:
    test_data = read_manifest(args.test_manifest)
    test_data_processed = apply_preprocessors(test_data, PREPROCESSORS)
    test_manifest_cleaned = write_processed_manifest(test_data_processed, args.test_manifest)