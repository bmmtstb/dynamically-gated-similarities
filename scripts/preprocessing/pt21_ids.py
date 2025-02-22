"""Helper file to show the different numbers of IDs used in the PoseTrack21 dataset."""

import json
import os
from glob import glob

ROOT = os.path.normpath(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def read_json(fp: str):
    with open(os.path.join(ROOT, fp), encoding="utf-8") as f:
        obj = json.load(f)
    return obj


if __name__ == "__main__":
    all_ids = set()
    pt_val_ids = set()
    pt_train_ids = set()
    val_ids = set()
    train_ids = set()
    query_ids = set()
    search_ids = set()
    track_ids = set()

    for json_path in glob("./data/PoseTrack21/posetrack_data/train/*.json"):
        annos = read_json(json_path)["annotations"]
        pt_train_ids.update(set(anno["person_id"] for anno in annos))
    all_ids.update(pt_train_ids)
    track_ids.update(pt_train_ids)

    for json_path in glob("./data/PoseTrack21/posetrack_data/val/*.json"):
        annos = read_json(json_path)["annotations"]
        pt_val_ids.update(set(anno["person_id"] for anno in annos))
    all_ids.update(pt_val_ids)
    track_ids.update(pt_val_ids)

    for json_path in glob("./data/PoseTrack21/posetrack_person_search/query.json"):
        annos = read_json(json_path)["annotations"]
        query_ids.update(set(anno["person_id"] for anno in annos))
    all_ids.update(query_ids)
    search_ids.update(query_ids)

    for json_path in glob("./data/PoseTrack21/posetrack_person_search/train.json"):
        annos = read_json(json_path)["annotations"]
        train_ids.update(set(anno["person_id"] for anno in annos))
    all_ids.update(train_ids)
    search_ids.update(train_ids)

    for json_path in glob("./data/PoseTrack21/posetrack_person_search/val.json"):
        annos = read_json(json_path)["annotations"]
        val_ids.update(set(anno["person_id"] for anno in annos))
    all_ids.update(val_ids)
    search_ids.update(val_ids)

    print("Dataset               Min - Max  - Unique")
    print(f"Search - Train:     {min(train_ids): >4} - {max(train_ids): >4} - {len(train_ids): >4}")
    print(f"Search - Query:     {min(query_ids): >4} - {max(query_ids): >4} - {len(query_ids): >4}")
    print(f"Search - Val:       {min(val_ids): >4} - {max(val_ids): >4} - {len(val_ids): >4}")
    print(f"Search total:       {min(search_ids): >4} - {max(search_ids): >4} - {len(search_ids): >4}")
    print("---")
    print(f"PTrack - Train:     {min(pt_train_ids): >4} - {max(pt_train_ids): >4} - {len(pt_train_ids): >4}")
    print(f"PTrack - Val/Query: {min(pt_val_ids): >4} - {max(pt_val_ids): >4} - {len(pt_val_ids): >4}")
    print(f"PTrack total:       {min(track_ids): >4} - {max(track_ids): >4} - {len(track_ids): >4}")
    print("---")
    print(f"All:                {min(all_ids): >4} - {max(all_ids): >4} - {len(all_ids): >4}")

# Dataset              Min - Max  - Unique
# Search - Train:       11 - 7529 - 5474
# Search - Query:        3 - 1600 - 1313
# Search - Val:          0 - 1655 - 1656
# Search total:          0 - 7529 - 7129
# ---
# PTrack - Train:     1600 - 6878 - 4172
# PTrack - Val/Query:    0 - 1601 - 1526
# PTrack total:          0 - 6878 - 5696
# ---
# All:                   0 - 7529 - 7130
