"""
Given 'total_AP_metrics.json' and 'total_MOT_metrics.json' in different sub-folders,
compute a csv containing all the combined data.
"""

import csv
import os
from glob import glob

from dgs.utils.files import read_json

JOINTS = [
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "neck",
    "nose",
    "head_top",
    "total",
]


FOLDERS: list[str] = [
    "FastPose",
    "HRNet",
    "SimpleBaseline",
    "DetectAndTrack_FastPose",
    "DetectAndTrack_HRNet",
    "DetectAndTrack_SimpleBaseline",
]

PT21_METRICS: list[str] = [
    "DetPr",
    "DetRe",
    "DetA",
    "AssPr",
    "AssRe",
    "AssA",
    "LocA",
    "FragA",
    "HOTA",
    "RHOTA",
    "FA-HOTA",
    "FA-RHOTA",
    "LocA(0)",
    "HOTALocA(0)",
    "HOTA(0)",
    "HOTA_TP",
    "HOTA_FP",
    "HOTA_FN",
    "HOTA_TP(0)",
    "HOTA_FP(0)",
    "HOTA_FN(0)",
]

BASE_DIR: str = "./results/ap_results/"

if __name__ == "__main__":

    # ####### #
    # POSEVAL #
    # ####### #

    with open(os.path.join(BASE_DIR, "./results_poseval.csv"), "w", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Dataset", "Type"] + JOINTS)

        for folder in FOLDERS:
            ap_data = read_json(f"./results/ap_results/{folder}/eval_data/total_AP_metrics.json")
            mot_data = read_json(f"./results/ap_results/{folder}/eval_data/total_MOT_metrics.json")

            # ap
            for k, new_k in {"ap": "AP", "pre": "AP precision", "rec": "AP recall"}.items():
                csv_writer.writerow([folder, new_k] + ap_data[k])
            # mot
            for k, new_k in {"mota": "MOTA", "motp": "MOTP", "pre": "MOT precision", "rec": "MOT recall"}.items():
                csv_writer.writerow([folder, new_k] + mot_data[k])

    # ######### #
    # PT21 EVAL #
    # ######### #

    with open(os.path.join(BASE_DIR, "./results_pt21.csv"), "w", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Name"] + PT21_METRICS)

        # Search for pose_hota_results.txt files recursively in the ./files/ directory
        files = glob(f"{BASE_DIR}/**/files/pose_hota_results.txt")

        for file_path in files:
            parent_folder_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))

            # Read data from input file and format as CSV
            with open(file_path, encoding="utf-8") as f:
                _ = f.readline()
                line = f.readline()
                line = line.strip().replace(r"\_", "_")
                values = line.split("&")
                values = [v.strip() for v in values[1:]]  # summary is empty
                assert len(values) == len(PT21_METRICS)
                csv_writer.writerow([parent_folder_name] + values)
