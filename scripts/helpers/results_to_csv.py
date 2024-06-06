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

BASE_DIR: str = "./results/own/eval/"

if __name__ == "__main__":

    # ####### #
    # POSEVAL #
    # ####### #

    poseval_out_file = os.path.join(BASE_DIR, "./results_poseval.csv")
    with open(poseval_out_file, "w", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Combined", "Dataset", "Key", "Type"] + JOINTS)

        AP_metrics = glob(f"{BASE_DIR}/**/eval_data/total_AP_metrics.json", recursive=True)

        for AP_file in AP_metrics:
            data_folder = os.path.dirname(os.path.dirname(os.path.realpath(AP_file)))
            MOT_file = os.path.join(data_folder, "./eval_data/total_MOT_metrics.json")

            if not os.path.isfile(MOT_file):
                raise FileNotFoundError(f"Found AP metrics file, but no MOT file at '{MOT_file}'.")

            ds, conf_key = os.path.split(data_folder)[-2:]
            ds_name = os.path.basename(ds)

            ap_data = read_json(AP_file)
            mot_data = read_json(MOT_file)

            # ap
            for k, new_k in {"ap": "AP", "pre": "AP precision", "rec": "AP recall"}.items():
                comb = f"{ds_name}_{conf_key}_{new_k}"
                csv_writer.writerow([comb, ds_name, conf_key, new_k] + ap_data[k])
            # mot
            for k, new_k in {"mota": "MOTA", "motp": "MOTP", "pre": "MOT precision", "rec": "MOT recall"}.items():
                comb = f"{ds_name}_{conf_key}_{new_k}"
                csv_writer.writerow([comb, ds_name, conf_key, new_k] + mot_data[k])
    print(f"Wrote poseval results to: {poseval_out_file}")

    # ######### #
    # PT21 EVAL #
    # ######### #

    pt21_out_file = os.path.join(BASE_DIR, "./results_pt21.csv")
    with open(pt21_out_file, "w", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Combined", "Dataset", "Key"] + PT21_METRICS)

        # Search for pose_hota_results.txt files recursively in the ./files/ directory
        files = glob(f"{BASE_DIR}/**/results_json/pose_hota_results.txt", recursive=True)

        for file_path in files:
            ds_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
            meth_key = os.path.basename(os.path.dirname(os.path.dirname(file_path)))

            # Read data from input file and format as CSV
            with open(file_path, encoding="utf-8") as f:
                _ = f.readline()
                line = f.readline()
                line = line.strip().replace(r"\_", "_")
                values = line.split("&")
                values = [v.strip() for v in values[1:]]  # summary is empty
                assert len(values) == len(PT21_METRICS)
                comb = f"{ds_name}_{meth_key}"
                csv_writer.writerow([comb, ds_name, meth_key] + values)
    print(f"Wrote PT21 results to: {pt21_out_file}")
