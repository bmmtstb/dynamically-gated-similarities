"""
Given 'total_AP_metrics.json' and 'total_MOT_metrics.json' in different sub-folders,
compute a csv containing all the combined data.
"""

import csv
import os
from glob import glob

from dgs.utils.files import read_json
from dgs.utils.types import FilePath

PT21_JOINTS = [
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


def replace_dots_with_commas(fp: FilePath) -> None:
    with open(fp, "r", encoding="utf-8") as file:
        content = file.read()

    modified_content = content.replace(".", ",")

    with open(fp, "w", encoding="utf-8") as file:
        file.write(modified_content)


if __name__ == "__main__":

    # ####### #
    # POSEVAL #
    # ####### #

    poseval_out_file = os.path.join(BASE_DIR, "./results_poseval.csv")
    with open(poseval_out_file, "w+", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=";")
        csv_writer.writerow(["Combined", "Dataset", "Key", "Type"] + PT21_JOINTS)

        MOT_metrics = glob(f"{BASE_DIR}/**/eval_data/total_MOT_metrics.json", recursive=True)

        for MOT_file in MOT_metrics:
            data_folder = os.path.dirname(os.path.dirname(os.path.realpath(MOT_file)))
            ds, conf_key = os.path.split(data_folder)[-2:]
            ds_name = os.path.basename(ds)

            mot_data = read_json(MOT_file)
            # mot
            for k, new_k in {"mota": "MOTA", "motp": "MOTP", "pre": "MOT precision", "rec": "MOT recall"}.items():
                comb = f"{ds_name}_{conf_key}_{new_k}"
                csv_writer.writerow([comb, ds_name, conf_key, new_k] + mot_data[k])

            AP_file = MOT_file.replace("total_MOT_metrics", "total_AP_metrics")
            if os.path.isfile(AP_file):
                ap_data = read_json(AP_file)
                # ap
                for k, new_k in {"ap": "AP", "pre": "AP precision", "rec": "AP recall"}.items():
                    comb = f"{ds_name}_{conf_key}_{new_k}"
                    csv_writer.writerow([comb, ds_name, conf_key, new_k] + ap_data[k])
    replace_dots_with_commas(poseval_out_file)

    print(f"Wrote poseval results to: {poseval_out_file}")

    # ######### #
    # PT21 EVAL #
    # ######### #

    pt21_out_file = os.path.join(BASE_DIR, "./results_pt21.csv")
    with open(pt21_out_file, "w+", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=";")
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
    replace_dots_with_commas(pt21_out_file)
    print(f"Wrote PT21 results to: {pt21_out_file}")

    # ########## #
    # DanceTrack #
    # ########## #

    dance_out_file = os.path.join(BASE_DIR, "./results_dance.csv")
    dance_files = glob("./data/DanceTrack/*/results_*/eval_data/pedestrian_detailed.csv")
    data: list[dict] = []
    for dance_file in dance_files:
        res_dir_name = os.path.basename(os.path.dirname(os.path.dirname(dance_file)))
        data_part_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(dance_file))))

        with open(dance_file, "r", encoding="utf-8") as in_file:
            csv_reader = csv.DictReader(in_file, delimiter=",", lineterminator="\r\n")
            line: dict
            for line in csv_reader:
                ds_name = line["seq"]
                comb = f"{data_part_name}_{res_dir_name}_{ds_name}"
                d = {**line, **{"Combined": comb, "Dataset": data_part_name, "Key": res_dir_name}}
                data.append(d)
    fieldnames = ["Combined", "Dataset", "Key"] + list(data[0].keys())
    with open(dance_out_file, "w+", encoding="utf-8") as out_file:
        csv_writer = csv.DictWriter(out_file, fieldnames=fieldnames, delimiter=";", lineterminator="\n")
        csv_writer.writeheader()
        for d in data:
            csv_writer.writerow(dict(d))
    print(f"Wrote DanceTrack results to: {dance_out_file}")
