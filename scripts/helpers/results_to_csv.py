"""
Given 'total_AP_metrics.json' and 'total_MOT_metrics.json' in different sub-folders,
compute a csv containing all the combined data.
"""

import csv
import os
from glob import glob
from typing import Union

from tqdm import tqdm

from dgs.utils.files import read_json
from dgs.utils.types import FilePath
from dgs.utils.utils import send_discord_notification

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

POSEVAL_FAULTY: list[str] = ["custom_total", "neck", "head_top", "head_bottom", "total"]

OUT_DIR: str = "./results/own/"
BASE_DIRS: list[str] = ["./results/own/eval/", "./results/own/train_single/"]


def replace_dots_with_commas(fp: FilePath) -> None:
    """Given a file path, replace all dots with commas in the file."""
    with open(fp, "r", encoding="utf-8") as file:
        content = file.read()

    modified_content = content.replace(".", ",")

    with open(fp, "w", encoding="utf-8") as file:
        file.write(modified_content)


def force_precision(num: Union[str, float, int], prec: int = 4) -> str:
    """Given a number, force a certain number of decimal points or precision. Does not round.

    Handles integer values separately.
    """
    if isinstance(num, int) or (isinstance(num, str) and not any(div in num for div in [",", "."])):
        return str(num)
    if isinstance(num, str):
        num = float(num)
    return f"{num:.{prec}f}"


if __name__ == "__main__":

    # ####### #
    # POSEVAL #
    # ####### #

    poseval_out_file = os.path.join(OUT_DIR, "./results_poseval.csv")
    with open(poseval_out_file, "w+", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=";")
        csv_writer.writerow(["Combined", "Dataset", "Key", "Type", "custom_total"] + PT21_JOINTS)

        for BASE_DIR in tqdm(BASE_DIRS, desc="poseval base", leave=False):
            MOT_metrics = glob(f"{BASE_DIR}/**/eval_data/total_MOT_metrics.json", recursive=True)

            for MOT_file in tqdm(MOT_metrics, desc="file", leave=False):
                data_folder = os.path.dirname(os.path.dirname(os.path.realpath(MOT_file)))
                ds, conf_key = os.path.split(data_folder)[-2:]
                ds_name = os.path.basename(ds)

                mot_data = read_json(MOT_file)
                # mot
                for k, new_k in {"mota": "MOTA", "motp": "MOTP", "pre": "MOT precision", "rec": "MOT recall"}.items():
                    comb = f"{ds_name}_{conf_key}_{new_k}"
                    custom_total = force_precision(
                        sum(
                            float(val)
                            for i, val in enumerate(mot_data[k])
                            if mot_data["names"][str(i)] not in POSEVAL_FAULTY
                        )
                        / float(
                            sum(
                                1 if mot_data["names"][str(i)] not in POSEVAL_FAULTY else 0
                                for i, val in enumerate(mot_data[k])
                            )
                        )
                    )
                    csv_writer.writerow(
                        [comb, ds_name, conf_key, new_k, custom_total] + [force_precision(mdk) for mdk in mot_data[k]]
                    )

                AP_file = MOT_file.replace("total_MOT_metrics", "total_AP_metrics")
                if not os.path.isfile(AP_file):
                    continue
                ap_data = read_json(AP_file)
                # ap
                for k, new_k in {"ap": "AP", "pre": "AP precision", "rec": "AP recall"}.items():
                    comb = f"{ds_name}_{conf_key}_{new_k}"
                    custom_total = force_precision(
                        sum(
                            float(val)
                            for i, val in enumerate(ap_data[k])
                            if ap_data["names"][str(i)] not in POSEVAL_FAULTY
                        )
                        / float(
                            sum(
                                1 if ap_data["names"][str(i)] not in POSEVAL_FAULTY else 0
                                for i, val in enumerate(ap_data[k])
                            )
                        )
                    )
                    csv_writer.writerow(
                        [comb, ds_name, conf_key, new_k, custom_total] + [force_precision(adk) for adk in ap_data[k]]
                    )
    replace_dots_with_commas(poseval_out_file)
    print(f"Wrote poseval results to: {poseval_out_file}")

    # ######### #
    # PT21 EVAL #
    # ######### #

    pt21_out_file = os.path.join(OUT_DIR, "./results_pt21.csv")
    with open(pt21_out_file, "w+", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=";")
        csv_writer.writerow(["Combined", "Dataset", "Key"] + PT21_METRICS)

        for BASE_DIR in BASE_DIRS:

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
                    values = [force_precision(v.strip()) for v in values[1:]]  # summary is empty
                    assert len(values) == len(PT21_METRICS)
                    comb = f"{ds_name}_{meth_key}"
                    csv_writer.writerow([comb, ds_name, meth_key] + values)
    replace_dots_with_commas(pt21_out_file)
    print(f"Wrote PT21 results to: {pt21_out_file}")

    # ########## #
    # DanceTrack #
    # ########## #

    dance_out_file = os.path.join(OUT_DIR, "./results_dance.csv")
    BASE_DIRS: list[str] = ["./data/DanceTrack/val/", "./data/DanceTrack/train/", "./data/DanceTrack/test/"]

    data: list[dict] = []

    for BASE_DIR in BASE_DIRS:
        dance_files = list(
            set(
                glob(f"{BASE_DIR}./results_*/eval_data/pedestrian_detailed.csv")
                + glob(f"{BASE_DIR}./results_*/*/eval_data/pedestrian_detailed.csv")
            )
        )
        for dance_file in dance_files:
            res_dir_name = os.path.relpath(os.path.dirname(os.path.dirname(dance_file)), start=BASE_DIR)
            data_part_name = os.path.basename(os.path.dirname(BASE_DIR))

            with open(dance_file, "r", encoding="utf-8") as in_file:
                csv_reader = csv.DictReader(in_file, delimiter=",", lineterminator="\r\n")
                line: dict
                for line in csv_reader:
                    ds_name = line["seq"]
                    # the file gets too big if we include all individual datasets
                    if ds_name != "COMBINED":
                        continue
                    comb = f"{data_part_name}_{res_dir_name}_{ds_name}"
                    line = {k: force_precision(v, 5) for k, v in line.items()}
                    d = {**{"Combined": comb, "Dataset": data_part_name, "Key": res_dir_name}, **line}
                    data.append(d)

    with open(dance_out_file, "w+", encoding="utf-8") as out_file:
        csv_writer = csv.DictWriter(out_file, fieldnames=list(data[0].keys()), delimiter=";", lineterminator="\n")
        csv_writer.writeheader()
        for d in data:
            csv_writer.writerow(dict(d))
    replace_dots_with_commas(dance_out_file)
    print(f"Wrote DanceTrack results to: {dance_out_file}")

    send_discord_notification("Finished writing results to csv")
