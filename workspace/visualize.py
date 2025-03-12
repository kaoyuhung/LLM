import os
import re
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def draw_throughput(folder):
    y_throughput = defaultdict(lambda: defaultdict(list))
    total_time = defaultdict(list)

    for file in os.listdir(folder):
        if not file.startswith("measure_"):
            continue

        if file.endswith("single_node.txt"):
            approach = file.split("_")[-3]
        else:
            approach = file.split("_")[-2]
            if "node" in approach:
                if approach == "node0":
                    approach = file.split("_")[-3]
                else:
                    continue

        prefill_y, decode_y, total_t = [], [], []
        file_path = os.path.join(folder, file)
        with open(file_path, "r") as file:
            lines = file.readlines()
            if not lines:
                raise ValueError("File is empty")

            for line in lines:
                if line.strip().endswith("tokens/s"):
                    prefill_et, decode_et, prefill_t, decode_t = re.findall(
                        r"\d+\.\d+", line.strip()
                    )
                    total_t.append(round(float(prefill_et)) + round(float(decode_et)))
                    prefill_y.append(round(float(prefill_t)))
                    decode_y.append(round(float(decode_t)))

        y_throughput["Prefill"][approach] = prefill_y
        y_throughput["Decode"][approach] = decode_y
        total_time[approach] = total_t

    assert len(y_throughput["Prefill"]) > 0
    x = [
        pow(2, i)
        for i in range(max([len(y) for y in y_throughput["Prefill"].values()]))
    ]
    x_indices = range(len(x))
    x = [str(_) for _ in x]
    versions = ["v0", "v1", "v2", "v3", "PP", "TP", "TP+EP", "EP", "PP+TP", "PP+EP"]

    fig, ax = plt.subplots()
    ax.set_title(f"elapsed time comparison")
    ax.set_xticks(x_indices)
    for version in versions:
        if version in total_time:
            ax.plot(
                x[: len(total_time[version])],
                total_time[version],
                label=version,
                linewidth=1.5,
                marker="o",
            )
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Elapsed Time(s) (Lower is better)")
    ax.legend()
    plt.savefig(f"{folder}/time_figure")

    for stage, y in y_throughput.items():
        fig, ax = plt.subplots()
        ax.set_title(f"{stage} Throughput Comparison")
        ax.set_xticks(x_indices)

        for version in versions:
            if version in y:
                ax.plot(
                    x[: len(y[version])],
                    y[version],
                    label=version,
                    linewidth=1.5,
                    marker="o",
                )

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (tokens/s)")
        ax.legend()
        plt.savefig(f"{folder}/{stage}_figure")


def draw_ACC(folder, dataset):
    x = []
    y_dict = dict()
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path) and subfolder.startswith("eval_"):
            model = subfolder.split("_")[1]
            with open(
                os.path.join(
                    subfolder_path,
                    dataset,
                    f"eval_{model}_EP",
                    "processed_results.json",
                )
            ) as f:
                acc_dict = json.load(f)["subcategories"]
            y_dict[model] = acc_dict

    x_labels = sorted(list(list(y_dict.values())[0].keys()))
    x = np.arange(len(x_labels)) * 1.2
    width = 0.15

    fig, ax = plt.subplots(figsize=(28, 6))
    ax.set_title(f"Accuracy on {dataset.upper()} benchmark")
    ax.set_xticks(x, x_labels)

    for i, (model, val) in enumerate(y_dict.items()):
        y = [val[label] for label in x_labels]
        ax.bar(
            x + (i * width) - (2 * width),
            np.array(y),
            label=model,
            width=width,
        )

    ax.set_xlabel("category")
    ax.set_ylabel("accuarcy")
    ax.legend()
    plt.savefig(f"{folder}/Accuarcy_{dataset.upper()}")

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        type=str,
        default="",
        help="draw the figure showing comparison between configurations of different M",
    )
    parser.add_argument(
        "-a",
        type=str,
        default="",
        help="draw the figure showing accuracy",
    )
    parser.add_argument(
        "-d",
        type=str,
        default="mmlu",
        help="dataset",
    )

    args = parser.parse_args()

    if args.t:
        draw_throughput(args.t)

    if args.a:
        draw_ACC(args.a, args.d)
