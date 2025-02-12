import os
import re
import argparse
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


# def draw_C(folder, n_thread):
#     x = []
#     y_ref_dict = defaultdict(list)
#     y_miss_dict = defaultdict(list)
#     for subfolder in os.listdir(folder):
#         if not subfolder.startswith("N"):
#             continue
#         x.append(int(subfolder.split("_")[1]))
#         subfolder_path = os.path.join(folder, subfolder, f"OMP_{n_thread}")
#         for file in os.listdir(subfolder_path):
#             if not file.startswith("cache_test_"):
#                 continue
#             method, file_path = file.split(".")[0][len("cache_test_") :], os.path.join(
#                 subfolder_path, file
#             )

#             n_cache_ref, n_cache_miss = None, None
#             with open(file_path, "r") as file:
#                 lines = file.readlines()
#                 if not lines:
#                     raise ValueError("File is empty")

#                 for line in lines:
#                     line = line.strip()
#                     if "cache-references" in line:
#                         n_cache_ref = int(line.split(" ")[0].replace(",", ""))

#                     if "cache-misses" in line:
#                         n_cache_miss = int(line.split(" ")[0].replace(",", ""))
#             y_ref_dict[method].append(n_cache_ref / 10e6)
#             y_miss_dict[method].append(n_cache_miss / n_cache_ref)

#     x = np.array(x)
#     width = 0.18
#     sort_idx = np.argsort(x)
#     x = np.sort(x)

#     fig, ax = plt.subplots()
#     ax.set_title("#Cache-Reference Comparison")
#     ax.set_xticks(x)

#     ax.bar(
#         x - 2.5 * width + width / 2,
#         np.array(y_ref_dict["baseline"])[sort_idx],
#         label="baseline",
#         width=width,
#     )
#     ax.bar(
#         x - 1.5 * width + width / 2,
#         np.array(y_ref_dict["QuEST"])[sort_idx],
#         label="QuEST",
#         width=width,
#     )
#     ax.bar(
#         x - 0.5 * width + width / 2,
#         np.array(y_ref_dict["merged_static"])[sort_idx],
#         label="batched_static",
#         width=width,
#     )
#     ax.bar(
#         x + 0.5 * width + width / 2,
#         np.array(y_ref_dict["merged_dynamic"])[sort_idx],
#         label="batched_dynamic",
#         width=width,
#     )
#     ax.bar(
#         x + 1.5 * width + width / 2,
#         np.array(y_ref_dict["merged_sort_dynamic"])[sort_idx],
#         label="batched_sort_dynamic",
#         width=width,
#     )
#     ax.set_xlabel("Number of Qubits")
#     ax.set_ylabel("#cache-refences (M)")
#     ax.legend()
#     plt.savefig(f"{folder}/cache_ref_figure")

#     fig, ax = plt.subplots()
#     ax.set_title("Cache-Miss Ratio Comparison")
#     ax.set_xticks(x)

#     ax.bar(
#         x - 2.5 * width + width / 2,
#         np.array(y_miss_dict["baseline"])[sort_idx],
#         label="baseline",
#         width=width,
#     )
#     ax.bar(
#         x - 1.5 * width + width / 2,
#         np.array(y_miss_dict["QuEST"])[sort_idx],
#         label="QuEST",
#         width=width,
#     )
#     ax.bar(
#         x - 0.5 * width + width / 2,
#         np.array(y_miss_dict["merged_static"])[sort_idx],
#         label="batched_static",
#         width=width,
#     )
#     ax.bar(
#         x + 0.5 * width + width / 2,
#         np.array(y_miss_dict["merged_dynamic"])[sort_idx],
#         label="batched_dynamic",
#         width=width,
#     )
#     ax.bar(
#         x + 1.5 * width + width / 2,
#         np.array(y_miss_dict["merged_sort_dynamic"])[sort_idx],
#         label="batched_sort_dynamic",
#         width=width,
#     )
#     ax.set_xlabel("Number of Qubits")
#     ax.set_ylabel("cache-miss ratio")
#     ax.legend()
#     plt.savefig(f"{folder}/cache_miss_figure")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        type=str,
        default="",
        help="draw the figure showing comparison between configurations of different M",
    )

    args = parser.parse_args()

    if args.t:
        draw_throughput(args.t)
