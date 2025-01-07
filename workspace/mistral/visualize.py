import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def draw_throughput(folder):
    model_name = folder.split("_")[-2]
    y_throughput = defaultdict(lambda: defaultdict(list))

    for file in os.listdir(folder):
        if not file.startswith("measure_"):
            continue

        approach = file.split("_")[-2]
        prefill_y, decode_y = [], []
        file_path = os.path.join(folder, file)
        with open(file_path, "r") as file:
            lines = file.readlines()
            if not lines:
                raise ValueError("File is empty")

            for line in lines:
                if line.strip().endswith("tokens/s"):
                    # print(line.strip())
                    # print(re.findall(r"\d+\.\d+", line.strip())[-2:])
                    prefill_t, decode_t = re.findall(r"\d+\.\d+", line.strip())[-2:]
                    prefill_y.append(round(float(prefill_t)))
                    decode_y.append(round(float(decode_t)))

        y_throughput["prefill"][approach] = prefill_y
        y_throughput["decode"][approach] = decode_y

        # print(y_prefill_throughput[approach], y_decode_throughput[approach])
        # exit()

    assert len(y_throughput["prefill"]) > 0
    x = [
        pow(2, i)
        for i in range(max([len(y) for y in y_throughput["prefill"].values()]))
    ]
    x_indices = range(len(x))
    x = [str(_) for _ in x]

    for stage, y in y_throughput.items():
        fig, ax = plt.subplots()
        ax.set_title(f"{stage} throughput comparison")
        ax.set_xticks(x_indices)
        ax.plot(
            x[: len(y["v0"])],
            y["v0"],
            label="v0",
            linewidth=1.5,
            marker="o",
        )
        ax.plot(
            x[: len(y["v1"])],
            y["v1"],
            label="v1",
            linewidth=1.5,
            marker="o",
        )
        ax.plot(
            x[: len(y["v2"])],
            y["v2"],
            label="v2",
            linewidth=1.5,
            marker="o",
        )
        ax.plot(
            x[: len(y["v3"])],
            y["v3"],
            label="v3",
            linewidth=1.5,
            marker="o",
        )
        ax.plot(
            x[: len(y["PP"])],
            y["PP"],
            label="PP",
            linewidth=1.5,
            marker="o",
        )
        ax.plot(
            x[: len(y["PP+TP-V1"])],
            y["PP+TP-V1"],
            label="PP+TP-V1",
            linewidth=1.5,
            marker="o",
        )
        # ax.plot(
        #     x[: len(y["PP+TP-V2"])],
        #     y["PP+TP-V2"],
        #     label="PP+TP-V2",
        #     linewidth=1.5,
        #     marker="o",
        # )
        # ax.plot(
        #     x[: len(y["PP+EP"])],
        #     y["PP+EP"],
        #     label="PP+EP",
        #     linewidth=1.5,
        #     marker="o",
        # )
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (tokens/s)")
        ax.legend()
        plt.savefig(f"{folder}/{stage}_figure")


def draw_C(folder, n_thread):
    x = []
    y_ref_dict = defaultdict(list)
    y_miss_dict = defaultdict(list)
    for subfolder in os.listdir(folder):
        if not subfolder.startswith("N"):
            continue
        x.append(int(subfolder.split("_")[1]))
        subfolder_path = os.path.join(folder, subfolder, f"OMP_{n_thread}")
        for file in os.listdir(subfolder_path):
            if not file.startswith("cache_test_"):
                continue
            method, file_path = file.split(".")[0][len("cache_test_") :], os.path.join(
                subfolder_path, file
            )

            n_cache_ref, n_cache_miss = None, None
            with open(file_path, "r") as file:
                lines = file.readlines()
                if not lines:
                    raise ValueError("File is empty")

                for line in lines:
                    line = line.strip()
                    if "cache-references" in line:
                        n_cache_ref = int(line.split(" ")[0].replace(",", ""))

                    if "cache-misses" in line:
                        n_cache_miss = int(line.split(" ")[0].replace(",", ""))
            y_ref_dict[method].append(n_cache_ref / 10e6)
            y_miss_dict[method].append(n_cache_miss / n_cache_ref)

    x = np.array(x)
    width = 0.18
    sort_idx = np.argsort(x)
    x = np.sort(x)

    fig, ax = plt.subplots()
    ax.set_title("#Cache-Reference Comparison")
    ax.set_xticks(x)

    ax.bar(
        x - 2.5 * width + width / 2,
        np.array(y_ref_dict["baseline"])[sort_idx],
        label="baseline",
        width=width,
    )
    ax.bar(
        x - 1.5 * width + width / 2,
        np.array(y_ref_dict["QuEST"])[sort_idx],
        label="QuEST",
        width=width,
    )
    ax.bar(
        x - 0.5 * width + width / 2,
        np.array(y_ref_dict["merged_static"])[sort_idx],
        label="batched_static",
        width=width,
    )
    ax.bar(
        x + 0.5 * width + width / 2,
        np.array(y_ref_dict["merged_dynamic"])[sort_idx],
        label="batched_dynamic",
        width=width,
    )
    ax.bar(
        x + 1.5 * width + width / 2,
        np.array(y_ref_dict["merged_sort_dynamic"])[sort_idx],
        label="batched_sort_dynamic",
        width=width,
    )
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("#cache-refences (M)")
    ax.legend()
    plt.savefig(f"{folder}/cache_ref_figure")

    fig, ax = plt.subplots()
    ax.set_title("Cache-Miss Ratio Comparison")
    ax.set_xticks(x)

    ax.bar(
        x - 2.5 * width + width / 2,
        np.array(y_miss_dict["baseline"])[sort_idx],
        label="baseline",
        width=width,
    )
    ax.bar(
        x - 1.5 * width + width / 2,
        np.array(y_miss_dict["QuEST"])[sort_idx],
        label="QuEST",
        width=width,
    )
    ax.bar(
        x - 0.5 * width + width / 2,
        np.array(y_miss_dict["merged_static"])[sort_idx],
        label="batched_static",
        width=width,
    )
    ax.bar(
        x + 0.5 * width + width / 2,
        np.array(y_miss_dict["merged_dynamic"])[sort_idx],
        label="batched_dynamic",
        width=width,
    )
    ax.bar(
        x + 1.5 * width + width / 2,
        np.array(y_miss_dict["merged_sort_dynamic"])[sort_idx],
        label="batched_sort_dynamic",
        width=width,
    )
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("cache-miss ratio")
    ax.legend()
    plt.savefig(f"{folder}/cache_miss_figure")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        type=str,
        default="",
        help="draw the figure showing comparison between configurations of different M",
    )
    # parser.add_argument(
    #     "-s",
    #     type=str,
    #     default="",
    #     help="draw the figure showing scalability result",
    # )
    # parser.add_argument(
    #     "-c",
    #     type=str,
    #     default="",
    #     help="draw the figure showing cache miss test result",
    # )
    # parser.add_argument(
    #     "-t",
    #     type=int,
    #     default=72,
    #     help="#threads to draw cache miss figure",
    # )
    args = parser.parse_args()

    if args.t:
        draw_throughput(args.t)

    # if args.s:
    #     draw_S(args.s)

    # if args.c:
    #     draw_C(args.c, args.t)
