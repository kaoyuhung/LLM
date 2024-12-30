import glob
import Path
import torch
from typing import Optional


class Mixtral8x7B_Partitioner:
    def __init__(
        self, model_path: str, ep_sz: Optional[int], tp_size: Optional[int]
    ) -> None:
        self.model_path = Path(model_path)
        # self.design_path = Path(design_path)
        # self.output_path = Path(output_path) if output_path else self.model_path
        self.model_config, self.expert_map, self.world_size, self.attn_tp_sizes = (
            self.get_configs()
        )

    def get_configs(self) -> tuple[dict, dict]:
        with open(self.model_path / "config.json", "r") as model_config_file:
            model_config = json.load(model_config_file)
        with open(self.design_path, "r") as design_file:
            design = json.load(design_file)

        expert_map = {}
        attn_tp_sizes = []
        interm_dim = model_config["intermediate_size"]
        next_expert, next_rank = 0, 0
        for d in design["design"]:
            n_experts, tp_size = d["n_experts"], d["tp_size"]
            step = -(interm_dim // -tp_size)  # ceil division
            m = (step, list(range(next_rank, next_rank + tp_size)))
            for ei in range(next_expert, next_expert + n_experts):
                expert_map[ei] = m

            if design["split_attn_weights"]:
                attn_tp_sizes.append(tp_size)

            next_expert += n_experts
            next_rank += tp_size

        assert next_expert == model_config["num_local_experts"]
        return model_config, expert_map, next_rank, attn_tp_sizes

    def load_weights(self) -> dict:
        weight_files = glob.glob(str(self.model_path / "consolidated.*.pt"))
        weights = {}
        for wf in weight_files:
            weights.update(torch.load(wf, weights_only=True, mmap=True))
        return weights

    def partition_expert_weights(self, ws: dict) -> None:
        interm_dim = self.model_config["intermediate_size"]
        partitions = [{} for _ in range(self.world_size)]

        for li in range(self.model_config["num_hidden_layers"]):
            w1: torch.Tensor = ws.pop(f"layers.{li}.block_sparse_moe.w1")
            w2: torch.Tensor = ws.pop(f"layers.{li}.block_sparse_moe.w2")
            w3: torch.Tensor = ws.pop(f"layers.{li}.block_sparse_moe.w3")

            for wi, w in enumerate([w1, w2, w3]):

                for ei, expert_slice in enumerate(torch.split(w, interm_dim)):
                    step, pis = self.expert_map[ei]

                    for pi, tp_slice in zip(pis, torch.split(expert_slice, step)):
                        partitions[pi][f"{li}.{ei}.w{wi + 1}"] = tp_slice.clone()

        for pi, partition in enumerate(partitions):
            torch.save(partition, self.output_path / f"experts-{pi}.pt")

        logging.info("finished partitioning expert weights")
        return ws

    def partition_non_expert_weights(self, ws: dict) -> None:
        for li in range(self.model_config["num_hidden_layers"]):
            ws[f"layers.{li}.feed_forward.gate.weight"] = ws.pop(
                f"layers.{li}.block_sparse_moe.gate.weight"
            )

        if len(self.attn_tp_sizes) > 0:
            non_attn_ws = {
                "tok_embeddings.weight": ws.pop("tok_embeddings.weight"),
                "norm.weight": ws.pop("norm.weight"),
                "output.weight": ws.pop("output.weight"),
            }
            for li in range(self.model_config["num_hidden_layers"]):
                non_attn_ws[f"layers.{li}.attention_norm.weight"] = ws.pop(
                    f"layers.{li}.attention_norm.weight"
                )
                non_attn_ws[f"layers.{li}.ffn_norm.weight"] = ws.pop(
                    f"layers.{li}.ffn_norm.weight"
                )
                non_attn_ws[f"layers.{li}.feed_forward.gate.weight"] = ws.pop(
                    f"layers.{li}.feed_forward.gate.weight"
                )

            n_attn_heads = self.model_config["num_attention_heads"]
            n_kv_heads = self.model_config["num_key_value_heads"]
            model_dim = self.model_config["hidden_size"]
            head_dim = model_dim // n_attn_heads

            for tp_size in self.attn_tp_sizes:
                partitions = [{} for _ in range(tp_size)]
                assert n_attn_heads % tp_size == 0
                assert n_kv_heads % tp_size == 0
                q_step = n_attn_heads * head_dim // tp_size
                kv_step = n_kv_heads * head_dim // tp_size
                o_step = model_dim // tp_size
                steps = [q_step, kv_step, kv_step, o_step]

                for li in range(self.model_config["num_hidden_layers"]):
                    wq: torch.Tensor = ws[f"layers.{li}.attention.wq.weight"]
                    wk: torch.Tensor = ws[f"layers.{li}.attention.wk.weight"]
                    wv: torch.Tensor = ws[f"layers.{li}.attention.wv.weight"]
                    wo: torch.Tensor = ws[f"layers.{li}.attention.wo.weight"].T

                    for wi, step, w in zip(
                        ["wq", "wk", "wv", "wo"], steps, [wq, wk, wv, wo]
                    ):

                        for partition, w_slice in zip(partitions, torch.split(w, step)):
                            partition[f"layers.{li}.attention.{wi}.weight"] = (
                                w_slice.clone() if wi != "wo" else w_slice.T.clone()
                            )

                for pi, partition in enumerate(partitions):
                    partition.update(non_attn_ws)
                    torch.save(
                        partition, self.output_path / f"non-experts-{tp_size}-{pi}.pt"
                    )

        else:
            torch.save(ws, self.output_path / f"non-experts-1-0.pt")

        logging.info("finished partitioning non-expert weights")
        return ws

    def start(self) -> None:
        ws = self.partition_expert_weights(self.load_weights())
        self.partition_non_expert_weights(ws)


# class Mixtral8x7B_Partitioner:

#     def __init__(self, model_path: str, design_path: str, output_path: str) -> None:
#         self.model_path = Path(model_path)
#         self.design_path = Path(design_path)
#         self.output_path = Path(output_path) if output_path else self.model_path
#         self.model_config, self.expert_map, self.world_size, self.attn_tp_sizes = (
#             self.get_configs()
#         )

#     def get_configs(self) -> tuple[dict, dict]:
#         with open(self.model_path / "config.json", "r") as model_config_file:
#             model_config = json.load(model_config_file)
#         with open(self.design_path, "r") as design_file:
#             design = json.load(design_file)

#         expert_map = {}
#         attn_tp_sizes = []
#         interm_dim = model_config["intermediate_size"]
#         next_expert, next_rank = 0, 0
#         for d in design["design"]:
#             n_experts, tp_size = d["n_experts"], d["tp_size"]
#             step = -(interm_dim // -tp_size)  # ceil division
#             m = (step, list(range(next_rank, next_rank + tp_size)))
#             for ei in range(next_expert, next_expert + n_experts):
#                 expert_map[ei] = m

#             if design["split_attn_weights"]:
#                 attn_tp_sizes.append(tp_size)

#             next_expert += n_experts
#             next_rank += tp_size

#         assert next_expert == model_config["num_local_experts"]
#         return model_config, expert_map, next_rank, attn_tp_sizes

#     def load_weights(self) -> dict:
#         weight_files = glob.glob(str(self.model_path / "consolidated.*.pt"))
#         weights = {}
#         for wf in weight_files:
#             weights.update(torch.load(wf, weights_only=True, mmap=True))
#         return weights

#     def partition_expert_weights(self, ws: dict) -> None:
#         interm_dim = self.model_config["intermediate_size"]
#         partitions = [{} for _ in range(self.world_size)]

#         for li in range(self.model_config["num_hidden_layers"]):
#             w1: torch.Tensor = ws.pop(f"layers.{li}.block_sparse_moe.w1")
#             w2: torch.Tensor = ws.pop(f"layers.{li}.block_sparse_moe.w2")
#             w3: torch.Tensor = ws.pop(f"layers.{li}.block_sparse_moe.w3")

#             for wi, w in enumerate([w1, w2, w3]):

#                 for ei, expert_slice in enumerate(torch.split(w, interm_dim)):
#                     step, pis = self.expert_map[ei]

#                     for pi, tp_slice in zip(pis, torch.split(expert_slice, step)):
#                         partitions[pi][f"{li}.{ei}.w{wi + 1}"] = tp_slice.clone()

#         for pi, partition in enumerate(partitions):
#             torch.save(partition, self.output_path / f"experts-{pi}.pt")

#         logging.info("finished partitioning expert weights")
#         return ws

#     def partition_non_expert_weights(self, ws: dict) -> None:
#         for li in range(self.model_config["num_hidden_layers"]):
#             ws[f"layers.{li}.feed_forward.gate.weight"] = ws.pop(
#                 f"layers.{li}.block_sparse_moe.gate.weight"
#             )

#         if len(self.attn_tp_sizes) > 0:
#             non_attn_ws = {
#                 "tok_embeddings.weight": ws.pop("tok_embeddings.weight"),
#                 "norm.weight": ws.pop("norm.weight"),
#                 "output.weight": ws.pop("output.weight"),
#             }
#             for li in range(self.model_config["num_hidden_layers"]):
#                 non_attn_ws[f"layers.{li}.attention_norm.weight"] = ws.pop(
#                     f"layers.{li}.attention_norm.weight"
#                 )
#                 non_attn_ws[f"layers.{li}.ffn_norm.weight"] = ws.pop(
#                     f"layers.{li}.ffn_norm.weight"
#                 )
#                 non_attn_ws[f"layers.{li}.feed_forward.gate.weight"] = ws.pop(
#                     f"layers.{li}.feed_forward.gate.weight"
#                 )

#             n_attn_heads = self.model_config["num_attention_heads"]
#             n_kv_heads = self.model_config["num_key_value_heads"]
#             model_dim = self.model_config["hidden_size"]
#             head_dim = model_dim // n_attn_heads

#             for tp_size in self.attn_tp_sizes:
#                 partitions = [{} for _ in range(tp_size)]
#                 assert n_attn_heads % tp_size == 0
#                 assert n_kv_heads % tp_size == 0
#                 q_step = n_attn_heads * head_dim // tp_size
#                 kv_step = n_kv_heads * head_dim // tp_size
#                 o_step = model_dim // tp_size
#                 steps = [q_step, kv_step, kv_step, o_step]

#                 for li in range(self.model_config["num_hidden_layers"]):
#                     wq: torch.Tensor = ws[f"layers.{li}.attention.wq.weight"]
#                     wk: torch.Tensor = ws[f"layers.{li}.attention.wk.weight"]
#                     wv: torch.Tensor = ws[f"layers.{li}.attention.wv.weight"]
#                     wo: torch.Tensor = ws[f"layers.{li}.attention.wo.weight"].T

#                     for wi, step, w in zip(
#                         ["wq", "wk", "wv", "wo"], steps, [wq, wk, wv, wo]
#                     ):

#                         for partition, w_slice in zip(partitions, torch.split(w, step)):
#                             partition[f"layers.{li}.attention.{wi}.weight"] = (
#                                 w_slice.clone() if wi != "wo" else w_slice.T.clone()
#                             )

#                 for pi, partition in enumerate(partitions):
#                     partition.update(non_attn_ws)
#                     torch.save(
#                         partition, self.output_path / f"non-experts-{tp_size}-{pi}.pt"
#                     )

#         else:
#             torch.save(ws, self.output_path / f"non-experts-1-0.pt")

#         logging.info("finished partitioning non-expert weights")
#         return ws

#     def start(self) -> None:
#         ws = self.partition_expert_weights(self.load_weights())
#         self.partition_non_expert_weights(ws)
