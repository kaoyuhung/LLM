import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, GenerationConfig
from engine.generate import generate
from tqdm import tqdm


@torch.no_grad
def evalGSM8K(
    nnodes,
    world_rank,
    local_rank,
    model_name,
    model_version,
    model,
    tokenizer,
    N_SHOT,
    max_new_tokens,
    T,
    P,
    COT_FLAG=True,
):

    # def generate(model, tokenizer, input_text, generate_kwargs):
    #     input_text = tokenizer(
    #         input_text,
    #         padding=False,
    #         add_special_tokens=True,
    #         return_tensors="pt",
    #     )
    #     input_ids = input_text.input_ids.to(model.device)
    #     attention_mask = input_text.attention_mask.to(model.device)
    #     output_ids = model.generate(
    #         input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs
    #     )
    #     response = []
    #     for i in range(output_ids.shape[0]):
    #         response.append(
    #             tokenizer.decode(
    #                 output_ids[i][input_ids.shape[1] :],
    #                 skip_special_tokens=True,
    #                 ignore_tokenization_space=True,
    #             )
    #         )

    #     if len(response) > 1:
    #         return response
    #     return response[0]

    sys.path.append("../GSM8K_eval")
    from GSM8K_eval.main import (
        build_prompt,
        clean_answer,
        is_correct,
        extract_answer_from_output,
    )
    from GSM8K_eval.utils import download_url, load_jsonl

    if "SLURM_JOB_ID" in os.environ:
        save_dir = (
            f"result/job{os.environ['SLURM_JOB_ID']}/eval_{model_name}_{model_version}"
        )
    else:
        if nnodes == 1:
            save_dir = f"result/eval_{model_name}_single_node/GSM8K/eval_{model_name}_{model_version}"
        else:
            save_dir = f"result/eval_{model_name}_multinode/GSM8K/eval_{model_name}_{model_version}"
    os.makedirs(save_dir, exist_ok=True)

    data_dir = Path("../dataset/GSM8K")
    test_filepath = os.path.join(data_dir, "gsm8k_test.jsonl")
    if not os.path.exists(test_filepath) and (
        world_rank == 0 or (local_rank == 0 and "SLURM_JOB_ID" not in os.environ)
    ):
        download_url(
            "https://raw.githubusercontent.com/openai/"
            "grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/"
            "grade_school_math/data/test.jsonl",
            data_dir,
        )
        os.rename(os.path.join(data_dir, "test.jsonl"), test_filepath)
    dist.barrier()

    list_data_dict = load_jsonl(test_filepath, instruction="question", output="answer")
    answers = []
    for sample in tqdm(list_data_dict):
        input_text = build_prompt(sample["instruction"], N_SHOT, COT_FLAG)
        # generate_kwargs = dict(max_new_tokens=256, top_p=0.95, temperature=0.8)
        # model_completion = generate(model, tokenizer, input_text, generate_kwargs)
        model_completion = generate(
            [input_text],
            tokenizer,
            model,
            max_new_tokens=max_new_tokens,
            batch_size=1,
            temperature=T,
            top_p=P,
            use_cache=True,
            eos_id=model.generation_config.eos_token_id,
        )[0]
        model_answer = clean_answer(model_completion)
        is_cor = is_correct(model_answer, sample["output"])
        answers.append(is_cor)

        if world_rank == 0:
            tqdm.write(
                f'Question: {sample["instruction"]}\n\n'
                f'Answers: {extract_answer_from_output(sample["output"])}\n\n'
                f"Model Answers: {model_answer}\n\n"
                f"Model Completion: {model_completion}\n\n"
                f"Is correct: {is_cor}\n\n"
            )
            tqdm.write(
                f"Num of total question: {len(answers)}, "
                f"Correct num: {sum(answers)}, "
                f"Accuracy: {float(sum(answers))/len(answers)}."
            )
        dist.barrier()

    if world_rank == 0:
        with open(os.path.join(save_dir, "results.txt"), "w") as f:
            for answer in answers:
                print(answer, file=f)

        with open(os.path.join(save_dir, "scores.txt"), "w") as f:
            print(
                f"Num of total question: {len(answers)}, "
                f"Correct num: {sum(answers)}, "
                f"Accuracy: {float(sum(answers))/len(answers)}.",
                file=f,
            )
    dist.barrier()
    return


@torch.no_grad()
def evalMMLU(
    nnodes,
    world_rank,
    local_rank,
    model_name,
    model_version,
    model,
    tokenizer,
    ntrain,
    dataset,
):
    sys.path.append("../llm_model_evaluation")
    from llm_model_evaluation.config.log_config import logging
    from llm_model_evaluation.evaluation_hf_testing import (
        get_subject,
        format_example,
        gen_prompt,
    )
    from llm_model_evaluation.categories import verify_categories
    from llm_model_evaluation.catogories_result_eval import (
        process_csv_file,
        find_main_category,
    )

    choices = ["A", "B", "C", "D"]

    def eval(subject, dev_df, test_df):
        """
        Evaluates the model on a given subject.

        Args:
        args (Namespace): Command line arguments.
        subject (str): The subject to evaluate on.
        model (AutoModelForCausalLM): The pre-trained model.
        tokenizer (AutoTokenizer): The tokenizer.
        dev_df (pd.DataFrame): The development set dataframe.
        test_df (pd.DataFrame): The test set dataframe.

        Returns:
        tuple: A tuple containing the correct answers array, accuracy, and probabilities array.
        """
        cors = []
        all_probs = []
        answers = choices[: test_df.shape[1] - 2]
        all_times = []
        all_preds = []

        for i in range(test_df.shape[0]):
            start_time = time.time()
            # get prompt and make sure it fits
            k = ntrain
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                model.device
            )

            while input_ids.shape[-1] > 2048:
                k -= 1
                train_prompt = gen_prompt(dev_df, subject, k)
                prompt = train_prompt + prompt_end
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                    model.device
                )

            label = test_df.iloc[i, test_df.shape[1] - 1]

            logits = model(input_ids=input_ids).logits[0, -1]

            probs = (
                torch.nn.functional.softmax(
                    torch.tensor(
                        [
                            logits[tokenizer("A").input_ids[-1]],
                            logits[tokenizer("B").input_ids[-1]],
                            logits[tokenizer("C").input_ids[-1]],
                            logits[tokenizer("D").input_ids[-1]],
                        ]
                    ).float(),
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

            cor = pred == label
            cors.append(cor)
            all_preds.append(pred)
            all_probs.append(probs)
            all_times.append(time.time() - start_time)

        acc = np.mean(cors)
        cors = np.array(cors)

        all_probs = np.array(all_probs)
        if world_rank == 0:
            logging.info("Average accuracy {:.3f} - {}".format(acc, subject))

        return cors, all_probs, all_preds, all_times

    """
    Main function to run the evaluation script.

    Args:
    args (Namespace): Command line arguments.
    """
    if "SLURM_JOB_ID" in os.environ:
        save_dir = (
            f"result/job{os.environ['SLURM_JOB_ID']}/eval_{model_name}_{model_version}"
        )
    else:
        if nnodes == 1:
            save_dir = f"result/eval_{model_name}_single_node/{dataset}/eval_{model_name}_{model_version}"
        else:
            save_dir = f"result/eval_{model_name}_multinode/{dataset}/eval_{model_name}_{model_version}"

    # Retrieve list of subjects
    data_dir = Path(f"../dataset/{dataset}")
    if not data_dir.exists() and (
        world_rank == 0 or ("SLURM_JOB_ID" not in os.environ and local_rank == 0)
    ):
        import shutil

        if dataset == "mmlu":
            import tarfile

            snapshot_download(
                repo_id="cais/mmlu",
                repo_type="dataset",
                allow_patterns=["data.tar"],
                local_dir="../dataset",
            )
            with tarfile.open(f"../dataset/data.tar", "r:") as tar:
                tar.extractall(path="../dataset")
            os.remove(f"../dataset/data.tar")

        elif dataset == "tmmluplus":
            snapshot_download(
                repo_id="ikala/tmmluplus",
                repo_type="dataset",
                allow_patterns=["data/*"],
                local_dir="../dataset",
            )

        shutil.rmtree(f"../dataset/.cache")
        os.rename(f"../dataset/data", data_dir)

        if dataset == "tmmluplus":
            for dir in ["auxiliary_train", "dev", "test", "val"]:
                os.mkdir(data_dir / dir)

            for filename in os.listdir(data_dir):
                file_path = os.path.join(data_dir, filename)

                if filename.endswith("train.csv"):
                    shutil.move(file_path, data_dir / "auxiliary_train" / filename)

                elif filename.endswith("dev.csv"):
                    shutil.move(file_path, data_dir / "dev" / filename)

                elif filename.endswith("test.csv"):
                    shutil.move(file_path, data_dir / "test" / filename)

                elif filename.endswith("val.csv"):
                    shutil.move(file_path, data_dir / "val" / filename)

    dist.barrier()

    if world_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        logging.info("===== [Start] Evaluation by huggingface model ===== ")
        start_time = time.time()
        old_checkpoint_time = start_time
        logging.info("<Spend Time> Starting time: {}".format(start_time))

    subjects = get_subject(data_dir)
    if dataset == "tmmluplus":
        subjects.remove("jce_humanities")

    # Loop through each subject in the 'subjects' list
    for subject in subjects:
        if world_rank == 0:
            logging.info("Start the subject: {}".format(subject))

        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )[:ntrain]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )

        dist.barrier()
        # Evaluate the model on the current subject's data
        cors, probs, all_preds, all_times = eval(subject, dev_df, test_df)

        if world_rank == 0:
            # Process and save the results
            test_df["{}_prediction".format(model_name)] = all_preds
            test_df["{}_correct".format(model_name)] = cors
            for j in range(probs.shape[1]):
                choice = choices[j]
                test_df["{}_choice{}_probs".format(model_name, choice)] = probs[:, j]
            test_df["{}_spend_time".format(model_name)] = all_times
            test_df.to_csv(
                os.path.join(
                    save_dir,
                    "{}.csv".format(subject),
                ),
                index=None,
            )
            # Logging the time spent on the current subject
            checkpoint_time = time.time()
            logging.info(
                "<Spend Time> In {}, spend time: {}.".format(
                    subject, checkpoint_time - old_checkpoint_time
                )
            )
            old_checkpoint_time = checkpoint_time

    if world_rank == 0:
        # Logging the total time spent
        end_time = time.time()
        logging.info(
            "<Spend Time> Total Spending Time: {}.".format(
                start_time, end_time, end_time - start_time
            )
        )
        logging.info("===== [Finish] Evaluation by huggingface model ===== ")

    if world_rank == 0:
        logging.info("===== Start the evaluate the category. ======")

        categories_mmlu, subcategories_mmlu, _, _ = verify_categories(dataset)

        results_dir = save_dir
        results = {
            "subcategories": {},
            "categories": {},
            "weighted_accuracy": 0,
            "cost_time": 0,
        }
        total_questions = 0
        total_accuracy = 0

        for file_name in os.listdir(results_dir):
            if file_name.endswith(".csv"):
                file_path = os.path.join(results_dir, file_name)
                subcategory_key = file_name.replace(".csv", "")
                if subcategory_key not in subcategories_mmlu:
                    continue  # Skip files without a corresponding key in subcategories_mmlu
                accuracy, time_spent = process_csv_file(file_path, model_name)

                # Aggregating results for subcategories
                broad_category = subcategories_mmlu[subcategory_key][0]

                results["subcategories"].setdefault(broad_category, []).append(accuracy)

                # Mapping broad category to main category
                if dataset == "mmlu":
                    main_category = find_main_category(broad_category, categories_mmlu)
                elif dataset == "tmmluplus":
                    main_category = find_main_category(subcategory_key, categories_mmlu)

                results["categories"].setdefault(main_category, []).append(accuracy)

                total_questions += len(pd.read_csv(file_path))
                total_accuracy += accuracy * len(pd.read_csv(file_path))
                results["cost_time"] += time_spent

        # Calculating averages for subcategories and categories
        for subcategory, accuracies in results["subcategories"].items():
            results["subcategories"][subcategory] = sum(accuracies) / len(accuracies)
        for category, accuracies in results["categories"].items():
            results["categories"][category] = sum(accuracies) / len(accuracies)

        results["weighted_accuracy"] = total_accuracy / total_questions

        # Saving the results to a JSON file
        output_file = os.path.join(results_dir, "processed_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

        logging.info(
            "===== Finish the evaluation, please check the result json file. ======"
        )
    dist.barrier()
    return


def getModelandTokenizeer(
    world_rank: int,
    world_size: int,
    local_rank: int,
    local_world_size: int,
    node_rank: int,
    nnodes,
    model_path: str,
    max_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    model_version: str,
):
    model_name = os.path.basename(model_path)
    model_path = Path(model_path)
    if not model_path.exists() and local_rank == 0:
        model_path.mkdir(parents=True)
        if model_name in [
            "deepseek-moe-16b-chat",
            "DeepSeek-V2-Lite",
            "DeepSeek-V2-Chat",
            "DeepSeek-R1",
        ]:
            repo_id = "deepseek-ai/" + model_name
        elif model_name == "Mixtral-8x7B-Instruct-v0.1":
            repo_id = "mistralai/" + model_name
        elif model_name in ["Qwen1.5-MoE-A2.7B-Chat", "Qwen2-57B-A14B-Instruct"]:
            repo_id = "Qwen/" + model_name

        snapshot_download(
            repo_id=repo_id,
            allow_patterns=["*.json", "model-*.safetensors", "*.py"],
            local_dir=model_path,
        )
    dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if model_name == "deepseek-moe-16b-chat":
        from engine.deepseekmoe.transformer import Transformer
    elif model_name == "DeepSeek-V2-Lite" or model_name == "DeepSeek-V2-Chat":
        from engine.deepseekv2.transformer import Transformer
    elif model_name == "DeepSeek-R1":
        from engine.deepseekv3.transformer import Transformer
    elif model_name == "Mixtral-8x7B-Instruct-v0.1":
        from engine.mixtral8x7Binstruct.transformer import Transformer

        tokenizer.pad_token = tokenizer.eos_token
    elif model_name in ["Qwen1.5-MoE-A2.7B-Chat", "Qwen2-57B-A14B-Instruct"]:
        from engine.qwenmoe.transformer import Transformer

    if model_version == "PP":
        model = Transformer.from_pretrained(
            model_path,
            pipeline_rank=world_rank,
            num_pipeline_ranks=world_size,
            torch_dtype=dtype,
            device_map=device,
        )
    elif model_version == "TP":
        assert world_size > 1
        model = Transformer.from_pretrained(
            model_path,
            tp_rank=world_rank,
            num_tp_ranks=world_size,
            tp_group=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
            torch_dtype=dtype,
            device_map=device,
        )
    elif model_version == "PP+TP":
        assert nnodes > 1
        assert local_world_size > 1
        model = Transformer.from_pretrained(
            model_path,
            pipeline_rank=node_rank,
            num_pipeline_ranks=nnodes,
            tp_rank=local_rank,
            num_tp_ranks=local_world_size,
            tp_group=dist.new_group(
                ranks=list(
                    range(
                        world_rank - local_rank,
                        world_rank - local_rank + local_world_size,
                    )
                ),
                backend="nccl",
            ),
            torch_dtype=dtype,
            device_map=device,
        )
    elif model_version == "EP":
        model = Transformer.from_pretrained(
            model_path,
            tp_rank=world_rank,
            num_tp_ranks=world_size,
            tp_group=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
            ep_rank=world_rank,
            num_ep_ranks=world_size,
            torch_dtype=dtype,
            device_map=device,
        )
    elif model_version == "PP+EP":
        assert nnodes > 1
        assert local_world_size > 1
        model = Transformer.from_pretrained(
            model_path,
            pipeline_rank=node_rank,
            num_pipeline_ranks=nnodes,
            tp_rank=local_rank,
            num_tp_ranks=local_world_size,
            tp_group=dist.new_group(
                ranks=list(
                    range(
                        world_rank - local_rank,
                        world_rank - local_rank + local_world_size,
                    )
                ),
                backend="nccl",
            ),
            ep_rank=local_rank,
            num_ep_ranks=local_world_size,
            torch_dtype=dtype,
            device_map=device,
        )
    elif model_version == "TP+EP":
        assert nnodes > 1
        assert local_world_size > 1
        model = Transformer.from_pretrained(
            model_path,
            tp_rank=world_rank,
            num_tp_ranks=world_size,
            tp_group=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
            ep_rank=node_rank,
            num_ep_ranks=nnodes,
            torch_dtype=dtype,
            device_map=device,
        )
    elif model_version == "TP+EP-1-1-2":
        assert nnodes == 1 and world_size == 3
        model = Transformer.from_pretrained(
            model_path,
            tp_rank=world_rank,
            num_tp_ranks=world_size,
            tp_group=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
            ep_rank=0 if world_rank == 0 else 1,
            num_ep_ranks=2,
            torch_dtype=dtype,
            device_map=device,
        )
    elif model_version == "TP+EP-1-2-2":
        assert nnodes == 1 and world_size == 4
        model = Transformer.from_folder(
            model_path,
            tp_rank=world_rank,
            num_tp_ranks=world_size,
            tp_group=dist.new_group(ranks=list(range(world_size)), backend="nccl"),
            ep_rank=world_rank // 2,
            num_ep_ranks=world_size // 2,
            torch_dtype=dtype,
            device_map=device,
        )

    if model_name in [
        "deepseek-moe-16b-chat",
        "DeepSeek-V2-Lite",
        "DeepSeek-V2-Chat",
        "DeepSeek-R1",
    ]:
        model.generation_config = GenerationConfig.from_pretrained(model_path)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return model, tokenizer
