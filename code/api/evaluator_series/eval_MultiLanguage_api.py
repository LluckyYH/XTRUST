import os
import argparse
import openai
import pandas as pd
import torch
import time
from evaluators.all_api import All_api_Evaluator

openai.api_key = "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"


choices = ["A", "B"]


def main(args):
    if (
        "ChatGPT-api" in args.model_name
        or "Davinci-api" in args.model_name
        or "GPT4-api" in args.model_name
        or "Geminipro-api" in args.model_name
        or "Qwen-api" in args.model_name
        or "Baichuan-api" in args.model_name
        or "Wenxin-api" in args.model_name
        or "ChatGLM-api" in args.model_name
    ):
        if args.cuda_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        device = torch.device("cuda")
        evaluator = All_api_Evaluator(
            choices=choices,
            k=args.ntrain,
            t=args.task,
            s=args.scenario,
            pt=args.prompt,
            es=args.evasive,
            l=args.language,
            model_name=args.model_name,
            device=device,
        )
    else:
        print("Unknown model name")
        return -1
    if not os.path.exists(rf"logs_{args.model_name}"):
        os.mkdir(rf"logs_{args.model_name}")

    run_date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    logs_model_language_dir = rf"logs_{args.model_name}/{args.language}"
    if not os.path.exists(logs_model_language_dir):
        os.mkdir(logs_model_language_dir)

    logs_model_language_task_dir = (
        rf"logs_{args.model_name}/{args.language}/{args.task}"
    )
    if not os.path.exists(logs_model_language_task_dir):
        os.mkdir(logs_model_language_task_dir)

    if args.prompt and args.prompt != 0:
        save_result_dir = os.path.join(
            logs_model_language_task_dir,
            f"{args.language}_{args.task}_{args.scenario}_{args.prompt}_{run_date}",
        )
    elif args.evasive and args.evasive != 0:
        save_result_dir = os.path.join(
            logs_model_language_task_dir,
            f"{args.language}_{args.task}_{args.scenario}_{args.evasive}_{run_date}",
        )
    else:
        save_result_dir = os.path.join(
            logs_model_language_task_dir,
            f"{args.language}_{args.task}_{args.scenario}_{run_date}",
        )
    if not os.path.exists(save_result_dir):
        os.mkdir(save_result_dir)
    # print(subject_name)

    # 指定可能的编码方式
    possible_encodings = ["utf-8", "GB2312", "gbk", "ISO-8859-1"]

    if args.task == "OOD" and args.few_shot:
        val_file_path = os.path.join(
            rf"../../data/val_{args.language}/{args.task}", f"{args.task}_val.csv"
        )
        # val_df = pd.read_csv(val_file_path, encoding="utf-8")
        # 尝试不同的编码方式
        for encoding in possible_encodings:
            try:
                val_df = pd.read_csv(val_file_path, encoding=encoding)
                print(f"File successfully read using {encoding} encoding.")
                break  # 如果成功读取，跳出循环
            except UnicodeDecodeError:
                print(
                    f"Failed to read file using {encoding} encoding. Trying next encoding."
                )
        dev_file_path = os.path.join(
            rf"../../data/dev_{args.language}", f"{args.task}_dev.csv"
        )
        for encoding in possible_encodings:
            try:
                dev_df = pd.read_csv(dev_file_path, encoding=encoding)
                print(f"File successfully read using {encoding} encoding.")
                break  # 如果成功读取，跳出循环
            except UnicodeDecodeError:
                print(
                    f"Failed to read file using {encoding} encoding. Trying next encoding."
                )
        evaluator.eval_MultiLanguage(
            args.task,
            val_df,
            few_shot=args.few_shot,
            dev_df=dev_df,
            save_result_dir=save_result_dir,
        )

    elif args.task == "Privacy" or args.task == "Machine_Ethics":
        file_list = os.listdir(
            f"../../data/val_{args.language}/{args.task}/{args.scenario}"
        )
        sorted_file_list = sorted(file_list, key=lambda x: x.lower())

        for file in sorted_file_list:
            if file.endswith("xlsx"):
                continue
            val_file_path = os.path.join(
                f"../../data/val_{args.language}/{args.task}",
                f"{args.scenario}/{file}",
            )

            for encoding in possible_encodings:
                try:
                    val_df = pd.read_csv(val_file_path, encoding=encoding)
                    print(f"File successfully read using {encoding} encoding.")
                    break  # 如果成功读取，跳出循环
                except UnicodeDecodeError:
                    print(
                        f"Failed to read file using {encoding} encoding. Trying next encoding."
                    )
            pii_name = val_file_path.split("/")[-1].replace("_val.csv", "")
            if args.scenario == "moral_judgement":
                if args.few_shot:
                    dev_file_path = os.path.join(
                        rf"../../data/dev_{args.language}",
                        file.replace("_val.csv", "_dev.csv"),
                    )

                    for encoding in possible_encodings:
                        try:
                            dev_df = pd.read_csv(dev_file_path, encoding=encoding)
                            print(f"File successfully read using {encoding} encoding.")
                            break  # 如果成功读取，跳出循环
                        except UnicodeDecodeError:
                            print(
                                f"Failed to read file using {encoding} encoding. Trying next encoding."
                            )
                    evaluator.eval_MultiLanguage(
                        subject_name=file.replace("_val.csv", ""),
                        test_df=val_df,
                        dev_df=dev_df,
                        few_shot=True,
                        save_result_dir=save_result_dir,
                        cot=args.cot,
                        pii=pii_name,
                        file=file.replace("_val.csv", ""),
                    )
                else:
                    evaluator.eval_MultiLanguage(
                        args.task,
                        test_df=val_df,
                        dev_df=None,
                        few_shot=args.few_shot,
                        save_result_dir=save_result_dir,
                        cot=args.cot,
                        pii=pii_name,
                        file=file.replace("_val.csv", ""),
                    )
            elif args.scenario == "PII" or args.scenario == "PrivacyEvents":
                evaluator.eval_MultiLanguage(
                    args.task,
                    test_df=val_df,
                    dev_df=None,
                    few_shot=args.few_shot,
                    save_result_dir=save_result_dir,
                    cot=args.cot,
                    pii=pii_name,
                    file=file.replace("_val.csv", ""),
                )
            else:
                evaluator.eval_MultiLanguage(
                    args.task,
                    test_df=val_df,
                    dev_df=None,
                    few_shot=args.few_shot,
                    save_result_dir=save_result_dir,
                    cot=args.cot,
                    file=file.replace("_val.csv", ""),
                )

    else:
        val_file_path = os.path.join(
            rf"../../data/val_{args.language}/{args.task}", f"{args.task}_val.csv"
        )

        for encoding in possible_encodings:
            try:
                val_df = pd.read_csv(val_file_path, encoding=encoding)
                print(f"File successfully read using {encoding} encoding.")
                break  # 如果成功读取，跳出循环
            except UnicodeDecodeError:
                print(
                    f"Failed to read file using {encoding} encoding. Trying next encoding."
                )
        evaluator.eval_MultiLanguage(
            args.task,
            val_df,
            few_shot=args.few_shot,
            dev_df=None,
            save_result_dir=save_result_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int)
    parser.add_argument(
        "--openai_key",
        type=str,
        default="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",
    )
    parser.add_argument("--few_shot", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="ChatGPT-api")
    parser.add_argument("--task", "-t", type=str, default="Illegal_activity")
    parser.add_argument("--prompt", "-pt", type=int, default=0)
    parser.add_argument("--evasive", "-es", type=int, default=0)
    parser.add_argument("--language", "-l", type=str, default="Korean")
    parser.add_argument("--scenario", "-s", type=str, default="")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--cuda_device", type=str)
    args = parser.parse_args()
    main(args)
