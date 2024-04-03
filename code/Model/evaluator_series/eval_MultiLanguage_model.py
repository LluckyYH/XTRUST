
# -*- coding: utf-8 -*-
import csv
import os
import argparse
import subprocess
import time
import pandas as pd
import torch
from evaluators.ChatGLM_1 import ChatGLM_1_Evaluator
from evaluators.ChatGLM_2 import ChatGLM_2_Evaluator
from evaluators.ChatGLM_3 import ChatGLM_3_Evaluator
from evaluators.InternLM import InternLM_Evaluator
from evaluators.Qwen import Qwen_Evaluator


choices = ["A", "B", "C", "D"]


def main(args):
    
    if "ChatGLM_1" in args.model_name:
        if args.cuda_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        device = torch.device("cuda")
        evaluator = ChatGLM_1_Evaluator(
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
    elif "ChatGLM_2" in args.model_name:
        if args.cuda_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        device = torch.device("cuda")
        evaluator = ChatGLM_2_Evaluator(
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
    elif "ChatGLM_3" in args.model_name:
        if args.cuda_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        device = torch.device("cuda")
        evaluator = ChatGLM_3_Evaluator(
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
    elif "Qwen" in args.model_name:
        if args.cuda_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        device = torch.device("cuda")
        evaluator = Qwen_Evaluator(
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
    elif "InternLM" in args.model_name:
        if args.cuda_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        device = torch.device("cuda")
        evaluator = InternLM_Evaluator(
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

    if args.task == "OOD" and args.few_shot:
        val_file_path = os.path.join(
            rf"../../data/val_{args.language}/{args.task}", f"{args.task}_val.csv"
        )
        val_df = pd.read_csv(val_file_path, encoding="utf-8")
        dev_file_path = os.path.join(
            rf"../../data/dev_{args.language}", f"{args.task}_dev.csv"
        )
        dev_df = pd.read_csv(dev_file_path, encoding="utf-8")
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

            val_df = pd.read_csv(val_file_path, encoding="utf-8")
            pii_name = val_file_path.split("/")[-1].replace("_val.csv", "")
            if args.scenario == "moral_judgement":
                if args.few_shot:
                    dev_file_path = os.path.join(
                        rf"../../data/dev_{args.language}",
                        file.replace("_val.csv", "_dev.csv"),
                    )
                    dev_df = pd.read_csv(dev_file_path, encoding="utf-8")
                    evaluator.eval_MultiLanguage(
                        subject_name=file.replace("_val.csv", ""),
                        test_df=val_df,
                        dev_df=dev_df,
                        few_shot=True,
                        save_result_dir=save_result_dir,
                        cot=args.cot,
                        pii=pii_name,
                        file=file.replace("_val.csv", "")
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
                        file=file.replace("_val.csv", "")
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
                    file=file.replace("_val.csv", "")
                )
            else:
                evaluator.eval_MultiLanguage(
                    args.task,
                    test_df=val_df,
                    dev_df=None,
                    few_shot=args.few_shot,
                    save_result_dir=save_result_dir,
                    cot=args.cot,
                    file=file.replace("_val.csv", "")
                )

    else:
        val_file_path = os.path.join(
            rf"../../data/val_{args.language}/{args.task}", f"{args.task}_val.csv"
        )
        val_df = pd.read_csv(val_file_path, encoding="utf-8")

        evaluator.eval_MultiLanguage(
            args.task,
            val_df,
            few_shot=args.few_shot,
            dev_df=None,
            save_result_dir=save_result_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ntrain", "-k", type=int,default=5)
    parser.add_argument("--ntrain", "-k", type=int)
    # parser.add_argument("--openai_key",type=str,default="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")
    # parser.add_argument("--few_shot", type=bool,default=True)
    parser.add_argument("--few_shot", type=bool,default=False)
    parser.add_argument("--model_name", type=str, default="Qwen") # 
    parser.add_argument("--task", "-t", type=str, default="Bias")
    parser.add_argument("--prompt", "-pt", type=int,default=0)
    parser.add_argument("--evasive", "-es", type=int,default=0)
    parser.add_argument("--language", "-l", type=str, default="Chinese")
    parser.add_argument("--scenario", "-s", type=str, default="untarget")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--cuda_device", type=str)
    args = parser.parse_args()
    main(args)
