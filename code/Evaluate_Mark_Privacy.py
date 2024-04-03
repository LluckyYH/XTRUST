import pandas as pd
import os


def extract_value_from_file(file_path, task):
    value = None
    if file.endswith(".txt"):
        with open(file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                if "positive response rate" in line:
                    value = float(line.split(":")[1].strip()) * 100
                    break
    return value


def process_directory(folder_path):
    df = pd.DataFrame()
    for root, dirs, files in os.walk(folder_path):
        for sub_dir in dirs:
            sub_dir_path = os.path.join(root, sub_dir)
            for file in os.listdir(sub_dir_path):
                task = file.split("_")[0]
                if task == "moral":
                    task = "Machine"
                if task not in [
                    "Physical",
                    "Hallucination",
                    "Mental",
                    "Misinformation",
                ]:
                    continue
                file_path = os.path.join(sub_dir_path, file)
                value = extract_value_from_file(file_path, task)
                if value is not None:
                    language = file_path.split("\\")[-2].split("_")[0]
                    df.loc[language, task] = value
    return df


def main():
    folder_paths = [
        r"code\evaluator_series\logs_Baichuan-api",
        r"code\evaluator_series\logs_ChatGLM-api",
        # Add more folder paths here
    ]
    for folder_path in folder_paths:
        df = process_directory(folder_path)
        file_name = folder_path.split("\\")[-1] + "_Privacy.csv"
        df.to_csv(file_name, index=True)


if __name__ == "__main__":
    main()
