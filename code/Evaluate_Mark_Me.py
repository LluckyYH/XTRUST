# Open the file
import pandas as pd
import os

tasks = ["Physical", "Hallucination", "Mental", "Misinformation"]
# folder_path = r"/home/liyahan/MultiLanguage/ceval-main/code/evaluator_series/logs_Baichuan-api"
# folder_path = r"/home/liyahan/MultiLanguage/ceval-main/code/evaluator_series/logs_Qwen"
flag = True
four_digits = ""
file_path = ""
task = ""
folder_paths = [
    "code\evaluator_series\logs_Baichuan-api",
    "code\evaluator_series\logs_ChatGLM-api",
    "code\evaluator_series\logs_Qwen-api",
    "code\evaluator_series\logs_Wenxin-api"
    "code\evaluator_series\logs_ChatGPT-api",
    "code\evaluator_series\logs_Davinci-api",
    "code\evaluator_series\logs_GPT4-api",
    "code\evaluator_series\logs_Geminipro-api",
    "code\Model\logs_ChatGLM_1",
    "code\Model\logs_ChatGLM_2",
    "code\Model\logs_ChatGLM_3",
    "code\Model\logs_Qwen",
    "code\Model\logs_InternLM",
]

# with open(txt_path, "w") as f:
# Machine_flag=[0,0,1,1]
index = 0
for folder_path in folder_paths:
    df = pd.DataFrame(
        # columns=["Physical", "Hallucination", "Mental_Health", "Misinformation"]
    )
    for root, dirs, files in os.walk(folder_path):
        # Iterate over subdirectories
        for sub_dir in dirs:
            # Construct the full path of the subdirectory
            sub_dir_path = os.path.join(root, sub_dir)
            for file in os.listdir(sub_dir_path):
                new_file = file
                for i in range(1, 6):
                    keyword = f"Machine_Ethics_evasive_sentence_{i}"
                    if keyword in file:
                        new_file = keyword

                for i in range(1, 6):
                    keyword = f"Machine_Ethics_jailbreaking_prompt_{i}"
                    if keyword in file:
                        new_file = keyword
                if not file.endswith("txt") and not file.endswith("csv"):
                    if "Machine_Ethics_moral_judgement_2" in file:

                        if flag:
                            new_file = "Machine_Ethics_moral_judgement_ZeroShot"
                            flag = False
                        else:
                            new_file = "Machine_Ethics_moral_judgement_FewShot"
                            flag = True

                target_path = os.path.join(sub_dir_path, new_file)
                # Check if the target file already exists
                if not os.path.exists(target_path):
                    os.rename(
                        os.path.join(sub_dir_path, file),
                        target_path,
                    )
                else:
                    # print(f"Skipping renaming. Target file {target_path} already exists.")
                    file_path = os.path.join(sub_dir_path, new_file)

                # file_path = os.path.join(sub_dir_path, new_file)
                # task = file.split("_")[0]
                task = file
                if (
                    "evasive_sentence" in task
                    and "OOD" not in task
                    or "jailbreaking" in task
                    or "moral" in task
                ):
                    # task = sub_dir_path.split('\\')[-1]
                    if file.endswith(".txt"):
                        # os.remove(file_path)
                        # continue
                        with open(file_path, "r") as file:
                            # print("file_path:",file_path)
                            if "evasive" in task:
                                task = (
                                    sub_dir_path.split("\\")[-1]
                                    + "_"
                                    + file_path.split("\\")[-1].split("_")[5]
                                    # + '_' +file_path.split("\\")[-1].split("_")[5]
                                )
                            elif "jailbreaking" in task:
                                task = (
                                    sub_dir_path.split("\\")[-1]
                                    + "_"
                                    + file_path.split("\\")[-1].split("_")[-5]
                                )
                            else:
                                task = (
                                    sub_dir_path.split("\\")[-1]
                                    + "_"
                                    + file_path.split("\\")[-1].split("_")[-3]
                                )

                            # Read the file content line by line
                            lines = file.readlines()

                            # Search for lines containing "positive response rate"
                            value = 0.0
                            for line in lines:

                                if "Accuracy" in line:
                                    value = line.split(":")[1].strip()

                                    # Extract the first four significant digits
                                value_str = str(value)
                                dot_index = value_str.index(".")
                                four_digits = value_str[: dot_index + 4]
                                # Convert back to float
                                # print("dad", four_digits)
                                four_digits = four_digits.replace("+", "").replace(
                                    "'", ""
                                )
                                # four_digits = float(four_digits.replace("'", ""))

                                result = float(four_digits) * 100

                                # Output the result
                                language = file_path.split("\\")[7]
                                # if task in tasks:
                                df.loc[language, task] = result
    print(df)
    file_name = folder_path.split("\\")[-1] + "_Me" + ".csv"
    print(file_name)
    df.to_csv(file_name, index=True)
