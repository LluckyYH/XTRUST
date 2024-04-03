#!/bin/bash
#SBATCH --job-name=model_Job        # Job name
#SBATCH --output=output_model.txt   # Redirect output to a file
#SBATCH --time=7-00:00:00              # Execution time limit is one week
#SBATCH --ntasks=1                     # Number of tasks is 1
#SBATCH --mem=60G                      # Memory per task is 60GB
#SBATCH --partition=gpujl              # Queue name is gpujl
#SBATCH --gres=gpu:1                   # If needed, use 1 GPU

model_list=('ChatGLM_1' 'ChatGLM_2' 'ChatGLM_3' 'InternLM' 'Qwen')
lan_list=('Chinese')

# Illegal_activity evaluation
for model in "${model_list[@]}"; do
    for lan in "${lan_list[@]}"; do
        echo "$model $lan"
        python eval_MultiLanguage_model.py --model_name="$model" --language="$lan" --task="Illegal_activity" --scenario="Illegal_activity"
    done
done


# Hallucination  evaluation

for model in "${model_list[@]}"; do
    for lan in "${lan_list[@]}"; do
        echo "$model $lan"
        python eval_MultiLanguage_model.py --model_name="$model" --language="$lan" --task="Hallucination" --scenario="Hallucination"
    done
done

# Misinformation  evaluation
for model in "${model_list[@]}"; do
    for lan in "${lan_list[@]}"; do
        echo "$model $lan"
        python eval_MultiLanguage_model.py --model_name="$model" --language="$lan" --task="Misinformation" --scenario="Misinformation"
    done
done


# # # Privacy evaluation
scenario_list=("PII" "PrivacyEvents")

for model in "${model_list[@]}"; do
    for lan in "${lan_list[@]}"; do
        for scenario in "${scenario_list[@]}"; do
            echo "$model $lan $scenario"
            python eval_MultiLanguage_model.py --model_name="$model" --language="$lan" --task="Privacy" --scenario="$scenario"
        done
    done
done


# Machine_Ethics evaluation
for model in "${model_list[@]}"; do
    for lan in "${lan_list[@]}"; do
        echo "$model $lan"

        scenarios=("evasive_sentence" "jailbreaking_prompt" "moral_judgement")
        for scenario in "${scenarios[@]}"; do
            if [ "$scenario" == "moral_judgement" ]; then
                python eval_MultiLanguage_model.py --model_name="$model" --language="$lan" --task="Machine_Ethics" --scenario="$scenario"
                python eval_MultiLanguage_model.py --model_name="$model" --language="$lan" --task="Machine_Ethics" --scenario="$scenario" --few_shot=True --ntrain=5
            elif [ "$scenario" == "evasive_sentence" ]; then
                for es in {1..5}; do
                    python eval_MultiLanguage_model.py --model_name="$model" --language="$lan" --task="Machine_Ethics" --scenario="$scenario" --evasive="$es"
                done
            else
                for pt in {1..5}; do
                    python eval_MultiLanguage_model.py --model_name="$model" --language="$lan" --task="Machine_Ethics" --scenario="$scenario" --prompt="$pt"
                done
            fi
        done
    done
done


# # Mental_Health evaluation
for model in "${model_list[@]}"; do
    for lan in "${lan_list[@]}"; do
        echo "$model $lan Mental_Health"
        python eval_MultiLanguage_model.py --model_name="$model" --language="$lan" --task="Mental_Health"  --scenario="Mental_Health"
    done
done

# # Physical_Health evaluation
for model in "${model_list[@]}"; do
    for lan in "${lan_list[@]}"; do
        echo "$model $lan Physical_Health"
        python eval_MultiLanguage_model.py --model_name="$model" --language="$lan" --task="Physical_Health" --scenario="Physical_Health"
    done
done


# # # OOD evaluation
for model in "${model_list[@]}"; do
    for lan in "${lan_list[@]}"; do
        echo "$model $lan "
        python eval_MultiLanguage_model.py --model_name="$model" --language="$lan" --task="OOD"
        python eval_MultiLanguage_model.py --model_name="$model" --language="$lan" --task="OOD"  --few_shot=True --ntrain=3 --scenario="3shot"
    done
done


# # Toxicity evaluation
scenario_list=('benign' 'adv')

for model in "${model_list[@]}"; do
    for lan in "${lan_list[@]}"; do
        for scenario in "${scenario_list[@]}"; do
            echo "$model $lan $scenario"
            python eval_MultiLanguage_model.py --model_name="$model" --language="$lan" --task="Toxicity" --scenario="$scenario"
        done
    done
done

# Bias
scenario_list=('benign' 'untarget' 'target')

for model in "${model_list[@]}"; do
    for lan in "${lan_list[@]}"; do
      for scenario in "${scenario_list[@]}"; do
        echo "$model $lan $scenario"
        python eval_MultiLanguage_model.py --model_name="$model" --language="$lan" --task="Bias" --scenario="$scenario"
      done
    done
done
