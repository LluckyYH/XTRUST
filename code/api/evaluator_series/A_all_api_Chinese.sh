#!/bin/bash
#SBATCH --job-name=api_Chinese_Job        # Job name
#SBATCH --output=output_api_Chinese.txt   # Redirect output to a file


model_list=("ChatGLM-api" "Qwen-api" "Wenxin-api")
lan_list=('Chinese')

# Illegal_activity evaluation
for model in "${model_list[@]}"; do
    for lan in "${lan_list[@]}"; do
        echo "$model $lan"
        python eval_MultiLanguage_api.py --model_name="$model" --language="$lan" --task="Illegal_activity" --scenario="Illegal_activity"
    done
done


# Hallucination  evaluation

for model in "${model_list[@]}"; do
    for lan in "${lan_list[@]}"; do
        echo "$model $lan"
        python eval_MultiLanguage_api.py --model_name="$model" --language="$lan" --task="Hallucination" --scenario="Hallucination"
    done
done


# Privacy evaluation

scenario_list=("PII" "PrivacyEvents")

for model in "${model_list[@]}"; do
    for lan in "${lan_list[@]}"; do
        for scenario in "${scenario_list[@]}"; do
            echo "$model $lan $scenario"
            python eval_MultiLanguage_api.py --model_name="$model" --language="$lan" --task="Privacy" --scenario="$scenario" 
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
                python eval_MultiLanguage_api.py --model_name="$model" --language="$lan" --task="Machine_Ethics" --scenario="$scenario" --few_shot=True --ntrain=5
            elif [ "$scenario" == "evasive_sentence" ]; then
                python eval_MultiLanguage_api.py --model_name="$model" --language="$lan" --task="Machine_Ethics" --scenario="$scenario" 
            else
                for pt in {1..5}; do
                    python eval_MultiLanguage_api.py --model_name="$model" --language="$lan" --task="Machine_Ethics" --scenario="$scenario" --prompt="$pt" 
                done
            fi
        done
    done
done


# Mental_Health evaluation
for model in "${model_list[@]}"; do
    for lan in "${lan_list[@]}"; do
        echo "$model $lan Mental_Health"
        python eval_MultiLanguage_api.py --model_name="$model" --language="$lan" --task="Mental_Health"  --scenario="Mental_Health" 
    done
done
python eval_MultiLanguage_api.py --model_name="GPT4-api" --language="Korean" --task="Physical_Health"  --scenario="Mental_Health" 

# Physical_Health evaluation
for model in "${model_list[@]}"; do
    for lan in "${lan_list[@]}"; do
        echo "$model $lan Physical_Health"
        python eval_MultiLanguage_api.py --model_name="$model" --language="$lan" --task="Physical_Health" --scenario="Physical_Health" 
    done
done


# OOD evaluation
for model in "${model_list[@]}"; do
    for lan in "${lan_list[@]}"; do
        echo "$model $lan "
        python eval_MultiLanguage_api.py --model_name="$model" --language="$lan" --task="OOD" 
        python eval_MultiLanguage_api.py --model_name="$model" --language="$lan" --task="OOD" --scenario="3shot" --few_shot=True --ntrain=3
    done
done



# Toxicity evaluation
scenario_list=('benign' 'adv')

for model in "${model_list[@]}"; do
    for lan in "${lan_list[@]}"; do
        for scenario in "${scenario_list[@]}"; do
            echo "$model $lan $scenario"
            python eval_MultiLanguage_api.py --model_name="$model" --language="$lan" --task="Toxicity" --scenario="$scenario" 
        done
    done
done

# Bias
scenario_list=('benign' 'untarget' 'target')

for model in "${model_list[@]}"; do
    for lan in "${lan_list[@]}"; do
      for scenario in "${scenario_list[@]}"; do
        echo "$model $lan $scenario"
        python eval_MultiLanguage_api.py --model_name="$model" --language="$lan" --task="Bias" --scenario="$scenario" 
      done
    done
done

# # Misformation evaluation
for model in "${model_list[@]}"; do
    for lang in "${lan_list[@]}"; do
        python eval_MultiLanguage_api.py --model_name="$model" --language="$lang" --task="Misinformation" --scenario="Misinformation" 
    done
done