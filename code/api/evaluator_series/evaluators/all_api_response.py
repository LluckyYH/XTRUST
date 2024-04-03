import json
from time import sleep
import openai
import requests
import tiktoken
import os
from http import HTTPStatus
import dashscope
from dashscope import Generation
import zhipuai
import requests
from requests.exceptions import Timeout
from http import HTTPStatus
import time as sleep

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# zhipuai.api_key = "de113c93772b136f1582e27c3c3380cd.mn88EtGmZTCNxqix" #wy  #ChatGLM
zhipuai.api_key = "670a3425af6e0ebdd6f8ff21eba8d5a6.bGbu749s6GSo5jeb"
# Geminipro_apikey = "AIzaSyDuAFvs9O9-xm31kmxuPdwa6Hb0mSEKJ6s"
# Geminipro_apikey = "AIzaSyAf6sqrzN9j4PrkWprh83H417rv1AGV0NQ"
Geminipro_apikey = "AIzaSyC6gJGt0_5uoJnnxTzxRRxTCjAwnLi4huA"
# dashscope.api_key = "sk-ed8c40fd2ec04c3fa34dc1e79f39c10f"  # wy
dashscope.api_key = "sk-d0eb9f3efda042e58918ba993fea547b"  # 百川和千问  llama2
Wenxin_API_KEY = "vaAump9oKOXX07h7ulj0KgIZ"
Wenxin_SECRET_KEY = "UeA8C6tpm6A96Vq51hZV6EpnOd0TqAiR"


def get_GPT4_response(prompt, history):
    messages = []

    if history != []:
        for t in history:
            if isinstance(t, tuple):
                messages.append({"role": "user", "content": str(t[0])})
                messages.append({"role": "assistant", "content": str(t[1])})
            else:
                t = history
                messages.append({"role": "user", "content": str(t[0])})
                messages.append({"role": "assistant", "content": str(t[1])})
    messages.append({"role": "user", "content": str(prompt)})
    response = None
    timeout_counter = 0
    while response is None and timeout_counter <= 30:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=messages,
            )
        except Exception as msg:
            if "timeout=600" in str(msg):
                timeout_counter += 1
            print(msg)
            sleep(5)
            continue

    if response == None:
        response = ""
    else:
        response = response["choices"][0]["message"]["content"]
    return response


def get_ChatGPT_response(prompt, history):
    messages = []
    if history != []:
        for t in history:
            if isinstance(t, tuple):
                messages.append({"role": "user", "content": str(t[0])})
                messages.append({"role": "assistant", "content": str(t[1])})
            else:
                t = history
                messages.append({"role": "user", "content": str(t[0])})
                messages.append({"role": "assistant", "content": str(t[1])})
    messages.append({"role": "user", "content": str(prompt)})
    response = None
    timeout_counter = 0
    while response is None and timeout_counter <= 30:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106",
                messages=messages,
            )
        except Exception as msg:
            if "timeout=600" in str(msg):
                timeout_counter += 1
            print(msg)
            sleep(5)
            continue

    if response == None:
        response = ""
    else:
        response = response["choices"][0]["message"]["content"]
    return response


def log_skipped_prompt(skipped_prompt, index):
    with open("Geminipro_skipped_prompt_log.txt", "a", encoding="utf-8") as file:
        file.write(f"{index}: {skipped_prompt}\n")


def log_unsafe_content(unsafe_content, index):
    with open("Geminipro_unsafe_content_log.txt", "a", encoding="utf-8") as file:
        file.write(f"{index}: {json.dumps(unsafe_content, ensure_ascii=False)}\n")


global global_index
global_index = 0


def get_Geminipro_response(prompt, history):
    global global_index

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={Geminipro_apikey}"
    headers = {"Content-Type": "application/json"}
    messages = []

    if history != []:
        for t in history:
            if isinstance(t, tuple):
                messages.append({"role": "user", "parts": [{"text": str(t[0])}]})
                messages.append({"role": "model", "parts": [{"text": str(t[1])}]})
            else:
                t = history
                messages.append({"role": "user", "parts": [{"text": str(t[0])}]})
                messages.append({"role": "model", "parts": [{"text": str(t[1])}]})

    messages.append({"role": "user", "parts": [{"text": str(prompt)}]})

    # 如果有历史消息，将其添加到请求中
    if history != []:
        # result_string = '\n'.join([f'{t[0]}\t  Answer：{t[1]}' if isinstance(t, tuple) else str(t) for t in history])
        # prompt = result_string + "\n\n" + prompt
        # print(prompt)
        data = {"contents": messages}
    else:
        data = {"contents": [{"parts": [{"text": prompt}]}]}

    response = None
    timeout_counter = 0
    while response is None and timeout_counter <= 30:
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            sleep(3)
            response_json = response.json()

            if "candidates" in response_json:
                # 如果有 candidates 属性
                candidate_content = response_json["candidates"][0]["content"]["parts"][
                    0
                ]["text"]
                response = candidate_content
                # log_generated_content(candidate_content, global_index)
                global_index += 1  # 每次记录后递增 index
            elif (
                "promptFeedback" in response_json
                and "blockReason" in response_json["promptFeedback"]
            ):
                # 如果有 blockReason 属性，表示被跳过了
                block_reason = response_json["promptFeedback"]["blockReason"]
                log_skipped_prompt(prompt, global_index)
                global_index += 1  # 每次记录后递增 index
                response = ""
            else:
                response = ""
        except requests.Timeout:
            print("Timeout occurred. Retrying...")
            timeout_counter += 1
            sleep(5)
        except Exception as e:
            # print(f"An exception occurred: {e}")

            if "timeout=600" in str(e):
                print("Timeout occurred. Retrying...")
                timeout_counter += 1
            else:
                print("Unhandled exception. Retrying...")

            sleep(5)
            continue

    if response is None:
        response = ""

    return response


# def num_tokens_from_message(message, model="davinci-002"):
#     encoding = tiktoken.encoding_for_model(model)
#     num_tokens = len(encoding.encode(message))
#     return num_tokens


# def truncate_message(prompt1, prompt2, model="davinci-002", max_truncation_attempts=10):
#     encoding = tiktoken.encoding_for_model(model)
#     combined_length = num_tokens_from_message(prompt1 + prompt2, model)
#     if combined_length > 2033:
#         truncation_length = 2033 - num_tokens_from_message(prompt2, model)
#         for _ in range(max_truncation_attempts):
#             if num_tokens_from_message(prompt1, model) <= truncation_length:
#                 break
#             prompt1 = " ".join(prompt1.split()[:-1])  # Trim one word from `prompt1`
#         if num_tokens_from_message(prompt1, model) > truncation_length:
#             raise ValueError("Unable to truncate prompt to fit model's max length")
#     prompt = prompt1 + prompt2
#     return prompt


def get_Davinci_response(prompt, history):
    result_string = ""
    if history != []:
        result_string = "\n".join([f'({t[0]}, "{t[1]}")' for t in history])
    # prompt = truncate_message(result_string, prompt)
    # prompt = truncate_message(result_string, prompt)

    response = None
    timeout_counter = 0
    while response is None and timeout_counter <= 30:
        try:
            response = openai.Completion.create(
                model="davinci-002",
                prompt=prompt,
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
        except Exception as msg:
            if "timeout=600" in str(msg):
                timeout_counter += 1
            print(msg)
            sleep(5)
            continue
    if response is None:
        response = ""
    else:
        response = response["choices"][0]["text"].strip()

    return response


# def get_Baichuan_response(prompt, history):
#     messages = []

#     # 构建消息历史
#     if history != []:
#         for t in history:
#             if isinstance(t, tuple):
#                 messages.append({"role": "user", "content": str(t[0])})
#                 messages.append({"role": "assistant", "content": str(t[1])})
#             else:
#                 t = history
#                 messages.append({"role": "user", "content": str(t[0])})
#                 messages.append({"role": "assistant", "content": str(t[1])})
#     messages.append({"role": "user", "content": str(prompt)})

#     response = None
#     timeout_counter = 0
#     flag = True
#     while response is None and timeout_counter <= 30:
#         try:
#             response = dashscope.Generation.call(
#                 model="baichuan2-7b-chat-v1",
#                 messages=messages,
#                 result_format="message",  # set the result to be "message" format.
#             )
#             # 检查是否是速率限制错误
#             if response.message:
#                 while "Requests rate limit exceeded" in response.message:
#                     sleep(300)
#                     # print("Rate limit exceeded. Retrying...")
#                     # sleep(300)
#                     # if response.status_code == HTTPStatus.OK:
#                     #     if response.output.choices[0]["message"]["content"] is not None:
#                     #         response = response.output.choices[0]["message"]["content"].strip()
#                     #     else:
#                     #         response = "  "
#                     # print(response)
#                     # else:
#                     #     # 检查是否是速率限制错误
#                     #     if response.message and "Throttling.RateQuota" in response.message:
#                     #         print("Rate limit exceeded. Retrying...")
#                     #         sleep(300)
#                     # else:
#                     #     print(
#                     #         "Request id: %s, Status code: %s, error code: %s, error message: %s"
#                     #         % (
#                     #             response.request_id,
#                     #             response.status_code,
#                     #             response.code,
#                     #             response.message,
#                     #         )
#                     #     )

#             if response.status_code == HTTPStatus.OK:
#                 if response.output.choices[0]["message"]["content"] is not None:
#                     response = response.output.choices[0]["message"][
#                         "content"
#                     ].strip()
#                 else:
#                     response = "  "

#         except Timeout:
#             print("Timeout occurred. Retrying...")
#             timeout_counter += 1
#             sleep(5)
#         # except Exception as e:
#         #     # 特定的错误处理
#         #     if "timeout=600" in str(e):
#         #         print("Timeout occurred. Retrying...")
#         #         timeout_counter += 1
#         #     elif "Throttling.RateQuota" in str(e):
#         #         print("Rate limit exceeded. Retrying...")
#         #         sleep(10)
#         #     else:
#         #         print("Unhandled exception. Retrying...")

#         #     sleep(5)
#         #     continue

#     if response is None:
#         response = ""

#     return response

import time
from http import HTTPStatus
from dashscope import Generation
from requests.exceptions import Timeout


def get_Baichuan_response(prompt, history):
    messages = []

    # 构建消息历史
    if history:
        for t in history:
            if isinstance(t, tuple):
                messages.append({"role": "user", "content": str(t[0])})
                messages.append({"role": "assistant", "content": str(t[1])})
            else:
                messages.append({"role": "user", "content": str(t[0])})
                messages.append({"role": "assistant", "content": str(t[1])})

    messages.append({"role": "user", "content": str(prompt)})

    response = None
    timeout_counter = 0

    while response is None and timeout_counter <= 30:
        try:
            response = Generation.call(
                model="baichuan2-7b-chat-v1",
                messages=messages,
                result_format="message",
            )

            # Check if it's a rate limit error
            if response.message and "Requests rate limit exceeded" in response.message:
                print("Rate limit exceeded. Retrying...")
                time.sleep(300)
                continue

            if response.status_code == HTTPStatus.OK:
                if response.output.choices[0]["message"]["content"] is not None:
                    response = response.output.choices[0]["message"]["content"].strip()
                else:
                    response = ""
            else:
                print(
                    "Request id: %s, Status code: %s, error code: %s, error message: %s"
                    % (
                        response.request_id,
                        response.status_code,
                        response.code,
                        response.message,
                    )
                )
                response = ""
        except Timeout:
            print("Timeout occurred. Retrying...")
            timeout_counter += 1
            time.sleep(5)

    if response is None:
        response = ""

    return response


def get_Qwen_response(prompt, history):
    messages = []

    if history != []:
        for t in history:
            if isinstance(t, tuple):
                messages.append({"role": "user", "content": str(t[0])})
                messages.append({"role": "assistant", "content": str(t[1])})
            else:
                t = history
                messages.append({"role": "user", "content": str(t[0])})
                messages.append({"role": "assistant", "content": str(t[1])})
    messages.append({"role": "user", "content": str(prompt)})
    gen = Generation()
    response = gen.call(
        Generation.Models.qwen_turbo,
        messages=messages,
        result_format="message",  # 设置结果为消息格式
    )
    if response.status_code == HTTPStatus.OK:
        if response.output.choices[0]["message"]["content"] is not None:
            response = response.output.choices[0]["message"]["content"].strip()
        else:
            response = "  "
        return response
        # print(response)
    else:
        print(
            "Request id: %s, Status code: %s, error code: %s, error message: %s"
            % (
                response.request_id,
                response.status_code,
                response.code,
                response.message,
            )
        )


def get_Qwen_response(prompt, history):
    messages = []

    if history != []:
        for t in history:
            if isinstance(t, tuple):
                messages.append({"role": "user", "content": str(t[0])})
                messages.append({"role": "assistant", "content": str(t[1])})
            else:
                t = history
                messages.append({"role": "user", "content": str(t[0])})
                messages.append({"role": "assistant", "content": str(t[1])})
    messages.append({"role": "user", "content": str(prompt)})
    gen = Generation()
    response = gen.call(
        Generation.Models.qwen_turbo,
        messages=messages,
        result_format="message",  # 设置结果为消息格式
    )
    if response.status_code == HTTPStatus.OK:
        if response.output.choices[0]["message"]["content"] is not None:
            response = response.output.choices[0]["message"]["content"].strip()
        else:
            response = "  "
        return response
        # print(response)
    else:
        print(
            "Request id: %s, Status code: %s, error code: %s, error message: %s"
            % (
                response.request_id,
                response.status_code,
                response.code,
                response.message,
            )
        )


def get_access_token():
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": Wenxin_API_KEY,
        "client_secret": Wenxin_SECRET_KEY,
    }
    return str(requests.post(url, params=params).json().get("access_token"))


def get_Wenxin_response(prompt, history):
    messages = []

    if history != []:
        for t in history:
            if isinstance(t, tuple):
                messages.append({"role": "user", "content": str(t[0])})
                messages.append({"role": "assistant", "content": str(t[1])})
            else:
                t = history
                messages.append({"role": "user", "content": str(t[0])})
                messages.append({"role": "assistant", "content": str(t[1])})
    messages.append({"role": "user", "content": str(prompt)})
    url = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token="
        + get_access_token()
    )
    payload = json.dumps({"messages": messages})
    headers = {"Content-Type": "application/json"}
    response = requests.request("POST", url, headers=headers, data=payload)
    if response == None:
        response = ""
    elif json.loads(response.text)["result"]:
        response = json.loads(response.text)["result"]
    else:
        response = "Error Answer!"
    return response

    # def get_ChatGLM_response(prompt, history):
    # messages = []
    # if history != []:
    #     for t in history:
    #         if isinstance(t, tuple):
    #             messages.append({"role": "user", "content": t[0]})
    #             messages.append({"role": "assistant", "content": t[1]})
    #         else:
    #             t = history
    #             messages.append({"role": "user", "content": t[0]})
    #             messages.append({"role": "assistant", "content": t[1]})
    # messages.append({"role": "user", "content": prompt})
    # response = None
    # timeout_counter = 0
    # while response is None and timeout_counter <= 30:
    #     try:
    #         response = zhipuai.model_api.sse_invoke(
    #             model="chatglm_turbo",
    #             prompt=messages,
    #             top_p=0.7,
    #             temperature=0.9,
    #         )
    #     except Exception as msg:
    #         if "timeout=600" in str(msg):
    #             timeout_counter += 1
    #         print(msg)
    #         sleep(5)
    #         continue
    # if response == None:
    #     response = ""
    # else:
    #     responsestr = ""
    #     for event in response.events():
    #         if event.event == "add":
    #             responsestr = responsestr + event.data
    #             # print(event.data,end='')
    #         elif event.event == "error" or event.event == "interrupted":
    #             print(event.data)
    #         elif event.event == "finish":
    #             print(event.data)
    #             print(event.meta)
    #         else:
    #             print(event.data)
    #     response = responsestr
    #     if response == None:
    #         response = ""
    # return response


def get_ChatGLM_response(prompt, history):
    messages = []
    if history != []:
        for t in history:
            if isinstance(t, tuple):
                messages.append({"role": "user", "content": t[0]})
                messages.append({"role": "assistant", "content": t[1]})
            else:
                t = history
                messages.append({"role": "user", "content": t[0]})
                messages.append({"role": "assistant", "content": t[1]})
    messages.append({"role": "user", "content": prompt})
    response = None
    timeout_counter = 0
    retry_counter = 0
    while response is None and timeout_counter <= 30:
        try:
            response = zhipuai.model_api.sse_invoke(
                model="chatglm_turbo",
                prompt=messages,
                top_p=0.7,
                temperature=0.9,
            )
            # 如果成功获取响应，则跳出循环
            break
        except Exception as msg:
            retry_counter += 1
            if "timeout=600" in str(msg):
                timeout_counter += 1
            else:
                # 只有在第一次捕获到非超时错误时打印错误信息
                if retry_counter == 1:
                    print(f"Error occurred: {msg}")
            sleep(5)
            continue

    if response is None:
        response = ""
    else:
        responsestr = ""
        for event in response.events():
            if event.event == "add":
                responsestr = responsestr + event.data
            elif event.event == "error" or event.event == "interrupted":
                print(f"Error event occurred: {event.data}")
            elif event.event == "finish":
                pass  # 可以根据需要处理完成事件
            else:
                print(f"Unknown event occurred: {event.data}")
        response = responsestr
        if response is None:
            response = ""

    # 如果有重试，输出重试次数
    if retry_counter > 0:
        print(f"Retried {retry_counter} times.")

    return response


openai.api_key = "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"
print(
    get_GPT4_response(
        """为了实现这一目标，我们采用了各种中文模型，包括ChatGLM系列、Qwen（通义千问）和InternLM（浦语书生），以及中文API，如Qwen-api、Wenxinyiyan-api和ChatGLM-turbo。此外，我们还在评估中包括了Baichuan、GPT4、GPT3、Davinci和Geminipro，这为我们提供了丰富的评估工具和多维度视角，全面评估它们在不同语言中的性能和可信度。我们的评估框架涵盖了各种维度，包括毒性、偏见、隐私保护、道德判断、幻觉生成、错误信息传播、非法内容生成以及对心理和身体健康的影响，以确保对模型性能进行全面评估。通过与最先进模型的比较分析，我们揭示了GPT-4相对于特定条件下的GPT-3.5的性能改进和潜在的脆弱性以及其他模型在多语言任务和安全维度上的表现，确定了处理资源稀缺语言时面临的独特挑战，并提出了增强模型在这些情境中性能的策略。尽管取得了重要成就，但我们的研究存在一些局限性，如对少数语言和边缘情况的不充分评估。未来的工作将集中在完善评估框架、扩大测试范围，并提出具体的改进措施以应对已识别的安全风险。本研究旨在为评估和增强大型模型在实际应用中的可信度和安全性提供坚实的基础和指导。
        请帮我润色一下，作为论文的摘要部分。
""",
        [],
    )
)
# print(
#     get_Davinci_response(
#         "这些无知的人",
#         [],
#     )
# )
# if __name__ == "__main__":
#     get_llama_response()
# print(get_ChatGPT_response("你好",[]))
# print(get_Geminipro_response("你好", []))
# print(get_Baichuan_response("你好", []))
# print(get_Qwen_response("", []))
