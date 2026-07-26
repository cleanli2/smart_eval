import requests
import json
import re
import os
from datetime import datetime

# ===================== CONFIGURATION AREA =====================
LLAMA_SERVER_URL = "http://127.0.0.1:8080/completion"
QUESTION_BANK_PATH = "question_bank.json"
TOTAL_SCORE = 100
# ==================================================================

def get_safe_model_name() -> str:
    """Get current loaded model name from llama-server api"""
    model_info_url = "http://127.0.0.1:8080/v1/models"
    resp = requests.get(model_info_url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    raw_name = data["data"][0]["id"]
    # Replace illegal characters for file name
    safe_name = raw_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    return safe_name

def load_question_bank(file_path: str) -> list:
    """Load question list from local json file"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Question bank root must be list array")
    return data

def ask_llama_stream(prompt_text: str) -> str:
    """Send prompt to llama-server stream api, return full response string"""
    payload = {
        "prompt": prompt_text,
        "temperature": 0.0,
        "stream": True,
        "repeat_penalty": 1.3,
        "repeat_last_n": 16
    }
    headers = {"Content-Type": "application/json; charset=utf-8"}
    full_output = ""

    resp = requests.post(
        LLAMA_SERVER_URL,
        json=payload,
        headers=headers,
        stream=True,
        timeout=180
    )
    resp.raise_for_status()
    resp.encoding = 'utf-8'

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            raw_data = line.removeprefix("data: ")
            if raw_data == "[DONE]":
                break
            try:
                chunk_json = json.loads(raw_data)
                token_text = chunk_json.get("content", "")
                if token_text:
                    print(token_text, end="", flush=True)
                    full_output += token_text
            except json.JSONDecodeError:
                continue
    return full_output

def extract_answer_letter(raw_text: str) -> str:
    """Extract single uppercase ABCD letter from model output, match format [Answer]#X"""
    match = re.search(r"\[Answer\]#([ABCD])", raw_text)
    if match:
        return match.group(1)
    # Fallback: extract any ABCD if format missing
    fallback_match = re.search(r"[ABCD]", raw_text)
    return fallback_match.group(0) if fallback_match else ""

def build_single_question_prompt(q_data: dict) -> str:
    """Construct standardized prompt"""
    q_text = q_data["question"]
    opt_lines = "\n".join([f"{k}: {v}" for k, v in q_data["options"].items()])
    prompt = f"""Answer this question by choose correct option. Answer after thinking.
Rule: Output final answer with fixed format and stop immediately: [Answer]#X
Replace X with A/B/C/D.

Question: {q_text}
Options:
{opt_lines}

Answer:"""
    return prompt

def main():
    # Step 1: Get model name and timestamp for log file
    model_name = get_safe_model_name()
    # Generate compact datetime string: YYYYMMDD_HHMMSS
    run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_time_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_storage = []
    log_storage.append(f"Evaluation Model: {model_name}")
    log_storage.append(f"Evaluation Time: {run_time_human}")
    log_storage.append(f"File Creation Timestamp: {run_datetime}")

    # Step 2: Load all exam questions
    question_list = load_question_bank(QUESTION_BANK_PATH)
    total_question_count = len(question_list)
    score_per_question = TOTAL_SCORE / total_question_count
    log_storage.append(f"Total Questions: {total_question_count}, Full Mark: 100, Points per question: {score_per_question:.2f}")
    log_storage.append("=" * 80 + "\n")

    print(f"===== Start Evaluation | Model: {model_name} | Total {total_question_count} Questions =====\n")
    correct_count = 0

    # Step3: Iterate each test question
    for idx, q in enumerate(question_list, start=1):
        q_id = q["id"]
        q_category = q["category"]
        std_answer = q["answer"]
        prompt = build_single_question_prompt(q)

        # Print question info to console
        print(f"[Question {idx} ID:{q_id} Category:{q_category}]")
        print(f"Question Content: {q['question']}")
        for opt_k, opt_v in q["options"].items():
            print(f"  {opt_k}: {opt_v}")
        print(f"Standard Answer: {std_answer}")
        #print("\nPrompt Sent To LLM:")
        #print(prompt)
        print("\nStreaming Model Output:")

        # Request stream answer from llama
        model_raw_response = ask_llama_stream(prompt)
        model_answer = extract_answer_letter(model_raw_response)

        print(f"\nExtracted Model Answer: {model_answer}")
        is_correct = (model_answer == std_answer)
        if is_correct:
            correct_count += 1
            print(f"✅ Correct, +{score_per_question:.2f} points")
        else:
            print(f"❌ Wrong, 0 points")
        print("-" * 60 + "\n")

        # Write full question record to log buffer
        log_storage.append(f"===== Question {idx} ID:{q_id} Category:{q_category} =====")
        log_storage.append(f"Question Content: {q['question']}")
        for opt_k, opt_v in q["options"].items():
            log_storage.append(f"  {opt_k}: {opt_v}")
        log_storage.append(f"Standard Answer: {std_answer}")
        log_storage.append(f"Sent Prompt:\n{prompt}")
        log_storage.append(f"Model Raw Full Response:\n{model_raw_response}")
        log_storage.append(f"Extracted Model Answer: {model_answer}")
        log_storage.append(f"Answer Correct: {is_correct}\n")

    # Step4: Calculate final score
    final_score = correct_count * score_per_question
    summary_text = f"""
==================== EVALUATION SUMMARY ====================
Total Test Questions: {total_question_count}
Correct Count: {correct_count}
Wrong Count: {total_question_count - correct_count}
Single Question Score: {score_per_question:.2f}
Final Score (Full Mark 100): {final_score:.2f}
===============================================================
"""
    print(summary_text)
    log_storage.append(summary_text)

    # Step5: Save log to result file with timestamp
    output_filename = f"result_{model_name}_{run_datetime}.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(log_storage))
    print(f"Full evaluation log saved to file: {output_filename}")

if __name__ == "__main__":
    # Install dependency: uv add requests
    main()

