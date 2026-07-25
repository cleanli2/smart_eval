import json
import re
from openai import OpenAI

# --- Configuration ---
API_BASE = "http://localhost:8080/v1" 
API_KEY = "sk-no-key-required"       
QUESTIONS_FILE = "questions.json"
RESULT_FILE = "result.txt"

def get_model_name(client):
    try:
        models = client.models.list()
        return models.data[0].id
    except Exception as e:
        print(f"Error fetching model name: {e}")
        return "unknown-model"

def main():
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    model_name = get_model_name(client)
    print(f"Connected to llama-server. Model detected: {model_name}")

    try:
        with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
            questions = json.load(f)
    except FileNotFoundError:
        print(f"Error: {QUESTIONS_FILE} not found.")
        return

    total_score = 0
    detailed_results = []

    print(f"Starting benchmark. Total questions: {len(questions)}\n")
    print("=" * 60)

    for item in questions:
        q_id = item['id']
        question_text = item['question']
        options = item['options']
        correct_answer = item['answer']

        options_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
        prompt = f"Question: {question_text}\nOptions:\n{options_str}\n\nPlease provide the correct option letter (A, B, C, or D) only. Answer:"

        print(f"[Q{q_id}] Sending Prompt:\n{prompt}")
        print("\n--- Model Thought & Response ---")

        full_response_content = ""
        try:
            # Use stream=True to capture every token as it's generated (including <think>)
            stream = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer with only the option letter."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                stream=True 
            )

            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True) # Print token by token
                    full_response_content += content

            print("\n--- End of Response ---")
            
            # Clean response for scoring
            # 1. Remove <think>...</think> block if it exists
            clean_content = re.sub(r'<think>.*?</think>', '', full_response_content, flags=re.DOTALL).strip()
            
            # 2. Get the first uppercase letter from the remaining text
            match = re.search(r'[A-D]', clean_content.upper())
            final_answer = match.group(0) if match else "N/A"

            print(f"Extracted Answer: {final_answer}")
            
            is_correct = (final_answer == correct_answer)
            if is_correct:
                total_score += 1

            print(f"Result: {'Correct' if is_correct else 'Wrong'} (Expected: {correct_answer})")
            print("-" * 60)

            detailed_results.append({
                "id": q_id,
                "prompt": prompt,
                "raw_answer": full_response_content,
                "final_answer": final_answer,
                "correct": correct_answer,
                "status": "Correct" if is_correct else "Wrong"
            })

        except Exception as e:
            print(f"\nError processing Q{q_id}: {e}")
            detailed_results.append({"id": q_id, "prompt": prompt, "status": "Error"})

    accuracy = (total_score / len(questions)) * 100 if questions else 0

    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Final Score: {total_score}/{len(questions)} ({accuracy:.2f}%)\n")
        f.write("=" * 60 + "\n\n")
        for res in detailed_results:
            f.write(f"ID: {res.get('id')}\n")
            f.write(f"Prompt:\n{res.get('prompt')}\n")
            f.write(f"Full Response (including thoughts):\n{res.get('raw_answer')}\n")
            f.write(f"Model Final Answer: {res.get('final_answer')}\n")
            f.write(f"Correct Answer: {res.get('correct')}\n")
            f.write(f"Status: {res.get('status')}\n")
            f.write("-" * 40 + "\n")

    print(f"\nBenchmark complete. Full report saved to {RESULT_FILE}")

if __name__ == "__main__":
    main()

