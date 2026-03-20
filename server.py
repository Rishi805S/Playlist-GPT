import ollama

response = ollama.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'Explain vector DB simply'}],
    options={
        "num_predict": 200   # limit output tokens
    }

)

print(response['message']['content'])