import json
import os
import openai

openai.api_key = os.environ.get("OPENAI_API_KEY")

def get_gpt4_response(messages, model_name="gpt-4o", temperature=0.00001, response_format="text"):
    generation_params = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        **({"response_format": {"type": "json_object"}} if response_format == "json" else {}),
    }
    chat_completions = openai.chat.completions.create(**generation_params)
    response = chat_completions.choices[0].message.content
    if response_format == "json":
        try:
            return json.loads(response)
        except:
            return None
    return response.strip()