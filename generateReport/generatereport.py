import openai
import os

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_report(query: str, response: str) -> dict:
    """
    Generate a title and summary from a query and response using GPT-3.5 Turbo (OpenAI SDK v1.69.0).
    """
    prompt = f"""
    You are an AI that writes concise reports.

    Given this conversation:
    ---
    Query: {query}
    Response: {response}
    ---

    Generate:
    1. Title (1 line)
    2. Summary (2-4 lines)

    Return exactly:
    Title: <title>
    Summary: <summary>
    """

    try:
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )

        result_text = chat_completion.choices[0].message.content.strip()
        lines = result_text.splitlines()

        title = next((line.replace("Title:", "").strip() for line in lines if line.lower().startswith("title:")), "")
        summary = next((line.replace("Summary:", "").strip() for line in lines if line.lower().startswith("summary:")), "")

        return {"title": title, "summary": summary}

    except Exception as e:
        return {"title": "Error", "summary": str(e)}
