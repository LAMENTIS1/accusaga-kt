import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def generate_combined_heading(user_query: str, subheading: Optional[str]=None) -> str:
    prompt = (
        f"Create a concise and attention-grabbing news-style heading using the following:\n"
        f"Query: '{user_query}'\n"
        f"Subheading category: '{subheading}'\n"
        f"Combine them into a single coherent heading suitable for an article or report.\n"
        f"Example format: '[Live] Latest Statistics on Anxiety and Depression Diagnoses'\n"
        f"Now, generate the heading:"
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )

    heading = response.choices[0].message.content.strip()
    return heading

