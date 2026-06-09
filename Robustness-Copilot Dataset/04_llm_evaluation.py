from groq import Groq
from utils import load_dataset


MODEL_GROQ = "llama-3.1-8b-instant"

client = Groq(api_key="...")

df = load_dataset()

sample = df.sample(20, random_state=42)

for _, row in sample.iterrows():

    code = row["body"]
    reference = row["javaDocFirstSentence"]

    prompt = f"""
Generate a concise JavaDoc
for the following method:

{code}
"""

    response = client.chat.completions.create(
        model=MODEL_GROQ,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    generated = response.choices[0].message.content

    print("=" * 80)
    print("REFERENCE:")
    print(reference)
    print()
    print("GENERATED:")
    print(generated)
