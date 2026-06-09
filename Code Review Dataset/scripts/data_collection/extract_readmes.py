import pandas as pd
import os

df = pd.read_csv("data/raw/YOUR_FILE.csv")

output_dir = "data/processed/readmes/"
os.makedirs(output_dir, exist_ok=True)

# Посмотри как называется колонка с текстом README — замени 'readme_content'
for i, row in df.iterrows():
    readme_text = row.get("readme_content", "")
    if isinstance(readme_text, str) and len(readme_text) > 10:
        filename = f"{output_dir}readme_{i}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(readme_text)

print(f"Extracted {i+1} README files")