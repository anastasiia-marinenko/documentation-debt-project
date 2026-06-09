from rouge_score import rouge_scorer

from utils import load_dataset

df = load_dataset()

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rougeL"],
    use_stemmer=True
)

scores = []

for _, row in df.iterrows():

    original = str(
        row["javaDocFirstSentence"]
    )

    paraphrase = str(
        row["perturbed_eval_1"]
    )

    result = scorer.score(
        original,
        paraphrase
    )

    scores.append(
        result["rougeL"].fmeasure
    )

print(
    "Average ROUGE-L:",
    sum(scores) / len(scores)
)