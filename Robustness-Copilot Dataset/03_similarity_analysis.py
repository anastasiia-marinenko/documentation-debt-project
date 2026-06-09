from sentence_transformers import (
    SentenceTransformer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)

from utils import load_dataset

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

df = load_dataset()

similarities = []

for _, row in df.iterrows():

    original = str(
        row["javaDocFirstSentence"]
    )

    paraphrase = str(
        row["perturbed_eval_1"]
    )

    emb1 = model.encode([original])

    emb2 = model.encode([paraphrase])

    sim = cosine_similarity(
        emb1,
        emb2
    )[0][0]

    similarities.append(sim)

print(
    "Average similarity:",
    sum(similarities) /
    len(similarities)
)