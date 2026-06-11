"""
evaluation/metrics.py — all metrics from the daily log, each isolated so a
missing library degrades to None instead of crashing the whole run.

  ROUGE-1 / ROUGE-L  -> rouge_score (Lin, 2004)
  BLEU               -> nltk sentence_bleu, smoothing (Papineni et al., 2002)
  METEOR             -> nltk meteor_score (Banerjee & Lavie, 2005)
  BERTScore F1       -> bert_score (Zhang et al., 2020)
  Cosine             -> sentence-transformers embeddings; TF-IDF fallback
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

# ---- ROUGE ----
try:
    from rouge_score import rouge_scorer
    _RS = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
except Exception:
    _RS = None

def rouge(ref, hyp):
    if not _RS:
        return None, None
    s = _RS.score(ref, hyp)
    return round(s["rouge1"].fmeasure, 4), round(s["rougeL"].fmeasure, 4)

# ---- BLEU / METEOR (nltk) ----
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
        try: nltk.data.find(pkg)
        except Exception:
            try: nltk.download(pkg, quiet=True)
            except Exception: pass
    _SMOOTH = SmoothingFunction().method1
    _NLTK = True
except Exception:
    _NLTK = False

def bleu(ref, hyp):
    if not _NLTK: return None
    try:
        return round(sentence_bleu([ref.split()], hyp.split(), smoothing_function=_SMOOTH), 4)
    except Exception:
        return None

def meteor(ref, hyp):
    if not _NLTK: return None
    try:
        return round(meteor_score([ref.split()], hyp.split()), 4)
    except Exception:
        return None

# ---- BERTScore (batched separately for speed) ----
def bertscore_batch(refs, hyps, lang="en"):
    try:
        from bert_score import score as bscore
        _, _, F1 = bscore(hyps, refs, lang=lang, verbose=False)
        return [round(float(x), 4) for x in F1]
    except Exception:
        return [None] * len(refs)

# ---- Cosine similarity ----
_ST = None
def _get_st():
    global _ST
    if _ST is None:
        try:
            from sentence_transformers import SentenceTransformer
            _ST = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _ST = False
    return _ST

def cosine_batch(refs, hyps):
    st = _get_st()
    if st:
        import numpy as np
        a = st.encode(refs, show_progress_bar=False)
        b = st.encode(hyps, show_progress_bar=False)
        num = (a * b).sum(1)
        den = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-9
        return [round(float(x), 4) for x in (num / den)]
    # TF-IDF fallback
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        out = []
        for r, h in zip(refs, hyps):
            try:
                m = TfidfVectorizer().fit_transform([r, h])
                out.append(round(float(cosine_similarity(m[0], m[1])[0][0]), 4))
            except Exception:
                out.append(None)
        return out
    except Exception:
        return [None] * len(refs)
