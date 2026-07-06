# Literature Review — Rewriting the Narrative: Leveraging LLMs for Documentation Debt Management

**Author:** Anastasiia Marinenko · Mitacs Globalink Research Internship, University of Saskatchewan
**Supervisor:** Dr. Zadia Codabux
**Updated:** 06/26/2026

This document consolidates the papers reviewed during the internship, organized by theme. Part A lists the **core reviewed papers** (the running literature-review set, ~32 papers). Part B lists **additional methodological references** that ground specific design choices (prompting foundations, evaluation metrics, datasets, and standards) and are cited throughout the experiments.

---

## Part A — Core Reviewed Papers

### A.1 Documentation & Technical Debt — foundations
1. **Rios et al. (2020)** — *Hearing the Voice of Software Practitioners on Causes, Effects, and Practices to Deal with Documentation Debt.* Causes, effects, and practitioner practices; establishes documentation debt as an underrepresented debt type.
2. **Sierra et al. (2019)** — *A Survey of Self-Admitted Technical Debt (SATD).* ~55% of SATD work targets detection, only ~10% repayment; motivates non-code-artifact analysis.
3. **Zhi et al. (2015)** — *Cost, Benefits and Quality of Software Development Documentation.* Systematic mapping; documentation cost is severely understudied.
4. **Velasco-Elizondo et al. (2024)** — Documentation debt / quality perspective.
5. **Tom et al. (2013)** — Technical-debt taxonomy: dimensions (code, design, environment, knowledge/documentation, testing) and forms.
6. **Avgeriou et al. (2021)** — Comparison of TD measurement tools; most tools see only code-level debt, documentation debt is invisible to tooling.
7. **Hermann & Fehr (2022)** — Documentation in engineering research software; quality follows lifecycle waves tied to personnel turnover.
8. **Ernst & Robillard (2023)** — Architecture documentation format study; format (narrative vs. structured) has little effect, prior code familiarity dominates.
9. **Silva et al. (2024)** — Customer-facing documentation debt taxonomy; most defects are content-related; proposes Dynamic Documentation Generation and Automated Documentation Testing.
10. **Le Hai et al. (2026)** — *Detection of Technical Debt in Java Source Code.* Automated TD detection in Java; complements documentation-debt analysis.
11. **Heikkala (2026)** — *Reducing Technical Debt in a Java Enterprise Application* (MSc thesis). Practitioner perspective on documentation/architectural debt repayment.

### A.2 README structure & quality
12. **Prana et al. (2019)** — *Categorizing the Content of GitHub README Files.* 8-category README taxonomy (What, Why, How, When, Who, Contribution, References, Other).
13. **Treude et al. (2020)** — *Beyond Accuracy: Assessing Software Documentation Quality.* 10-dimension quality framework; README files score highest across genres.
14. **Foidl et al. (2020)** — README structure & documentation quality.

### A.3 LLM-based documentation automation
15. **Gao et al. (2026)** — *Does My README File Need To Be Updated? Exploring LLM-Based README Maintenance.* Surgical-update pipeline; README updates are rare, highly imbalanced events.
16. **Xiao et al. (2024)** — *Generative AI for Pull Request Descriptions.* Real-world Copilot impact on review time and merge likelihood.
17. **Liu et al. (2019)** — *Automatic Generation of Pull Request Descriptions (PRSummarizer).* Seq2seq with pointer-generator and RL for ROUGE.
18. **Ahmed et al. (2024)** — *Can LLMs Replace Manual Annotation of SE Artifacts?* Model–model agreement predicts human–model agreement; confidence-based filtering.
19. **Alqaimi et al. (2019)** — *Automatically Generating Documentation for Lambda Expressions (LAMBDADOC).* Only ~6% of Java lambdas documented; generated summaries rated complete/concise.
20. **Trigui et al. (2025)** — *AI-Driven Code Documentation: Comparative Evaluation of LLMs for Commit Message Generation.*
21. **Dvivedi et al.** — *A Comparative Analysis of Large Language Models for Code Documentation Generation* (IIIT Delhi).
22. **Vitale et al.** — *Optimizing Datasets for Code Summarization: Is Code-Comment Coherence Enough?*

### A.4 Code–comment consistency & the primary dataset
23. **Zhong et al. (2026)** — *CCISOLVER: End-to-End Detection and Repair of Method-Level Code-Comment Inconsistency* (IEEE TSE). Directly aligned with method-level documentation debt; CCIBench dataset candidate.
24. **Lin et al. (2026)** — *Leveraging Reviewer Experience in Code Review Comment Generation* (ICSE). Source of the Code Review dataset used for method-level extraction.

### A.5 Prompt engineering
25. **White et al. (2023)** — *A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT.* Reusable patterns (Persona, Few-Shot, CoT, Template, etc.); basis for the persona pattern.
26. **White et al. (2023)** — *Towards a Catalog of Prompt Patterns to Enhance the Discipline of Prompt Engineering.* Additional patterns and anti-patterns.
27. **Agade et al. (2025)** — *Prompt Engineering for Large Language Models: Zero-shot, Few-shot, and Beyond.* Unified taxonomy; supports testing all base patterns as independent conditions.
28. **Cheng et al. (2025)** — *Revisiting Chain-of-Thought Prompting: Zero-shot Can Be Stronger than Few-shot* (EMNLP Findings). Motivates zero-shot as a competitive baseline.
29. **Beri & Srivastava (2024)** — *Advanced Techniques in Prompt Engineering for LLMs: A Comprehensive Study.* Supports CoT as a standalone method.

### A.6 Meta-prompting
30. **Suzgun & Kalai (2024)** — *Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding.* Theoretical basis for the meta-refinement layer.
31. **de Wynter et al.** — *On Meta-Prompting.* Conditions under which meta-prompting improves vs. degrades output; same-LLM refinement framing.
32. **Zhang et al. (Tsinghua)** — *Meta Prompting for AI Systems.* Structured orchestrator framing; scope and limits of meta-refinement.

---

## Part B — Additional Methodological References

These ground specific design and evaluation decisions and are cited in the experiments/slides.

### B.1 Prompting foundations
- **Brown et al. (2020)** — Few-shot in-context learning (rationale for one-/two-shot).
- **Wei et al. (2022)** — Chain-of-Thought prompting.
- **Kojima et al. (2022)** — Zero-shot Chain-of-Thought.
- **Madaan et al. (2023)** — Self-Refine; gains concentrate in the first 1–2 iterations.

### B.2 Evaluation metrics
- **Zhang et al. (2020, ICLR)** — BERTScore.
- **Sellam et al. (2020)** — BLEURT.
- **Liu et al. (2023, Microsoft)** — G-Eval (LLM-as-judge).
- **Roy, Eberhart, Fakhoury & Arnaoudova (2021, ESEC/FSE)** — Reassessing automatic evaluation metrics for code summarization.
- **Haque, Eberhart, Bansal & McMillan (ICPC)** — Semantic similarity metrics for source-code summarization.
- **Hu, Chen, Wang, Lo & Zimmermann (2022, TOSEM)** — Correlating automated and human evaluation of code documentation generation quality.
- **Feng et al. (2020)** — CodeBERT (semantic embeddings for code).

### B.3 Datasets & preprocessing
- **Husain et al. (2019)** — CodeSearchNet Challenge (primary dataset).
- **Shi et al. (2022, ESEC/FSE)** — *Are We Building on the Rock?* CAT preprocessing rules.
- **Panthaplackel et al. (2020, ACL)** — JITDATA (comment-update / consistency).
- **Hu et al. (2018, IJCAI)** — TL-CodeSum.
- **LeClair et al. (2019)** — Funcom.
- **Yu et al. (2024)** — CoderEval.
- **Mastropaolo et al. (2022)** — Robustness-Copilot dataset.

### B.4 Complexity, sampling & standards
- **McCabe (1976)** — Cyclomatic complexity measure (complexity categorization).
- **Cochran (1977)** — Stratified sampling.
- **van der Lee et al. (2019, INLG)** — Best practices for human evaluation of automatically generated text.
- **Schreck et al. (2007, IWPSE)** — How documentation evolves over time (Javadoc quality metrics).
- **Khamis et al. (2010, NLDB)** — Automatic quality assessment of source-code comments (JavadocMiner).
- **Oracle / Sun Microsystems** — *How to Write Doc Comments for the Javadoc Tool* (Javadoc contract: summary, @param, @return, @throws).

---

## Thematic synthesis

The reviewed work traces a clear path: **identify** documentation debt (Rios, Sierra, Zhi) → **categorize** documentation structure (Prana, Foidl) → **measure** quality (Treude; Schreck; Khamis) → **automate** generation and maintenance with LLMs (Gao, Xiao, Liu, Ahmed, Alqaimi, Trigui, Dvivedi, Vitale). This project extends that path with (i) a method-level Javadoc generation study across multiple open-source LLMs, (ii) a literature-grounded, single-factor prompt-engineering design (zero-/one-/two-shot, CoT, persona, with per-pattern meta-refinement), and (iii) a validated human ground truth evaluated with complementary lexical and semantic metrics — laying the foundation for method-level documentation-debt detection.
