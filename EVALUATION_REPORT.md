# Evaluation Report
## AI Investigator — Evidence-Based Multi-Document Investigation Engine

**Program:** Talent Growth Program — Applied AI  
**Project:** AI Investigator  
**Repository:** [MrPrincee/AiInvestigator](https://github.com/MrPrincee/AiInvestigator)  
**Status:** Recorded retrieval, answer-correctness, and citation experiments; robustness results pending verification.

## 1. Objectives and Scope

The evaluation examines whether AI Investigator retrieves relevant documents, answers multi-document questions correctly, provides traceable document references, and recognizes limitations. Three retrieval configurations are compared: **semantic**, **keyword**, and **hybrid**. Two answer-generation configurations are compared: **standard RAG** and **Gemini tool calling**.

All reported numbers below come from the project's recorded evaluation artifacts or the output of its analysis scripts. No unexecuted experiment is reported as successful.

## 2. Dataset and Test Design

The main benchmark is **HotpotQA (`distractor`)**, with **50 answerable questions**, comprising **25 bridge** and **25 comparison** questions. Evaluation examples contain the question, reference answer, question type, supporting facts, and expected supporting-document IDs. `evaluation/prepare_dataset.py` selects questions whose supporting document IDs are present in the configured Qdrant collection.

The benchmark tests multi-hop and cross-document retrieval. It is **not a separately validated simple single-hop benchmark**. Additional robustness scenarios are defined in code: **five unanswerable questions** and **five conflicting-evidence cases**. Their actual completion and pass rates are not yet recorded in the supplied results.

The three evaluation populations differ:

| Evaluation | Questions | Basis |
| --- | ---: | --- |
| Retrieval | 50 | All selected answerable HotpotQA questions |
| Answer correctness | 40 per mode | Questions with both a generated answer and a correctness judgment in both modes |
| Citation evaluation | 48 per mode | Questions with generated answers in both modes |

The stored `answer_results.jsonl` contains **98 records** and `correctness_results.jsonl` contains **80 records**. The 40-question paired subset, rather than all 50, is used for the answer-correctness comparison.

## 3. Retrieval Evaluation

### Method

`evaluation/retrieval_evaluation.py` queries the same benchmark in three modes: `semantic`, `keyword`, and `hybrid`. The retrieval service returns up to ten chunks; duplicate document IDs are removed before metrics are computed. Relevance is defined by HotpotQA's supporting-document IDs.

- **Recall@K:** fraction of gold supporting documents found among the first K unique returned document IDs, averaged over questions.
- **MRR@K:** reciprocal rank of the first relevant document within K, averaged over questions. MRR measures first-hit rank, **not** whether all documents required for a multi-hop answer were found.

### Results

| Metric | Semantic | Keyword | Hybrid |
| --- | ---: | ---: | ---: |
| Recall@1 | 0.4100 | — | — |
| Recall@3 | 0.6700 | — | — |
| Recall@5 | 0.7000 | 0.6100 | **0.7200** |
| Recall@10 | 0.7200 | 0.7000 | **0.8000** |
| MRR@10 | 0.8583 | 0.7218 | **0.8687** |

The table reports selected confirmed values; `—` denotes metrics not transcribed into this report, not missing functionality. The complete per-configuration results, including all K values, are available in `data/evaluation/retrieval_results.json`.

**Interpretation:** Hybrid improves Recall@10 by **8 percentage points** over semantic and **10 points** over keyword. Its MRR@10 is also the highest, although the gain over semantic is small (**0.0104**). This supports the use of combined semantic and sparse keyword retrieval on this particular benchmark.

**Methodological limitation:** Because the search initially returns at most ten **chunks**, and deduplication happens afterward, fewer than ten unique documents may remain. Thus the measured document-level Recall@10 is influenced by chunk duplication and should not be described as a true top-ten-document retrieval budget in all cases.

## 4. Answer Correctness Evaluation

### Method

`evaluation/answer_evaluation.py` collects generated answers in `standard` and `tool_calling` modes. It records exact match, token F1, latency, answerability, and source-coverage information. `evaluation/evaluate_existing_answers.py` uses `AnswerCorrectnessEvaluator` (Gemini as an LLM judge) to score responses against the expected answers:

- **1.0** — correct;
- **0.5** — partially correct or incomplete;
- **0.0** — incorrect or missing.

The judge also supplies a reason and `needs_review` flag. `evaluation/error_analysis.py` compares the **40 common questions** with available judgments in both modes. These judgments are not independently human-verified ground truth.

### Results

| Metric | Standard RAG | Tool Calling |
| --- | ---: | ---: |
| Common questions evaluated | 40 | 40 |
| Mean answer-correctness score | **0.8250** | **0.8250** |
| Questions with score below 1.0 | 7 | 7 |
| Needs manual review | 1 | 0 |
| Incomplete retrieved-source coverage | 14 | 16 |

**Paired comparison:** Standard wins **1**, Tool Calling wins **1**, and **38** are ties.

**Interpretation:** No answer-correctness advantage for tool calling was observed in the 40-question paired subset. The 0.8250 figure is a **mean graded score**, not necessarily an 82.5% exact-answer accuracy rate. This sample is too small to establish equivalence or superiority across all question types.

## 5. Citation Evaluation

### Method

`evaluation/citation_evaluation.py` extracts numeric citation markers (such as `[1]`) from generated answers and maps them to source entries. It compares **cited document IDs** with HotpotQA gold supporting-document IDs. The reported precision and F1 are explicitly **document-level proxies**, not claim-level faithfulness measures.

- **Citation precision proxy:** proportion of cited documents that belong to the gold supporting-document set.
- **Citation recall:** proportion of gold supporting documents that were cited.
- **Citation F1 proxy:** harmonic mean of those two values, computed per question and then averaged.
- **Unresolved citations:** citation markers without a resolvable source entry.

The separate `citation_coverage` field printed by `error_analysis.py` checks the response's `sources` collection against gold supporting-document IDs; it is **not identical** to citation recall.

### Results

| Metric | Standard RAG | Tool Calling |
| --- | ---: | ---: |
| Common questions evaluated | 48 | 48 |
| Citation precision proxy | 0.7750 | **0.8028** |
| Citation recall | 0.7500 | **0.7812** |
| Citation F1 proxy | 0.7310 | **0.7552** |
| Answers with citations | 42/48 | **45/48** |
| Unresolved citation references | 0 | 0 |

**Interpretation:** Tool calling scores **2.78 percentage points** higher in citation precision proxy, **3.12 points** in citation recall, and **2.42 points** in citation F1 proxy. All generated citation markers resolved to a source entry. However, a resolvable citation can still point to a passage that does not substantiate the precise claim being made.

## 6. Error Analysis

The recorded `error_analysis.py` output identifies several recurring failure modes:

| Failure mode | Observed example | Likely issue / potential improvement |
| --- | --- | --- |
| Over-abstention | `Catching Fire` protagonist's age, James Henry Miller's wife's nationality, or the `Great Outdoors` Walk of Fame year: answerable references, but the system abstained. | Inspect retrieved chunks and evidence thresholds; improve query decomposition and second-hop retrieval. A reference answer alone does not prove that the system retrieved sufficient evidence. |
| Wrong entity attribute | Longs Drugs / Warren Bryant: answered company headquarters (**California**) when the reference sought store location (**Hawaii**). | Preserve the precise target relation during retrieval planning and response validation. |
| Incomplete multi-hop answer | Tool calling identified **Metallica** in the *Hurt Locker The Musical* question but did not list the requested band members. | Add a final check that all requested fields have supporting evidence. |
| Incorrect entity selection | Tool calling returned other performers instead of **Carol Lawrence** for *New Faces of 1952*. | Improve evidence ranking and answer extraction. |
| Question/reference inconsistency | Moth question expects **Crambidae** as a *genus*, although Crambidae is a *family*; system returned genus names. | Flag for human adjudication; do not treat the LLM judge's 0.0 score as conclusive evidence of system failure. |

For the paired correctness sample, **7/40** cases in each mode received a score below 1.0. The examples above are illustrative rather than a statistical breakdown of error categories.

## 7. Robustness and Contradictory Evidence

`evaluation/robustness_evaluation.py` defines:

- **Five unanswerable scenarios**, checking `is_answerable == false`.
- **Five contradiction scenarios**, each creating two documents with incompatible claims, asking a question, checking that both documents were retrieved, and expecting a contradiction flag plus abstention.

**Status:** No verified end-to-end pass rates are included in the current recorded results. Therefore this report makes **no claim** that these tests passed.

**Safety:** The contradiction script **creates and deletes documents via the API**. Run it only with a local backend configured to use an isolated test collection. It must not be run against production Qdrant. The script also makes real Gemini requests and consumes API quota.

## 8. Chunking and Dataset-Scale Verification

Two chunking algorithms exist in `dataset/chunking.py`:

1. `fixed_size_chunking` — character-based segments with configurable overlap.
2. `sentence_aware_chunking` — groups sentences within a character budget, splitting overlong sentences if necessary.

**Chunking comparison verified:** Both strategies were evaluated on the same
sample of 1,000 HotpotQA documents. Sentence-aware chunking achieved a
98.39% sentence-ending rate, compared with 59.77% for fixed-size chunking.
Detailed results and limitations are provided in the
"Chunking Strategy Comparison" section.

**Dataset scale verified:** The production Qdrant collection
`hotpotqa_50k_hybrid` contains 78,809 vector points representing
50,157 unique document IDs. The count was verified by scrolling through
the collection payloads and counting distinct `document_id` values.
The requirement of at least 50,000 indexed documents is satisfied.
## 9. Reproducibility and Artifacts

| Artifact | Purpose |
| --- | --- |
| `data/evaluation/hotpotqa_evaluation.jsonl` | Questions and gold supporting documents |
| `data/evaluation/retrieval_results.json` | Per-mode retrieval results and per-question metrics |
| `data/evaluation/answer_results.jsonl` | Generated answers and associated evidence |
| `data/evaluation/correctness_results.jsonl` | LLM-judge scores, reasons, review flags |
| `data/evaluation/citation_results.json` | Per-question citation proxy results |
| `evaluation/error_analysis.py` | Offline correctness comparison and error inspection |
| `evaluation/citation_evaluation.py` | Offline citation metric calculation |
| `evaluation/robustness_evaluation.py` | Live robustness scenarios; mutates documents |

From the project root, after installing dependencies, the **offline** summaries can be reproduced without a Gemini call:

```powershell
python -m evaluation.error_analysis
python -m evaluation.citation_evaluation
```

The full retrieval benchmark requires a compatible populated Qdrant collection. Generating new answers and LLM-judge scores requires Gemini quota and may be nondeterministic. Preserve the existing result files when rerunning experiments.

## 10. Threats to Validity and Next Steps

- **Small and uneven samples:** Retrieval has 50 questions; answer correctness 40 common questions; citations 48. Results should not be combined into one overall accuracy figure.
- **LLM-judge bias:** Gemini evaluates Gemini-generated answers against references; manual review is needed for ambiguous or contested cases.
- **Citation proxies:** Document-ID overlap does not measure sentence-level entailment or hallucination rate.
- **Retrieval budget:** Chunk-level top-10 followed by document deduplication may return fewer than ten distinct documents.
- **Dataset specificity:** HotpotQA bridge/comparison questions do not cover every document format or real-world investigation type.
- **Incomplete validation:** simple single-hop benchmark, and end-to-end robustness pass rates remain to be demonstrated.

Recommended improvements: (1) run robustness tests in isolated infrastructure;
(2) compare chunking strategies using retrieval metrics on separately indexed collections;
(3) add a small human-audited citation faithfulness set;
(4) expand question-type coverage and manual review.
## 11. Conclusion

On the recorded 50-question HotpotQA retrieval benchmark, **hybrid retrieval outperformed semantic-only and keyword-only retrieval**. On 40 paired answer-correctness questions, **standard RAG and tool calling tied at 0.8250 mean score**. On 48 paired citation questions, **tool calling produced modestly higher document-level citation relevance proxies**, with no unresolved citation markers in either mode. These results support the value of hybrid search and source tracking, while also identifying concrete retrieval, reasoning, evaluation, and robustness limitations.

## Chunking Strategy Comparison

Two chunking strategies were evaluated on the same sample of 1,000 HotpotQA documents:

- **Fixed-size chunking:** 500 characters per chunk, with a 50-character overlap.
- **Sentence-aware chunking:** maximum 500 characters per chunk, preserving sentence boundaries where possible.

### Results

| Metric | Fixed Size | Sentence Aware |
|---|---:|---:|
| Documents evaluated | 1,000 | 1,000 |
| Total chunks | 1,653 | 1,679 |
| Average chunk length (characters) | 362.98 | 335.77 |
| Chunks ending with sentence punctuation | 59.77% | 98.39% |

### Analysis

Sentence-aware chunking produced 26 more chunks (approximately 1.57%) than fixed-size chunking.

However, sentence-aware chunking achieved a substantially higher sentence-ending rate: 98.39%, compared with 59.77% for fixed-size chunking.

This represents an improvement of 38.62 percentage points in the proportion of chunks ending with sentence punctuation.

The sentence-aware strategy also produced slightly shorter chunks on average.

### Conclusion

Sentence-aware chunking better preserves sentence boundaries, potentially improving the readability and coherence of retrieved evidence.

However, the sentence-ending rate is only a proxy for boundary preservation. This experiment does not establish that sentence-aware chunking improves retrieval accuracy or answer correctness.

A controlled retrieval evaluation using separately indexed collections would be required to compare the strategies using Recall@K and MRR.

The experiment is reproducible using:

`python -m evaluation.chunking_evaluation`

Results are stored in:

`data/evaluation/chunking_results.json`