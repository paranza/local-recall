# Design notes

Questions I would expect about this project, with short answers and where to look in the code. Every number comes from `results/RESULTS.md`.

## The basics

**What is BM25 and what do k1 and b do?**
A score for how well a document matches a query by its words. For each query term it multiplies how rare the term is (idf) by how often it appears in the document (tf), with two corrections. `k1` caps the effect of repetition: the tenth occurrence of a word adds much less than the second. `b` corrects for length: a long document contains more words by chance, so its counts are scaled down. `b = 0` turns length correction off, `b = 1` applies it fully. Code: `bm25.py`, `BM25Index.scores`.

**How is the index stored?**
An inverted index: for each term, an array of document ids and an array of term counts. A query only touches the postings of its own terms, so cost grows with how common the query words are, not with the size of the collection. FiQA (57,638 documents) takes 30.9 MB and answers in 0.6 ms.

**How do you know your BM25 is right?**
Two tests in `tests/test_core.py`: one score computed by hand, and a comparison with the `rank_bm25` library on the same tokens, equal to five decimal places. One subtlety: that library replaces negative idf values with a floor, so the comparison uses terms that appear in fewer than half the documents. My default idf is the Lucene form, `log(1 + (N - n + 0.5) / (n + 0.5))`, which cannot go negative.

**What does the tokenizer do, and why that stemmer?**
Lowercase, strip accents, split on non-alphanumerics, drop a short stopword list, then fold plurals with Harman's S-stemmer (three rules). An aggressive stemmer such as Porter merges unrelated words ("university" and "universe"). The plural-only one cannot. I measured it instead of assuming: +0.014 nDCG@10 on NFCorpus (significant), no measurable change on SciFact or FiQA.

**What is a bi-encoder and what is a cross-encoder?**
A bi-encoder turns the query and each document into a vector separately. Documents are encoded once, ahead of time, and search is a dot product. A cross-encoder reads the query and one document together and outputs a score, so every word of the query can attend to every word of the document. It is more accurate and cannot be precomputed. Here: 17.5 ms to embed a query against about 1.3 s to cross-encode 50 candidates on the CPU. That cost gap is why the cross-encoder only sees a short candidate list.

**Why normalize embeddings?**
With unit-length vectors the dot product equals cosine similarity, so one matrix product ranks the whole collection (`dense.py`, `ExactIndex`).

**What is the query prefix for?**
The bge v1.5 models were trained with an instruction on the query side only ("Represent this sentence for searching relevant passages: "). Queries are short and documents long, and the prefix tells the model which role the text plays. Leaving it out costs accuracy on short queries.

## Search and retrieval

**How does HNSW work?**
It builds a layered graph where each vector links to its near neighbours. Upper layers are sparse and act as an express lane. A search starts at the top, walks greedily toward the query, and drops a layer when it cannot get closer. At the bottom it keeps a list of the best `ef_search` candidates. `M` is links per node (memory against recall), `ef_construction` is build effort, `ef_search` is the recall and latency dial at query time.

**What did the HNSW sweep show?**
On FiQA, `ef = 8` gives 0.758 recall of the exact top 10 at 0.07 ms, `ef = 64` gives 0.975 at 0.22 ms, `ef = 256` gives 0.996 at 0.94 ms. Exact search takes 1.72 ms. Lower recall costs ranking quality: nDCG@10 drops from 0.403 to 0.352 at `ef = 8`.

**So would you ship HNSW?**
Not for one person's data. Exact search over tens of thousands of vectors is under 2 ms, it has no recall loss, and it combines cleanly with filters. An approximate index starts to pay off in the millions of vectors, and it complicates filtering: if you filter after the search you can end up with fewer than k results, and filtering during the graph walk needs index support.

**Why reciprocal rank fusion?**
BM25 scores are unbounded and depend on query length. Cosine similarities sit in a narrow band. Adding them needs calibration. RRF uses only ranks: each list gives a document `1 / (k + rank)`, and the sums are sorted. `k = 60` is the value from the original paper and I did not tune it.

**Hybrid is worse than dense on FiQA. Why, and what would you do?**
BM25 is weak there (0.242 against 0.403) because financial questions and their answers share few words. RRF gives both lists equal weight, so a poor list pushes good dense results down: -0.048, significant. Options, in order: weight the lists and choose the weight on held-out queries, fall back to dense when the two lists barely overlap, or drop fusion for that domain. I did not tune anything on the test queries, so the table shows the default.

**Why did the re-ranker not help on SciFact and NFCorpus?**
It was trained on MS MARCO, which is web search queries and short passages. Scientific claims and medical queries are a different distribution, and the first stage is already strong there. On mailbox questions, which look more like web queries, it adds +0.029 MRR@10. A re-ranker is a domain-dependent component, and it should be measured before it is paid for.

**Why does BM25 beat dense on email?**
Two reasons. People ask about email with the exact names, numbers and terms that are in it, and one vector for a whole email blurs a single detail. And the benchmark questions were written by a model that had read the email, so they reuse its words. The second reason means the gap is probably overstated.

## Evaluation

**Define nDCG@10, recall@k and MRR.**
- nDCG@10: each of the top 10 results adds its relevance grade divided by `log2(rank + 1)`. The sum is divided by the best possible sum for that query. It rewards putting the most relevant documents first.
- Recall@k: the share of a query's relevant documents found in the top k. For a first stage this is the number that matters, because a re-ranker cannot recover what was never retrieved.
- MRR@10: one over the rank of the first relevant result. Right for tasks with one correct answer, such as finding one email.

**How did you check the metric code?**
`tests/test_core.py` compares my nDCG, recall and MRR with the `ranx` library on random runs, equal to nine decimal places, plus hand-computed cases.

**Why bootstrap confidence intervals?**
A test set is a sample of queries. Resampling the queries with replacement 10,000 times and recomputing the mean shows how much the number would move with a different sample. With 300 queries the interval on nDCG@10 is about plus or minus 0.045, so a 0.02 difference between two rows of the table means nothing on its own.

**Then how can you claim any difference?**
Paired tests. Both systems answer the same queries, so I test the per-query differences, which removes the query difficulty that dominates the variance. The test is a randomization test: if the two systems were equivalent, the sign of each difference would be a coin flip. Flip the signs at random 10,000 times and count how often the mean difference is as large as the observed one. Code: `stats.py`.

**Why Holm correction?**
Each dataset has seven to nine comparisons. Testing many pairs at 0.05 produces false positives by chance. Holm controls the chance of any false positive across the family and is less strict than Bonferroni. Example: dense against BM25 on SciFact has a difference interval of [+0.013, +0.081], yet the corrected p-value is 0.078, so I do not claim it.

**Did you tune on the test set?**
No. There is no tuning in the project. All parameters are the textbook defaults. That keeps the test numbers honest and means they are a baseline, not a ceiling.

**What are the weaknesses of your evaluation?**
Three BEIR sets out of eighteen. Synthetic mailbox questions that favour lexical match. A small candidate pool per question (436 emails on average). BEIR judgments are incomplete, so an unjudged relevant document counts as wrong. And I did not check the pretrained models' training data for overlap with the test sets.

## On-device

**What is dynamic int8 quantization?**
Weights are stored as 8-bit integers instead of 32-bit floats, with a scale and zero point per tensor. Activations are quantized on the fly at inference, so no calibration data is needed. The file shrinks from 133.1 MB to 34.1 MB.

**What did it cost?**
Embeddings stay close: mean cosine 0.995 to the full-precision ones. nDCG@10 changes by -0.005 on SciFact and -0.003 on NFCorpus, both inside the confidence interval. Query encoding goes from 6.4 ms to 5.6 ms on SciFact.

**Where did the speed actually come from?**
From the runtime, not the quantization. PyTorch to ONNX Runtime at the same precision is 20.8 ms to 6.4 ms. For a single short query the graph optimizations and lower overhead matter more than integer arithmetic.

**What would you do next for a phone?**
Static quantization with calibration data, a smaller or distilled encoder, Core ML conversion to use the neural engine, and embedding compression (float16 or product quantization) since at scale the vectors, not the model, dominate storage: 88.5 MB for 57,638 documents here.

## Query understanding

**Why rules and not a model for filters?**
A hard filter that is wrong removes the right answer with no way back, so precision matters more than coverage. Rules are transparent, testable and never invent a filter. The parser handles "emails from greg whalley in June 2000" and deliberately ignores "Avi from Mars Corp." (`query_filters.py`, `tests/test_query_filters.py`).

**How much is a filter worth?**
The date filter alone is +0.059 MRR@10 on the mailbox benchmark, more than fusion or re-ranking. It moved the right email to rank 1 for 132 questions.

**Why filter before ranking?**
It is exact and the candidate set is small. Post-filtering an approximate top-k can leave fewer than k results.

## Agent

**How does the agent work?**
A local chat model (Qwen2.5 1.5B) gets two tools described by JSON schemas: `search_emails(query, sender?, date_from?, date_to?)` and `read_email(email_id)`. The loop in `agent.py` sends the conversation to the model. If the reply contains a tool call, the tool runs and its output is appended. The first reply without a tool call is the final answer, which must end with a "Sources:" line. At most six model calls.

**Why is the final answer plain text and not a tool?**
My first version had a `final_answer` tool. The model kept answering in prose after seeing tool results, because that is how chat models are trained to use tools. Working with that habit instead of against it removed most wasted steps.

**Which rules are enforced in code, and why not in the prompt?**
An email can only be cited after `read_email` showed its full text. An answer before any email was opened is sent back. Giving up after one search is sent back once. An identical repeated call is refused. A filter that matches nothing falls back to an unfiltered search. A small model does not follow instructions reliably: in 150 episodes the loop sent back 180 answers given from previews alone.

**What was the result?**
On 150 questions, the agent cites a gold source email 61.3% of the time and single-step retrieval 64.0%, a difference well inside the noise (p = 0.695). The agent's answers score lower (token F1 0.255 against 0.454), it takes 7.6 s against 1.9 s, and when the source emails are hidden it says NOT_FOUND 18.7% of the time against 52.0%.

**Why did the agent not win?**
The retriever already puts the right email in the top 5 for 96% of questions, so there is little for extra searches to recover. The single step shows the model five full emails at once. The agent has to choose which one to open from a preview, and a 1.5B model chooses and summarises poorly. More steps also mean more chances to find an email that looks relevant, which is why it abstains less when the real evidence is gone.

**So are agents a bad idea?**
No. This says a multi-step loop has to earn its latency, and that for single-fact lookups with a strong retriever and a small model it does not. I would expect the loop to pay off for questions that need several emails, for weaker first-stage retrieval, and for larger models. Those are the next experiments.

**How do you score the agent without a human?**
Two automatic scores. Source hit: did it cite an email that the benchmark marks as a source. Token F1 against the reference answer, the SQuAD metric. F1 is rough: a correct answer in other words scores low, and the agent's longer answers (median 20 words against 10) are penalised. That is why I treat the citation rate as the main number. I did not use a language model as a judge, because the judge's errors would be hard to separate from the system's.

**How do you test abstention?**
By hiding the gold emails for 75 questions and counting how often each system still answers. One caveat: the mailbox may hold other emails with the same fact (replies that quote the original), so some of those answers are correct. The comparison between the two systems is still fair, since both see the same mailbox.

**How is the loop tested?**
With a scripted fake model in `tests/test_agent.py`, so the tests need no GPU: tool parsing, each push-back rule, the step limit, hidden evidence and the filter fallback.

## What I would do with more time

1. Index long emails as passages and aggregate scores per email, so a single detail is not blurred by one vector per email.
2. Learn the fusion weight on a held-out split, and test a fallback to dense when the lists disagree.
3. Build a small set of real, human-written mailbox questions, since the synthetic ones favour BM25.
4. Evaluate the filter parser on labelled queries.
5. Run the agent comparison across model sizes to see where multi-step search starts to pay.
6. Port the embedding model to Core ML and measure on a phone.
