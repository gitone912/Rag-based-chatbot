# Post-retrieval-fix evaluation summary

**Run date:** 2026-11-17 (Asia/Kolkata)
**Application model:** Groq `openai/gpt-oss-20b`
**Independent review model:** Qwen `qwen/qwen3.8-27b`
**Corpus and set:** six prepared documents, 36 author-curated questions (15 core plus 21 source-linked local holdout)

## Live run

| Metric | Result |
| --- | ---: |
| Completed / errors / pending | 36 / 0 / 0 |
| Median / p95 latency | 4,346.6 ms / 5,328.6 ms |
| Inline citation presence | 36/36 (100%) |
| Citation ID validity | 36/36 (100%) |
| Citation source metadata mapping | 36/36 (100%) |
| Input / output tokens reported by API | 47,225 / 2,786 |
| Context / output limits | 4,000 characters / 512 tokens |
| Browser search | Disabled |

## Separate-model review

| Measure | Mean |
| --- | ---: |
| Answer correctness | 0.9583 |
| Context faithfulness | 0.9819 |
| Citation support | 0.9806 |
| Evidence sufficient | 36/36 |

The reviewer scored the previously missed RBAC case 1.0 for correctness, faithfulness, and citation support. Its retrieved evidence was `cronjobs.docx` chunk 10 at citation slot 2.

## Scope and limitations

This is a model-assisted evaluation, not human certification. The question/reference set was authored from the same six-document prepared corpus and is not an independently sampled production benchmark. A non-author should review the questions and references before using these values as release gates. These metrics characterize this fixed model, prompt, corpus, and retrieval profile only.

Raw evaluation checkpoints contain retrieved source excerpts and per-case answers. They are intentionally excluded from this public repository; the source-linked holdout is local-only for the same reason. The sanitized aggregate above is the public report.

## Reproducibility identifiers

- Evaluation profile SHA-256: `ef9eeeb162e4d663375617adf26379a95e16d9cf7396ee5784094286327fd2f2`
- Question-set SHA-256: `1996342c616089a070719bee7a845ffa4bb1e666723527658adc2cbf977cdcb7`
- Corpus SHA-256: `cc4c782532eb824fdd1d0eee7e8fa7e51015a22e3a32087936e9ef49af67a9df`
- Separate-review profile match: verified
