# data/sample_pdfs

Put your own lecture PDFs here to use them with the developer tools and the
evaluation script. **Everything in this folder except this README is
git-ignored**, because lecture slides and notes are usually copyrighted by
your university or lecturer and should not be published on GitHub.

## What works well

- PDFs with **selectable text**: exported slides, typed notes, handouts.
  Quick test: open the PDF and try to select a sentence with the mouse.
- Several related PDFs from one course, so retrieval has to pick the right file.

## What does not work

- **Scanned or photographed pages** (image-only PDFs). NoteMind does not do
  OCR, so it shows a "looks scanned" warning and cannot search those pages.
- Files renamed to `.pdf` without being exported. Use **Save as PDF** or
  **Print to PDF** instead; the loader reports a clear error otherwise.

## Commands that use this folder

```powershell
python -m notemind.pdf_loader data\sample_pdfs\lecture1.pdf   # inspect pages and chunks
python -m notemind.hybrid "your question"                     # compare search modes
python -m notemind.generator "your question"                  # answer with citations
python -m notemind.quiz data\sample_pdfs\lecture1.pdf 5       # make a 5-question quiz
```

## Evaluating on your own notes

The included evaluation set (`eval/eval_set.json`) is labelled against the demo
notes in `assets/demo/`. To evaluate on your own course:

1. Copy `eval/eval_set.json` to `eval/my_eval_set.json` and set
   `"pdf_dir": "data/sample_pdfs"`.
2. Replace the questions. For each one, open the PDF, find the page(s) that
   answer it, and list them under `expected`. **Paraphrase** instead of copying
   sentences from the page, or keyword search gets an unfair advantage.
3. Add 8-10 questions your notes do **not** answer (`"answerable": false`,
   `"expected": []`), including a few near-misses on the same subject.
4. Run `python eval/run_eval.py --eval-set eval/my_eval_set.json`.

Aim for at least 30 answerable questions. With fewer, differences of a few
percentage points between configurations are within the noise (the results
table shows 95% confidence intervals for this reason).
