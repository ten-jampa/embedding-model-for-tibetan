# Lessons

- RKTS collection pages use inconsistent legacy table markup around the collection source label, so the parser should try more than one boundary pattern instead of assuming clean closing tags.
- The RKTS raw text files are consistently line-addressed transliteration sources, and preserving the original file while parsing line references separately keeps downstream cleanup reversible.
- For conjoined outputs, counting explicit `xxx` placeholders separately from empty-body drops keeps quality stats interpretable across collections with many blank addressed lines.
- Space-joining retained line bodies is a practical default for later `pyewts` conversion because it avoids accidental token fusion at original line breaks.
- Running pipeline verification through the intended conda env (`embedding-tibetan-env`) avoids environment drift between local installs and reproducible project execution.
- In OCR-heavy collections, Unicode outputs can still contain ASCII-like fragments after conversion; warning manifests are useful for triaging noisy segments without stopping the full run.
