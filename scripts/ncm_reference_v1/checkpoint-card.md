---
license: mit
pipeline_tag: text-generation
---

# NCM checkpoints

Released tokenizer and generator pairs for Next-Concept Modeling. Each folder holds a `config.yaml` with the architecture and sampling settings and the `.pt` checkpoints it names.

- Paper: [Next-Concept Modeling](https://huggingface.co/papers/2610.14162)
- Project page: [Hazy Research blog](https://hazyresearch.stanford.edu/blog/2026-10-12-ncm)
- Code: [GitHub](https://github.com/HazyResearch/ncm)

| folder | task |
|---|---|
| `xsum/` | XSum summarization |
| `wmt14/` | WMT14 German-English translation |
| `sudoku/` | Sudoku, easy / medium / hard |
| `lm1b/` | One Billion Word, small / medium / large with a shared tokenizer |
| `tinystories/` | TinyStories |
| `owt/` | OpenWebText |

## Citation

```bibtex
@misc{kumbong2026nextconcept,
  title   = {Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation},
  author  = {Kumbong, Hermann and Khare, Ishan S. and Mlauzi, Iddah and Woen, Andrew and Ding, Charles and Buchanan, E. Kelly and Arora, Simran and Mirhoseini, Azalia and R{\'e}, Christopher},
  year    = {2026},
  eprint={2610.14162},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2610.14162},
}
```