# Third-Party Notices

This repository contains selected or adapted material from the following projects. The repository-level Apache 2.0 license does not replace the licenses listed here.

## MAMO benchmark

- Project: **Mamo: a Mathematical Modeling Benchmark with Solvers**
- Authors: Xuhan Huang, Qingning Shen, Yan Hu, Anningzhe Gao, and Benyou Wang
- Source: <https://github.com/FreedomIntelligence/Mamo>
- License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- Material used: 30 English problem descriptions selected from the EasyLP and ComplexLP files.
- Changes: selected a two-topic subset, normalized whitespace, removed trailing answer-request boilerplate, and added stable identifiers, topic mappings, source row indices, and selection rationales.

The derived file `data/benchmark.jsonl` remains available under CC BY-SA 4.0. A copy of the license is available at <https://creativecommons.org/licenses/by-sa/4.0/legalcode>.

Suggested citation:

```bibtex
@misc{huang2024mamo,
  title={Mamo: a Mathematical Modeling Benchmark with Solvers},
  author={Xuhan Huang and Qingning Shen and Yan Hu and Anningzhe Gao and Benyou Wang},
  year={2024},
  eprint={2405.13144},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

## AgentMILO knowledge graphs

- Project: **AgentMILO: A Knowledge-Based Framework for Complex MILP Modelling Conversations with LLMs**
- Authors: Jyotheesh Gaddam, Lele Zhang, Vicky Mak-Hau, John Yearwood, Bahadorreza Ofoghi, and Diego Molla-Aliod
- Source: <https://github.com/arc2022-deakin/AgentMILO>
- License: Apache License 2.0
- Material used: Production Planning and Traveling Salesman Problem graph structure.
- Changes: adapted and exported into DOT, Mermaid-like text, and rendered PNG representations for controlled modality comparisons.

The upstream repository identifies Vicky Mak-Hau as the Production Planning graph's creator and Lele Zhang as the Traveling Salesman Problem graph's creator.
