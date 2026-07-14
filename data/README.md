# Benchmark Data

`benchmark.jsonl` contains the exact 30 MAMO problems used in the reported experiment. Records are sorted by stable `id`, and the experiment loader enforces 30 unique records split evenly between the two graph topics.

Each JSON object contains:

- `id`: stable experiment identifier;
- `description`: normalized problem description shown only to the simulated user and evaluator;
- `kg_topic`: `Production Planning` or `Travel Salesman Problem`;
- `difficulty`: original MAMO split, `easy` or `complex`;
- `source_row_index`: zero-based row index in the corresponding upstream MAMO JSONL file;
- `selection_reason`: rationale for mapping the instance to the graph topic; and
- `source`: upstream dataset name.

The descriptions are derived from [MAMO](https://github.com/FreedomIntelligence/Mamo) and remain licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Changes comprise subset selection, whitespace normalization, removal of trailing answer-request boilerplate, and addition of metadata. See the repository's `THIRD_PARTY_NOTICES.md` for full attribution.
