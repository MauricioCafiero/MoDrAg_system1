<img src="https://github.com/MauricioCafiero/MauricioCafiero.github.io/blob/main/images/comp_chem_2_small.jpg" height="200" align="top" style="height:240px">

# MoDrAg System 1
MoDrAg's drug-design tool library on the command line, driven by **no autoregressive LLM anywhere in the control flow**. Every tool selection, confirmation, missing-data request, and follow-up is decided by a small local decision model ([opendecider-nano](https://huggingface.co/manjunathshiva/opendecider-nano), ~400M parameters) fused with an embedding scorer — the entire routing stack runs on a laptop.

This repo is a System 1 replacement for [MoDrAg_CLI](https://github.com/MauricioCafiero/MoDrAg)'s driver: upstream interprets your query with a chat LLM that calls tools; here the tool library is kept, and the LLM driver is removed and replaced by deterministic code plus three small models.

- [How a query flows](#how-a-query-flows)
- [The tool set](#the-tool-set)
- [Install and run](#install-and-run)
- [Talking to the agent](#talking-to-the-agent)
- [Benchmark](#benchmark)
- [Testing](#testing)
- [Repo layout](#repo-layout)
- [Adding a tool](#adding-a-tool)
- [Credits](#credits)
- [License](#license)

## How a query flows
All of this lives in `code/` — `sys1_router.py` implements it, `decision_layer.py` wraps the decision engine with caching, `modrag_cli.py` drives the REPL loop. There is **no chat LLM**: tool results are shown to you raw and unedited.

1. **Entity intake** (`sys1_intake`) — [GLiNER-biomed](https://huggingface.co/knowledgator/gliner-biomed-large-v1.0) plus regex pull proteins, molecule names, diseases, SMILES strings, Uniprot accession codes, PDB IDs, and ChEMBL IDs out of your query.
2. **Embedding scoring** — [EmbeddingGemma 300M](https://huggingface.co/google/embeddinggemma-300M) scores each tool's *decision criterion* ("Use when…") against your query, by cosine similarity. The tool embeddings are cached in a content-keyed `.tool_embeddings.npz` (rebuilt automatically whenever a tool description changes).
3. **Decision-model choice** — opendecider-nano answers a calibrated `Choice` question over the tool descriptions, conditioned on an entity-state JSON of what your query and the session contain.
4. **Fusion** — the embedding similarity and the decision model's choice score are combined by geometric mean (decision weight 0.7, default). An optional second `Choice` stage re-ranks the fused top 5 (off by default; trades Top-5 recall for Top-1).
5. **Confidence gating** — if the fused confidence of rank #1 is ≥ `--auto-threshold` (0.6), the tool runs without asking. Below that, the agent shows you the ranked tools and you confirm.
6. **Missing data** — before running, `define_tool_reqs` checks what the tool needs against the entities found (and the carried-over session entities). Anything missing is asked for in **one concise prompt**, and the exact gap is filled from your reply.
7. **Sequential tools** — after a tool runs, a calibrated `Noul` question ("to fully complete the request, is another tool needed after this one?") gates a second route, computed on the query-plus-results context, with the first tool's entities carried forward — e.g. *get actives for DNA gyrase then generate new molecules*, or *retrieve SMILES then dock them*, up to `--max-tools` (2) tools.
8. **Session state** — a full transcript (queries, tools, results) is kept for the `memory`/`recall` vault, and entity memory carries entities across turns until `reset`.

**Why no LLM?** Routing over a flat library of named tools is a classification problem, not a language-generation problem — a 400M decision model plus a 300M embedder handle it deterministically, offline, on a laptop, with calibrated probabilities you can gate on and cache.

## The tool set
23 tools are wired through the router (tool registry: `code/input_parsing.py`; graphs and glue: `code/modrag_task_graphs.py`, `code/chain_tools.py`; implementations ported from MoDrAg's library).

**Lookups** — protein/target/data retrieval:
| Tool | Does |
|---|---|
| `uniprot_node` | Uniprot accession codes (with organisms/gene names) for protein names |
| `target_node` | protein targets for a disease (Open Targets) |
| `find_node` | PDB IDs available for a protein |
| `pdb_node` | sequence, ligand, chain count for a PDB ID |
| `listbioactives_node` | ChEMBL IDs attached to Uniprot IDs |
| `getbioactives_node` / `get_actives_for_protein` | bioactive molecules + IC50s for a ChEMBL ID / for a protein name directly |

**Molecule conversion and search**:
| Tool | Does |
|---|---|
| `smiles_node` / `name_node` | names → SMILES, SMILES → names |
| `structure_node` | drawn structure for a name/SMILES |
| `canonical_node` | canonical SMILES form |
| `related_node` / `similarity_node` | related/similar molecules via PubChem |

**Properties**:
| Tool | Does |
|---|---|
| `lipinski_node` | full Lipinski/QED property panel for SMILES |
| `pharmfeature_node` | pharmacophore feature comparison of two molecules |

**Modelling (ChEMBL bioactives)**:
| Tool | Does |
|---|---|
| `predict_node` / `get_predictions_for_protein` | LightGBM IC50 prediction trained on the (found) bioactives of a ChEMBL ID / a protein |
| `gpt_node` | fine-tunes the ZN305 GPT checkpoint on the bioactives and generates novel ligands |

**Docking (Vina)**:
| Tool | Does |
|---|---|
| `docking_node` | dock SMILES into a named target via Dockstring; scores + pose |
| `blind_dock_node` | blind dock into a PDB receptor when the binding site is unknown |
| `dock_check_node` | verify the blind-dock pose landed in a plausible binding site |
| `dock_from_names` | multi-tool graph: molecule + protein names → docking score |

## Install and run
```
git clone https://github.com/MauricioCafiero/MoDrAg_system1.git
cd MoDrAg_system1
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
bash setup_alias.sh
modragsys1
```
`setup_alias.sh` installs a `modragsys1()` shell function (not `modrag` — that name stays free for MoDrAg_CLI, so both CLIs can be installed side by side). Without the alias, run `.venv/bin/python code/modrag_cli.py` from anywhere: the CLI self-heals its working directory to `code/` (the tool modules resolve repo-root dirs as `../data`, `../outputs`, …) and creates the runtime dirs on first launch.

The GPT weights and vocabularies used by the generative tool ship in `data/`. The three routing models download from the HuggingFace hub on first launch, are cached in `~/.cache/huggingface/hub`, and after that the CLI detects the cache and runs fully offline. Everything is local and free — no OpenAI/Anthropic/paid API anywhere.

CLI flags: `--auto-threshold` (0.6), `--followup-threshold` (0.6), `--max-tools` (2), `--router {sys1,embedding}` (embedding-only argmax routing for comparison), `--model` (opendecider checkpoint), `--no-cache` (decision caching off), and `MODRAG_DEBUG=1` for tracebacks from tool errors.

## Talking to the agent
At the REPL:
- your query, in plain English — `dock aspirin in drd2`, `generate novel ligands for CHEMBL213`,
- when confidence is below threshold, the ranked tool list waits for your choice: `enter` for #1, `2`/`3` to pick another, `no` to start the query over,
- when data is missing, the agent names the exact gap and waits for just that piece (a reply of `MAOB` fills the slot and runs the tool) — accepted unparsed text is stored as-is,
- `memory` (or `save memory` / `remember`) — save the session transcript to the vault; `recall` lists sessions; `recall <date>` / `recall last` restores one, whose molecules/proteins/results become reusable state,
- `reset` — new session, clears carried-over entities; `quit` — exit.

## Benchmark
Measured on **95 real drug-design queries** (`test/real_queries.json`), compared against the LLM-router-era baseline from the workbook where the queries came from and the embedding-argmax router without decision fusion:

| Router | Top-1 | All-expected-in-Top-3 | Top-5 |
|---|---|---|---|
| Human baseline (workbook) | 0.337 | 0.642 | 0.832 |
| Embedding argmax alone | 0.326 | 0.516 | 0.789 |
| **Fused System 1** | **0.379** | **0.653** | **0.779** |

(At 18 tools the fusion scored 0.389 / 0.705 / 0.853, beating the baseline on all three metrics; adding richer docking/generation tools diluted Top-5 — the routing descriptions retune is in progress.)

Reproduce: `.venv/bin/python test/eval_real.py sys1` (per-query miss listing included), or sweep the fusion parameters with `test/sweep_real.py`.

## Testing
Suites live in `test/` and self-configure (they add `../code`, cd there, create the runtime dirs, then run) — call them from anywhere:

- Fast, offline (no external APIs; the shipped GPT checkpoint plus a local blind-dock round-trip):
  ```
  .venv/bin/python test/tool_tests.py
  ```
- Full, every wired tool — `MODRAG_LIVE=1` hits the live APIs (PubChem, ChEMBL, RCSB), docks for real, and fine-tunes the GPT on live CHEMBL213 bioactives (minutes on CPU):
  ```
  MODRAG_LIVE=1 .venv/bin/python test/tool_tests_full.py
  ```
- Routing checks: `test/eval_router.py` (synthetic queries), `test/probe_router.py` (feed one query and see the fused distribution), `test/single_test.py`, `test/proteins_test.py`, `test/smiles_node_test.py`.

## Repo layout
```
setup_alias.sh           # installs the `modragsys1` shell function
requirements.txt         # dependencies
code/                    # the CLI, the System 1 router, and the tool library
  modrag_cli.py                  # REPL entry point + routing gates (this is the agent)
  sys1_router.py                 # intake → embedding scoring → decision fusion → route()
  decision_layer.py              # cached DecisionEngine wrapper over opendecider
  input_parsing.py               # tool registry: descriptions, hash, required-data lists
  chain_tools.py                 # sequential-glue wrappers + file-path extraction
  modrag_task_graphs.py          # multi-tool graphs (dock_from_names)
  modrag_protein_functions.py    # Uniprot/PDB/ChEMBL/docking nodes
  modrag_molecule_functions.py   # SMILES/name/analogue/similarity/bioactives drawing nodes
  modrag_property_functions.py   # lipinski / pharmacophore feature nodes
  gpt_node.py / finetune_gpt.py / CafChemGPT.py / smiles_tokenizer.py  # generation stack
  modrag_memory.py               # session vault (memory/recall)
  vina_dock.py / subs_code.py / modrag_ref.py / ...  # docking substrate, substitution helpers
test/                    # benchmark + suites (all runnable from anywhere)
data/                    # shipped GPT weights + tokenizer vocabularies
```

## Adding a tool
A tool is a function of that library shape (SMILES, molecule names, protein names, diseases, ChEMBL IDs, Uniprot codes, or PDB IDs as input; returns exactly `(list, string, images_or_None)`):

1. Implement `whatever_node(...)` in `code/modrag_protein_functions.py`, `code/modrag_molecule_functions.py`, or `code/modrag_property_functions.py` (a graph of tools goes in `code/modrag_task_graphs.py`).
2. Register it three places in `code/input_parsing.py`: a one-line decision criterion in `tool_descriptions` (what the embedder scores and the decision model reads — phrase it as *when to use the tool*), the call in `define_tool_hash`, and the required-data entries in `define_tool_reqs` (this is what lets the agent ask you for missing data).
3. If it should chain to/from other tools, wire the glue into `code/chain_tools.py`, and put any file-path extraction in `extract_file_paths`.
4. Write it a test in `test/tool_tests_full.py`, and note new dependencies in `requirements.txt`.

The decision model routes to it as soon as its description is embedded (the embedding cache is content-keyed, so it rebuilds itself).

## Credits
The tool library is ported from [MoDrAg / MoDrAg_CLI](https://github.com/MauricioCafiero/MoDrAg) (CafChem) — docking via [Dockstring](https://github.com/dockstring/dockstring) + [AutoDock Vina](https://github.com/ccsb-scripps/AutoDock-Vina), properties via RDKit/LightGBM, lookups via PubChem/ChEMBL/Open Targets/RCSB. What is *new* in this repo is the control plane: opendecider-nano, EmbeddingGemma-300M, and GLiNER-biomed replacing an autoregressive driver, with the fusion/caching/gating stack in this repo's `code/`.

## License
MIT — see [LICENSE](LICENSE).