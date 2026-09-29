<img src="https://github.com/MauricioCafiero/MauricioCafiero.github.io/blob/main/images/comp_chem_2_small.jpg" height="200" align="top" style="height:240px">

# MoDrAg System 1
A *Mo*dular *Dr*ug-design AI *Ag*ent, driven entirely by a small decision model — **no autoregressive LLM anywhere in the control flow**, and it runs on the command line!

- [System 1 routing](#system-1-routing)
- [Abilities](#abilities)
- [Philosophy](#philosophy)
- [Getting started](#getting-started)
- [Using the agent](#using-the-agent)
- [Session memory and recall](#session-memory-and-recall)
- [Testing](#testing)
- [Benchmark](#benchmark)
- [Repo layout](#repo-layout)
- [How to add a function to the agent](#how-to-add-a-function-to-the-agent)

## System 1 routing
Where most agent frameworks ask an autoregressive chat LLM "which tool should I call?", this agent asks a **small decision model** ([opendecider-nano](https://huggingface.co/manjunathshiva/opendecider-nano), ~400M parameters). The flow:

- **Input parsing** — a GLiNER biomedical NER model plus REGEX pull the entities out of your query: protein names, molecule names, SMILES strings, diseases, Uniprot accession codes, PDB IDs, and ChEMBL IDs.
- **Routing** — an embedding model ([EmbeddingGemma 300M](https://huggingface.co/google/embeddinggemma-300m)) scores every tool description against the query, and the decision model fuses these scores with its own calibrated tool-choice scores (geometric-mean fusion) to pick the tool.
- **Confidence gating** — above the confidence threshold (default 0.6) the chosen tool executes automatically; below it, the agent shows you the ranked tools and you confirm (`enter` to accept #1, `2`/`3` to pick another, `no` to start over).
- **Sequential tool use** — after a tool runs, a second gate asks "is another tool needed after this one?" and, if so, re-routes on the results — carrying molecules, proteins, and generated files forward. If data for the chosen tool is missing, the agent knows exactly what it is asking for and prompts you for only the missing pieces.

The decision model, the embedder, and the NER model together are smaller than most single chat LLMs — the whole stack runs locally on a laptop.

## Abilities
A nicely featured drug-design helper pipeline, including:

**Premium Features**
- Dock a molecule in a protein using AutoDock Vina — you get the score *and* the pose (Docking score and pose XYZ via Dockstring; blind docking into receptor structures from the PDB/DUDE database).
- Use LightGBM to create a model that predicts IC50 values for novel molecules. It trains itself on a ChEMBL bioactives dataset, which MoDrAg can find for you.
- Fine-tune a GPT on a ChEMBL bioactives dataset (found by MoDrAg as well!) to generate novel ligands for a protein.

##### *these features take a bit longer than the standard features. The last two require a ChEMBL dataset ID, which MoDrAg can find for you.

**Standard Features**
- Find targets for a disease,
- Find Uniprot IDs for a protein,
- Find ChEMBL IDs for a given Uniprot ID,
- Find bioactive molecules for a given ChEMBL ID,
- Find PDB IDs for a given protein,
- Find sequences, ligands, and numbers of chains for a PDB ID,
- Get SMILES strings for molecules, or names for SMILES strings,
- Canonicalize SMILES,
- Search PubChem for similar molecules or generate analogues,
- Substitute groups on a molecule at a chosen position,
- Find Lipinski properties of molecules,
- Check pharmacophore overlap between two molecules, and
- Chain these steps together — e.g. *dock dopamine in MAOB* (by name), or *find actives for a protein and train a predictor on them* — without an LLM writing any of the glue.

## Philosophy
- **No autoregressive LLM in the loop.** A small decision model selects the tools and gates follow-ups; deterministic code does everything else. Tool results are shown to you raw — not reinterpreted by an LLM.
- **Strong human-in-the-loop functionality:** you approve the tools before deployment; you check (and can edit) the data going to the tools before it is sent; anything below the confidence threshold waits for your confirmation.
- **Let each model do what it does best:** a NER model plus REGEX to read the query, an embedding model plus a small decision model to route, and task-specific models (dockstring, Vina, LightGBM, a fine-tuned GPT) to do the chemistry.
- **Everything 'open,'** avoiding paid services — no OpenAI or Anthropic APIs, etc. The models run locally from the HuggingFace hub.
- **Small models, so it can be deployed almost anywhere!**

## Getting started
```
git clone https://github.com/MauricioCafiero/MoDrAg_system1.git
cd MoDrAg_system1
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Then run the agent — easiest is the `modragsys1` shell function, which `setup_alias.sh` installs (`bash setup_alias.sh`); it cds into `code/` in a subshell so the CWD-relative writes always land right. Named after the original CLI (`modrag`, which stays free for [MoDrAg_CLI](https://github.com/MauricioCafiero/MoDrAg) so both can be installed side by side):
```
modragsys1
```
or directly:
```
.venv/bin/python code/modrag_cli.py
```
(from any directory — the CLI self-heals its CWD to `code/` and creates the runtime dirs on first launch).

The GPT weights and tokenizer vocab needed by the generative tool ship in `data/`. The three routing models (decision model, embedding model, GLiNER NER) are pulled from the HuggingFace hub on first launch and cached in `~/.cache/huggingface/hub` — every launch after that detects the cache and runs fully offline.

## Using the agent
The REPL understands, at any prompt:
- your query, in plain English (`dock aspirin in drd2`, `calculate the lipinski properties of caffeine`),
- `enter` to accept tool choice #1 when the agent is waiting for confirmation, `2`/`3` to pick another, `no` to start over,
- the missing pieces, when the agent tells you exactly which data a tool still needs (`MAOB`, `CC(=O)OC1=CC=CC=C1C(=O)O`, ...),
- `1` or `2` after a multi-step answer to run follow-on graphs (*dock these molecules*, *get actives for this protein*),
- `memory` (or `save memory`) — save this session to the vault; `recall` — list saved sessions; `recall <date>` — reload one; `reset` — start a fresh session (clears carried-over entities); `quit` — exit.

Optional flags for `modrag_cli.py`:
| Flag | Default | Meaning |
|---|---|---|
| `--auto-threshold` | `0.6` | fused confidence above which the chosen tool auto-executes |
| `--followup-threshold` | `0.5` | gate probability needed to chain a second tool |

Set `MODRAG_DEBUG=1` for full tracebacks out of the tool nodes.

## Session memory and recall
The agent keeps a full session transcript (your queries, the tools chosen, and every tool result). `memory` writes the session into `vault/`; `recall` lists what is in the vault, and `recall <date>` restores a session so its molecules, proteins, and results can be reused in new queries.

## Testing
Two suites live in `test/` — runnable from **anywhere** (each one self-configures: it adds `../code` to `sys.path`, cd's there, and creates the runtime dirs):

- **Fast, offline** (no external APIs; uses the shipped model + a local blind-dock round-trip):
  ```
  .venv/bin/python test/tool_tests.py
  ```
- **Full, all 23 wired tools** — add `MODRAG_LIVE=1` to hit the live APIs (PubChem, ChEMBL, RCSB) and actually dock and fine-tune:
  ```
  MODRAG_LIVE=1 .venv/bin/python test/tool_tests_full.py
  ```

## Benchmark
Routing was measured on **95 real drug-design queries** (`real_queries.json`) — this is the System 1 replacement for the autoregressive router:

| Router | Top-1 | All-expected-in-Top-3 | Top-5 |
|---|---|---|---|
| Excel workbook (human baseline) | 0.337 | 0.642 | 0.832 |
| Embedding argmax alone | 0.326 | 0.516 | 0.789 |
| **Fused System 1 (18 tools)** | **0.389** | **0.705** | **0.853** |
| Fused System 1 (21 tools) | 0.379 | 0.653 | 0.779 |

Metrics: fraction of queries where the expected tool is ranked 1st; where all expected tools appear in the top 3; and in the top 5. Fusion (geometric mean of embedding score and decision-model choice score, decision weight 0.7) beats the human baseline on all three metrics at 18 tools; the work to retune the newer tools' routing descriptions back to that level is ongoing. Re-run the sweep yourself with `test/sweep_real.py`.

## Repo layout
```
setup_alias.sh           # installs the `modragsys1` shell function into your shell config
requirements.txt         # full dependency list
code/                    # the CLI + System 1 router + the tool library (nodes)
  modrag_cli.py                  # the command-line REPL (entry point)
  decision_layer.py              # cached System 1 decision engine
  sys1_router.py                 # intake parsing → embedding + decision fusion → route()
  modrag_protein_functions.py    # docking, Uniprot, PDB, bioactives
  modrag_molecule_functions.py   # SMILES, names, analogues, Lipinski, ...
  modrag_property_functions.py   # molecular properties
  modrag_task_graphs.py          # multi-tool graphs (dock_from_names, ...)
  input_parsing.py               # tool registry: descriptions, hashes, requirements
  chain_tools.py                 # sequential-tool glue + file-path extraction
  gpt_node.py / finetune_gpt.py  # ligand generation + fine-tuning
  modrag_memory.py               # session vault (memory/recall)
test/
  eval_router.py         # quick routing checks
  probe_router.py        # probe single queries through the router
  eval_real.py           # benchmark over real_queries.json
  sweep_real.py          # hyperparameter sweep for the router
  real_queries.json      # the 95-query benchmark set
  tool_tests.py          # fast offline suite
  tool_tests_full.py     # full 23-tool suite (MODRAG_LIVE=1)
  single_test.py / proteins_test.py / smiles_node_test.py  # focused node tests
data/                    # GPT weights, tokenizer vocabularies
```

## How to add a function to the agent
- A function can easily be added if it only requires input in the form of: SMILES, molecule names, protein names, disease names, ChEMBL IDs, Uniprot accession codes, or PDB IDs.
- The function should be of the form (substitute the function's task for the word function):
```
def function_node(arg1: type, arg2: type):
  '''
  Doc string with args and returns
  '''
  [function body]

  returns a_list, a_string, an_image_list
```
The function can take any number of arguments but must return exactly 3: a list (can be nested), a string containing the function results in text form (shown to the user), and an image or list of images (optional — given as a list either way, `None` if none). If the function is a *graph* of other tools (like `dock_from_names`), put it in `modrag_task_graphs.py`.
- The function can be added to *`modrag_protein_functions`, `modrag_molecule_functions`,* or *`modrag_property_functions`*. Dependencies go at the top of the file, and into `requirements.txt` for installation.
- A description should be added to the ```tool_descriptions``` dictionary in `code/input_parsing.py`. This description is what the System 1 router embeds and fuses against the user's query — phrase it as a decision criterion for when the tool *should* be chosen (e.g. *"Use when a new molecule must be drawn from a name"*), because that is what the small decision model reads.
- The function name and argument list should be added to the ```define_tool_hash``` function in `input_parsing.py`. This hash table is used to run the selected tool.
- The function name, argument list, and a human-readable version of the arguments should be added to the ```define_tool_reqs``` function in `input_parsing.py`. This hash table is used to check for the required data before running a tool, and for asking the user to provide any missing data.
- If the tool should be run as a follow-up to another tool (or should trigger one), wire it into `code/chain_tools.py` and the follow-up gate in `code/modrag_cli.py`.

That should be it! The decision model should be able to select for and deploy the new function.

## Credits
Built on the [MoDrAg](https://github.com/MauricioCafiero/MoDrAg) tool library (CafChem). Routing uses [opendecider-nano](https://huggingface.co/manjunathshiva/opendecider-nano), [EmbeddingGemma 300M](https://huggingface.co/google/embeddinggemma-300m), and [GLiNER-biomed](https://huggingface.co/knowledgator/gliner-biomed-large-v1.0). Generation fine-tunes the ZN305 GPT in `data/`; docking uses [Dockstring](https://github.com/dockstring/dockstring) and [AutoDock Vina](https://github.com/ccsb-scripps/AutoDock-Vina); properties use RDKit and LightGBM; lookups hit PubChem, ChEMBL, and RCSB — all open and free.

## License
MIT — see [LICENSE](LICENSE).