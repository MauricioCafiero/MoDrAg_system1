'''
System 1 routing for semantic MoDrAg - drop-in replacements for
input_parsing.intake() and input_parsing.second_intake().

Routing pipeline (probed in probe_router.py; labeled-set results in a comment):
  1. Local embedding model scores all tools (cached to disk; recomputed only
     when tool_descriptions change).
  2. GLiNER parses entities; their counts are fed to the decision model as a
     structured state ('detected' counts + the request) - entity-aware state
     was the single largest accuracy gain (7/20 -> 12/20).
  3. opendecider-nano answers one Choice question over all 19 tools
     (response-cached in decision_layer).
  4. The geometric mean of the decision model's probability distribution and
     the embedding similarity distribution gives the final ranking
     (16/20 fused vs 14/20 embedding-only on the labeled set).
Extraction stays local: GLiNER NER + regex (a decision model cannot return spans).
'''
import os
import numpy as np

import input_parsing as ip
from decision_layer import DecisionEngine

# Capability-phrased criteria: verb-first "Use when..." descriptions beat the
# example-style tool_descriptions for the decision model, and remain stable
# when the example text changes.
TOOL_CRITERIA = {
    'uniprot_node': 'Use when the user wants UNIPROT accession codes, gene names or organisms for a named protein or gene.',
    'canonical_node': 'Use when the user wants the canonical SMILES form of a given SMILES string, or to check if a SMILES is canonical.',
    'similarity_node': 'Use when the user wants Tanimoto-similarity between a reference molecule and a set of other molecules.',
    'listbioactives_node': 'Use when the user gives a UNIPROT accession code and wants ChEMBL IDs and counts of bioactive molecules.',
    'getbioactives_node': 'Use when the user gives a ChEMBL ID and wants bioactive molecule SMILES strings and IC50 values.',
    'predict_node': 'Use when the user wants a predicted IC50 or activity for a given molecule SMILES against a ChEMBL ID.',
    'gpt_node': 'Use when the user wants to generate new/novel molecules for a protein target, by fine-tuning a generative model on its ChEMBL bioactives dataset.',
    'pdb_node': 'Use when the user gives a PDB ID and wants the protein sequence, ligands or chains of that crystal structure.',
    'find_node': 'Use when the user wants PDB structure IDs for a named protein (search the PDB by name).',
    'docking_node': 'Use when the user wants docking (docking scores/poses) for given SMILES strings in a named protein.',
    'blind_dock_node': 'Use when the user wants molecules docked into a PDB receptor structure of unknown binding site (no search box), especially when the protein is not in the named docking target list.',
    'dock_check_node': 'Use when the user wants to verify that a completed blind docking pose landed in the correct binding site (compare against co-crystallized ligands).',
    'target_node': 'Use when the user wants protein targets for a named disease.',
    'substitution_node': 'Use when the user wants analogues of a molecule by substituting groups, with QED scores.',
    'lipinski_node': 'Use when the user wants Lipinski properties (QED, LogP, H-bond donors/acceptors, molar mass, TPSA) for molecules.',
    'pharmfeature_node': 'Use when the user wants a comparison of pharmacophore features between two molecules.',
    'name_node': 'Use when the user wants the name of a molecule identified by its SMILES string.',
    'smiles_node': 'Use when the user wants the SMILES string of a molecule identified by name.',
    'related_node': 'Use when the user wants molecules similar/related to a given molecule.',
    'structure_node': 'Use when the user wants the 2D/3D structure image or chemical formula of a molecule by name or SMILES.',
    'get_actives_for_protein': 'Use when the user wants all bioactive molecules known for a named protein.',
    'get_predictions_for_protein': 'Use when the user wants a predicted IC50 for a given molecule SMILES in a named protein.',
    'dock_from_names': 'Use when the user wants molecules named in plain words docked in a named protein.',
}

ROUTE_INSTRUCTIONS = 'Which tool should execute this request?'


def start_embedding_cached(tool_descriptions_values: list[str], cache_file: str = '.tool_embeddings.npz'):
  '''
  Computes or loads tool-description embeddings keyed by a content hash,
  so they are recomputed only when the descriptions change.
    Returns:
        document_embeddings, embed_model
  '''
  spec = 'v1|' + '|'.join(tool_descriptions_values)

  embed_model = ip.SentenceTransformer("google/embeddinggemma-300m")

  loaded = None
  if os.path.exists(cache_file):
    try:
      data = np.load(cache_file, allow_pickle=False)
      data_spec = data['spec'].item()
      if data_spec == spec:
        loaded = data['emb']
    except Exception:
      pass

  if loaded is not None:
    document_embeddings = loaded
  else:
    document_embeddings = embed_model.encode_document(tool_descriptions_values)
    try:
      np.savez(cache_file, emb=document_embeddings, spec=spec)
    except Exception:
      pass  # cache failure is non-fatal

  return document_embeddings, embed_model


def _ranked_from_probs(probabilities: dict, fallback_tool: str) -> list[str]:
  '''Returns tool names ordered by probability, capped at 3.'''
  ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
  tools = [k for k, _ in ranked if k in ip.tool_descriptions_keys]
  if not tools:
    tools = [fallback_tool]
  return tools[:3]


def route(query: str, parse_model, embed_model, document_embeddings, engine: DecisionEngine,
          context: str = None, fuse_weight: float = 0.7, two_stage: bool = False,
          stage2_candidates: int = 5, stage2_alpha: float = 0.15):
  '''
  Shared routing core for intake and second_intake.
    Args:
        query: The user query.
        context: Optional context string (for second_intake-style routing).
        fuse_weight: Exponent on the decision-model distribution in the
          geometric mean; the embedding distribution gets (1 - fuse_weight).
          0.7 was best on the 95-query real test set.
        two_stage: If True, a second Choice question re-ranks the top
          stage2_candidates tools from stage 1. Its distribution is applied
          softly (exponent stage2_alpha) so it reorders near the top without
          collapsing top-5 recall. 0.15 was best on the real set.
        stage2_candidates: How many of stage-1's best tools enter stage 2.
    Returns:
        best_tools: Top-3 tool names by final probability.
        present, proteins_list, names_list, diseases_list, smiles_list,
        uniprot_list, pdb_list, chembl_list: as parse_input() returns.
    Also sets engine.last_choice (choice/probabilities/confidence derived from
    the final distribution) for the CLI's auto-execute gate.
  '''
  present, proteins_list, names_list, diseases_list, smiles_list, uniprot_list, pdb_list, chembl_list = \
      ip.parse_input(query if context is None else context, parse_model)

  # 1) embedding prior over the same option ordering
  state_text = query if context is None else f'Query: {query}\nContext: {context}'
  q_emb = embed_model.encode_query(state_text)
  sims = np.asarray(embed_model.similarity(q_emb, document_embeddings))[0]
  lo, hi = float(sims.min()), float(sims.max())
  norm = (sims - lo) / (hi - lo + 1e-9) if hi > lo else np.zeros_like(sims)
  emb_p = np.exp(10 * norm)
  emb_p = emb_p / emb_p.sum()

  # 2) System 1 decision over a structured, entity-aware state
  state = {'request': state_text, 'detected_entities': dict(present)}
  choice_ans = engine.choice(state=state, name='tool',
                             question=ROUTE_INSTRUCTIONS,
                             options=dict(TOOL_CRITERIA))
  s1_p = np.array([float(choice_ans.get('probabilities', {}).get(k, 1e-9))
                   for k in ip.tool_descriptions_keys])
  s1_p = s1_p / s1_p.sum()

  # 3) fuse: geometric mean (both distributions are normalized)
  fused = (s1_p ** fuse_weight) * (emb_p ** (1 - fuse_weight))
  fused = fused / fused.sum()

  # 4) optional stage 2: re-rank only the leading candidates with a focused
  #    Choice question; the answer is combined multiplicatively onto the
  #    stage-1 fused distribution (tools outside the candidate set keep their
  #    stage-1 probability, slightly damped).
  if two_stage:
    cand_idx = list(np.argsort(fused)[::-1][:stage2_candidates])
    keys = [ip.tool_descriptions_keys[i] for i in cand_idx]
    cand_options = {k: k + ': ' + TOOL_CRITERIA[k] for k in keys}
    shortlist = ', '.join(keys)
    ans2 = engine.choice(state=state, name='tool_stage2',
                         question=ROUTE_INSTRUCTIONS
                         + f' Choose the single best fit among: {shortlist}.',
                         options=cand_options)
    stage2 = np.array([float(ans2.get('probabilities', {}).get(k, 1e-9))
                       for k in cand_options])
    stage2 = stage2 / stage2.sum()
    final = np.copy(fused)
    for rank, i in enumerate(cand_idx):
      final[i] = fused[i] * (stage2[rank] ** stage2_alpha)
    final = final / final.sum()
  else:
    final = fused

  fused_map = {k: float(v) for k, v in zip(ip.tool_descriptions_keys, final)}

  best_tools = _ranked_from_probs(fused_map, 'uniprot_node')
  top = best_tools[0]
  engine.last_choice = {'choice': top, 'probabilities': fused_map,
                        'confidence': fused_map.get(top, 0.0)}

  return best_tools, present, proteins_list, names_list, diseases_list, \
      smiles_list, uniprot_list, pdb_list, chembl_list


def sys1_intake(query: str, parse_model, embed_model, document_embeddings,
                engine: DecisionEngine):
  '''
  System 1 version of input_parsing.intake(). Same return signature.
  '''
  best_tools, present, proteins_list, names_list, diseases_list, smiles_list, \
      uniprot_list, pdb_list, chembl_list = route(query, parse_model, embed_model,
                                                  document_embeddings, engine)

  proteins_list, smiles_list = _role_disambiguate(best_tools[0], proteins_list, smiles_list, engine, query)

  if present['molecules'] > 0 and present['smiles'] == 0:
    smiles_list, _, _ = ip.smiles_node(names_list)

  return best_tools, present, proteins_list, names_list, diseases_list, \
      smiles_list, uniprot_list, pdb_list, chembl_list


def sys1_second_intake(query: str, context: str, parse_model, embed_model, document_embeddings,
                       engine: DecisionEngine):
  '''System 1 version of input_parsing.second_intake(). Same return signature.'''
  best_tools, present, proteins_list, names_list, diseases_list, smiles_list, \
      uniprot_list, pdb_list, chembl_list = route(query, parse_model, embed_model,
                                                  document_embeddings, engine, context=context)

  proteins_list, smiles_list = _role_disambiguate(best_tools[0], proteins_list, smiles_list, engine, query)

  if present['molecules'] > 0 and present['smiles'] == 0:
    smiles_list, _, _ = ip.smiles_node(names_list)

  return best_tools, present, proteins_list, names_list, diseases_list, \
      smiles_list, uniprot_list, pdb_list, chembl_list


def _role_disambiguate(tool_key: str, proteins_list, smiles_list, engine: DecisionEngine, query: str):
  '''
  When a tool binds entities to roles and the query is ambiguous (e.g. several
  proteins detected but only one receptor), asks the decision model a Choice
  question instead of blindly taking proteins_list[0].
  Returns (proteins_list, smiles_list), possibly reordered.
  '''
  receptor_tools = {'docking_node', 'get_predictions_for_protein', 'dock_from_names'}
  if tool_key in receptor_tools and len(proteins_list) > 1:
    options = {p: f'detected protein/gene: {p}' for p in proteins_list}
    ans = engine.choice(state={'request': query, 'proteins': list(proteins_list)},
                        name='receptor',
                        question='Which protein is the receptor/target for this docking or prediction request?',
                        options=options)
    picked = ans.get('choice')
    if picked in proteins_list:
      proteins_list = [picked] + [p for p in proteins_list if p != picked]
  return proteins_list, smiles_list