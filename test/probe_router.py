'''
Probe: which state/criteria formulation makes opendecider-nano route best?
Variants: raw state + example-criteria (current), entity-state, short rewritten criteria.
'''
import json
import sys

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'code'))
os.chdir(os.path.join(_HERE, '..', 'code'))

from opendecider import load, Choice
from opendecider.questions import as_dict

from eval_router import LABELED_QUERIES
import input_parsing as ip

# Shorter, capability-phrased criteria (verb-first: "Use when ...")
SHORT_CRITERIA = {
    'uniprot_node': 'Use when the user wants UNIPROT accession codes, gene names or organisms for a named protein or gene.',
    'listbioactives_node': 'Use when the user gives a UNIPROT accession code and wants ChEMBL IDs and counts of bioactive molecules.',
    'getbioactives_node': 'Use when the user gives a ChEMBL ID and wants bioactive molecule SMILES strings and IC50 values.',
    'predict_node': 'Use when the user wants a predicted IC50 or activity for a given molecule SMILES against a ChEMBL ID.',
    'gpt_node': 'Use when the user wants to generate new/novel molecules for a target, by fine-tuning a generative model on a ChEMBL dataset.',
    'pdb_node': 'Use when the user gives a PDB ID and wants the protein sequence, ligands or chains of that crystal structure.',
    'find_node': 'Use when the user wants PDB structure IDs for a named protein (search the PDB by name).',
    'docking_node': 'Use when the user wants docking (docking scores/poses) for given SMILES strings in a named protein.',
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


def entity_state(query: str, parse_model) -> dict:
  present, _, _, _, _, _, _, _ = ip.parse_input(query, parse_model)
  return {'request': query, 'detected': present}


def main():
  from gliner import GLiNER
  print('loading NER for entity-state variant...')
  ner = ip.start_ner()
  model = load('manjunathshiva/opendecider-nano')

  # embedding prior for fusion
  import numpy as np
  embed_model = ip.SentenceTransformer('google/embeddinggemma-300m')
  doc_emb, _ = ip.start_embedding(ip.tool_descriptions_values)
  q_embs = embed_model.encode_query([q for q, _ in LABELED_QUERIES])
  sims = np.asarray(embed_model.similarity(q_embs, doc_emb))
  sims = (sims - sims.min(1, keepdims=True)) / (sims.max(1, keepdims=True) - sims.min(1, keepdims=True) + 1e-9)
  emb_probs = np.exp(10 * sims)
  emb_probs /= emb_probs.sum(1, keepdims=True)

  variants = {
      'raw+example': lambda query: (query, 'Which tool should execute this request?', dict(ip.tool_descriptions)),
      'raw+short': lambda query: (query, 'Which tool should execute this request?', SHORT_CRITERIA),
      'entities+short': lambda query: (entity_state(query, ner), 'Which tool should execute this request?', SHORT_CRITERIA),
      'entities+example': lambda query: (entity_state(query, ner), 'Which tool should execute this request?', dict(ip.tool_descriptions)),
  }

  s1_rows = {}
  for label, fn in variants.items():
    correct = 0
    misses = []
    for (query, want), emb_p in zip(LABELED_QUERIES, emb_probs):
      state, instructions, options = fn(query)
      r = model.system_one(state, {'tool': Choice(instructions, options)})
      a = r['answers']['tool']
      got = a['choice']
      if got == want:
        correct += 1
      else:
        misses.append(f'  got {got:<28} want {want:<28} <- {query}')

    # fusion: geometric mean of normalized sys1 probs and embedding distribution
    fused_correct = 0
    for (query, want), emb_p in zip(LABELED_QUERIES, emb_probs):
      state, instructions, options = fn(query)
      r = model.system_one(state, {'tool': Choice(instructions, options)})
      s1_p = np.array([r['answers']['tool']['probabilities'].get(k, 1e-9) for k in ip.tool_descriptions_keys])
      s1_p = s1_p / s1_p.sum()
      fused = (s1_p * emb_p) ** 0.5
      if ip.tool_descriptions_keys[int(np.argmax(fused))] == want:
        fused_correct += 1

    print(f'\n{label}: sys1 {correct}/{len(LABELED_QUERIES)}, '
          f'fused(embed prior + sys1 probs): {fused_correct}/{len(LABELED_QUERIES)}')
    for m in misses:
      print(m)


if __name__ == '__main__':
  main()