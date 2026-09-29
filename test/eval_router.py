'''
Router comparison harness: embedding-router vs System 1 decision-model router
on hand-labeled paraphrased queries (the tool_descriptions themselves are
training-like, so labels here deliberately use different phrasings/things).
Usage: python eval_router.py [--sys1]
'''
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'code'))
os.chdir(os.path.join(_HERE, '..', 'code'))

import input_parsing as ip
from decision_layer import DecisionEngine, DECISION_MODEL_ID

# (query, correct tool) - paraphrases, not the description text itself
LABELED_QUERIES = [
    ('what organism makes the protein TP53?', 'uniprot_node'),
    ('get me accession numbers for the BRCA1 gene', 'uniprot_node'),
    ('how many active compounds are known for the protein with accession P49959?', 'listbioactives_node'),
    ('show me the inhibitor molecules and IC50 values for CHEMBL3265', 'getbioactives_node'),
    ('estimate the inhibition constant of "CC(=O)Oc1ccccc1C(=O)O" against CHEMBL2119632', 'predict_node'),
    ('design some new compounds that bind acetylcholinesterase', 'gpt_node'),
    ('what ligands were in the crystal structure with accession code 8HGK?', 'pdb_node'),
    ('search the protein databank for structures of HIV-1 protease', 'find_node'),
    ('dock imatinib into BCR-ABL and give me the score', 'docking_node'),
    ('which genes are linked to Alzheimer disease', 'target_node'),
    ('make analogs of sildenafil and rank their drug-likeness', 'substitution_node'),
    ('compute QED and logP for "c1ccccc1O"', 'lipinski_node'),
    ('are the binding sites of naproxen and flurbiprofen similar?', 'pharmfeature_node'),
    ('can you tell me the identity of this string CC(=O)Nc1ccc(O)cc1', 'name_node'),
    ('give me the smiles for lisdexamfetamine', 'smiles_node'),
    ('find me compounds that look like adenosine', 'related_node'),
    ('draw out what this is: aspirin', 'structure_node'),
    ('give me the inhibitors of the protein encoded by MAOB', 'get_actives_for_protein'),
    ('what IC50 would "O=C(O)c1ccccc1O" have in the protein HMGCR?', 'get_predictions_for_protein'),
    ('dock naproxen and ibuprofen in COX-2', 'dock_from_names'),
]

RECEPTOR_ROLES = {'docking_node', 'get_predictions_for_protein', 'dock_from_names'}


def embedding_top1(query, embed_model, document_embeddings):
  import numpy as np
  q = embed_model.encode_query(query)
  scores = embed_model.similarity(q, document_embeddings)
  return ip.tool_descriptions_keys[int(scores[0].argmax())]


def sys1_top1(query, engine):
  ans = engine.choice(state=query, name='tool',
                      question='Which tool should execute this request?',
                      options=dict(ip.tool_descriptions))
  return ans.get('choice')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--sys1', action='store_true', help='also run the decision-model router')
  args = ap.parse_args()

  embed_model = ip.SentenceTransformer('google/embeddinggemma-300m')
  document_embeddings, _ = ip.start_embedding(ip.tool_descriptions_values)

  print(f'{len(LABELED_QUERIES)} labeled queries\n')

  emb_correct = 0
  results = []
  for query, label in LABELED_QUERIES:
    emb = embedding_top1(query, embed_model, document_embeddings)
    results.append((query, label, emb))
    if emb == label:
      emb_correct += 1
  print(f'embedding router top-1: {emb_correct}/{len(LABELED_QUERIES)}')

  if args.sys1:
    engine = DecisionEngine(model_id=DECISION_MODEL_ID, cache_path='decision_cache.sqlite3')
    s1_correct = 0
    for query, label, emb in results:
      s1 = sys1_top1(query, engine)
      marks = ('HIT' if s1 == label else 'MISS  ', 'emb-miss' if emb != label else '')
      print(f'[{marks[0]}|emb {emb:<22}] {s1:<24} <- {query}')
      if s1 == label:
        s1_correct += 1
    print(f'System 1 router top-1: {s1_correct}/{len(LABELED_QUERIES)}')

  print('\nEmbedding disagreements (query -> correct vs embedding pick):')
  for query, label, emb in results:
    if emb != label:
      print(f'  {label:<24} | emb chose {emb:<24} <- {query}')


if __name__ == '__main__':
  main()