'''
Sweep fusion weight x stage-2 softness on the 95 real queries, all reads from
the decision cache where possible. Precomputes parse_input (the slow GLiNER
step) once per query, then recombines distributions offline.
'''
import json
import sys

sys.path.insert(0, 'code_new')

import numpy as np
import input_parsing as ip
from decision_layer import DecisionEngine, DECISION_MODEL_ID
from sys1_router import TOOL_CRITERIA, ROUTE_INSTRUCTIONS, start_embedding_cached
from eval_real import QUERIES, expected_list


def main():
  print('Loading models...')
  parse_model = ip.start_ner()
  document_embeddings, embed_model = start_embedding_cached(ip.tool_descriptions_values)
  engine = DecisionEngine(model_id=DECISION_MODEL_ID, cache_path='decision_cache.sqlite3')
  engine.load()

  keys = ip.tool_descriptions_keys
  pre = []
  for row in QUERIES:
    prompt = str(row['Prompt'])
    present, proteins, names, diseases, smiles, uniprot, pdb, chembl = ip.parse_input(prompt, parse_model)
    q = embed_model.encode_query(prompt)
    sims = np.asarray(embed_model.similarity(q, document_embeddings))[0]
    lo, hi = float(sims.min()), float(sims.max())
    norm = (sims - lo) / (hi - lo + 1e-9) if hi > lo else np.zeros_like(sims)
    emb_p = np.exp(10 * norm)
    emb_p /= emb_p.sum()

    state = {'request': prompt, 'detected_entities': dict(present)}
    a1 = engine.choice(state=state, name='tool', question=ROUTE_INSTRUCTIONS, options=dict(TOOL_CRITERIA))
    s1_p = np.array([float(a1.get('probabilities', {}).get(k, 1e-9)) for k in keys])
    s1_p /= s1_p.sum()

    fused_05 = (s1_p ** 0.5) * (emb_p ** 0.5)
    fused_05 /= fused_05.sum()
    cand_idx = list(np.argsort(fused_05)[::-1][:5])
    cand_keys = [keys[i] for i in cand_idx]
    cand_options = {k: k + ': ' + TOOL_CRITERIA[k] for k in cand_keys}
    a2 = engine.choice(state=state, name='tool_stage2',
                       question=ROUTE_INSTRUCTIONS + ' Choose the single best fit among: '
                       + ', '.join(cand_keys) + '.',
                       options=cand_options)
    s2 = np.array([float(a2.get('probabilities', {}).get(k, 1e-9)) for k in cand_keys])
    s2 /= s2.sum()

    expected = [t.strip() for t in str(row['Expected Tools']).split(',')]
    single = row['Multi-step'] == 'N' and len(expected) == 1
    pre.append({'prompt': prompt, 'expected': expected, 'single': single,
                'state': {'request': prompt, 'detected_entities': dict(present)},
                'emb_p': emb_p, 's1_p': s1_p, 'cand_idx': cand_idx, 'stage2': s2})

  def score(final_rows):
    n = len(pre)
    t1 = sum(1 for r, f in final_rows
             if r['single'] and keys[int(np.argmax(f))] == r['expected'][0]) / n
    t3 = sum(1 for r, f in final_rows
             if all(k in [keys[i] for i in np.argsort(f)[::-1][:3]] for k in r['expected'])) / n
    t5 = sum(1 for r, f in final_rows
             if all(k in [keys[i] for i in np.argsort(f)[::-1][:5]] for k in r['expected'])) / n
    return t1, t3, t5

  print('\nmode                weight  Top-1   Top-3   Top-5')
  for fw in (0.3, 0.5, 0.7, 1.0):
    for alpha in (False, 0.0, 0.15, 0.25, 0.4, 0.6, 1.0):
      final_rows = []
      for r in pre:
        fused = (r['s1_p'] ** fw) * (r['emb_p'] ** (1 - fw))
        fused /= fused.sum()
        f = np.copy(fused)
        if alpha is not False:
          # shortlist (and stage-2 request) derived from THIS config's stage-1 fused
          cand = choose_stage2(r, fused, fw, alpha, engine)
          if cand is None:
            for rank, i in enumerate(r['cand_idx']):
              f[i] = fused[i]
          else:
            for rank, i in enumerate(cand['cand_idx']):
              f[i] = fused[i] * (cand['stage2'][rank] ** alpha)
          f /= f.sum()
        final_rows.append((r, f))
      t1, t3, t5 = score(final_rows)
      stage = 'no-s2  ' if alpha is False else f'a={alpha}'
      print(f'{stage:<19}  {fw:<5}  {t1:.3f}   {t3:.3f}   {t5:.3f}')


def choose_stage2(r, fused, fw, alpha, engine):
  '''Builds the stage-2 shortlist from a given config's stage-1 fused distribution,
  asking the (cached) decision question if not solved for this config before.'''
  from sys1_router import ROUTE_INSTRUCTIONS as _RI  # same question wording as route()
  sig = hash((tuple(np.round(fused, 6)), alpha))
  if sig in stage2_map:
    return stage2_map[sig]
  keys = ip.tool_descriptions_keys
  cand_idx = list(np.argsort(fused)[::-1][:5])
  cand_keys = [keys[i] for i in cand_idx]
  a2 = engine.choice(state=r['state'], name='tool_stage2',
                     question=ROUTE_INSTRUCTIONS + ' Choose the single best fit among: '
                     + ', '.join(cand_keys) + '.',
                     options={k: k + ': ' + TOOL_CRITERIA[k] for k in cand_keys})
  s2 = np.array([float(a2.get('probabilities', {}).get(k, 1e-9)) for k in cand_keys])
  entry = {'cand_idx': cand_idx, 'stage2': s2 / s2.sum()}
  stage2_map[hash((tuple(np.round(fused, 6)), alpha))] = entry
  return entry


stage2_map = {}


if __name__ == '__main__':
  main()