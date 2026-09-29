'''
Evaluation on the 95 real test queries (tool testing full evaluation.xlsx).
Metrics mirror the workbook's Summary sheet so results are directly comparable:
  - single-step Top-1 accuracy (queries whose Primary Expected Tool is a
    single expected tool and Multi-step == N)
  - strict Top-3 / Top-5 recall (ALL expected tools appear)
Routers compared: embedding argmax, System 1 fused (one stage), two-stage fused.
Usage: python eval_real.py [embedding|sys1|sys1_2stage|all]
'''
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'code'))
os.chdir(os.path.join(_HERE, '..', 'code'))

import numpy as np
import input_parsing as ip
from decision_layer import DecisionEngine, DECISION_MODEL_ID
from sys1_router import route, start_embedding_cached

QUERIES = json.load(open(os.path.join(os.path.dirname(__file__), 'real_queries.json')))


def expected_list(row):
  return [t.strip() for t in str(row['Expected Tools']).split(',')]


def run(mode, parse_model, embed_model, document_embeddings, engine):
  top1s, top3s, top5s, details = 0, 0, 0, []
  for row in QUERIES:
    prompt = str(row['Prompt'])
    expected = expected_list(row)
    single = row['Multi-step'] == 'N' and ',' not in str(row['Expected Tools'])

    if mode == 'embedding':
      q = embed_model.encode_query(prompt)
      sims = np.asarray(embed_model.similarity(q, document_embeddings))[0]
      order = [ip.tool_descriptions_keys[i] for i in np.argsort(sims)[::-1]]
    else:
      route(prompt, parse_model, embed_model, document_embeddings,
            engine, two_stage=(mode == 'sys1_2stage'))
      probs = engine.last_choice['probabilities']
      order = sorted(probs, key=probs.get, reverse=True)

    top1 = order[0] if (row['Multi-step'] == 'N' and len(expected) == 1
                        and order[0] == expected[0]) else False
    top3 = all(t in order[:3] for t in expected)
    top5 = all(t in order[:5] for t in expected)
    top1s += bool(top1)
    top3s += top3
    top5s += top5
    details.append((row, order, top1, top3, top5))

  n = len(QUERIES)
  print(f'\n=== {mode} ===')
  print(f'single-step Top-1 (strict): {top1s}/{n} = {top1s / n:.3f}')
  print(f'ALL-expected Top-3 recall:  {top3s}/{n} = {top3s / n:.3f}')
  print(f'ALL-expected Top-5 recall:  {top5s}/{n} = {top5s / n:.3f}')
  return details


def main():
  print('Loading models...')
  parse_model = ip.start_ner()
  document_embeddings, embed_model = start_embedding_cached(ip.tool_descriptions_values)
  engine = DecisionEngine(model_id=DECISION_MODEL_ID, cache_path='decision_cache.sqlite3')
  engine.load()
  ip.intake_cache = None  # silence unused

  mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
  all_details = {}
  for m in (['embedding', 'sys1', 'sys1_2stage'] if mode == 'all' else [mode]):
    all_details[m] = run(m, parse_model, embed_model, document_embeddings, engine)

  # per-tool misses for the two-stage variant (or the single one run)
  shown = 'sys1_2stage' if 'sys1_2stage' in all_details else mode
  print(f'\n=== {shown}: per-query misses (top-3 not covering all expected) ===')
  for row, order, top1, top3, top5 in all_details[shown]:
    if not top5:
      exp = expected_list(row)
      print(f"[{row['ID']}] {row['Prompt'][:70]}")
      print(f'     want: {exp}   got: {order[:5]}')
    elif not top3:
      exp = expected_list(row)
      print(f"[{row['ID']}] top5-only miss: {exp} got {order[:5]}")


if __name__ == '__main__':
  main()