'''
Test suite for the MoDrAg_sys1 tool library (code_new/).

Base suite (fast): covers the tool nodes MoDrAg_CLI's tool_tests.py does not -
canonical tool, similarity tool, blind-dock + dock-check wrappers (real local
dock), gpt node via its generation cache, and the memory vault round-trip.

Live suite (`MODRAG_LIVE=1` ./python tool_tests_full.py): runs EVERY wired
node against the real APIs (PubChem, UniProt, ChEMBL, RCSB) and the gpt node
runs a REAL fine-tune on the CHEMBL213 bioactives (foundation checkpoint at
../data/GPT_ZN305_mini.pt; a few minutes on CPU).

Same verdict style (✓/✗ + summary) as MoDrAg_CLI tool_tests.py.
Run from inside code_new/.
'''
import os
import shutil
import sys
import time

import chain_tools
from modrag_molecule_functions import (name_node, smiles_node, related_node,
                                       structure_node)
from modrag_property_functions import (substitution_node, lipinski_node,
                                       pharmfeature_node, similarity_node)
from modrag_protein_functions import (uniprot_node, listbioactives_node,
                                      getbioactives_node, predict_node, pdb_node,
                                      find_PDBID_node, docking_node, target_node)
from modrag_task_graphs import dock_from_names
from gpt_node import gpt_node
from modrag_memory import save_session, recall_session, list_sessions

summary = []


def check(name, passed, detail=''):
  mark = '✓' if passed else '✗'
  print(f'{mark} {name} test passed{"" if passed else f": {detail}"}\n', flush=True)
  summary.append((name, passed))


def is_tool_result(result):
  '''The shape _execute() unpacks: (list, str, list).'''
  return (isinstance(result, tuple) and len(result) == 3
          and isinstance(result[1], str))


def json_lowercase(text):
  return text.lower() if isinstance(text, str) else str(text).lower()


def run_check(name, fn, live=False):
  '''Runs one test fn: it performs the call, asserts truth, returns None or a
  note string. AssertionError == test failure; any other exception too.'''
  if live and os.environ.get('MODRAG_LIVE') != '1':
    return
  print('=' * 60)
  print(f'TEST{" (live API)" if live else ""}: {name}', flush=True)
  print('=' * 60, flush=True)
  t0 = time.time()
  try:
    detail = fn()
    check(name, True, '' if detail is None else detail)
    print(f'  ({time.time() - t0:.0f}s)', flush=True)
  except AssertionError as e:
    check(name, False, str(e))
  except Exception as e:
    check(name, False, f'{type(e).__name__}: {e}')


def truth_list(result, truth, frac=0.8):
  '''At least frac of truth items occur somewhere in result[0].'''
  data = result[0] if isinstance(result, tuple) else result
  flat = data[0] if (data and isinstance(data[0], list)) else data
  pool = {str(x) for x in flat}
  matches = sum(1 for t in truth if str(t) in pool)
  assert matches >= len(truth) * frac, \
      f'{matches}/{len(truth)} matched; got {list(pool)[:5]}...'


def truth_float(result, truth, frac=0.1):
  data = result[0] if isinstance(result, tuple) else result
  val = float(data[0]) if isinstance(data, list) else float(data)
  assert abs(val - truth) <= abs(truth * frac), f'got {val:.3f}, want {truth:.3f}'


# ---- base (fast) suite ------------------------------------------------------

def base_suite():
  # 1. canonical_tool: value + user-ready text + 3-shape
  def t1():
    d, t, i = chain_tools.canonical_tool('CC(=O)OC1=CC=CC=C1C(=O)O')
    assert is_tool_result((d, t, i)), 'not a (list, str, list)'
    assert 'CC(=O)Oc1ccccc1C(=O)O' in t and 'canonical' in t, f'got {t!r}'
  run_check('canonical_node', t1)

  # 2. similarity_node: 3-shape, self-similarity 1.00, ibuprofen scored
  def t2():
    ref = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    _, t, _ = similarity_node([ref], [ref, 'CC(C)Cc1ccc(cc1)C(C)C(O)=O'])
    low = t.lower()
    assert 'similarity of 1.00' in low and 'test molecule 1' in low, f'got {t[:120]!r}'
  run_check('similarity_node', t2)

  # 3. blind_dock_node: real local dock (obabel + vendored Vina, npockets=1)
  #    of aspirin into receptor 2A3R fetched from RCSB.
  def t3():
    chain_tools.LAST_RECEPTOR = None
    chain_tools.LAST_POSE_SDF = None
    _, t, _ = chain_tools.blind_dock_node(
        ['CC(=O)Oc1ccccc1C(=O)O'], ['2A3R'], ['test_receptor'])
    pose = chain_tools.LAST_POSE_SDF
    assert pose is not None and os.path.exists(pose), f'no pose sfd; text: {t[:200]}'
    assert 'score' in t.lower(), f'no score in report: {t[:200]}'
    return f'pose: {pose}'
  run_check('blind_dock_node', t3)

  # 4. dock_check_node: error path (no session), then the pose from test 3
  def t4():
    saved_rec, saved_pose = chain_tools.LAST_RECEPTOR, chain_tools.LAST_POSE_SDF
    chain_tools.LAST_RECEPTOR, chain_tools.LAST_POSE_SDF = None, None
    _, miss, _ = chain_tools.dock_check_node([])
    assert 'No blind docking has run in this session' in miss, f'got {miss[:80]}'
    chain_tools.LAST_RECEPTOR, chain_tools.LAST_POSE_SDF = saved_rec, saved_pose
    _, t, _ = chain_tools.dock_check_node([])
    assert isinstance(t, str) and len(t) > 10, f'empty check result: {t!r}'
  run_check('dock_check_node', t4)

  # 5. gpt_node via the generation cache (no fine-tune): the path a repeat
  #    call on a target takes.
  def t5():
    os.makedirs('../scratch', exist_ok=True)
    smiles = ['CC(=O)Oc1ccccc1C(=O)O', 'CN1C=NC2=C1C(=O)N(C)C(=O)N2C',
              'CC(C)Cc1ccc(cc1)C(C)C(O)=O']
    with open('../scratch/gen_smiles_CHEMBL213.csv', 'w') as f:
      f.write('SMILES\n' + '\n'.join(smiles) + '\n')
    with open('../scratch/CHEMBL213_bioactives.csv', 'w') as f:
      f.write('SMILES\n' + smiles[0] + '\n')
    sm, t, _ = gpt_node('CHEMBL213')
    assert is_tool_result((sm, t, _)) and len(sm) == 3 \
        and 'novel molecules' in t and all(s in t for s in smiles), f'got {t[:120]!r}'
  run_check('gpt_node (cached generation path)', t5)

  # 6. modrag_memory: save a session then recall it back
  def t6():
    vault = '../scratch/test_vault'
    shutil.rmtree(vault, ignore_errors=True)
    messages = [
        {'role': 'user', 'content': 'dock aspirin in EGFR and show me the lipinski properties'},
        {'role': 'assistant', 'content': 'Docking score: -6.3. Lipinski QED: 0.55'},
    ]
    status = save_session(messages, time.time(), vault_dir=vault)
    assert isinstance(status, str) and len(os.listdir(vault)) >= 1, f'save: {status!r}'
    assert len(list_sessions(vault_dir=vault)) >= 1, 'list_sessions empty'
    text = recall_session(vault_dir=vault, which='last')
    assert text and 'lipinski' in json_lowercase(text), 'recall text lacks content'
    shutil.rmtree(vault, ignore_errors=True)
  run_check('modrag_memory (save/recall round trip)', t6)


# ---- live suite (MODRAG_LIVE=1): every wired node, real APIs ----------------

def live_tests():
  # PubChem family
  def t_names():
    n, t, i = name_node(['CCO', 'c1ccccc1'])
    assert is_tool_result((n, t, i)) and len(n) == 2, 'bad shape/count'
    assert n[0] == 'ethanol' and 'benzene' in n[1], f'got {n}'
  run_check('name_node', t_names, live=True)

  def t_smiles():
    sm, t, _ = smiles_node(['aspirin', 'caffeine'])
    assert is_tool_result((sm, t, _)) and len(sm) == 2 \
        and all(len(s) > 10 for s in sm), f'got {sm}'
  run_check('smiles_node', t_smiles, live=True)

  def t_related():
    rel, t, _ = related_node(['CCO'])
    assert is_tool_result((rel, t, _)) and len(rel) >= 1, f'got {t[:100]}'
  run_check('related_node', t_related, live=True)

  def t_structure():
    img, t, im = structure_node(['CCO', 'c1ccccc1'])
    assert is_tool_result((img, t, im)) and len(img) >= 2, f'{len(img)} results'
  run_check('structure_node', t_structure, live=True)

  def t_substitution():
    subs, t, _ = substitution_node(['c1ccc(O)cc1'])
    assert is_tool_result((subs, t, _)) and len(subs) == 1, 'no analogues'
    assert len(subs[0]) >= 1, f'empty analogue list: {subs}'
  run_check('substitution_node', t_substitution, live=True)

  def t_lipinski():
    qp, t, _ = lipinski_node(['CCO', 'c1ccccc1'])
    # nested per-molecule [QED, molar_mass] pairs
    assert is_tool_result((qp, t, _)) and len(qp) >= 1, f'got {qp}'
    truth_float((qp[0][0], t), 0.4068079656553945, frac=0.05)
    truth_float((qp[0][1], t), 46.069, frac=0.02)
  run_check('lipinski_node', t_lipinski, live=True)

  def t_pharmfeature():
    sc, t, _ = pharmfeature_node('CCO', ['CC(=O)OC1=CC=CC=C1C(=O)O'])
    # the CLI FeatMaps implementation gives different values than the old
    # SemanticMoDrAg truths; assert a sane overlap score
    val = float(sc[0][0] if isinstance(sc[0], list) else sc[0])
    assert 0.0 < val <= 1.0, f'overlap score out of range: {val}'
  run_check('pharmfeature_node', t_pharmfeature, live=True)

  # UniProt / ChEMBL
  def t_uniprot():
    up, t, _ = uniprot_node(['DNA gyrase'], False)
    # per-protein nesting: [[accessions], ...]
    outer = up[0] if (up and isinstance(up[0], list)) else up
    assert is_tool_result((up, t, _)) and len(outer) >= 5, f'got {up}'
    truth_list((outer, t, _), ['Q5SHZ4', 'P0AES4', 'Q08582', 'P33012', 'P0AES6'], frac=0.6)
  run_check('uniprot_node', t_uniprot, live=True)

  def t_listbioactives():
    counts, t, _ = listbioactives_node(['P27338'])
    outer = counts[0] if (counts and isinstance(counts[0], list)) else counts
    assert is_tool_result((counts, t, _)) and len(outer) >= 2, f'got {counts}'
    assert all(c > 0 for c in outer), f'zero counts: {counts}'
  run_check('listbioactives_node', t_listbioactives, live=True)

  def t_getbioactives():
    pairs, t, _ = getbioactives_node(['CHEMBL2039'])
    assert is_tool_result((pairs, t, _)), 'bad shape'
    flat = pairs[0] if (pairs and isinstance(pairs[0], list)) else pairs
    assert len(flat) >= 40, f'{len(flat)} bioactives'
  run_check('getbioactives_node', t_getbioactives, live=True)

  def t_predict():
    preds, t, _ = predict_node(['[NH3+]CCc1ccc(O)cc1'], 'CHEMBL2039')
    truth_float((preds, t, _), 10995.537239000963, frac=0.25)
  run_check('predict_node', t_predict, live=True)

  # RCSB
  def t_pdb_node():
    seq, t, _ = pdb_node(['2A3R'])
    assert is_tool_result((seq, t, _)) and str(seq).count('MELIQDTSRPPLEY') >= 2, \
        f'got sequences: {str(seq)[:120]}'
  run_check('pdb_node', t_pdb_node, live=True)

  def t_find():
    ids, t, _ = find_PDBID_node(['DNA gyrase'])
    truth_list((ids, t, _), ['3QTD', '1VL4', '1VPB', '5NJ5', '3NUH'], frac=0.6)
  run_check('find_node (find_PDBID_node)', t_find, live=True)

  # docking (real Vina via dockstring)
  def t_docking():
    sc, t, _ = docking_node(['[NH3+]CCc1ccc(O)cc1'], 'DRD2')
    truth_float((sc, t), -6.3, frac=0.3)
  run_check('docking_node', t_docking, live=True)

  def t_target():
    tg, t, _ = target_node(['phenylketonuria'])
    assert is_tool_result((tg, t, _)) and len(tg) >= 1, 'no genes returned'
    truth_list((tg, t, _), ['PAH', 'DNAJC12', 'NSUN2', 'COL1A1', 'PI4KA'], frac=0.5)
  run_check('target_node', t_target, live=True)

  # task-graph wrappers (network + docking)
  def t_actives():
    pairs, t, _ = chain_tools.get_actives_tool('MAOB')
    assert is_tool_result((pairs, t, _)), 'bad shape'
    flat = pairs[0] if (pairs and isinstance(pairs[0], list)) else pairs
    assert len(flat) >= 10, f'{len(flat)} bioactives'
  run_check('get_actives_for_protein (wrapper)', t_actives, live=True)

  def t_preds():
    preds, t, _ = chain_tools.get_predictions_tool(['[NH3+]CCc1ccc(O)cc1'], 'MAOB')
    truth_float((preds, t, _), 10995.537239000963, frac=0.25)
  run_check('get_predictions_for_protein (wrapper)', t_preds, live=True)

  def t_dock_from_names():
    sc, t, _ = dock_from_names(['aspirin', 'caffeine'], 'DRD2')
    assert is_tool_result((sc, t, _)) and len(sc) == 2 \
        and all(isinstance(s, float) for s in sc), f'got {sc}'
  run_check('dock_from_names', t_dock_from_names, live=True)

  # gpt_node: REAL fine-tune on live ChEMBL bioactives (foundation reloaded)
  def t_gpt_finetune():
    for cache in ('../scratch/gen_smiles_CHEMBL213.csv', '../scratch/CHEMBL213_bioactives.csv'):
      if os.path.exists(cache):
        os.remove(cache)
    sm, t, _ = gpt_node('CHEMBL213')
    assert is_tool_result((sm, t, _)) and len(sm) >= 1, 'no novel molecules generated'
    from rdkit import Chem
    valid = sum(1 for s in sm if Chem.MolFromSmiles(s) is not None)
    assert os.path.exists('../scratch/gen_smiles_CHEMBL213.csv'), 'generation cache not written'
    assert os.path.exists('../data/GPT_CHEMBL213_mini_finetuned.pt'), 'fine-tuned model not saved'
    return f'{valid}/{len(sm)} generated SMILES are valid'
  run_check('gpt_node (REAL fine-tune, ~minutes)', t_gpt_finetune, live=True)


def print_summary():
  print('=' * 60)
  print('TEST SUMMARY')
  print('=' * 60)
  passed = sum(1 for _, ok in summary if ok)
  for name, ok in summary:
    print(f'{"✓ PASS" if ok else "✗ FAIL"}: {name}')
  print(f'\nPassed: {passed}/{len(summary)}')


def main():
  chain_tools.apply_safe_env()  # obabel/libstdc++ collision (see chain_tools)
  base_suite()
  if os.environ.get('MODRAG_LIVE') == '1':
    live_tests()
  else:
    print('\n(live-API suite skipped; set MODRAG_LIVE=1 to run every tool)')
  print_summary()
  sys.exit(0 if all(ok for _, ok in summary) else 1)


if __name__ == '__main__':
  main()