'''
System 1 decision layer for semantic MoDrAg.

Wraps an opendecider decision model (default: opendecider-nano, ~400M encoder,
~17ms/question on CPU, Apache-2.0) behind a small interface, with a sqlite
key-value cache so identical (query, question, options) triples never re-run.

Primitives used from MoDrAg's flow:
  - Choice: tool routing (replaces embedding-only argmax), entity-role
    disambiguation when multiple entities were detected
  - Noul:   calibrated gates (is this query a tool task? is this candidate
    string really a SMILES?)
  - Score:  not used in the first pass
Extraction stays local: GLiNER NER + regex (a decision model cannot return spans).
'''
import hashlib
import json
import os
import sqlite3

DECISION_MODEL_ID = 'manjunathshiva/opendecider-nano'

class DecisionEngine:
  '''
  Thin wrapper around opendecider with a persistent response cache.

  Args:
      model_id: opendecider checkpoint id or local path.
      device: 'cpu' | 'cuda' | 'mps' (default: auto).
      cache_path: sqlite file for cached decision results; None disables caching.
  '''

  def __init__(self, model_id: str = DECISION_MODEL_ID, device: str = None,
               cache_path: str = 'decision_cache.sqlite3'):
    self.model_id = model_id
    self.cache_path = cache_path
    self._model = None  # lazy: heavy load only on first real call
    if device == 'cpu':
      self.device = 'cpu'
    else:
      self.device = device  # None -> opendecider auto-selects (CUDA -> MPS -> CPU)

    if cache_path:
      self._conn = sqlite3.connect(cache_path)
      self._conn.execute(
          'CREATE TABLE IF NOT EXISTS decisions ('
          ' key TEXT PRIMARY KEY,'
          ' payload TEXT NOT NULL)')
      self._conn.commit()
    else:
      self._conn = None

  def load(self):
    '''
    Eagerly loads the decision model (optional; first call loads lazily).
    '''
    if self._model is None:
      from opendecider import load
      self._model = load(self.model_id) if self.device is None else load(self.model_id, device=self.device)
    return self._model

  def _cache_key(self, state: str, questions: list) -> str:
    payload = self.model_id + '\x00' + state + '\x00' + json.dumps(questions, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()

  def _cache_get(self, key: str):
    if self._conn is None:
      return None
    row = self._conn.execute('SELECT payload FROM decisions WHERE key = ?', (key,)).fetchone()
    return json.loads(row[0]) if row else None

  def _cache_put(self, key: str, answers: dict):
    if self._conn is None:
      return
    self._conn.execute('INSERT OR REPLACE INTO decisions VALUES (?, ?)',
                       (key, json.dumps(answers)))
    self._conn.commit()

  def ask(self, state, questions: dict) -> dict:
    '''
    Runs a batch of Choice/Score/Noul questions against the decision model,
    returning the answers dict keyed by question name.
      questions: dict of name -> opendecider helper objects: Choice(question, options),
                 Noul(question), Score(question, levels).
    Results (per model_id + state + questions) are cached across runs.
    Args:
        state: The text (or JSON-serializable object) the questions are judged on.
        questions: A dict of named questions for opendecider's system_one().
    Returns:
        answers: A dict mapping question name -> {choice/probabilities/confidence}.
    '''
    key_spec = {'names': list(questions.keys()),
                'qs': [json.dumps({'type': q._answer_type if hasattr(q, '_answer_type') else type(q).__name__,
                                   'text': q.__dict__}, sort_keys=True, default=str)
                       for q in questions.values()]}
    key = self._cache_key(_state_str(state), key_spec)
    cached = self._cache_get(key)
    if cached is not None:
      return cached

    self.load()
    result = self._model.system_one(state, questions)
    answers = result.get('answers', {})
    self._cache_put(key, answers)
    return answers

  def choice(self, state, name: str, question: str, options: dict) -> dict:
    '''
    Convenience wrapper for a single Choice question.
      options: dict(option name -> description).
    Returns:
        {'choice': ..., 'probabilities': {...}, 'confidence': ...}
    '''
    answers = self.ask(state, {name: _Choice(question, options)})
    return answers.get(name, {})

  def noul(self, state, name: str, question: str) -> float:
    '''
    Convenience wrapper for a single Noul (yes/no) question.
    Returns:
        probability of 'yes' in [0, 1].
    '''
    answers = self.ask(state, {name: _Noul(question)})
    a = answers.get(name, {})
    return float(a.get('noul', a.get('probabilities', {}).get('true', 0.0)))


def _state_str(state) -> str:
  '''Normalizes state (string or dict) into the cache-key string.'''
  return state if isinstance(state, str) else json.dumps(state, sort_keys=True, default=str)


def _Choice(question: str, options: dict):
  '''Returns an opendecider Choice, importing lazily to keep model load light.'''
  from opendecider import Choice as _C
  return _C(question, options)


def _Noul(question: str):
  from opendecider import Noul as _N
  return _N(question)