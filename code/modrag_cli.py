'''
MoDrAg_sys1: command-line drug-design agent driven by a System 1 decision
model (opendecider) instead of an autoregressive chat LLM.

Interaction model follows MoDrAg_CLI (rich markdown rendering, banner-as-help,
"memory"/"recall" vault, quit loop); tool routing follows the SemanticMoDrAg
input-parsing/intake flow, replaced as follows:
  - embedding argmax        -> cached embedding scores + one System 1 Choice call
                               (full calibrated distribution, response-cached)
  - always-confirm state    -> auto-execute when top-1 confidence >= threshold;
                               confirm only when ambiguous
  - chat LLM interpretation -> none; tool output is rendered directly
No autoregressive LLM is consulted for routing or responses.

Run from inside code/ — or from anywhere via the `modrag` shell
function installed by `bash setup_alias.sh` (it cds into code/ in a
subshell, so the tools' repo-root-relative writes always land right).
'''
import argparse
import os
import re
import sys
import time
import warnings

warnings.filterwarnings('ignore', message='`torch.jit.script` is deprecated')
warnings.filterwarnings('ignore', message='The `resume_download` argument is deprecated')

# All models (GLiNER NER, embeddinggemma, opendecider) live in the HF hub cache
# after the first run. If every model is already cached locally, go offline so
# launches don't re-check the hub ("Fetching N files" gone); otherwise stay
# online so a first run can download. Override with HF_HUB_OFFLINE=0/1.
_MODEL_CACHE_IDS = ('manjunathshiva--opendecider-nano',
                    'google--embeddinggemma-300m',
                    'anthonyyazdaniml--gliner-biomed-large-v1.0-disease-chemical-gene-variant-species-cellline-ner')
_hub = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
if os.environ.get('HF_HUB_OFFLINE') is None:
  if 'HF_HOME' in os.environ:
    _hub = os.path.join(os.environ['HF_HOME'], 'hub')
  if all(os.path.isdir(os.path.join(_hub, f'models--{mid}'))
         for mid in _MODEL_CACHE_IDS):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

import input_parsing as ip
from decision_layer import DecisionEngine, DECISION_MODEL_ID
from sys1_router import sys1_intake, sys1_second_intake, start_embedding_cached
import chain_tools
import modrag_memory
from modrag_memory import save_session, recall_session, list_sessions

# the session vault sits at the repo root (../vault from code/, as in MoDrAg)
VAULT_DIR = '../vault'

console = Console()

_SETETH_SEP = re.compile(r'[=\-]{3,}')


def safe_markdown(text: str) -> Markdown:
  '''Markdown-render free-form tool output.

  A node's separator lines (e.g. lipinski_node's trailing
  "===================") are valid Markdown *setext heading* syntax when
  they follow a text line: "===" turns the whole preceding paragraph into
  an h1, which rich renders centered — the wrapped property lines come
  out looking centered. Inserting a blank line before such a separator
  detaches it from the paragraph, so it renders as the plain separator
  characters it was meant to be.
  '''
  lines = text.split('\n')
  out = []
  for line in lines:
    if out and out[-1].strip() and _SETETH_SEP.fullmatch(line.strip()):
      out.append('')
    out.append(line)
  return Markdown('\n'.join(out))

# the entity lists the router parses (order matches parse_input's present dict)
ENTITY_ATTRS = ('proteins_list', 'names_list', 'diseases_list', 'smiles_list',
                'uniprot_list', 'pdb_list', 'chembl_list')
ENTITY_KEYS = ('proteins', 'molecules', 'diseases', 'smiles',
               'uniprot', 'pdb', 'chembl')

BANNER = '''\x1b[1;36m****************************************\x1b[0m
\x1b[1;35m* _   _ _     ___ _                    *\x1b[0m
\x1b[1;36m*| | | (_)   |_ _( )_ __ ___           *\x1b[0m
\x1b[1;35m*| |_| | |    | |/ | '_ ` _ \\          *\x1b[0m
\x1b[1;36m*|  _  | |_   | |  | | | | | |         *\x1b[0m
\x1b[1;35m*|_| |_|_( ) |___| |_| |_| |_|         *\x1b[0m
\x1b[1;36m*        |/                            *\x1b[0m
\x1b[38;5;208m* __  __       ____        _         _ *\x1b[0m
\x1b[1;35m*|  \\/  | ___ |  _ \\ _ __ / \\   __ _| |*\x1b[0m
\x1b[1;36m*| |\\/| |/ _ \\| | | | '__/ _ \\ / _` | |*\x1b[0m
\x1b[1;35m*| |  | | (_) | |_| | | / ___ \\ (_| |_|*\x1b[0m
\x1b[38;5;208m*|_|  |_|\\___/|____/|_|/_/   \\_\\__, (_)*\x1b[0m
\x1b[1;36m*                              |___/   *\x1b[0m
\x1b[1;35m* (System 1 edition)                   *\x1b[0m
\x1b[38;5;208m* A CafChem project!                   *\x1b[0m
\x1b[1;36m****************************************\x1b[0m
\x1b[1;35mThe MOdular DRug design AGent!\x1b[0m
\x1b[1;36mA command-line interface (CLI) for drug\x1b[0m
\x1b[1;35mdiscovery and molecular design.\x1b[0m
\x1b[38;5;208mType `memory` to save this session, `recall` to revisit one, `quit` to exit.\x1b[0m
\x1b[0m'''


class Sys1Agent:
  '''The confirm/auto state machine, System 1 router, no LLM.'''

  def __init__(self, router: str = 'sys1', auto_threshold: float = 0.6,
               engine: DecisionEngine = None, max_tools: int = 2,
               followup_threshold: float = 0.6):
    self.router_kind = router
    self.auto_threshold = auto_threshold
    self.engine = engine
    self.max_tools = max_tools          # how many tools may chain in one request
    self.followup_threshold = followup_threshold
    self.chat_idx = 0
    # session memory: transcript + artifacts for the vault "memory"/"recall"
    self.session_start = time.time()
    self.messages = []
    # cross-turn entity memory: entities from the last query are carried into
    # follow-up queries that don't mention their own (e.g. "now the lipinski
    # properties" after docking a named molecule still knows the SMILES)
    self.last_entities = {}

  def start(self):
    print('Loading GLiNER biological NER model...')
    self.parse_model = ip.start_ner()

    print('Loading embedding model (tool descriptions cached to disk)...')
    self.document_embeddings, self.embed_model = start_embedding_cached(ip.tool_descriptions_values)

    if self.router_kind == 'sys1':
      print(f'Loading System 1 decision model ({self.engine.model_id})...')
      self.engine.load()

    self.reset_chat()

  def reset_chat(self):
    # archive any entity context for the next query before clearing
    if any(getattr(self, a, None) for a in ENTITY_ATTRS):
      self.last_entities = {a: list(getattr(self, a)) for a in ENTITY_ATTRS
                            if getattr(self, a)}
    self.chat_idx = 0
    self.best_tools = []
    self.confidences = []
    self.tools_run = []
    self.proteins_list = []
    self.names_list = []
    self.diseases_list = []
    self.smiles_list = []
    self.uniprot_list = []
    self.pdb_list = []
    self.chembl_list = []
    self.query = ''
    self.file_paths = []
    self.present = {}

  def _intake(self, query):
    if self.router_kind == 'sys1':
      return sys1_intake(query, self.parse_model, self.embed_model,
                         self.document_embeddings, self.engine)
    return ip.intake(query, self.parse_model, self.embed_model, self.document_embeddings)

  def ask(self, query: str):
    '''One REPL turn.'''
    # file-path extraction (like the other entity extractors) feeds chain tools
    from chain_tools import extract_file_paths
    self.file_paths = extract_file_paths(query)
    chain_tools.SESSION_FILE_PATHS = self.file_paths

    if self.chat_idx == 0:
      self.query = query
      self.messages.append({'role': 'user', 'content': query})
      (self.best_tools, self.present, self.proteins_list, self.names_list,
       self.diseases_list, self.smiles_list, self.uniprot_list, self.pdb_list,
       self.chembl_list) = self._intake(query)

      # cross-turn entity memory: entity types the new query doesn't mention
      # keep the previous query's values (present counts only parsed matches)
      for i, attr in enumerate(ENTITY_ATTRS):
        if not getattr(self, attr) and self.present.get(ENTITY_KEYS[i], 0) == 0:
          old = self.last_entities.get(attr)
          if old:
            setattr(self, attr, list(old))
      # the carried entities count as known information for the tool hash
      for i, attr in enumerate(ENTITY_ATTRS):
        self.present[ENTITY_KEYS[i]] = len(getattr(self, attr))
      self.last_entities = {a: list(getattr(self, a)) for a in ENTITY_ATTRS
                            if getattr(self, a)}

      will_auto = (self.router_kind == 'sys1' and len(self.best_tools) > 0 and
                   getattr(self.engine, 'last_choice', {}).get('confidence', 0.0)
                   >= self.auto_threshold)
      print('')
      self._show_tools_and_entities(prompt=not will_auto)

      if self.router_kind == 'sys1' and len(self.best_tools) > 0:
        top = self.best_tools[0]
        conf = getattr(self.engine, 'last_choice', {}).get('confidence', 0.0)
        if conf >= self.auto_threshold:
          console.print(Markdown(f'Confidence **{conf:.2f}** >= threshold **{self.auto_threshold:.2f}** - auto-executing **{top}**. ("no" to override)'))
          self.chat_idx = 1
          self.tool_choice = 0
          self._execute(top)
          return
        else:
          print('')
          console.print(Markdown(f'Low routing confidence (**{conf:.2f}**) - please confirm below.'))
          self.chat_idx = 1
          return

    if self.chat_idx == 1:
      if query.lower() == 'no':
        self.reset_chat()
        console.print(Markdown('OK - start over with a new query.'))
        return
      if query == '':
        self.tool_choice = 0
      elif query in ('1', '2', '3'):
        self.tool_choice = int(query) - 1
      else:
        console.print(Markdown('Enter 1, 2 or 3 (or enter for #1), or "no" to start over.'))
        return
      self._execute(self.best_tools[self.tool_choice])
      return

  def _execute(self, tool_key: str):
    reqs = ip.define_tool_reqs(tool_key, self.proteins_list, self.names_list,
                               self.diseases_list, self.smiles_list, self.uniprot_list,
                               self.pdb_list, self.chembl_list)
    reqs_list, list_names = reqs[tool_key]
    missing = [n for l, n in zip(reqs_list, list_names) if len(l) == 0]
    if missing:
      print('')
      console.print(Markdown(f'**Missing information for: {", ".join(missing)}.** '
                             'Reply with the missing data.'))
      self.chat_idx = 999
      return

    tool_hash = ip.define_tool_hash(tool_key, self.proteins_list, self.names_list,
                                    self.diseases_list, self.smiles_list, self.uniprot_list,
                                    self.pdb_list, self.chembl_list)
    fn, args = tool_hash[tool_key]
    try:
      _, results_string, _ = fn(*args)
    except Exception as e:
      console.print(Markdown(f'**Tool {tool_key} failed:** `{e}`'))
      self.reset_chat()
      return
    print('')
    # the response text is the tool's own formatted block (one property per
    # line etc.): render it as rich Text so the line structure survives and
    # long lines wrap at the console width — Markdown was merging those
    # lines into one paragraph and re-wrapping them at random points (and,
    # before safe_markdown, a trailing "===" separator promoted the whole
    # block to a centered h1). The nodes' own stdout (tool name header
    # lines etc.) stays raw, as before.
    console.print(Text(results_string.rstrip()))
    print('')
    self.tools_run.append(tool_key)
    self.messages.append({'role': 'tool', 'tool_name': tool_key,
                          'content': results_string})
    self._maybe_followup(tool_key, results_string)

  def _maybe_followup(self, tool_key: str, results_string: str):
    '''System 1 sequential-tool gate: a calibrated Noul question decides whether
    the user's request needs another tool after this one; a second_intake-style
    re-route on the results context proposes it (todo #4: 2 sequential tools).'''
    if len(self.tools_run) >= self.max_tools:
      self.reset_chat()
      return

    noul = self.engine.noul(
        state={'request': self.query, 'tools_run': list(self.tools_run),
               'results_excerpt': str(results_string)[:1500]},
        name='more_tools',
        question=f'The user\'s request may involve several steps. To fully complete '
                 f'the user\'s original request, is another tool needed after '
                 f'{tool_key}?')
    if noul < self.followup_threshold:
      self.reset_chat()
      return

    from sys1_router import sys1_second_intake
    (best_tools, present, proteins_list, names_list, diseases_list, smiles_list,
     uniprot_list, pdb_list, chembl_list) = sys1_second_intake(
        self.query, str(results_string)[:2000], self.parse_model, self.embed_model,
        self.document_embeddings, self.engine)
    best_tools = [t for t in best_tools if t not in self.tools_run][:3]

    # merge: keep entities already known, fill gaps from the results context
    for attr, new_list in (('proteins_list', proteins_list), ('names_list', names_list),
                           ('diseases_list', diseases_list), ('smiles_list', smiles_list),
                           ('uniprot_list', uniprot_list), ('pdb_list', pdb_list),
                           ('chembl_list', chembl_list)):
      current = getattr(self, attr)
      setattr(self, attr, (current + [e for e in new_list if e not in current]) or new_list)
    self.present = present
    self.best_tools = best_tools

    top = best_tools[0]
    conf = getattr(self.engine, 'last_choice', {}).get('confidence', 0.0)
    print('')
    console.print(Markdown(f'**Request has more steps** (model says {noul:.2f}) - '
                           f'next suggested tool: **{top}** (confidence {conf:.2f}).'))
    self.chat_idx = 1
    if conf >= self.auto_threshold:
      print('')
      console.print(Markdown(f'Auto-executing **{top}** ("no" to override).'))
      self.tool_choice = 0
      self._execute(top)
      return
    self._show_tools_and_entities()

  def parse_followup(self, query):
    '''chat_idx 999: the user is replying with exactly the missing data. Fill
    the gaps from the reply (parsed entities, or the plain text stashed into
    the slot that was asked for) and run the chosen tool directly - one prompt
    per turn, no second confirmation round.'''
    self.messages.append({'role': 'user', 'content': query})
    present, proteins_list, names_list, diseases_list, smiles_list, uniprot_list, pdb_list, chembl_list = \
        ip.parse_input(query, self.parse_model)
    for attr, new_list in (('proteins_list', proteins_list), ('names_list', names_list),
                           ('diseases_list', diseases_list), ('smiles_list', smiles_list),
                           ('uniprot_list', uniprot_list), ('pdb_list', pdb_list),
                           ('chembl_list', chembl_list)):
      if len(getattr(self, attr)) == 0 and len(new_list) > 0:
        setattr(self, attr, new_list)

    self._fill_missing_slots(query)

    self.chat_idx = 1
    self._execute(self.best_tools[getattr(self, 'tool_choice', 0)])

  @staticmethod
  def _slot_attr(req_name: str):
    '''Maps a define_tool_reqs requirement name (e.g. "protein names") to the
    matching entity-list attribute.'''
    low = req_name.lower()
    if 'smiles' in low:
      return 'smiles_list'
    if 'protein' in low:
      return 'proteins_list'
    if 'uniprot' in low:
      return 'uniprot_list'
    if 'pdb' in low:
      return 'pdb_list'
    if 'chembl' in low:
      return 'chembl_list'
    if 'disease' in low:
      return 'diseases_list'
    if 'name' in low:
      return 'names_list'
    return None

  def _fill_missing_slots(self, query: str):
    '''When the reply doesn't parse into entities (GLiNER misses some gene
    symbols and short names), stash the reply text into the first requirement
    that was asked for and is still empty.'''
    tool = self.best_tools[getattr(self, 'tool_choice', 0)]
    reqs = ip.define_tool_reqs(tool, self.proteins_list, self.names_list,
                               self.diseases_list, self.smiles_list, self.uniprot_list,
                               self.pdb_list, self.chembl_list)
    reqs_list, list_names = reqs[tool]
    reply = query.strip()
    if not reply or not any(len(l) == 0 for l in reqs_list):
      return
    for name, req in zip(list_names, reqs_list):
      if len(req) == 0:
        attr = self._slot_attr(name)
        if attr and not getattr(self, attr):
          setattr(self, attr, [reply])
          return

  def prompt_for(self):
    '''The REPL's input prompt reflects what the agent is waiting for:
    a tool confirmation, or the specific missing data (chat_idx 999).
    New-query turns use the default MoDrAg_CLI prompt.'''
    if self.chat_idx == 1:
      return '\x1b[1;36mEnter to accept #1, 2/3, or "no" > \x1b[0m'
    if self.chat_idx == 999:
      try:
        tool = self.best_tools[getattr(self, 'tool_choice', 0)]
        reqs = ip.define_tool_reqs(tool, self.proteins_list, self.names_list,
                                   self.diseases_list, self.smiles_list,
                                   self.uniprot_list, self.pdb_list, self.chembl_list)
        reqs_list, list_names = reqs[tool]
        missing = [list_names[j] for j, l in enumerate(reqs_list) if len(l) == 0]
      except Exception:
        missing = ['the missing data']
      return ('\x1b[1;38;5;208mReply with the missing '
              f'{", ".join(missing) or "data"} > \x1b[0m')
    return None

  def _show_tools_and_entities(self, prompt=True):
    md = '## The tools chosen based on your query are:\n'
    for i, tool in enumerate(self.best_tools):
      md += f'{i + 1}. **{tool}**\n'
    md += '\n## Information found in your query:\n'
    for entity_type, entity_list in zip(self.present, [self.proteins_list, self.names_list,
                                                       self.diseases_list, self.smiles_list,
                                                       self.uniprot_list, self.pdb_list, self.chembl_list]):
      if self.present.get(entity_type, 0) > 0:
        md += f'**{entity_type}**: '
        md += ', '.join(entity_list) + '\n'
    console.print(Markdown(md))
    if not prompt and len(md.strip()) > 0:
      return
    # the state-aware REPL prompt (see prompt_for) carries the call to action


def main():
  # the tool modules resolve ../images, ../scratch, ../vault, ../pdb_files
  # and ../data relative to this code/ directory — self-heal the CWD so
  # the CLI runs correctly from anywhere (fresh clones get the runtime
  # dirs created here, since none of them are committed)
  os.chdir(os.path.dirname(os.path.abspath(__file__)))
  for _d in ('../images', '../outputs', '../pdb_files', '../scratch', '../vault'):
    os.makedirs(_d, exist_ok=True)
  ap = argparse.ArgumentParser(description='MoDrAg_sys1 CLI')
  ap.add_argument('--router', choices=['sys1', 'embedding'], default='sys1',
                  help='tool router: System 1 decision model (default) or the original embedding argmax')
  ap.add_argument('--model', default=DECISION_MODEL_ID, help='opendecider checkpoint id')
  ap.add_argument('--no-cache', action='store_true', help='disable decision-response caching')
  ap.add_argument('--auto-threshold', type=float, default=0.6,
                  help='auto-exec tool #1 at confidence >= this (0 = always confirm)')
  ap.add_argument('--max-tools', type=int, default=2,
                  help='max tools to chain per request (2 = sequential tools on)')
  ap.add_argument('--followup-threshold', type=float, default=0.6,
                  help='Noul probability above which another tool is chained')
  args = ap.parse_args()

  # obabel (used by dockstring + vina_dock) crashes under an LD_LIBRARY_PATH
  # carrying an out-of-date libstdc++ (e.g. ORCA's bundled copy); sanitize this
  # process's env once so docking tools run in shells where ORCA is set up.
  import chain_tools
  chain_tools.apply_safe_env()

  engine = DecisionEngine(model_id=args.model,
                          cache_path=None if args.no_cache else 'decision_cache.sqlite3')
  agent = Sys1Agent(router=args.router, auto_threshold=args.auto_threshold, engine=engine,
                    max_tools=args.max_tools, followup_threshold=args.followup_threshold)
  # MoDrAg_CLI order: load the models first, THEN the banner sits right above
  # the first prompt instead of being scrolled off by the loading output.
  agent.start()
  print(BANNER)

  first = True
  while True:
    try:
      prompt = agent.prompt_for()
      if prompt is None:
        prompt = ('\x1b[1;36mWhat can I help with today? > \x1b[0m' if first
                  else '\x1b[1;36mWhat else can I help with? > \x1b[0m')
      query = input(prompt)
      print('')
    except (KeyboardInterrupt, EOFError):
      break
    first = False
    query = query.strip()
    t0 = time.time()

    if query == 'quit':
      print('\x1b[1;35mResponse > \x1b[0mGoodbye!')
      break
    elif query == 'reset':
      agent.reset_chat()
      agent.last_entities = {}  # "reset" clears the retained context too
      console.print(Markdown('(cleared - start over)'))
      continue
    elif query == 'memory' or query in ('save memory', 'remember'):
      # save the transcript + this session's dock outputs to the vault
      status = save_session(agent.messages, agent.session_start, vault_dir=VAULT_DIR)
      agent.session_start = time.time()  # each memory covers work since the last
      agent.messages = []
      console.print(Markdown(status))
      continue
    elif query == 'recall' or query.startswith('recall '):
      which = query.split(maxsplit=1)[1].strip() if len(query.split(maxsplit=1)) > 1 else ''
      if which == '':
        console.print(Markdown(list_sessions(vault_dir=VAULT_DIR)))
      else:
        text = recall_session(vault_dir=VAULT_DIR,
                              which='last' if which in ('last', 'latest') else which)
        if text is None:
          console.print(Markdown(f'No saved session matches `{which}`. '
                                 'Type `recall` alone to list them.'))
        else:
          console.print(safe_markdown(text.strip()[:4000]))
      continue
    elif query == '':
      continue

    try:
      if agent.chat_idx == 999:
        agent.parse_followup(query)
      else:
        agent.ask(query)
    except Exception as exc:
      console.print(Markdown(f'**Error:** `{exc}` (state reset)'))
      agent.reset_chat()

    # the elapsed-time marker is for completed tool turns; when the agent is
    # waiting on the user (confirm / missing data) the state-aware input prompt
    # below is the only call to action
    if agent.chat_idx == 0:
      time_for_inf = (time.time() - t0) / 60
      print(f'\x1b[1;35mResponse {time_for_inf:.2f}m > \x1b[0m')


if __name__ == '__main__':
  main()