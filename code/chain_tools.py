'''
Chain-tool wrappers for the MoDrAg_CLI docking nodes that operate on file
paths rather than the agent's parsed entity lists:

  blind_dock_node  - fetches the receptor PDB (get_pdb_file) when the user
                     gives a PDB ID, then calls vina_dock.blind_dock_agent.
                     Records the receptor and pose file paths for follow-ups.
  dock_check_node  - wraps check_nearby_molecules using the paths recorded by
                     the last blind docking run in this session (the README's
                     recommended verification step after blind docking).

Wrappers return the (list, str, list) shape the router's define_tool_hash
expects. No autoregressive LLM involved.
'''
import glob
import os
import re

from modrag_protein_functions import get_pdb_file, check_nearby_molecules
from modrag_molecule_functions import canonical_node as _canonical_node
from modrag_task_graphs import get_actives_for_protein as _get_actives, \
    get_predictions_for_protein as _get_predictions
from vina_dock import blind_dock_agent

# session state: set by blind_dock_node, consumed by dock_check_node
LAST_RECEPTOR = None
LAST_POSE_SDF = None

POSE_SDF_RE = re.compile(r'([\w./-]+\.sdf)', re.IGNORECASE)

FILE_PATH_RE = re.compile(r'([\w./-]+\.(?:pdb|sdf|mol|mol2|csv|pt))', re.IGNORECASE)


def extract_file_paths(text: str) -> list[str]:
  '''
  Regex extractor for molecular file paths mentioned in a query or tool output
  (`.pdb`, `.sdf`, `.mol`, `.mol2`, `.csv`, `.pt`), keeping only paths that
  exist on disk - the chain tools consume them like the other entity extractors.
    Returns:
        paths: A de-duplicated list of existing file paths.
  '''
  candidates = list(dict.fromkeys(FILE_PATH_RE.findall(text)))
  return [p for p in candidates if os.path.exists(p)]


# set by the CLI from extract_file_paths on the latest query/results, so the
# router's define_tool_hash signatures stay stable
SESSION_FILE_PATHS = []


def _find_pose_sdf(text: str):
  '''Pulls the most recent SDF path mentioned in a tool output string.'''
  global LAST_POSE_SDF
  matches = POSE_SDF_RE.findall(text)
  # pick the newest existing file match (the pose SDF, not receptor PDBQT)
  for m in reversed(list(dict.fromkeys(matches))):
    if os.path.exists(m):
      return m
  return LAST_POSE_SDF


def blind_dock_node(smiles_list: list[str], pdb_list: list[str],
                    proteins_list: list[str] = None):
  '''
  Fetches the receptor PDB (from the user's PDB ID, falling back to the
  detected protein name) and blind-docks the ligand SMILES with AutoDock Vina.
    Args:
        smiles_list: Ligand SMILES strings.
        pdb_list: A PDB ID for the receptor (used to fetch its file).
        proteins_list: Optional protein name used when saving the PDB file.
    Returns:
        results_list: An empty list (tool writes files and text).
        results_string: Blind-docking results (pose affinities, SDF paths).
        results_images: An empty list for consistency with other nodes.
  '''
  global LAST_RECEPTOR, LAST_POSE_SDF
  print('Blind docking tool')
  print('===================================================')

  if not pdb_list:
    return [], 'A PDB ID for the receptor is required for blind docking (the ' \
               'protein is not in docking_node\'s named-target list).', []

  # a user-supplied receptor PDB path wins; otherwise fetch it by PDB ID
  receptor_path = next((p for p in SESSION_FILE_PATHS if p.lower().endswith('.pdb')), None)
  if receptor_path is None:
    # get_pdb_file returns a status string, not a path; the file is saved to
    # ../pdb_files/<name>_<pdb_id>.pdb (reused when already cached by ID)
    get_pdb_file(pdb_list[0], proteins_list[0] if proteins_list else pdb_list[0])
    pdb_upper = pdb_list[0].upper()
    hits = [p for p in glob.glob('../pdb_files/*.pdb')
            if os.path.basename(p)[:-4].upper().endswith('_' + pdb_upper)]
    receptor_path = hits[0] if hits else None
  if receptor_path is None:
    return [], 'Could not locate the receptor PDB file for blind docking ' \
               f'(PDB ID {pdb_list[0]}).', []
  LAST_RECEPTOR = receptor_path

  results = blind_dock_agent(receptor_path, smiles_list) if smiles_list else \
      'No ligand SMILES were provided to blind dock.'

  pose = _find_pose_sdf(results)
  LAST_POSE_SDF = pose

  text = str(results)
  if pose:
    text += '\n\nNext step suggestion: verify with the check_nearby_molecules tool.'
  return [], text, []


def dock_check_node(names_list: list[str] = None):
  '''
  Verifies the previous blind docking pose landed in the correct binding site
  by comparing against co-crystallized ligands.
    Args:
        names_list: Unused placeholder to match the router's arg convention.
    Returns:
        (list, str, list) as above.
  '''
  print('Nearby molecules check tool')
  print('===================================================')

  if not (LAST_RECEPTOR and LAST_POSE_SDF and os.path.exists(LAST_POSE_SDF)):
    return [], 'No blind docking has run in this session yet (or its pose file is ' \
               'gone). Run blind docking first, then this check.', []

  nearby = check_nearby_molecules(LAST_RECEPTOR, LAST_POSE_SDF)
  return [], str(nearby), []


def canonical_tool(smiles: str):
  '''
  Wraps the CLI repo's canonical_node (which returns a bare SMILES string, no
  explanatory text - its own chat LLM phrased it) so the tool output is
  user-ready and matches the (list, str, list) shape _execute() unpacks.
    Args:
        smiles: A SMILES string for the molecule.
    Returns:
        (list, str, list) as above.
  '''
  canonical = _canonical_node(smiles)
  return [], f'The canonical SMILES for {smiles} is: {canonical}', []


def _as_tool_result(result: tuple):
  '''Normalizes a wrapped node's return to (list, str, list); some CLI nodes
  return 2-tuples exactly on their "no bioactives / error" paths.'''
  if len(result) == 3:
    return result
  text = ' '.join(str(x) for x in result if isinstance(x, str))
  return [], text, []


def get_actives_tool(query_protein: str):
  '''Wraps get_actives_for_protein, normalizing its 2-tuple error returns.'''
  return _as_tool_result(_get_actives(query_protein))


def get_predictions_tool(smiles_list: list[str], query_protein: str):
  '''Wraps get_predictions_for_protein, normalizing its 2-tuple error returns.'''
  return _as_tool_result(_get_predictions(smiles_list, query_protein))


def apply_safe_env():
  '''
  Neutralizes the one environment collision dockstring hits: its internal
  `obabel -V` availability test (and every conversion it runs) inherits THIS
  process's LD_LIBRARY_PATH, and if a dir like ~/.local/orca_*/lib carries a
  libstdc++.so.6 too old for libopenbabel (no CXXABI_1.3.15), obabel crashes
  on load and dockstring reports "The test command obabel -V failed".

  Call once at CLI startup: it copies vina_dock._safe_env()'s filtered
  LD_LIBRARY_PATH (dirs with an outdated libstdc++ removed) into this
  process's environment. Never touches the parent shell, so ORCA itself
  keeps working from any other terminal.
  '''
  import vina_dock
  env = vina_dock._safe_env()
  for key, value in env.items():
    if key in os.environ and os.environ[key] != value:
      os.environ[key] = env[key]