from sentence_transformers import SentenceTransformer
from gliner import GLiNER
import re
import numpy as np
from rdkit import Chem
from modrag_molecule_functions import name_node, smiles_node, related_node, structure_node, canonical_node
from modrag_task_graphs import get_actives_for_protein, get_predictions_for_protein, dock_from_names
# MoDrAg_CLI tool library: find_PDBID_node (renamed vs SemanticMoDrAg)
from modrag_protein_functions import uniprot_node, listbioactives_node, getbioactives_node, predict_node, pdb_node, docking_node, target_node
from modrag_protein_functions import find_PDBID_node as find_node
from chain_tools import blind_dock_node, dock_check_node, canonical_tool, \
    get_actives_tool, get_predictions_tool
from gpt_node import gpt_node
from modrag_property_functions import substitution_node, lipinski_node, pharmfeature_node, similarity_node

smiles_pattern = r'[CHONFClBrISPKacnosp0-9@+\-\[\]\(\)\/.=#$%]{5,}'
UPA_pattern = r'[OPQ][0-9][A-Z0-9]{3}[0-9]'
UPA_pattern_2 = r'[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9][A-Z]?[A-Z0-9]*[0-9]?'
PDB_pattern = r'\s[0-9][A-Z0-9]{3}'
chembl_pattern = r'[Cc][Hh][Ee][Mm][Bb][Ll][0-9]{3,}'

tool_descriptions = {
    # modrag_protein_functions.py
    'uniprot_node': 'Find the UNIPROT Accession codes (IDs) for this DNA Gyrase. Report the organisms and gene names.',
    'listbioactives_node': 'Find the Chembl IDs for the protein with UNIPROT Accession code P091H7 and \
report the number of bioactive molecules for each Chembl ID.',
    'getbioactives_node': 'Find all of the bioactives molecule SMILES and IC50s for the Chembl ID CHEMBL8999.',
    'predict_node': 'Predict the IC50 for dopamine based on the chembl ID chembl908564.',
    'gpt_node': 'Generate novel molecules for the protein with Chembl ID CHEMBL9999 by fine-tuning a generative model on its bioactives.',
    'blind_dock_node': 'Blind dock the molecules c1cccc1 into the protein structure with PDB ID 6YT5 when the binding site is unknown.',
    'dock_check_node': 'Check that the blind docked ligand landed in the correct binding site of the receptor.',
    'pdb_node': 'Find the protein sequence and and ligands (small molecules) present in the crystal structure \
represented by the PDB ID 6YT5.',
    'find_node': 'Find all the PDB IDs in the protein databank for DNA gyrase.',
    'docking_node': 'Find the docking scores for c1cccc1 and CCCCC=O in DNA gyrase. Dock c1cccc1 and CCCCC=O in the protein DNA gyrase.',
    'target_node': 'Find possible protein targets for the disease phenylketonuria.',
    # modrag_property_functions.py
    'substitution_node': 'Generate analogues of O=C([O-])CCc1ccc(O)cc1 by substitution of different groups. Report the QED values as well.',
    'lipinski_node': 'Find the Lipinski properties for c1cccc1 and CCCCC=O; report the\
QED, LogP, number of hydrogen bond donors and acceptors, molar mass, and polar surface area.',
    'canonical_node': 'Convert the SMILES string c1ccccc1O into its canonical form.',
    'similarity_node': 'Find which molecules are similar to the reference molecule c1ccccc1O among c1ccccc1, CCCCC=O, and CCCCCC=O.',
    'pharmfeature_node': 'Find the similarity in the pharmacophores between c1cccc1 and CCCCC=O. \
Find the similarity in the pharmacophores between ibuprofen and aspirin.',
    
    # modrag_molecule_functions.py
    'name_node': 'Find the name of this molecule c1cc(O)ccc1',
    'smiles_node': 'Finds SMILES strings for cyclohexane and aspirin',
    'related_node': 'Find molecules similar to c1cc(O)ccc1',
    'structure_node': 'Find the structure of the molecule with SMILES string c1cc(O)ccc1, or the name Aspirin.',
    # modrag_task_graphs.py
    'get_actives_for_protein': 'Find the bioactive molecules for the protein DNA gyrase.',
    'get_predictions_for_protein': 'Predict the IC50 value for c1cc(O)ccc1 in the protein DNA gyrase.',
    'dock_from_names': 'Dock 1,3-butadiene and levadopa in the protein MAOB.'
}

tool_descriptions_keys = list(tool_descriptions.keys())

tool_descriptions_values = list(tool_descriptions.values())

def start_ner():
  '''
  Starts the NER model for biomedical named entity recognition.
    Returns:
        model: The NER model.
  '''
  model_name =  "Shoriful025/biomedical_ner_roberta_base"
  model = GLiNER.from_pretrained("anthonyyazdaniml/gliner-biomed-large-v1.0-disease-chemical-gene-variant-species-cellline-ner")

  return model

def smiles_regex(query: str):
  '''
  Accepts a query string and returns the detected SMILES strings.
    Args:
        query: The input query string.
    Returns:
        matches: A list of detected SMILES strings.
  '''
  matches = re.findall(smiles_pattern, query)
  matches = [m for m in matches if any(char not in ['a','c','n','o','s','p','l','r'] for char in m)]
  matches = [m for m in matches if any(char not in ['0','1','2','3','4','5','6','7','8','9','l','P','O','Q'] for char in m)]
  matches = [m for m in matches if any(char not in ['0','1','2','3','4','5','6','7','8','9','.','-','+'] for char in m)]
  #matches = [m.strip(' ') for m in matches]
  print(f'Initial SMILES matches: {matches}')
  modified_matches = []
  for m in matches:
    try:
      mol = Chem.MolFromSmiles(m)
      if mol is not None:
        modified_matches.append(m)
    except:
      continue
  print(f'Modified SMILES matches: {modified_matches}')
  return modified_matches

def uniprot_regex(query: str):
  '''
  Accepts a query string and returns the detected Uniprot IDs.
    Args:
        query: The input query string.
    Returns:
        matches: A list of detected Uniprot IDs.
  '''
  matches = re.findall(UPA_pattern, query)
  matches = matches + re.findall(UPA_pattern_2, query)
  
  return matches

def pdb_regex(query: str):
  '''
  Accepts a query string and returns the detected PDB IDs.
    Args:
        query: The input query string.
    Returns:
        matches: A list of detected PDB IDs.
  '''
  matches = re.findall(PDB_pattern, query)
  matches = [m[1:] for m in matches]

  return matches

def chembl_regex(query: str):
  '''
  Accepts a query string and returns the detected ChEMBL IDs.
    Args:
        query: The input query string.
    Returns:
        matches: A list of detected ChEMBL IDs.
  '''
  matches = re.findall(chembl_pattern, query)

  return matches

def name_protein_ner(query: str, model):
  '''
  Accepts a query string and returns the detected protein, disease and molecule entities.
    Args:
        query: The input query string.
        model: The NER model to use.
    Returns:
        proteins: A list of detected protein names.
        molecules: A list of detected molecule names.
        diseases: A list of detected disease names.
  '''
  labels = ['Disease or phenotype', 'Chemical entity', 'Gene or gene product',
  'Sequence variant', 'Organism', 'Cell line']

  entities = model.predict_entities(query, labels, threshold=0.90)
  molecules = []
  proteins = []
  diseases = []

  for entity in entities:
    if entity['label'] == 'Gene or gene product':
      start_idx = entity['start']
      end_idx = entity['end']
      if ' ' not in query[start_idx:end_idx]:
        proteins.append(query[start_idx:end_idx])

    elif entity['label'] == 'Chemical entity':
      start_idx = entity['start']
      end_idx = entity['end']
      if ' ' not in query[start_idx:end_idx]:
        molecules.append(query[start_idx:end_idx])

    elif entity['label'] == 'Disease or phenotype':
      start_idx = entity['start']
      end_idx = entity['end']
      print('Found disease label: ', query[start_idx:end_idx])
      diseases.append(query[start_idx:end_idx].strip())

  molecules = [m.lower() for m in molecules]
  molecules = list(set(molecules))
  proteins = list(set(proteins))
  diseases = list(set(diseases))

  return proteins, molecules, diseases

def parse_input(query: str, model):
  '''
  Accepts a query string and returns the detected entities.
    Args:
        query: The input query string.
        model: The NER model to use.
    Returns:
        present: A dictionary with counts of each entity type found.
        proteins_list: A list of detected protein names.
        molecules_list: A list of detected molecule names.
        smiles_list: A list of detected SMILES strings.
        uniprot_list: A list of detected Uniprot IDs.
        pdb_list: A list of detected PDB IDs.
        chembl_list: A list of detected ChEMBL IDs.
  '''
  proteins_list, molecules_list, diseases_list = name_protein_ner(query, model)
  smiles_list = smiles_regex(query)
  uniprot_list = uniprot_regex(query)
  pdb_list = pdb_regex(query)
  chembl_list = chembl_regex(query)

  # drop duplicates in each list
  proteins_list = list(set(proteins_list))
  molecules_list = list(set(molecules_list))
  diseases_list = list(set(diseases_list))
  smiles_list = list(set(smiles_list))
  uniprot_list = list(set(uniprot_list))
  pdb_list = list(set(pdb_list))
  chembl_list = list(set(chembl_list))
  
  present = {
      'proteins': len(proteins_list),
      'molecules': len(molecules_list),
      'diseases': len(diseases_list),
      'smiles': len(smiles_list),
      'uniprot': len(uniprot_list),
      'pdb': len(pdb_list),
      'chembl': len(chembl_list)
  }

  return present, proteins_list, molecules_list, diseases_list, smiles_list, uniprot_list, pdb_list, chembl_list


def start_embedding(tool_descriptions_values: list[str]):
  '''
  Starts the embedding model and encodes the tool descriptions.
    Args:
        tool_descriptions_values: A list of tool description strings.
    Returns:
        document_embeddings: The encoded document embeddings.
        embed_model: The embedding model.
  '''
  embed_model = SentenceTransformer("google/embeddinggemma-300m")
  document_embeddings = embed_model.encode_document(tool_descriptions_values)

  return document_embeddings, embed_model

def define_tool_hash(tool: str, proteins_list, names_list, diseases_list, smiles_list, uniprot_list, pdb_list, chembl_list):
  '''
  Defines the tool function hash based on the selected tool and input entities.
    Args:
        tool: The selected tool name.
        proteins_list: A list of detected protein names.
        names_list: A list of detected molecule names.
        diseases_list: A list of detected disease names.
        smiles_list: A list of detected SMILES strings.
        uniprot_list: A list of detected Uniprot IDs.
        pdb_list: A list of detected PDB IDs.
        chembl_list: A list of detected ChEMBL IDs.
    Returns:
        tool_function_hash: A dictionary mapping tool names to their function and arguments.
  '''
  global tool_function_hash

  if tool == 'smiles_node':
    tool_function_hash = {
        'smiles_node': [smiles_node, [names_list]]}
  elif tool == 'name_node':
    tool_function_hash = {
        'name_node': [name_node, [smiles_list]]}
  elif tool == 'related_node':
    tool_function_hash = {
        'related_node': [related_node, [smiles_list]]}
  elif tool == 'structure_node':
    tool_function_hash = {
        'structure_node': [structure_node, [smiles_list]]}
  elif tool == 'get_predictions_for_protein':
    tool_function_hash = {
        'get_predictions_for_protein': [get_predictions_tool, [smiles_list, proteins_list[0]]]}
  elif tool == 'dock_from_names':
    tool_function_hash = {
        'dock_from_names': [dock_from_names, [names_list, proteins_list[0]]]}
  elif tool == 'get_actives_for_protein':
    tool_function_hash = {
        'get_actives_for_protein': [get_actives_tool, [proteins_list[0]]]}
  elif tool == 'uniprot_node':
    tool_function_hash = {
        'uniprot_node': [uniprot_node, [proteins_list]]}
  elif tool == 'listbioactives_node':
    tool_function_hash = {
        'listbioactives_node': [listbioactives_node, [uniprot_list]]}
  elif tool == 'getbioactives_node':
    tool_function_hash = {
        'getbioactives_node': [getbioactives_node, [chembl_list]]}
  elif tool == 'predict_node':
    tool_function_hash = {
        'predict_node': [predict_node, [smiles_list, chembl_list[0]]]}
  elif tool == 'pdb_node':
    tool_function_hash = {
        'pdb_node': [pdb_node, [pdb_list]]}
  elif tool == 'gpt_node':
    tool_function_hash = {
        'gpt_node': [gpt_node, [chembl_list[0]]]}
  elif tool == 'canonical_node':
    tool_function_hash = {
        'canonical_node': [canonical_tool, [smiles_list[0]]]}
  elif tool == 'similarity_node':
    tool_function_hash = {
        'similarity_node': [similarity_node, [smiles_list[:1], smiles_list]]}
  elif tool == 'blind_dock_node':
    tool_function_hash = {
        'blind_dock_node': [blind_dock_node, [smiles_list, pdb_list, proteins_list]]}
  elif tool == 'dock_check_node':
    tool_function_hash = {
        'dock_check_node': [dock_check_node, [names_list]]}
  elif tool == 'find_node':
    tool_function_hash = {
        'find_node': [find_node, [proteins_list]]}
  elif tool == 'docking_node':
    tool_function_hash = {
        'docking_node': [docking_node, [smiles_list, proteins_list[0]]]}
  elif tool == 'target_node':
    tool_function_hash = {
        'target_node': [target_node, [diseases_list]]}
  elif tool == 'substitution_node':
    tool_function_hash = {
        'substitution_node': [substitution_node, [smiles_list]]}
  elif tool == 'lipinski_node':
    tool_function_hash = {
        'lipinski_node': [lipinski_node, [smiles_list]]}
  elif tool == 'pharmfeature_node':
    tool_function_hash = {
        'pharmfeature_node': [pharmfeature_node, [smiles_list[0], smiles_list[1:]]]}
  
  return tool_function_hash

def define_tool_reqs(tool: str, proteins_list, names_list, diseases_list, smiles_list, uniprot_list, pdb_list, chembl_list):
  '''
  Defines the tool function requirements based on the selected tool and input entities.
    Args:
        tool: The selected tool name.
        proteins_list: A list of detected protein names.
        names_list: A list of detected molecule names.
        diseases_list: A list of detected disease names.
        smiles_list: A list of detected SMILES strings.
        uniprot_list: A list of detected Uniprot IDs.
        pdb_list: A list of detected PDB IDs.
        chembl_list: A list of detected ChEMBL IDs.
    Returns:
        tool_function_reqs: A dictionary mapping tool names to their required arguments.
  '''
  global tool_function_reqs

  if tool == 'smiles_node':
    tool_function_reqs = {
        'smiles_node': [[names_list], ['molecule names']]}
  elif tool == 'name_node':
    tool_function_reqs = {
        'name_node': [[smiles_list], ['SMILES strings']]}
  elif tool == 'related_node':
    tool_function_reqs = {
        'related_node': [[smiles_list], ['SMILES strings']]}
  elif tool == 'structure_node':
    tool_function_reqs = {
        'structure_node': [[smiles_list], ['SMILES strings']]}
  elif tool == 'get_predictions_for_protein':
    tool_function_reqs = {
        'get_predictions_for_protein': [[smiles_list, proteins_list], ['SMILES strings', 'protein names']]}
  elif tool == 'dock_from_names':
    tool_function_reqs = {
        'dock_from_names': [[names_list, proteins_list], ['molecule names', 'protein names']]}
  elif tool == 'get_actives_for_protein':
    tool_function_reqs = {
        'get_actives_for_protein': [[proteins_list], ['protein names']]}
  elif tool == 'uniprot_node':
    tool_function_reqs = {
        'uniprot_node': [[proteins_list], ['protein names']]}
  elif tool == 'listbioactives_node':
    tool_function_reqs = {
        'listbioactives_node': [[uniprot_list], ['Uniprot Accession codes']]}
  elif tool == 'getbioactives_node':
    tool_function_reqs = {
        'getbioactives_node': [[chembl_list], ['ChEMBL IDs']]}
  elif tool == 'predict_node':
    tool_function_reqs = {
        'predict_node': [[smiles_list, chembl_list], ['SMILES strings', 'ChEMBL IDs']]}
  elif tool == 'pdb_node':
    tool_function_reqs = {
        'pdb_node': [[pdb_list], ['PDB IDs']]}
  elif tool == 'gpt_node':
    tool_function_reqs = {
        'gpt_node': [[chembl_list], ['ChEMBL IDs']]}
  elif tool == 'canonical_node':
    tool_function_reqs = {
        'canonical_node': [[smiles_list], ['SMILES strings']]}
  elif tool == 'similarity_node':
    tool_function_reqs = {
        'similarity_node': [[smiles_list, smiles_list], ['reference molecule SMILES', 'molecules to compare SMILES']]}
  elif tool == 'blind_dock_node':
    tool_function_reqs = {
        'blind_dock_node': [[smiles_list, pdb_list], ['ligand SMILES strings', 'a PDB ID or receptor PDB file path']]}
  elif tool == 'dock_check_node':
    tool_function_reqs = {
        'dock_check_node': [[pdb_list], ['a receptor PDB ID from a previous blind docking']]}
  elif tool == 'find_node':
    tool_function_reqs = {
        'find_node': [[proteins_list], ['protein names']]}
  elif tool == 'docking_node':
    tool_function_reqs = {
        'docking_node': [[smiles_list, proteins_list], ['SMILES strings', 'protein names']]}
  elif tool == 'target_node':
    tool_function_reqs = {
        'target_node': [[diseases_list], ['disease names']]}
  elif tool == 'substitution_node':
    tool_function_reqs = {
        'substitution_node': [[smiles_list], ['SMILES strings']]}
  elif tool == 'lipinski_node':
    tool_function_reqs = {
        'lipinski_node': [[smiles_list], ['SMILES strings']]}
  elif tool == 'pharmfeature_node':
    tool_function_reqs = {
        'pharmfeature_node': [[smiles_list], ['SMILES strings']]}
  return tool_function_reqs

def intake(query: str, parse_model, embed_model, document_embeddings):
  '''
  Accepts a query string and returns the best tool choices and detected entities.
    Args:
        query: The input query string.
        parse_model: The NER model to use.
        embed_model: The embedding model.
        document_embeddings: The encoded document embeddings.
    Returns:
        best_tools: A list of the best tool choices.
        present: A dictionary with counts of each entity type found.
        proteins_list: A list of detected protein names.
        names_list: A list of detected molecule names.
        diseases_list: A list of detected disease names.
        smiles_list: A list of detected SMILES strings.
        uniprot_list: A list of detected Uniprot IDs.
        pdb_list: A list of detected PDB IDs.
        chembl_list: A list of detected ChEMBL IDs.
  '''
  query_embeddings = embed_model.encode_query(query)

  scores = embed_model.similarity(query_embeddings, document_embeddings)

  best_tools = []
  for i in range(3):
    try:
      best_idx = np.argmax(scores[0])
      this_tool = tool_descriptions_keys[best_idx]
      scores[0][best_idx] = -1
    except:
      this_tool = 'None'
    best_tools.append(this_tool)

  print(f"Chosen tool is: {best_tools[0]} for query: {query}")
  print(f"Second choice is: {best_tools[1]}")
  print(f"Third choice is: {best_tools[2]}")

  present, proteins_list, names_list, diseases_list, smiles_list, uniprot_list, pdb_list, chembl_list = parse_input(query, parse_model)
  for (entity_type, entity_list) in zip(present, [proteins_list, names_list, diseases_list, smiles_list, uniprot_list, pdb_list, chembl_list]):
    if present[entity_type] > 0:
      print(f'{entity_type}: {present[entity_type]}')
      for entity in entity_list:
        print(f'{entity_type}: {entity}')
  
  if present['molecules'] > 0 and present['smiles'] == 0:
    smiles_list, _, _ = smiles_node(names_list)
    print(f'Retrieved SMILES for {len(smiles_list)} molecules.')

  return best_tools, present, proteins_list, names_list, diseases_list, smiles_list, uniprot_list, pdb_list, chembl_list

def second_intake(query: str, context: str, parse_model, embed_model, document_embeddings):
  '''
  Accepts a query string and returns the best tool choices and detected entities.
    Args:
        query: The input query string.
        context: The context string.
        parse_model: The NER model to use.
        embed_model: The embedding model.
        document_embeddings: The encoded document embeddings.
    Returns:
        best_tools: A list of the best tool choices.
        present: A dictionary with counts of each entity type found.
        proteins_list: A list of detected protein names.
        diseases_list: A list of detected disease names.
        names_list: A list of detected molecule names.
        smiles_list: A list of detected SMILES strings.
        uniprot_list: A list of detected Uniprot IDs.
        pdb_list: A list of detected PDB IDs.
        chembl_list: A list of detected ChEMBL IDs.
  '''
  query_embeddings = embed_model.encode_query(query)

  scores = embed_model.similarity(query_embeddings, document_embeddings)

  best_tools = []
  for i in range(3):
    try:
      best_idx = np.argmax(scores[0])
      this_tool = tool_descriptions_keys[best_idx]
      scores[0][best_idx] = -1
    except:
      this_tool = 'None'
    best_tools.append(this_tool)

  print(f"Chosen tool is: {best_tools[0]} for query: {query}")
  print(f"Second choice is: {best_tools[1]}")
  print(f"Third choice is: {best_tools[2]}")

  present, proteins_list, names_list, diseases_list, smiles_list, uniprot_list, pdb_list, chembl_list = parse_input(context, parse_model)
  for (entity_type, entity_list) in zip(present, [proteins_list, names_list, diseases_list, smiles_list, uniprot_list, pdb_list, chembl_list]):
    if present[entity_type] > 0:
      print(f'{entity_type}: {present[entity_type]}')
      for entity in entity_list:
        print(f'{entity_type}: {entity}')
  
  if present['molecules'] > 0 and present['smiles'] == 0:
    smiles_list, _, _ = smiles_node(names_list)
    print(f'Retrieved SMILES for {len(smiles_list)} molecules.')

  return best_tools, present, proteins_list, names_list, diseases_list, smiles_list, uniprot_list, pdb_list, chembl_list
