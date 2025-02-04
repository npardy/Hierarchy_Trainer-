import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

def get_parent_child_labels(label: str) -> Tuple[Optional[str], str]:
    """
    Extract parent and child labels from a hierarchical label.
    
    Args:
        label (str): The full label string
    
    Returns:
        Tuple[Optional[str], str]: (parent label, full label)
    """
    # If no '>' in label, it's a top-level label
    if '>' not in label:
        return None, label
    
    # Split the label by '>'
    parts = label.split('>')
    
    # Construct parent label (all parts except the last)
    if len(parts) > 1:
        parent = '>'.join(parts[:-1])
        return parent, label
    
    return None, label

def create_comprehensive_label_hierarchy(surveys):
    """Create hierarchy from actual labels present in dataset"""
    hierarchy = {
        'structure': OrderedDict(),
        'order': OrderedDict(),
        'counts': OrderedDict(),
        'levels': OrderedDict(),
        'strict_sequences': {}  # Remove predefined sequences
    }
    
    for survey_idx, (texts, label_pairs) in enumerate(surveys):
        for label_pair_idx, (parent, label) in enumerate(label_pairs):
            # Derive parent from label structure
            if '>' in label:
                parts = label.split('>')
                derived_parent = '>'.join(parts[:-1])
                if parent != derived_parent:
                    print(f"Correcting parent for {label} from {parent} to {derived_parent}")
                    parent = derived_parent
            
            # Count label occurrences
            hierarchy['counts'][label] = hierarchy['counts'].get(label, 0) + 1
            
            # Track first appearance
            if label not in hierarchy['order']:
                hierarchy['order'][label] = {
                    'survey': survey_idx,
                    'index': label_pair_idx
                }
            
            # Determine label depth and build full hierarchy
            parts = label.split('>')
            current_parent = None
            
            # Process each part of the label to build hierarchy
            for depth in range(len(parts)):
                current_label = '>'.join(parts[:depth+1])
                
                # Initialize label in hierarchy structure if not exists
                if current_label not in hierarchy['structure']:
                    hierarchy['structure'][current_label] = {
                        'depth': depth,
                        'parent': current_parent,
                        'children': OrderedDict(),
                        'siblings': set()
                    }
                
                # Add to levels tracking
                if depth not in hierarchy['levels']:
                    hierarchy['levels'][depth] = set()
                hierarchy['levels'][depth].add(current_label)
                
                # Add as child to parent if exists
                if current_parent:
                    parent_children = hierarchy['structure'][current_parent]['children']
                    if current_label not in parent_children:
                        parent_children[current_label] = len(parent_children)
                
                # Update siblings for current level
                if current_parent:
                    for sibling in hierarchy['structure'][current_parent]['children']:
                        if sibling != current_label:
                            hierarchy['structure'][current_label]['siblings'].add(sibling)
                            hierarchy['structure'][sibling]['siblings'].add(current_label)
                
                current_parent = current_label

    # Post-process to remove invalid relationships
    for label in list(hierarchy['structure'].keys()):
        parent = hierarchy['structure'][label]['parent']
        if parent and parent not in hierarchy['structure']:
            print(f"Removing invalid parent reference: {label} -> {parent}")
            hierarchy['structure'][label]['parent'] = None

    # Generate sequences dynamically based on actual parent-child relationships
    hierarchy['strict_sequences'] = generate_dynamic_sequences(hierarchy)
    
    # Debug and validation printing
    print("\nLabel Hierarchy Analysis:")
    print("\nStrict Sequences:")
    for seq_name, sequence in hierarchy['strict_sequences'].items():
        print(f"{seq_name}: {' -> '.join(sequence)}")
    
    print("\nHierarchy Levels:")
    for depth, labels in hierarchy['levels'].items():
        print(f"Depth {depth}: {labels}")
    
    print("\nLabel Occurrence Counts:")
    for label, count in hierarchy['counts'].items():
        print(f"{label}: {count} times")
    
    print("\nFirst Appearance:")
    for label, appearance in hierarchy['order'].items():
        print(f"{label}: First in Survey {appearance['survey']}, Index {appearance['index']}")
    
    print("\nDetailed Hierarchy Structure:")
    for label, details in hierarchy['structure'].items():
        print(f"\nLabel: {label}")
        print(f"  Depth: {details['depth']}")
        print(f"  Parent: {details['parent']}")
        print("  Children (in order):")
        for child, child_order in details['children'].items():
            print(f"    - {child} (Order: {child_order})")
        print(f"  Siblings: {details['siblings']}")
    
    return hierarchy

def generate_dynamic_sequences(hierarchy):
    sequences = {}
    
    # Correct depth progression sequence
    main_flow = [
        'Full_Text',  # depth 0
        'Full_Text>Survey',  # depth 1
        'Full_Text>Survey>Preamble',  # depth 2
        'Full_Text>Survey>Beginning',  # depth 2
        'Full_Text>Survey>Boundary',  # depth 2
        'Full_Text>Survey>End'  # depth 2
    ]
    
    # Convert to sibling-based flow
    sequences['main_flow'] = [
        label for label in main_flow 
        if label in hierarchy['structure']
    ]
    
    # Add sibling order validation
    survey_children = hierarchy['structure']['Full_Text>Survey']['children']
    ordered_children = sorted(survey_children.items(), key=lambda x: x[1])
    sequences['sibling_flow'] = [child[0] for child in ordered_children]
    
    return sequences

def detect_survey_boundaries(df, label_col):
    """Group ALL text segments by original Full_Text document"""
    document_ids = []
    current_doc = 0
    in_document = False
    
    for label in df[label_col]:
        # New document starts with root Full_Text label
        if label == "Full_Text":
            current_doc += 1
            in_document = True
        elif in_document and label.startswith("Full_Text>"):
            # Still in same document
            pass
        else:
            in_document = False
            
        document_ids.append(current_doc)
    
    return document_ids

def create_batches(surveys, batch_size=1):
    """Create batches containing full documents with all their surveys"""
    # Each "survey" is actually a Full_Text document
    return [surveys[i:i+batch_size] 
          for i in range(0, len(surveys), batch_size)]

def load_data(file_path: str, test_size: float = 0.2):
    """
    Load data from either Excel or TSV file.
    
    Args:
        file_path (str): Path to the input file
        test_size (float): Proportion of data to use for testing
    
    Returns:
        Tuple of training and test surveys with hierarchy information
    """
    # Read file based on extension
    if file_path.endswith('.txt'):
        df = pd.read_csv(file_path, sep='\t', dtype=str, keep_default_na=False)
    else:
        df = pd.read_excel(file_path, dtype=str, keep_default_na=False)
    
    # Detect columns dynamically
    text_col = next((col for col in ['Component Text', 'text', 'Text'] if col in df.columns), None)
    label_col = next((col for col in ['Component Label', 'label', 'Label'] if col in df.columns), None)
    
    if not text_col or not label_col:
        raise ValueError(f"Required columns not found. Available: {df.columns.tolist()}")
    
    print(f"Using columns: Text='{text_col}', Label='{label_col}'")
    
    # Ensure text column contains non-empty strings
    df[text_col] = df[text_col].astype(str)
    df = df[df[text_col].str.strip() != '']
    
    # Modified section: Add Full_Text parent where missing
    df['Label'] = df.apply(
        lambda row: f"Full_Text>{row[label_col]}" 
        if not row[label_col].startswith('Full_Text') 
        else row[label_col],
        axis=1
    )
    
    # Keep your existing boundary detection
    df['survey_id'] = detect_survey_boundaries(df, 'Label')
    
    # Add standardization labels (new)
    df = add_standardization_labels(df)
    
    grouped = df.groupby('survey_id')
    
    all_surveys = []
    for survey_id, group in grouped:
        texts = group[text_col].tolist()
        label_pairs = [get_parent_child_labels(label) for label in group['Label']]
        all_surveys.append((texts, label_pairs))
        
        # Debug first few surveys
        if survey_id < 3:
            print(f"Survey {survey_id + 1}: {len(texts)} texts, {len(label_pairs)} labels")
            print(f"Sample texts: {texts[:2]}")
            print(f"Sample labels: {label_pairs[:2]}")
    
    # Analyze comprehensive label hierarchy
    label_hierarchy_info = create_comprehensive_label_hierarchy(all_surveys)
    
    # Split surveys
    num_surveys = len(all_surveys)
    num_test = int(num_surveys * test_size)
    test_indices = np.random.choice(num_surveys, num_test, replace=False)
    test_mask = np.zeros(num_surveys, dtype=bool)
    test_mask[test_indices] = True
    
    train_surveys = [s for i, s in enumerate(all_surveys) if not test_mask[i]]
    test_surveys = [s for i, s in enumerate(all_surveys) if test_mask[i]]
    
    print(f"Split into {len(train_surveys)} training and {len(test_surveys)} test surveys")
    
    # Debug first document
    if all_surveys:
        first_doc_texts, first_doc_labels = all_surveys[0]
        print(f"\nFirst Full Document Contents:")
        print(f"Total segments: {len(first_doc_texts)}")
        print("Label distribution:")
        print(pd.Series([label for _, label in first_doc_labels]).value_counts())
    
    print("\nPrediction-Ready Batch Example:")
    full_document = all_surveys[0][0]  # First document's texts
    first_doc_labels = [label for _, label in all_surveys[0][1]]  # Get labels from (texts, label_pairs)
    print(f"Contains {len([l for l in first_doc_labels if l == 'Full_Text>Survey'])} surveys")  # Exact match
    print("Sample segments:", full_document[:3])
    
    return train_surveys, test_surveys, label_hierarchy_info

def add_standardization_labels(df):
    new_rows = []
    for _, row in df.iterrows():
        new_rows.append(row)
        
        # Add coordinate standardization
        if 'Coord>N' in row['Label'] or 'Coord>E' in row['Label']:
            val_row = row.copy()
            val_row['Label'] = f"{row['Label']}>Val"
            new_rows.append(val_row)
            
        # Add BD standardization
        if row['Label'].endswith('BD'):
            b_row = row.copy()
            b_row['Label'] = f"{row['Label']}>B"
            new_rows.append(b_row)
            
            d_row = row.copy()
            d_row['Label'] = f"{row['Label']}>D"
            new_rows.append(d_row)
    
    return pd.DataFrame(new_rows)

def get_unique_labels(surveys, hierarchy_info=None):
    """
    Extract unique parent and child labels.
    
    Args:
        surveys (List[Tuple]): List of surveys with texts and labels
        hierarchy_info (Dict, optional): Precomputed hierarchy information
    
    Returns:
        Tuple[List[str], List[str]]: Parent and child labels
    """
    # If hierarchy wasn't precomputed, compute it
    if hierarchy_info is None:
        hierarchy_info = create_comprehensive_label_hierarchy(surveys)
    
    # Separate parent and child labels based on depth
    parent_labels = [label for label, details in hierarchy_info['structure'].items() if details['depth'] == 0]
    child_labels = [label for label, details in hierarchy_info['structure'].items() if details['depth'] > 0]
    
    print(f"\nFound {len(parent_labels)} parent labels and {len(child_labels)} child labels")
    
    return parent_labels, child_labels

def analyze_hierarchy(df):
    """Analyze label hierarchy from dataframe with document awareness"""
    hierarchy = {
        'components': {},
        'structure': OrderedDict(),
        'doc_counts': {},
        'order': OrderedDict()
    }
    
    # First pass - build structure and order
    all_labels = df['Label'].unique()
    for label in all_labels:
        parts = label.split('>')
        parent = '>'.join(parts[:-1]) if len(parts) > 1 else None
        
        hierarchy['structure'][label] = {
            'depth': len(parts)-1,
            'parent': parent,
            'children': []
        }
        
        # Track label order
        if parent not in hierarchy['order']:
            hierarchy['order'][parent] = []
        if label not in hierarchy['order'][parent]:
            hierarchy['order'][parent].append(label)
    
    # Second pass - document counts
    grouped = df.groupby('Document_ID')
    for doc_id, group in grouped:
        hierarchy['doc_counts'][doc_id] = {
            label: (group['Label'] == label).sum()
            for label in all_labels
        }
    
    # Third pass - component rules
    for label in all_labels:
        parts = label.split('>')
        
        if label == 'Full_Text':
            hierarchy['components'][label] = {
                'min': 1, 'max': 1, 'per_document': True
            }
        elif label == 'Full_Text>Survey':
            hierarchy['components'][label] = {
                'min': 1, 'max': None, 'per_document': True
            }
        elif any(part in ['Preamble', 'Beginning', 'End'] for part in parts):
            hierarchy['components'][label] = {
                'min': 1, 'max': 1, 'per_document': False  # Per survey
            }
        elif 'Easement' in label:
            hierarchy['components'][label] = {
                'min': 0, 'max': 1, 'per_document': False  # Optional per survey
            }
        else:
            hierarchy['components'][label] = {
                'min': 0, 'max': None, 'per_document': False
            }
    
    # Populate children relationships
    for label in hierarchy['structure']:
        parent = hierarchy['structure'][label]['parent']
        if parent in hierarchy['structure']:
            hierarchy['structure'][parent]['children'].append(label)
    
    return hierarchy

def detect_documents(df):
    """Auto-detect document boundaries with multiple surveys per doc"""
    df['Document_ID'] = 0
    doc_counter = 0
    
    # Track document starts using Full_Text label
    for i, row in df.iterrows():
        if row['Label'] == 'Full_Text':
            doc_counter += 1
        df.at[i, 'Document_ID'] = doc_counter
    
    print(f"Detected {doc_counter} documents containing multiple surveys")
    return df
