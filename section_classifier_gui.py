import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import sys
import os
import threading
import queue
import json
import webbrowser
from typing import Optional, Dict, Set, List
import pandas as pd
from collections import defaultdict
from data_loader import add_standardization_labels
import re

class RedirectText:
    def __init__(self, text_widget, widget_type, queue):
        self.text_widget = text_widget
        self.widget_type = widget_type
        self.queue = queue

    def write(self, string):
        self.queue.put((self.widget_type, string))

    def flush(self):
        pass

class GUILogger:
    def __init__(self, text_widget, widget_type):
        self._text_widget = text_widget
        self._type = widget_type

    def write(self, message):
        self._text_widget.configure(state='normal')
        self._text_widget.insert('end', message)
        self._text_widget.see('end')
        self._text_widget.configure(state='disabled')
        self._text_widget.update()

class SectionClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Survey Section Classifier")
        self.root.geometry("1200x800")
        
        # Add current_hierarchy_info storage
        self.current_hierarchy_info = None
        
        # Message queue for thread-safe logging
        self.message_queue = queue.Queue()
        
        # Create main container
        self.main_container = ttk.Frame(root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create notebook for multiple tabs
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_training_tab()
        self.create_testing_tab()
        self.create_label_hierarchy_tab()
        self.create_validation_tab()
        
        # Start message queue checking
        self.check_messages()

        self.validation_status = {
            'hierarchy': False,
            'standardization': False,
            'survey_integrity': False
        }

    def create_training_tab(self):
        """Create the training tab with input selection and training controls"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="Model Training")
        
        # Training data input section
        train_input_frame = ttk.LabelFrame(training_frame, text="Training Configuration", padding=10)
        train_input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Training data file
        ttk.Label(train_input_frame, text="Training Data File:").pack(anchor=tk.W)
        train_file_frame = ttk.Frame(train_input_frame)
        train_file_frame.pack(fill=tk.X, pady=5)
        self.train_input_var = tk.StringVar()
        ttk.Entry(train_file_frame, textvariable=self.train_input_var, width=80).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(train_file_frame, text="Browse", command=self.browse_train_input).pack(side=tk.LEFT)
        
        # Output directory
        ttk.Label(train_input_frame, text="Model Output Directory:").pack(anchor=tk.W)
        train_output_frame = ttk.Frame(train_input_frame)
        train_output_frame.pack(fill=tk.X, pady=5)
        self.train_output_var = tk.StringVar()
        ttk.Entry(train_output_frame, textvariable=self.train_output_var, width=80).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(train_output_frame, text="Browse", command=self.browse_train_output).pack(side=tk.LEFT)
        
        # Training controls
        train_control_frame = ttk.Frame(training_frame)
        train_control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Add training button with reference
        self.train_button = ttk.Button(
            train_control_frame,
            text="Start Training",
            command=self.start_training,
            state=tk.DISABLED
        )
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(train_control_frame, text="View Label Hierarchy", command=self.view_label_hierarchy).pack(side=tk.LEFT, padx=5)
        ttk.Button(train_control_frame, text="Stop Training", command=self.stop_training).pack(side=tk.LEFT, padx=5)
        
        # Training output log
        ttk.Label(training_frame, text="Training Progress:").pack(anchor=tk.W, padx=10)
        self.train_output = scrolledtext.ScrolledText(training_frame, height=20, width=100)
        self.train_output.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def create_testing_tab(self):
        """Create the testing tab with model and test data selection"""
        testing_frame = ttk.Frame(self.notebook)
        self.notebook.add(testing_frame, text="Model Testing")
        
        # Model selection section
        model_frame = ttk.LabelFrame(testing_frame, text="Model Configuration", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Model directory
        ttk.Label(model_frame, text="Trained Model Directory:").pack(anchor=tk.W)
        model_dir_frame = ttk.Frame(model_frame)
        model_dir_frame.pack(fill=tk.X, pady=5)
        self.model_folder_var = tk.StringVar()
        ttk.Entry(model_dir_frame, textvariable=self.model_folder_var, width=80).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_dir_frame, text="Browse", command=self.browse_model_folder).pack(side=tk.LEFT)
        
        # Test data file
        ttk.Label(model_frame, text="Test Data File:").pack(anchor=tk.W)
        test_file_frame = ttk.Frame(model_frame)
        test_file_frame.pack(fill=tk.X, pady=5)
        self.test_input_var = tk.StringVar()
        ttk.Entry(test_file_frame, textvariable=self.test_input_var, width=80).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(test_file_frame, text="Browse", command=self.browse_test_input).pack(side=tk.LEFT)
        
        # Testing controls
        test_control_frame = ttk.Frame(testing_frame)
        test_control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(test_control_frame, text="Start Testing", command=self.start_testing).pack(side=tk.LEFT, padx=5)
        ttk.Button(test_control_frame, text="Open Results", command=self.open_prediction_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(test_control_frame, text="Stop Testing", command=self.stop_testing).pack(side=tk.LEFT, padx=5)
        
        # Testing output log
        ttk.Label(testing_frame, text="Testing Progress:").pack(anchor=tk.W, padx=10)
        self.test_output = scrolledtext.ScrolledText(testing_frame, height=20, width=100)
        self.test_output.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def create_label_hierarchy_tab(self):
        """Create a tab to display and explore label hierarchy"""
        hierarchy_frame = ttk.Frame(self.notebook)
        self.notebook.add(hierarchy_frame, text="Label Hierarchy")
        
        # Hierarchy display area
        hierarchy_text_frame = ttk.LabelFrame(hierarchy_frame, text="Label Hierarchy Visualization", padding=10)
        hierarchy_text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.hierarchy_text = scrolledtext.ScrolledText(hierarchy_text_frame, height=30, width=100)
        self.hierarchy_text.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        control_frame = ttk.Frame(hierarchy_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="Generate Hierarchy", command=self.view_label_hierarchy).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load From File", command=self.load_label_hierarchy).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export Hierarchy", command=self.export_label_hierarchy).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Clear", command=self.clear_hierarchy).pack(side=tk.LEFT, padx=5)

    def create_validation_tab(self):
        """Create a tab for validating label hierarchies and relationships"""
        validation_frame = ttk.Frame(self.notebook)
        self.notebook.add(validation_frame, text="Hierarchy Validation")
        
        # Validation controls
        control_frame = ttk.LabelFrame(validation_frame, text="Validation Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="Validate Current Hierarchy", 
                  command=self.validate_hierarchy).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Check Label Flow", 
                  command=self.check_label_flow).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Verify Parent-Child Relations", 
                  command=self.verify_relations).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Validate Standardization", 
                  command=self.validate_standardization).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Validate Survey Integrity", 
                  command=self.validate_survey_integrity).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Hierarchy Data",
                  command=self.load_hierarchy_data).pack(side=tk.LEFT, padx=5)
        
        # Validation output
        ttk.Label(validation_frame, text="Validation Results:").pack(anchor=tk.W, padx=10)
        self.validation_output = scrolledtext.ScrolledText(validation_frame, height=20, width=100)
        self.validation_output.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def create_enhanced_validation_tab(self):
        """Add validation checklist from guide"""
        val_frame = ttk.Frame(self.notebook)
        self.notebook.add(val_frame, text="Enhanced Validation")
        
        # Validation checklist
        self.val_tree = ttk.Treeview(
            val_frame,
            columns=('Check', 'Status'),
            show='headings'
        )
        self.val_tree.heading('Check', text='Validation Check')
        self.val_tree.heading('Status', text='Status')
        self.val_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Add default checks
        checks = [
            ('Hierarchy Structure', 'Pending'),
            ('Standardization', 'Pending'),
            ('Survey Integrity', 'Pending')
        ]
        for check in checks:
            self.val_tree.insert('', 'end', values=check)

    def browse_train_input(self):
        filename = filedialog.askopenfilename(
            title="Select Training Data File",
            filetypes=[("Excel files", "*.xlsx"), ("Text files", "*.txt")]
        )
        if filename:
            self.train_input_var.set(filename)

    def browse_train_output(self):
        directory = filedialog.askdirectory(title="Select Model Output Directory")
        if directory:
            self.train_output_var.set(directory)

    def browse_model_folder(self):
        directory = filedialog.askdirectory(title="Select Trained Model Directory")
        if directory:
            self.model_folder_var.set(directory)

    def browse_test_input(self):
        filename = filedialog.askopenfilename(
            title="Select Test Data File",
            filetypes=[("Excel files", "*.xlsx"), ("Text files", "*.txt")]
        )
        if filename:
            self.test_input_var.set(filename)

    def start_training(self):
        required = ['hierarchy', 'standardization', 'survey_integrity']
        if not all(self.validation_status[check] for check in required):
            messagebox.showerror(
                "Validation Required", 
                "Complete all validations first: Hierarchy Structure, Standardization, Survey Integrity"
            )
            return
        
        if not self.train_input_var.get() or not self.train_output_var.get():
            messagebox.showerror("Error", "Please select training input file and output folder")
            return

        # Clear previous output
        self.train_output.configure(state='normal')
        self.train_output.delete(1.0, tk.END)
        self.train_output.configure(state='disabled')

        # Start training in a separate thread
        self.training_thread = threading.Thread(target=self.run_training)
        self.training_thread.daemon = True
        self.training_thread.start()

    def stop_training(self):
        """Stop the training process"""
        if hasattr(self, 'training_thread') and self.training_thread.is_alive():
            # Implement training stop mechanism
            self.message_queue.put(("train", "Stopping training..."))

    def run_training(self):
        # Redirect stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = RedirectText(self.train_output, "train", self.message_queue)
        sys.stderr = RedirectText(self.train_output, "train", self.message_queue)

        try:
            from train_hiagm import HierarchicalTrainer  # More accurate naming
            trainer = HierarchicalTrainer(
                input_file=self.train_input_var.get(),
                output_dir=self.train_output_var.get(),
                hierarchy=self.current_hierarchy_info
            )
            
            # Fix the validation check
            if self.validation_status['hierarchy']:  # Direct boolean check
                trainer.train()
            else:
                self.log("Validation failed - cannot train")
                return
            
            self.message_queue.put(("train", "Training completed successfully!"))
        except Exception as e:
            self.message_queue.put(("train", f"Error during training: {str(e)}"))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def start_testing(self):
        if not self.model_folder_var.get() or not self.test_input_var.get():
            messagebox.showerror("Error", "Please select model folder and test input file")
            return

        # Clear previous output
        self.test_output.configure(state='normal')
        self.test_output.delete(1.0, tk.END)
        self.test_output.configure(state='disabled')

        # Start testing in a separate thread
        self.testing_thread = threading.Thread(target=self.run_testing)
        self.testing_thread.daemon = True
        self.testing_thread.start()

    def stop_testing(self):
        """Stop the testing process"""
        if hasattr(self, 'testing_thread') and self.testing_thread.is_alive():
            # Implement testing stop mechanism
            self.message_queue.put(("test", "Stopping testing..."))

    def run_testing(self):
        # Redirect stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = RedirectText(self.test_output, "test", self.message_queue)
        sys.stderr = RedirectText(self.test_output, "test", self.message_queue)

        try:
            import test_model
            test_model.main(
                model_dir=self.model_folder_var.get(),
                test_file=self.test_input_var.get()
            )
            self.message_queue.put(("test", "Testing completed successfully!"))
        except Exception as e:
            self.message_queue.put(("test", f"Error during testing: {str(e)}"))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def view_label_hierarchy(self):
        if not self.train_input_var.get():
            messagebox.showinfo("Info", "Please select a training input file first")
            return

        # Clear previous hierarchy
        self.clear_hierarchy()

        # Generate hierarchy in a separate thread
        thread = threading.Thread(target=self.generate_label_hierarchy)
        thread.daemon = True
        thread.start()

    def generate_label_hierarchy(self):
        """Generate and display label hierarchy"""
        try:
            from data_loader import load_data
            
            # Load data and generate hierarchy
            train_surveys, _, hierarchy_info = load_data(self.train_input_var.get())
            
            # Store hierarchy for validation functions
            self.current_hierarchy_info = hierarchy_info
            
            # Format and display hierarchy
            hierarchy_text = self.format_label_hierarchy(hierarchy_info)
            self.update_hierarchy_display(hierarchy_text)
            
            self.message_queue.put(("train", "Label hierarchy generated successfully!"))
        except Exception as e:
            self.message_queue.put(("train", f"Error generating label hierarchy: {str(e)}"))

    def format_label_hierarchy(self, hierarchy_info):
        """Format hierarchy information for display"""
        text = "Label Hierarchy Details:\n\n"
        
        # Main survey flow section
        text += "Main Survey Flow:\n"
        text += "  Preamble (Optional) -> Beginning -> Boundary -> End\n\n"
        
        # Strict sequence sections
        text += "Strict Label Sequences:\n\n"
        
        # Display strict sequences from hierarchy
        for seq_name, sequence in hierarchy_info['strict_sequences'].items():
            text += f"{seq_name.capitalize()} Sequence:\n"
            text += "   " + " -> ".join(sequence) + "\n\n"
        
        # Full hierarchy section
        text += "Complete Label Structure:\n"
        for label, details in hierarchy_info['structure'].items():
            text += f"Label: {label}\n"
            text += f"  Depth: {details['depth']}\n"
            text += f"  Parent: {details['parent'] or 'None'}\n"
            
            if details['children']:
                text += "  Children (in order):\n"
                for child, order in details['children'].items():
                    text += f"    - {child} (Order: {order})\n"
            
            if details['siblings']:
                text += "  Siblings:\n"
                for sibling in sorted(details['siblings']):
                    text += f"    - {sibling}\n"
            
            text += "\n"
        
        return text

    def update_hierarchy_display(self, text):
        """Update the hierarchy text display"""
        self.hierarchy_text.configure(state='normal')
        self.hierarchy_text.delete(1.0, tk.END)
        self.hierarchy_text.insert(tk.END, text)
        self.hierarchy_text.configure(state='disabled')

    def load_label_hierarchy(self):
        """Load label hierarchy from a JSON file"""
        filename = filedialog.askopenfilename(
            title="Load Label Hierarchy",
            filetypes=[("JSON files", "*.json")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    hierarchy_data = json.load(f)
                
                # Store hierarchy for validation functions
                self.current_hierarchy_info = hierarchy_data
                
                # Format and display hierarchy
                formatted_text = self.format_label_hierarchy(hierarchy_data)
                self.update_hierarchy_display(formatted_text)
                
                self.message_queue.put(("train", f"Label hierarchy loaded from {filename}"))
            except Exception as e:
                messagebox.showerror("Error", f"Could not load hierarchy: {str(e)}")

    def export_label_hierarchy(self):
        """Export label hierarchy to a JSON file"""
        if not self.hierarchy_text.get(1.0, tk.END).strip():
            messagebox.showinfo("Info", "No hierarchy data to export")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Export Label Hierarchy",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if filename:
            try:
                hierarchy_text = self.hierarchy_text.get(1.0, tk.END)
                
                # Parse hierarchy text to create JSON structure
                hierarchy_data = self.parse_hierarchy_text(hierarchy_text)
                
                with open(filename, 'w') as f:
                    json.dump(hierarchy_data, f, indent=2)
                
                messagebox.showinfo("Success", f"Hierarchy exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not export hierarchy: {str(e)}")

    def parse_hierarchy_text(self, text):
        """Parse hierarchy text display into JSON structure"""
        hierarchy_data = {
            'structure': {},
            'main_flow': ['Preamble', 'Beginning', 'Boundary', 'End'],
            'strict_sequences': {
                'beginning_coord': [
                    'Beginning', 
                    'Beginning>Coord', 
                    'Beginning>Coord>N', 
                    'Beginning>Coord>E'
                ],
                'beginning_mon': [
                    'Beginning', 
                    'Beginning>Mon', 
                    'Beginning>Mon>Mon_Name', 
                    'Beginning>Mon>Mon_Name>Mon_N', 
                    'Beginning>Mon>Mon_Name>Mon_E'
                ],
                'boundary_line': [
                    'Boundary', 
                    'Boundary>Line', 
                    'Boundary>Line>Landowner', 
                    'Boundary>Line>BD'
                ]
            }
        }
        
        # Parse the complete label structure section
        current_label = None
        current_section = None
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
                
            if line.startswith('Label: '):
                current_label = line.split('Label: ')[1]
                hierarchy_data['structure'][current_label] = {
                    'depth': 0,
                    'parent': None,
                    'children': {},
                    'siblings': set()
                }
                current_section = None
                
            elif current_label:
                if line.startswith('  Depth: '):
                    hierarchy_data['structure'][current_label]['depth'] = int(line.split('Depth: ')[1])
                    
                elif line.startswith('  Parent: '):
                    parent = line.split('Parent: ')[1]
                    hierarchy_data['structure'][current_label]['parent'] = None if parent == 'None' else parent
                    
                elif line.startswith('  Children:'):
                    current_section = 'children'
                    
                elif line.startswith('  Siblings:'):
                    current_section = 'siblings'
                    
                elif line.startswith('    - '):
                    if current_section == 'children':
                        # Parse child entry with order
                        child_text = line.split('    - ')[1]
                        child_name = child_text.split(' (Order: ')[0]
                        order = int(child_text.split('Order: ')[1].rstrip(')'))
                        hierarchy_data['structure'][current_label]['children'][child_name] = order
                    elif current_section == 'siblings':
                        # Parse sibling entry
                        sibling = line.split('    - ')[1]
                        hierarchy_data['structure'][current_label]['siblings'].add(sibling)
        
        return hierarchy_data

    def validate_hierarchy(self):
        try:
            # Add null check for hierarchy info
            if not self.current_hierarchy_info or 'structure' not in self.current_hierarchy_info:
                messagebox.showerror("Error", "No hierarchy data loaded")
                return {'error': "No hierarchy data loaded"}
            
            hierarchy_info = self.current_hierarchy_info
            validation_results = ["Hierarchy Validation Results:\n"]
            
            # Check required components
            required_labels = [
                'Full_Text>Survey',
                'Full_Text>Survey>Beginning',
                'Full_Text>Survey>Boundary',
                'Full_Text>Survey>End'
            ]
            
            # Check existence
            missing_labels = [label for label in required_labels 
                            if label not in hierarchy_info['structure']]
            
            # Check parent-child relationships
            relationship_issues = []
            for label in required_labels:
                if label == 'Full_Text>Survey':  # Skip root label
                    continue
                parent = '>'.join(label.split('>')[:-1])
                if parent not in hierarchy_info['structure']:
                    relationship_issues.append(f"Missing parent {parent} for {label}")
                elif label not in hierarchy_info['structure'][parent]['children']:
                    relationship_issues.append(f"{parent} missing child {label}")

            # Build results
            if missing_labels:
                validation_results.append("Missing required labels:\n- " + "\n- ".join(missing_labels))
            if relationship_issues:
                validation_results.append("\nRelationship Issues:\n- " + "\n- ".join(relationship_issues))
            
            # Add guide's standardization check
            std_check = self.validate_standardization()
            
            # Update status dictionary
            self.validation_status['hierarchy'] = not bool(missing_labels)
            self.validation_status['standardization'] = std_check
            
            # Add this line to update model readiness
            self.validation_status['model_ready'] = all(self.validation_status.values())
            
            # THEN handle display
            self.validation_output.configure(state='normal')
            self.validation_output.delete(1.0, tk.END)
            if missing_labels or relationship_issues:
                self.validation_output.insert(tk.END, "\n".join(validation_results))
                return False
            else:
                self.validation_output.insert(tk.END, "Validation passed!\n")
                return True
        
        except Exception as e:
            self.log(f"Validation error: {str(e)}")
            return False

    def check_label_flow(self):
        if not self.current_hierarchy_info:
            messagebox.showerror("Error", "Load hierarchy data first")
            return
        
        # Get the main flow sequence from hierarchy info
        main_flow = self.current_hierarchy_info.get('strict_sequences', {}).get('main_flow', [])
        
        # Validate sequence
        results = ["Label Flow Analysis:\n"]
        prev_label = None
        
        for label in main_flow:
            if prev_label:
                # Get hierarchy depth and relationships
                current_depth = self.current_hierarchy_info['structure'][label]['depth']
                prev_depth = self.current_hierarchy_info['structure'][prev_label]['depth']
                
                # Check valid transitions
                if current_depth > prev_depth:
                    # Should be direct parent-child relationship
                    parent = '>'.join(label.split('>')[:-1])
                    if parent == prev_label:
                        results.append(f"✓ Valid parent-child: {prev_label} -> {label}")
                    else:
                        results.append(f"✗ Invalid parent-child: {prev_label} -> {label}")
                elif current_depth == prev_depth:
                    # Should be siblings with correct order
                    parent = '>'.join(label.split('>')[:-1])
                    prev_parent = '>'.join(prev_label.split('>')[:-1])
                    
                    if parent == prev_parent:
                        siblings = list(self.current_hierarchy_info['structure'][parent]['children'].keys())
                        prev_idx = siblings.index(prev_label)
                        curr_idx = siblings.index(label)
                        if curr_idx == prev_idx + 1:
                            results.append(f"✓ Valid sibling order: {prev_label} -> {label}")
                        else:
                            results.append(f"✗ Out-of-order siblings: {prev_label} before {label}")
                    else:
                        results.append(f"✗ Not siblings: {prev_label} -> {label}")
                else:
                    results.append(f"✗ Invalid depth change: {prev_label} ({prev_depth}) -> {label} ({current_depth})")
                
            prev_label = label

        self.validation_output.configure(state='normal')
        self.validation_output.delete(1.0, tk.END)
        self.validation_output.insert(tk.END, "\n".join(results))
        self.validation_output.configure(state='disabled')

    def verify_relations(self):
        """Verify parent-child relationships in the hierarchy"""
        try:
            if not self.current_hierarchy_info:
                messagebox.showinfo("Info", "No hierarchy data to verify")
                return
            
            # Verify relationships
            relation_results = ["Parent-Child Relationship Verification:"]
            
            # Check each label's relationships
            for label, details in self.current_hierarchy_info['structure'].items():
                relation_results.append(f"\nLabel: {label}")
                
                # Check parent relationship
                parent = details['parent']
                if parent:
                    if parent in self.current_hierarchy_info['structure']:
                        if label in self.current_hierarchy_info['structure'][parent]['children']:
                            relation_results.append(f"✓ Valid parent: {parent}")
                        else:
                            relation_results.append(f"✗ Not listed as child of parent: {parent}")
                    else:
                        relation_results.append(f"✗ Parent not found: {parent}")
                else:
                    relation_results.append("- No parent (top-level label)")
                
                # Check children relationships
                children = details['children']
                if children:
                    relation_results.append("Children:")
                    for child in children:
                        if child in self.current_hierarchy_info['structure']:
                            if self.current_hierarchy_info['structure'][child]['parent'] == label:
                                relation_results.append(f"  ✓ Valid child: {child}")
                            else:
                                relation_results.append(f"  ✗ Parent mismatch for child: {child}")
                        else:
                            relation_results.append(f"  ✗ Child not found: {child}")
                else:
                    relation_results.append("- No children")
            
            # Display results
            self.validation_output.configure(state='normal')
            self.validation_output.delete(1.0, tk.END)
            self.validation_output.insert(tk.END, "\n".join(relation_results))
            self.validation_output.configure(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Error", f"Relationship verification failed: {str(e)}")

    def clear_hierarchy(self):
        """Clear the hierarchy text display"""
        self.hierarchy_text.configure(state='normal')
        self.hierarchy_text.delete(1.0, tk.END)
        self.hierarchy_text.configure(state='disabled')

    def open_prediction_results(self):
        """Open the prediction results Excel file"""
        results_path = 'prediction_results.xlsx'
        if os.path.exists(results_path):
            try:
                os.startfile(results_path)
            except Exception as e:
                messagebox.showerror("Error", f"Could not open results: {str(e)}")
        else:
            messagebox.showinfo("Info", "Prediction results file not found. Run testing first.")

    def check_messages(self):
        """Check message queue and update UI"""
        try:
            # Process all available messages
            while True:
                try:
                    widget_type, msg = self.message_queue.get_nowait()
                    if widget_type == "train":
                        self.train_output.configure(state='normal')
                        self.train_output.insert(tk.END, msg)
                        self.train_output.see(tk.END)
                        self.train_output.configure(state='disabled')
                    else:
                        self.test_output.configure(state='normal')
                        self.test_output.insert(tk.END, msg)
                        self.test_output.see(tk.END)
                        self.test_output.configure(state='disabled')
                except queue.Empty:
                    break
                
                # Update the GUI more frequently
                self.root.update()
        finally:
            # Schedule next check very frequently
            self.root.after(1, self.check_messages)

    def validate_standardization(self):
        try:
            self.validation_output.configure(state='normal')
            self.validation_output.delete(1.0, tk.END)
            
            df = pd.read_excel(self.train_input_var.get())
            
            # Coordinate values (N>Val/E>Val)
            coord_pattern = re.compile(r'^\d{1,3}(?:,\d{3})*(?:\.\d+)?$')  # 5,274,657.247

            # Bearings (BD>B)
            bearing_pattern = re.compile(
                r'^[NS]\s\d{1,3}°\s\d{1,2}\'\s\d{1,2}"\s[EW]$',  # N 01° 35' 34" E
                re.IGNORECASE
            )

            # Distances (BD>D)
            distance_pattern = re.compile(r'^\d{1,3}(?:,\d{3})*(?:\.\d+)?$')  # 36,374.25
            
            # Tracking
            coord_issues = 0
            bearing_issues = 0
            distance_issues = 0
            total_coords = 0
            total_bd = 0
            
            # Updated label checks
            for _, row in df.iterrows():
                label = row['Label']
                text = str(row['Text']).strip()
                
                if not text:
                    self.log(f"Empty text for {label}", warning=True)
                    continue
                
                # Coordinate standardization (N>Val and E>Val)
                if label.endswith(('N>Val', 'E>Val')):
                    total_coords += 1
                    if not coord_pattern.match(text):
                        coord_issues += 1
                        self.log(f"Invalid coord: {text} ({label})")
                
                # Bearing standardization (BD>B)
                elif label.endswith('BD>B'):
                    total_bd += 1
                    if not bearing_pattern.match(text):
                        bearing_issues += 1
                        self.log(f"Invalid bearing: {text} ({label})")
                
                # Distance standardization (BD>D)
                elif label.endswith('BD>D'):
                    total_bd += 1
                    if not distance_pattern.match(text):
                        distance_issues += 1
                        self.log(f"Invalid distance: {text} ({label})")
            
            # Calculate percentages
            coord_valid = ((total_coords - coord_issues)/total_coords) if total_coords else 1.0
            bd_valid = ((total_bd - (bearing_issues + distance_issues))/total_bd) if total_bd else 1.0
            
            # Update status with 95% threshold
            all_standardized = (coord_valid >= 0.95) and (bd_valid >= 0.95)
            self.validation_status['standardization'] = all_standardized
            self.toggle_training_button()
            
            # Display results
            output = [
                "Standardization Check Results:",
                f"- Coordinates: {coord_valid*100:.1f}% valid ({coord_issues} issues)",
                f"- Bearings/Distances: {bd_valid*100:.1f}% valid ({bearing_issues + distance_issues} issues)"
            ]
            self.validation_output.insert(tk.END, "\n".join(output))
            
            return all_standardized
            
        except Exception as e:
            self.validation_status['standardization'] = False
            self.toggle_training_button()
            raise

    def validate_survey_integrity(self):
        try:
            # Clear previous output
            self.validation_output.configure(state='normal')
            self.validation_output.delete(1.0, tk.END)
            
            # Modified to handle optional easement
            required_components = ['Full_Text', 'Survey', 'Beginning', 'Boundary', 'End']
            optional_components = ['Easement']
            
            from data_loader import detect_documents, analyze_hierarchy
            
            df = pd.read_excel(self.train_input_var.get())
            
            if 'Document_ID' not in df.columns:
                df = detect_documents(df)
            
            hierarchy = analyze_hierarchy(df)
            issues = []
            
            # Strict per-document checks only
            for doc_id, doc_counts in hierarchy['doc_counts'].items():
                for label, spec in hierarchy['components'].items():
                    if not spec.get('per_document', False):
                        continue
                    
                    count = doc_counts.get(label, 0)
                    if count < spec['min']:
                        issues.append(f"Doc {doc_id}: Missing {label}")
                    elif spec['max'] and count > spec['max']:
                        issues.append(f"Doc {doc_id}: Extra {label}")
            
            # Build clean output
            output = [
                f"Validated {len(hierarchy['doc_counts'])} documents",
                f"Validation Issues: {len(issues)}" if issues else "All documents valid"
            ]
            
            if issues:
                output.append("Per-document issues:")
                output.extend(issues)
            
            self.validation_output.insert(tk.END, "\n".join(output))
            
            # Update validation status
            self.validation_status['survey_integrity'] = not bool(issues)
            self.toggle_training_button()
            
            if not issues:
                self.validation_output.insert(tk.END, "✓ All surveys have valid structure")
            else:
                self.validation_output.insert(tk.END, f"Integrity Issues:\n{issues}")
            
            return not bool(issues)
            
        except Exception as e:
            messagebox.showerror("Error", f"Validation failed: {str(e)}")
            return False

    def log(self, message: str, warning: bool = False) -> None:
        """Log messages to the validation output area"""
        self.validation_output.configure(state='normal')
        self.validation_output.insert(tk.END, message + "\n")
        self.validation_output.see(tk.END)
        self.validation_output.configure(state='disabled')

    def load_hierarchy_data(self):
        try:
            # Load training data to build hierarchy
            train_file = self.train_input_var.get()
            if not train_file:
                messagebox.showerror("Error", "Select training data first")
                return
            
            # Use data_loader to get hierarchy info
            from data_loader import load_data
            _, _, self.current_hierarchy_info = load_data(train_file)
            
            messagebox.showinfo("Success", "Hierarchy data loaded from training file")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load hierarchy: {str(e)}")

    def toggle_training_button(self):
        required = ['hierarchy', 'standardization', 'survey_integrity']
        all_valid = all(self.validation_status[check] for check in required)
        
        # Debug output
        print(f"\n=== Training Button State ===")
        print(f"Hierarchy Valid: {self.validation_status['hierarchy']}")
        print(f"Standardization Valid: {self.validation_status['standardization']}")
        print(f"Survey Integrity Valid: {self.validation_status['survey_integrity']}")
        print(f"All Valid: {all_valid}\n")
        
        if all_valid:
            self.train_button.config(state=tk.NORMAL)
        else:
            self.train_button.config(state=tk.DISABLED)

    def run_comprehensive_validation(self):
        try:
            # ... existing validation code ...
            
            # Update validation status
            self.validation_status.update({
                'hierarchy': hierarchy_valid,
                'standardization': standard_valid,
                'survey_integrity': integrity_valid
            })
            
            # Update training button state
            self.toggle_training_button()
            
        except Exception as e:
            messagebox.showerror("Validation Error", str(e))

    def validate_hierarchy_structure(self, hierarchy_info):
        try:
            # ... validation logic ...
            
            self.validation_status['hierarchy'] = valid
            self.toggle_training_button()
            return valid
            
        except Exception as e:
            self.validation_status['hierarchy'] = False
            self.toggle_training_button()
            raise

    def verify_survey_integrity(self):
        try:
            # ... integrity checks ...
            
            self.validation_status['survey_integrity'] = integrity_valid
            self.toggle_training_button()
            return integrity_valid
            
        except Exception as e:
            self.validation_status['survey_integrity'] = False
            self.toggle_training_button()
            raise

if __name__ == "__main__":
    root = tk.Tk()
    app = SectionClassifierGUI(root)
    root.mainloop()