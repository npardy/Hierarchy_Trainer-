# Hierarchy_Trainer-
How to Use the Hierarchy_Trainer Repository for Review
🔗 GitHub Repository: Hierarchy Trainer

This repository contains a hierarchical text classification system designed for structured document processing, value normalization (_Val labels), and OCR correction learned by the model (not hardcoded). Below is a professional guide on how to use the program to view the hierarchy and understand the implementation required.

📌 How to Set Up & Run the Program
1️⃣ Install Dependencies
Before running the program, install the required packages:

bash
Copy
Edit
pip install -r requirements.txt
pip install -r requirements_cuda.txt  # If using CUDA for GPU acceleration
📌 Key dependencies:

Hugging Face Transformers for text classification
Pandas & OpenPyXL for data handling
PyTorch (CUDA-compatible) for deep learning
2️⃣ Running the GUI to View the Hierarchy
The project includes a GUI for training, testing, and viewing the hierarchical label structure.

▶️ To launch the GUI:
Run the following command:

bash
Copy
Edit
python section_classifier_gui.py
💡 In the GUI, you can:

Load a dataset and generate the hierarchical label structure.
View multi-label relationships between survey sections.
Validate the classification system before training.
3️⃣ Viewing the Hierarchical Structure
Inside the GUI:

Click "View Label Hierarchy"
The hierarchical classification structure will be displayed.
Labels are structured with parent-child relationships and _Val labels for numerical values.
📌 Understanding the Hierarchy:

Each Full_Text label represents an entire survey document.
The model must recognize multiple surveys inside a single Full_Text.
_Val labels track numerical values that must be normalized.
4️⃣ Training the Model
To train the model using the structured dataset:

bash
Copy
Edit
python setup_and_run.bat
📌 Key Training Details:

The batch size is set per Full_Text entry (ensuring no random shuffling).
The model is trained on full document context rather than isolated sentences.
OCR errors must be learned dynamically—no regex or preprocessing.
📌 Implementation Required
🔹 Transform the hierarchy into a model-compatible format
🔹 Ensure _Val labels are predicted correctly as numerical values.
🔹 Train a hierarchical model that learns OCR correction from context.
🔹 Verify that batch sizes respect Full_Text structures (confirm in data_loader.py).
🔹 Deploy the trained model for inference on raw .txt inputs.

📩 Next Steps
🔗 GitHub Repository: Hierarchy Trainer

🔹 Review section_classifier_gui.py to understand the GUI-based workflow.
🔹 Check data_loader.py to confirm batching logic for Full_Text.
🔹 Ensure the model learns OCR correction dynamically without preprocessing.
