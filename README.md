# Fine-tuning Mistral-7B for Automatic Short Answer Grading

This repository contains a comprehensive Kaggle notebook for fine-tuning Mistral-7B on the EngSAF dataset to perform automatic short answer grading with feedback generation. The implementation uses 4-bit quantization and LoRA (Low-Rank Adaptation) for efficient training on limited GPU resources.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Dataset Format](#dataset-format)
5. [Notebook Structure](#notebook-structure)
6. [Configuration](#configuration)
7. [Functions and Classes](#functions-and-classes)
8. [Usage Instructions](#usage-instructions)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Troubleshooting](#troubleshooting)

## Overview

This notebook implements a complete pipeline for:
- **Automatic Short Answer Grading**: Predicts scores (0-5) for student answers
- **Feedback Generation**: Generates constructive feedback for each answer
- **Multi-task Learning**: Simultaneously learns to predict scores and generate feedback
- **Parameter-Efficient Fine-tuning**: Uses LoRA/PEFT to fine-tune only a small subset of parameters
- **Memory-Efficient Training**: Employs 4-bit quantization to fit the model in limited GPU memory

### Key Features

- ✅ 4-bit quantization for memory efficiency (fits in 16GB GPU)
- ✅ LoRA/PEFT for parameter-efficient fine-tuning
- ✅ Unseen-question split methodology (realistic evaluation)
- ✅ Comprehensive evaluation metrics (QWK, BLEU, ROUGE, BERTScore)
- ✅ Full inference pipeline with configurable sampling
- ✅ Visualization and analysis tools
- ✅ Kaggle-optimized with checkpoint saving

## Requirements

### Hardware
- **GPU**: P100 (16GB) or better (T4, V100, A100 recommended)
- **GPU Memory**: ~12-14GB with 4-bit quantization
- **RAM**: 8GB+ recommended

### Software
- Python 3.8+
- CUDA-capable GPU with CUDA 11.8+
- Kaggle account (for Kaggle Notebooks) or local Jupyter environment

### Estimated Runtime
- **Full Training**: 3-5 hours (depending on dataset size and GPU)
- **Evaluation**: 10-30 minutes (depending on test set size)
- **Inference**: <1 second per answer

## Installation

The notebook automatically installs all required dependencies. The main packages include:

```python
transformers>=4.35.0      # Hugging Face transformers library
peft>=0.6.0                # Parameter-Efficient Fine-Tuning
bitsandbytes>=0.41.0       # 4-bit quantization support
datasets>=2.14.0           # Dataset handling
accelerate>=0.24.0         # Training acceleration
wandb                      # Experiment tracking (optional)
scikit-learn               # Evaluation metrics
nltk                       # Natural language processing
rouge-score                # ROUGE metric for feedback evaluation
bert-score                 # BERTScore for semantic similarity
torch                      # PyTorch deep learning framework
```

## Dataset Format

The EngSAF dataset is **pre-split** into four CSV files:

| File | Purpose | Description |
|------|---------|-------------|
| `train.csv` | Training set | Used for model training |
| `val.csv` | Validation set | Used for validation during training |
| `unseen_question.csv` | Test set (primary) | Test set with unseen questions (most realistic evaluation) |
| `unseen_answers.csv` | Test set (secondary) | Test set with unseen answers (additional evaluation) |

### CSV Column Structure

Each CSV file contains the following columns (with original names):

| Original Column Name | Mapped To | Description | Example |
|---------------------|-----------|-------------|---------|
| `Question` | `question` | The question being answered | "What is photosynthesis?" |
| `Student Answer` | `student_answer` | Student's answer to grade | "It's when plants make food using sunlight." |
| `Correct Answer` | `reference_answer` | Reference/ideal answer | "Photosynthesis is the process by which plants convert light energy..." |
| `output_label` | `score` | Numeric score (0-5) | 3 |
| `feedback` | `feedback` | Reference feedback text | "Your answer captures the basic concept but lacks detail..." |
| `Question_id` | `question_id` | Question identifier (optional) | 12345 |

**Note**: The notebook automatically maps the original column names to standardized names internally.

### Dataset Loading

The notebook includes two main functions for loading the dataset:

1. **`load_engsaf_split(dataset_dir, split)`**: Loads a specific split
   - `split` can be: `'train'`, `'val'`, `'unseen_question'`, `'unseen_answers'`
   - Automatically maps column names
   - Validates required columns
   - Cleans data (removes NaN values and empty strings)
   - Displays score distribution

2. **`load_all_engsaf_splits(dataset_dir)`**: Loads all splits at once
   - Returns: `(train_df, val_df, test_df_unseen_question, test_df_unseen_answers)`
   - Convenient for loading everything in one call

**Expected dataset directory structure:**
```
EngSAF dataset/
├── train.csv
├── val.csv
├── unseen_question.csv
├── unseen_answers.csv
└── Readme.md.docx (optional)
```

**Expected paths (searched in order):**
1. `EngSAF dataset/` (current directory)
2. `./EngSAF dataset/` (relative path)
3. `/kaggle/input/EngSAF dataset/` (Kaggle environment)
4. `/kaggle/input/engsaf-dataset/` (alternative Kaggle path)
5. `/kaggle/input/engsaf/` (alternative Kaggle path)

## Notebook Structure

The notebook is organized into 8 main sections:

### 1. Environment Setup

**Purpose**: Install dependencies and configure the environment.

**Components**:
- Library installation via pip
- Import statements for all required packages
- GPU availability check and memory reporting
- Kaggle environment detection
- Random seed setting for reproducibility

**Key Variables**:
- `KAGGLE_ENV`: Boolean indicating Kaggle environment
- `OUTPUT_DIR`: Directory for saving outputs (`/kaggle/working` or `./output`)
- `INPUT_DIR`: Directory for input data (`/kaggle/input` or `./input`)
- `WANDB_AVAILABLE`: Boolean for wandb availability

### 2. Data Loading & Preprocessing

**Purpose**: Load and prepare the EngSAF dataset for training.

**Components**:
- Configuration dictionary (`CONFIG`) with all hyperparameters
- Dataset loading functions for pre-split data (`load_engsaf_split()`, `load_all_engsaf_splits()`)
- Column name mapping (original → standardized)
- Prompt template creation functions
- Custom `GradingDataset` class for PyTorch

**Key Functions**:
- `load_engsaf_split(dataset_dir, split)`: Loads a specific split (train/val/unseen_question/unseen_answers) with automatic column mapping
- `load_all_engsaf_splits(dataset_dir)`: Loads all four splits at once, returns tuple of DataFrames
- `create_prompt_template(question, student_answer, rubric, system_prompt)`: Creates instruction-tuning prompts
- `format_instruction(sys_prompt, user_prompt, assistant_response)`: Formats prompts in Mistral's chat template
- `create_assistant_response(score, feedback)`: Formats model output

**Key Variables** (after loading):
- `train_df`: Training DataFrame
- `val_df`: Validation DataFrame
- `test_df_unseen_question`: Test set with unseen questions (primary test set)
- `test_df_unseen_answers`: Test set with unseen answers (secondary test set)
- `test_df`: Alias for `test_df_unseen_question` (used for primary evaluation)

**Classes**:
- `GradingDataset(Dataset)`: PyTorch dataset class for instruction tuning

**Note**: The dataset is already split, so no splitting function is needed. The notebook automatically loads all splits and creates PyTorch datasets.

### 3. Model Configuration

**Purpose**: Load and configure the Mistral-7B model with quantization and LoRA.

**Components**:
- Tokenizer loading and configuration
- 4-bit quantization configuration (`BitsAndBytesConfig`)
- Model loading with quantization
- LoRA configuration and application
- Gradient checkpointing setup

**Key Steps**:
1. Load tokenizer and set padding token
2. Configure 4-bit quantization (NF4 quantization type, float16 compute)
3. Load Mistral-7B model with quantization
4. Prepare model for k-bit training
5. Apply LoRA adapters to specified modules
6. Enable gradient checkpointing

**LoRA Configuration**:
- **Rank (r)**: 32 (number of trainable parameters)
- **Alpha**: 64 (scaling parameter)
- **Dropout**: 0.05
- **Target Modules**: `['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']`

### 4. Training Setup

**Purpose**: Configure and initialize the training process.

**Components**:
- Dataset instantiation
- Data collator configuration
- Training arguments setup
- Trainer initialization
- Optional wandb integration

**Training Configuration**:
- **Epochs**: 3
- **Batch Size**: 4 per device
- **Gradient Accumulation**: 4 steps (effective batch size = 16)
- **Learning Rate**: 2e-4
- **Scheduler**: Cosine with 100 warmup steps
- **Mixed Precision**: FP16
- **Checkpointing**: Every 500 steps
- **Evaluation**: Every 500 steps

**Callbacks**:
- `EarlyStoppingCallback`: Stops training if validation loss doesn't improve
- `MetricsCallback`: Custom callback for logging evaluation metrics

### 5. Evaluation Metrics

**Purpose**: Comprehensive evaluation of model performance on scoring and feedback generation.

**Components**:
- Score extraction functions
- Feedback extraction functions
- Evaluation metrics calculation
- Comprehensive evaluation function

**Key Functions**:
- `quadratic_weighted_kappa(y_true, y_pred)`: Calculates QWK (standard metric for automated essay scoring)
- `extract_score_from_response(response_text)`: Extracts numeric score from model output using regex patterns
- `extract_feedback_from_response(response_text)`: Extracts feedback text from model output
- `evaluate_feedback_quality(predicted, reference)`: Calculates BLEU, ROUGE, and BERTScore
- `evaluate_model(model, tokenizer, test_dataset, device, max_samples)`: Comprehensive evaluation function

**Metrics Calculated**:
- **For Scoring**:
  - Quadratic Weighted Kappa (QWK)
  - Cohen's Kappa
  - Accuracy
  - Confusion Matrix

- **For Feedback**:
  - BLEU Score
  - ROUGE-1, ROUGE-2, ROUGE-L
  - BERTScore F1

### 6. Inference Pipeline

**Purpose**: Provide functions for grading new student answers.

**Components**:
- Main inference function (`grade_answer()`)
- Model checkpoint saving function
- Example usage code

**Key Functions**:
- `grade_answer(model, tokenizer, question, student_answer, rubric, temperature, top_p, max_new_tokens, device)`: Grades a single answer and returns score + feedback
- `save_model_checkpoint(model, tokenizer, output_path)`: Saves fine-tuned model and tokenizer

**Inference Parameters**:
- **Temperature**: 0.7 (controls randomness)
- **Top-p**: 0.9 (nucleus sampling)
- **Max New Tokens**: 256 (maximum feedback length)

### 7. Visualization & Analysis

**Purpose**: Visualize training progress and model performance.

**Components**:
- Training/validation loss curve plotting
- Confusion matrix visualization
- Example prediction display

**Key Functions**:
- `plot_training_curves(log_history, save_path)`: Plots training and validation loss over steps
- `plot_confusion_matrix(y_true, y_pred, save_path)`: Creates heatmap of confusion matrix
- `display_examples(predictions, n_examples, show_good, show_bad)`: Displays example predictions (both correct and incorrect)

### 8. Optional Enhancements

**Purpose**: Placeholders for future improvements.

**Components**:
- RAG integration placeholder
- Chain-of-Thought verification placeholder
- Ensemble with RoBERTa placeholder

**Future Enhancements**:
- **RAG**: Retrieve relevant course materials to enhance grading context
- **Chain-of-Thought**: Add reasoning steps for transparent grading decisions
- **Ensemble**: Combine Mistral-7B with RoBERTa-based scoring model

## Configuration

All hyperparameters are centralized in the `CONFIG` dictionary for easy modification:

```python
CONFIG = {
    # Model config
    'model_name': 'mistralai/Mistral-7B-v0.1',
    'use_4bit': True,
    'bnb_4bit_compute_dtype': 'float16',
    'bnb_4bit_quant_type': 'nf4',
    
    # LoRA config
    'lora_r': 32,
    'lora_alpha': 64,
    'lora_dropout': 0.05,
    'lora_target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    
    # Training config
    'num_train_epochs': 3,
    'per_device_train_batch_size': 4,
    'gradient_accumulation_steps': 4,
    'learning_rate': 2e-4,
    'lr_scheduler_type': 'cosine',
    'warmup_steps': 100,
    'eval_steps': 500,
    'save_steps': 500,
    
    # Data config
    'max_length': 1024,
    # Note: test_size and val_size are not used - dataset is pre-split
    
    # Inference config
    'temperature': 0.7,
    'top_p': 0.9,
    'max_new_tokens': 256,
}
```

### Adjusting Hyperparameters

**For Memory Constraints**:
- Reduce `per_device_train_batch_size` (e.g., 2 or 1)
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Reduce `max_length` (e.g., 512 or 768)
- Reduce `lora_r` (e.g., 16)

**For Better Performance**:
- Increase `num_train_epochs` (e.g., 5)
- Increase `lora_r` and `lora_alpha` (e.g., r=64, alpha=128)
- Increase `per_device_train_batch_size` if memory allows
- Adjust learning rate (try 1e-4 to 5e-4)

## Functions and Classes

### Data Processing Functions

#### `load_engsaf_split(dataset_dir=None, split='train')`
Loads a specific split of the pre-split EngSAF dataset.

**Parameters**:
- `dataset_dir` (str, optional): Directory containing the dataset files. If None, searches common paths.
- `split` (str): Which split to load. Options:
  - `'train'`: Training set (`train.csv`)
  - `'val'` or `'validation'`: Validation set (`val.csv`)
  - `'unseen_question'` or `'test'`: Test set with unseen questions (`unseen_question.csv`)
  - `'unseen_answers'`: Test set with unseen answers (`unseen_answers.csv`)

**Returns**:
- `df` (DataFrame): Loaded and cleaned dataset with standardized column names

**Raises**:
- `FileNotFoundError`: If dataset directory or file not found
- `ValueError`: If required columns are missing or invalid split name

**Features**:
- Automatically maps column names (Question → question, Student Answer → student_answer, etc.)
- Validates required columns
- Cleans data (removes NaN values and empty strings)
- Displays score distribution

#### `load_all_engsaf_splits(dataset_dir=None)`
Loads all four splits of the EngSAF dataset at once.

**Parameters**:
- `dataset_dir` (str, optional): Directory containing the dataset files. If None, searches common paths.

**Returns**:
- `tuple`: `(train_df, val_df, test_df_unseen_question, test_df_unseen_answers)`
  - `train_df`: Training DataFrame
  - `val_df`: Validation DataFrame
  - `test_df_unseen_question`: Test set with unseen questions (primary test set)
  - `test_df_unseen_answers`: Test set with unseen answers (secondary test set)

**Note**: The dataset is already pre-split, so no splitting function is needed. This function simply loads all four CSV files and returns them as a tuple.

#### `create_prompt_template(question, student_answer, rubric=None, system_prompt=None)`
Creates instruction-tuning prompt template.

**Parameters**:
- `question` (str): The question text
- `student_answer` (str): Student's answer
- `rubric` (str, optional): Custom grading rubric
- `system_prompt` (str, optional): Custom system prompt

**Returns**:
- `system_prompt, user_prompt` (tuple): Formatted prompts

#### `format_instruction(system_prompt, user_prompt, assistant_response=None)`
Formats instruction in Mistral's chat template format.

**Parameters**:
- `system_prompt` (str): System prompt with rubric
- `user_prompt` (str): User prompt with question and answer
- `assistant_response` (str, optional): Expected model response (for training)

**Returns**:
- `prompt` (str): Formatted instruction string

### Dataset Class

#### `GradingDataset(Dataset)`
PyTorch dataset class for instruction tuning.

**Parameters**:
- `df` (DataFrame): Dataset DataFrame
- `tokenizer`: Hugging Face tokenizer
- `max_length` (int): Maximum sequence length
- `rubric` (str, optional): Custom rubric

**Methods**:
- `__len__()`: Returns dataset size
- `__getitem__(idx)`: Returns tokenized sample with labels

**Returns** (per sample):
- `input_ids`: Tokenized input
- `attention_mask`: Attention mask
- `labels`: Labels (same as input_ids for causal LM)
- `score`: Ground truth score

### Evaluation Functions

#### `quadratic_weighted_kappa(y_true, y_pred)`
Calculates Quadratic Weighted Kappa score.

**Parameters**:
- `y_true` (list): True scores
- `y_pred` (list): Predicted scores

**Returns**:
- `kappa` (float): QWK score (0-1, higher is better)

#### `extract_score_from_response(response_text)`
Extracts numeric score from model response using regex patterns.

**Parameters**:
- `response_text` (str): Model-generated text

**Returns**:
- `score` (int or None): Extracted score (0-5) or None if not found

**Patterns Tried**:
- `Score: X`, `score: X`, `Score X`
- `X out of`, `Grade: X`
- First number found (fallback)

#### `extract_feedback_from_response(response_text)`
Extracts feedback text from model response.

**Parameters**:
- `response_text` (str): Model-generated text

**Returns**:
- `feedback` (str): Extracted feedback text

#### `evaluate_feedback_quality(predicted_feedback, reference_feedback)`
Evaluates feedback quality using multiple metrics.

**Parameters**:
- `predicted_feedback` (str): Generated feedback
- `reference_feedback` (str): Reference feedback

**Returns**:
- `dict` with keys: `bleu`, `rouge1`, `rouge2`, `rougeL`, `bertscore_f1`

#### `evaluate_model(model, tokenizer, test_dataset, device='cuda', max_samples=None)`
Comprehensive evaluation function for the entire test set.

**Parameters**:
- `model`: Fine-tuned model
- `tokenizer`: Tokenizer
- `test_dataset`: Test dataset
- `device` (str): Device to run on
- `max_samples` (int, optional): Limit evaluation to N samples

**Returns**:
- `results` (dict): Dictionary with all metrics
- `predictions` (list): List of prediction dictionaries

### Inference Functions

#### `grade_answer(model, tokenizer, question, student_answer, rubric=None, temperature=0.7, top_p=0.9, max_new_tokens=256, device='cuda')`
Grades a single student answer and generates feedback.

**Parameters**:
- `model`: Fine-tuned model
- `tokenizer`: Tokenizer
- `question` (str): Question text
- `student_answer` (str): Student's answer
- `rubric` (str, optional): Custom rubric
- `temperature` (float): Sampling temperature (0.1-1.0)
- `top_p` (float): Nucleus sampling parameter (0.0-1.0)
- `max_new_tokens` (int): Maximum tokens to generate
- `device` (str): Device to run on

**Returns**:
- `dict` with keys: `score`, `feedback`, `full_response`

#### `save_model_checkpoint(model, tokenizer, output_path)`
Saves fine-tuned model and tokenizer.

**Parameters**:
- `model`: Fine-tuned model
- `tokenizer`: Tokenizer
- `output_path` (str): Path to save model

**Saves**:
- Model weights (LoRA adapters)
- Tokenizer files
- Training configuration JSON

### Visualization Functions

#### `plot_training_curves(log_history, save_path=None)`
Plots training and validation loss curves.

**Parameters**:
- `log_history` (list): Training log history from Trainer
- `save_path` (str, optional): Path to save figure

#### `plot_confusion_matrix(y_true, y_pred, save_path=None)`
Creates confusion matrix heatmap.

**Parameters**:
- `y_true` (list): True scores
- `y_pred` (list): Predicted scores
- `save_path` (str, optional): Path to save figure

#### `display_examples(predictions, n_examples=5, show_good=True, show_bad=True)`
Displays example predictions for analysis.

**Parameters**:
- `predictions` (list): List of prediction dictionaries
- `n_examples` (int): Number of examples per category
- `show_good` (bool): Show correct predictions
- `show_bad` (bool): Show incorrect predictions

## Usage Instructions

### Step 1: Prepare Dataset

The EngSAF dataset should be organized in a directory with the following structure:

```
EngSAF dataset/
├── train.csv
├── val.csv
├── unseen_question.csv
├── unseen_answers.csv
└── Readme.md.docx (optional)
```

**For Local Use:**
- Place the `EngSAF dataset` folder in the same directory as the notebook

**For Kaggle:**
- Upload the dataset folder to Kaggle as a dataset
- Add it to your notebook via the Kaggle UI

### Step 2: Load Dataset

The dataset loading cell automatically loads all splits. Simply run the cell:

```python
# The notebook automatically loads all splits
train_df, val_df, test_df_unseen_question, test_df_unseen_answers = load_all_engsaf_splits('EngSAF dataset')

# Set primary test set
test_df = test_df_unseen_question
```

**Note**: The notebook automatically:
- Searches for the dataset directory in common locations
- Maps column names from original format to standardized format
- Cleans the data (removes NaN and empty strings)
- Displays summary statistics

### Step 3: Create PyTorch Datasets

After loading the tokenizer (from Model Configuration section), the datasets are automatically created:

```python
# This happens automatically after tokenizer is loaded
train_dataset = GradingDataset(train_df, tokenizer, max_length=CONFIG['max_length'])
val_dataset = GradingDataset(val_df, tokenizer, max_length=CONFIG['max_length'])
test_dataset = GradingDataset(test_df, tokenizer, max_length=CONFIG['max_length'])

# Optional: Also create dataset for unseen_answers test set
test_dataset_unseen_answers = GradingDataset(test_df_unseen_answers, tokenizer, max_length=CONFIG['max_length'])
```

**Note**: The notebook includes error handling - if the tokenizer isn't loaded yet, it will prompt you to run the Model Configuration cells first.

### Step 4: Initialize Trainer

Uncomment the trainer initialization cell:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3), MetricsCallback()],
)
```

### Step 5: Train Model

Uncomment the training cell:

```python
print("Starting training...")
trainer.train()

# Save final model
final_model_path = os.path.join(OUTPUT_DIR, 'final_model')
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Model saved to {final_model_path}")
```

### Step 6: Evaluate Model

Uncomment the evaluation cell:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

results, predictions = evaluate_model(
    model,
    tokenizer,
    test_dataset,
    device=device,
    max_samples=100  # Adjust as needed
)

print("\nEvaluation Results:")
print(f"Quadratic Weighted Kappa (QWK): {results['qwk']:.4f}")
print(f"Cohen's Kappa: {results['cohen_kappa']:.4f}")
print(f"Accuracy: {results['accuracy']:.4f}")
```

### Step 7: Visualize Results

Uncomment visualization cells:

```python
# Training curves
plot_training_curves(trainer.state.log_history, save_path=os.path.join(OUTPUT_DIR, 'training_curves.png'))

# Confusion matrix
plot_confusion_matrix(true_scores, predicted_scores, save_path=os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))

# Example predictions
display_examples(predictions, n_examples=3)
```

### Step 8: Use for Inference

```python
result = grade_answer(
    model,
    tokenizer,
    question="What is photosynthesis?",
    student_answer="Plants use sunlight to make food.",
    temperature=0.7,
    top_p=0.9,
    device='cuda'
)

print(f"Score: {result['score']}")
print(f"Feedback: {result['feedback']}")
```

## Evaluation Metrics

### Scoring Metrics

1. **Quadratic Weighted Kappa (QWK)**
   - Standard metric for automated essay scoring
   - Range: -1 to 1 (higher is better)
   - Accounts for magnitude of errors (e.g., predicting 4 when true is 5 is better than predicting 0)

2. **Cohen's Kappa**
   - Measures inter-rater agreement
   - Range: -1 to 1 (higher is better)
   - Accounts for agreement by chance

3. **Accuracy**
   - Percentage of exact score matches
   - Range: 0 to 1 (higher is better)

4. **Confusion Matrix**
   - Shows distribution of predictions vs. true scores
   - Helps identify systematic biases

### Feedback Metrics

1. **BLEU Score**
   - Measures n-gram overlap with reference
   - Range: 0 to 1 (higher is better)
   - Good for measuring word-level similarity

2. **ROUGE Scores**
   - **ROUGE-1**: Unigram overlap (recall)
   - **ROUGE-2**: Bigram overlap (recall)
   - **ROUGE-L**: Longest common subsequence (recall)
   - Range: 0 to 1 (higher is better)

3. **BERTScore**
   - Semantic similarity using BERT embeddings
   - Range: 0 to 1 (higher is better)
   - Better captures semantic meaning than n-gram metrics

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms**: `CUDA out of memory` error during training

**Solutions**:
1. Reduce `per_device_train_batch_size` to 2 or 1
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `max_length` to 512 or 768
4. Reduce `lora_r` to 16
5. Enable `gradient_checkpointing` (already enabled by default)

### Slow Training

**Symptoms**: Training takes longer than expected

**Solutions**:
1. Check GPU utilization (`nvidia-smi`)
2. Reduce `eval_steps` frequency (e.g., 1000 instead of 500)
3. Reduce `max_samples` during evaluation
4. Use smaller `max_length` if possible

### Poor Model Performance

**Symptoms**: Low QWK score or inaccurate predictions

**Solutions**:
1. Increase training epochs (`num_train_epochs`)
2. Adjust learning rate (try 1e-4 to 5e-4)
3. Increase LoRA rank (`lora_r`) and alpha (`lora_alpha`)
4. Check data quality and balance
5. Ensure proper train/val/test splits

### Score Extraction Fails

**Symptoms**: Many `None` scores in predictions

**Solutions**:
1. Check model output format
2. Adjust `extract_score_from_response()` regex patterns
3. Increase `max_new_tokens` to ensure score is generated
4. Lower `temperature` for more deterministic outputs

### Dataset Loading Fails

**Symptoms**: `FileNotFoundError` or `ValueError` for missing columns or directory

**Solutions**:
1. Verify dataset directory structure:
   - Ensure `EngSAF dataset/` folder exists
   - Check that all four CSV files are present: `train.csv`, `val.csv`, `unseen_question.csv`, `unseen_answers.csv`
2. Check dataset directory path:
   - For local: `EngSAF dataset/` should be in the same directory as the notebook
   - For Kaggle: Add dataset via Kaggle UI, then use `/kaggle/input/[dataset-name]/`
3. Verify CSV file column names (original format):
   - `Question`, `Student Answer`, `Correct Answer`, `output_label`, `feedback`
   - The notebook automatically maps these to standardized names
4. Check for encoding issues (use UTF-8)
5. Ensure CSV files are not corrupted or empty

### Checkpoint Saving Issues

**Symptoms**: Model not saving or checkpoints corrupted

**Solutions**:
1. Verify `OUTPUT_DIR` has write permissions
2. Check available disk space
3. Reduce `save_total_limit` if disk is full
4. Use Kaggle's "Save Version" feature as backup

### Wandb Integration Issues

**Symptoms**: Wandb errors or not logging

**Solutions**:
1. Wandb is optional - training will work without it
2. Set `report_to='none'` in training arguments if not using wandb
3. Login to wandb: `wandb login` (if using)

## Kaggle-Specific Notes

### Session Limits
- Kaggle sessions have a 9-hour limit
- Save checkpoints every 500 steps (configured)
- Use "Save Version" regularly to persist work

### Output Persistence
- Files saved to `/kaggle/working` persist after session ends
- Use `save_model_checkpoint()` to save final model
- Download outputs before session expires

### Dataset Access
- Add dataset to notebook via Kaggle UI
- Dataset directory will be available at `/kaggle/input/[dataset-name]/`
- Ensure the dataset contains all four CSV files: `train.csv`, `val.csv`, `unseen_question.csv`, `unseen_answers.csv`
- The `load_engsaf_split()` and `load_all_engsaf_splits()` functions automatically search common paths
- If your dataset folder is named differently, specify the path: `load_all_engsaf_splits('/kaggle/input/your-dataset-name/')`

### GPU Selection
- P100 (16GB) is sufficient with 4-bit quantization
- T4, V100, or A100 will be faster
- CPU-only mode is possible but very slow

## File Structure

```
mistral_engsaf_finetuning.ipynb    # Main notebook
README.md                           # This documentation
```

After running the notebook, you'll have:
```
/kaggle/working/
├── checkpoints/                   # Training checkpoints
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── ...
├── final_model/                   # Final trained model
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
├── lora_config.json              # LoRA configuration
├── training_config.json          # Training configuration
├── training_curves.png           # Loss curves
└── confusion_matrix.png          # Confusion matrix
```

## Citation

If you use this notebook, please cite:

```bibtex
@misc{mistral_engsaf_finetuning,
  title={Fine-tuning Mistral-7B for Automatic Short Answer Grading},
  author={Your Name},
  year={2024},
  note={Kaggle Notebook}
}
```

## License

This notebook is provided as-is for educational and research purposes.

## Acknowledgments

- Mistral AI for the Mistral-7B model
- Hugging Face for transformers and PEFT libraries
- EngSAF dataset creators
- Kaggle for providing GPU resources

## Changelog

### Dataset Loading Updates (Latest)

**Changes Made:**
- Updated dataset loading to work with **pre-split** EngSAF dataset
- Dataset is now organized into 4 CSV files: `train.csv`, `val.csv`, `unseen_question.csv`, `unseen_answers.csv`
- Replaced `load_engsaf_dataset()` with `load_engsaf_split()` and `load_all_engsaf_splits()`
- Added automatic column name mapping:
  - `Question` → `question`
  - `Student Answer` → `student_answer`
  - `Correct Answer` → `reference_answer`
  - `output_label` → `score`
  - `feedback` → `feedback`
- Removed `create_unseen_question_splits()` function (no longer needed - dataset is pre-split)
- Updated notebook to automatically load all splits when run
- Added support for both test sets: `unseen_question` (primary) and `unseen_answers` (secondary)

**Benefits:**
- Simpler workflow - no need to manually split data
- Consistent splits across runs
- Supports both unseen-question and unseen-answer evaluation scenarios
- Automatic path detection for local and Kaggle environments

## Contact

For questions or issues, please open an issue on the repository or contact the maintainer.

---

**Last Updated**: 2024
**Notebook Version**: 1.0
**Python Version**: 3.8+
**PyTorch Version**: 2.0+

