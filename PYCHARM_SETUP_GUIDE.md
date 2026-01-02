# Step-by-Step Guide: Running Mistral Fine-tuning in PyCharm on University Station

## Overview
This guide will help you run the `mistral_engsaf_finetuning.ipynb` notebook using PyCharm on your university station with 2 Quadro RTX 5000 GPUs via remote desktop connection.

**Estimated Training Time**: 3-5 hours (you can safely leave it running for 9 hours)

---

## Prerequisites Checklist

- [ ] Remote desktop connection to university station
- [ ] PyCharm Professional (or Community with Jupyter plugin)
- [ ] Python 3.8+ installed on the station
- [ ] CUDA-capable GPU drivers installed
- [ ] Dataset folder (`EngSAF dataset`) accessible

---

## Step 1: Connect to University Station via Remote Desktop

1. **Establish Remote Desktop Connection**
   - Use Windows Remote Desktop Connection or your preferred RDP client
   - Connect to your university station
   - **IMPORTANT**: Ensure you have admin/user permissions

2. **Verify GPU Availability**
   - Open PowerShell or Command Prompt
   - Run: `nvidia-smi`
   - You should see 2 Quadro RTX 5000 GPUs listed
   - Note the CUDA version (should be 11.8+)

3. **Prevent Remote Desktop Disconnection**
   - **Option A (Recommended)**: Use `tscon` command to keep session active
   - **Option B**: Configure Windows to not lock/disconnect on idle
   - **Option C**: Use a tool like `KeepAlive` or `Caffeine` to prevent screen lock

---

## Step 2: Set Up Python Environment

### 2.1 Check Python Installation
```powershell
python --version
# Should be Python 3.8 or higher
```

### 2.2 Create Virtual Environment (Recommended)
```powershell
# Navigate to your project directory
cd "C:\Studying\9th Semester\GP 2\QA\mistral - quadro"

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
# If PowerShell execution policy blocks, use:
# .\venv\Scripts\activate.bat
```

### 2.3 Install Required Packages
```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install PyTorch with CUDA support (adjust CUDA version if needed)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other required packages
pip install transformers>=4.35.0 peft>=0.6.0 bitsandbytes>=0.41.0 datasets>=2.14.0 accelerate>=0.24.0 wandb scikit-learn nltk rouge-score bert-score jupyter ipykernel matplotlib seaborn tqdm
```

### 2.4 Verify CUDA Installation
```python
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

---

## Step 3: Configure PyCharm

### 3.1 Open Project in PyCharm
1. Launch PyCharm
2. **File → Open** → Select your project directory: `C:\Studying\9th Semester\GP 2\QA\mistral - quadro`
3. Wait for PyCharm to index files

### 3.2 Configure Python Interpreter
1. **File → Settings** (or `Ctrl+Alt+S`)
2. **Project → Python Interpreter**
3. Click the gear icon → **Add Interpreter → Existing Environment**
4. Select your virtual environment: `venv\Scripts\python.exe`
5. Click **OK**

### 3.3 Install Jupyter Support
1. In PyCharm, go to **File → Settings → Project → Python Interpreter**
2. Click the **+** button
3. Search for `jupyter` and `ipykernel`
4. Install both packages
5. Also install `ipywidgets` for better notebook experience

### 3.4 Configure Notebook Settings
1. **File → Settings → Tools → Jupyter**
2. Set **Jupyter Server** to **Managed Server**
3. Ensure **Kernel** is set to your virtual environment

---

## Step 4: Prepare Dataset

### 4.1 Verify Dataset Location
Ensure your dataset folder structure is:
```
mistral - quadro/
├── EngSAF dataset/
│   ├── train.csv
│   ├── val.csv
│   ├── unseen_question.csv
│   ├── unseen_answers.csv
│   └── Readme.md.docx
├── mistral_engsaf_finetuning.ipynb
└── ...
```

### 4.2 Test Dataset Loading (Optional)
Open a Python console in PyCharm and test:
```python
import pandas as pd
import os

dataset_dir = "EngSAF dataset"
train_path = os.path.join(dataset_dir, "train.csv")
if os.path.exists(train_path):
    df = pd.read_csv(train_path)
    print(f"Train dataset loaded: {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
else:
    print(f"Dataset not found at: {train_path}")
```

---

## Step 5: Configure Notebook for Remote Execution

### 5.1 Open Notebook in PyCharm
1. Double-click `mistral_engsaf_finetuning.ipynb` in the project tree
2. PyCharm will open it as a notebook

### 5.2 Review and Modify Configuration
Before running, check these settings in the notebook:

**In Cell 7 (CONFIG dictionary):**
- `num_train_epochs`: 3 (default, adjust if needed)
- `per_device_train_batch_size`: 4 (you can increase to 6-8 with 2 GPUs)
- `save_steps`: 500 (checkpoints every 500 steps)
- `eval_steps`: 500 (evaluation every 500 steps)

**For Multi-GPU Support:**
The notebook uses `device_map='auto'` which should automatically use both GPUs. If you want explicit control, you can modify Cell 15.

---

## Step 6: Run the Notebook

### 6.1 Execution Strategy

**Option A: Run All Cells Sequentially (Recommended)**
1. Click **Run All** button (or `Shift+Ctrl+F10`)
2. Monitor the first few cells to ensure everything loads correctly
3. Once training starts (Cell 23), you can minimize PyCharm

**Option B: Run Cells Manually**
1. Run cells 0-2: Setup and installation
2. Run cells 3-5: Imports and GPU check
3. Run cells 6-9: Data loading
4. Run cells 10-16: Model configuration
5. Run cells 17-22: Training setup
6. **Uncomment Cell 23** to start training
7. Run Cell 23: Training begins

### 6.2 Before Starting Training

**IMPORTANT**: Uncomment the training code in Cell 23:
```python
# Change from:
# print("Starting training...")
# trainer.train()

# To:
print("Starting training...")
trainer.train()

# Save final model
final_model_path = os.path.join(OUTPUT_DIR, 'final_model')
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Model saved to {final_model_path}")
```

### 6.3 Start Training
1. Run Cell 23 (or all cells)
2. Monitor the first few minutes to ensure:
   - GPU is being used (check `nvidia-smi` in terminal)
   - No errors occur
   - Checkpoints are being saved

---

## Step 7: Monitor Training (Optional but Recommended)

### 7.1 Check GPU Usage
Open a new terminal in PyCharm (`Alt+F12`) and run:
```powershell
# Monitor GPU usage every 5 seconds
nvidia-smi -l 5
```

You should see:
- GPU memory usage (~12-14GB per GPU)
- GPU utilization (should be 80-100%)
- Temperature (should be reasonable)

### 7.2 Monitor Training Progress
- Check PyCharm's notebook output for loss values
- Checkpoints are saved every 500 steps in `./output/checkpoints/`
- Training logs show progress every 50 steps (logging_steps)

### 7.3 Check Disk Space
```powershell
# Check available disk space
Get-PSDrive C | Select-Object Used,Free
```

Ensure you have at least 20GB free for:
- Model checkpoints (~2-3GB each)
- Final model (~5-10GB)
- Training logs

---

## Step 8: Leave It Running Safely

### 8.1 Before Leaving

**Checklist:**
- [ ] Training has started successfully (no errors in first 5-10 minutes)
- [ ] GPU is being utilized (check `nvidia-smi`)
- [ ] Checkpoints are being saved (verify `./output/checkpoints/` folder)
- [ ] Remote desktop won't disconnect (see Step 1.3)
- [ ] PyCharm is set to not sleep/hibernate
- [ ] Windows power settings allow long-running processes

### 8.2 Prevent Disconnection

**Windows Power Settings:**
1. **Control Panel → Power Options**
2. Set **Turn off display**: Never
3. Set **Put computer to sleep**: Never
4. Set **When I close the lid**: Do nothing (if laptop)

**Keep Session Alive:**
```powershell
# Run this in a separate PowerShell window to prevent lock
while ($true) {
    [System.Windows.Forms.SendKeys]::SendWait("{SCROLLLOCK}")
    Start-Sleep -Seconds 60
}
```

Or use a simple script (see `keep_session_alive.ps1`)

### 8.3 What Happens During Training

- **Training runs for 3-5 hours** (with 3 epochs)
- **Checkpoints saved every 500 steps** (in `./output/checkpoints/`)
- **Evaluation every 500 steps** (validation loss calculated)
- **Final model saved** at the end (in `./output/final_model/`)
- **Training logs** show progress every 50 steps

---

## Step 9: After Training Completes

### 9.1 Verify Training Completion
1. Check notebook output for "Training completed" message
2. Verify final model saved: `./output/final_model/` should exist
3. Check training logs for final metrics

### 9.2 Run Evaluation (Optional)
1. Uncomment Cell 28 (evaluation code)
2. Run Cell 28 to evaluate on test set
3. Results will show QWK, accuracy, and other metrics

### 9.3 Save Results
1. **File → Export → Export Notebook to HTML** (for documentation)
2. Copy checkpoints and final model to a safe location
3. Download results if needed

---

## Troubleshooting

### Issue: Remote Desktop Disconnects
**Solution**: 
- Use `tscon` command or keep-alive script
- Configure Windows to not lock on idle
- Consider using `screen` or `tmux` if using SSH instead

### Issue: GPU Not Detected
**Solution**:
```python
# In notebook, check:
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```
- Verify CUDA drivers: `nvidia-smi`
- Reinstall PyTorch with correct CUDA version

### Issue: Out of Memory (OOM)
**Solution**:
- Reduce `per_device_train_batch_size` to 2
- Increase `gradient_accumulation_steps` to 8
- Reduce `max_length` to 512
- Reduce `lora_r` to 16

### Issue: Training Stops Unexpectedly
**Solution**:
- Check PyCharm console for errors
- Verify disk space
- Check Windows Event Viewer for system errors
- Ensure power settings don't hibernate

### Issue: Checkpoints Not Saving
**Solution**:
- Verify `OUTPUT_DIR` has write permissions
- Check disk space
- Ensure `save_steps` is set correctly

### Issue: Slow Training
**Solution**:
- Verify GPU utilization (`nvidia-smi`)
- Check if both GPUs are being used
- Reduce `eval_steps` frequency
- Ensure `fp16=True` is enabled

---

## Quick Reference Commands

```powershell
# Check GPU status
nvidia-smi

# Monitor GPU continuously
nvidia-smi -l 5

# Check Python/CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check disk space
Get-PSDrive C

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install packages
pip install [package-name]
```

---

## Estimated Timeline

| Step | Time |
|------|------|
| Environment Setup | 15-30 min |
| PyCharm Configuration | 10-15 min |
| Dataset Verification | 5 min |
| First Run & Testing | 10-15 min |
| **Training** | **3-5 hours** |
| Evaluation (optional) | 10-30 min |
| **Total** | **4-6 hours** |

**You can safely leave it for 9 hours** - training will complete and save checkpoints automatically.

---

## Additional Tips

1. **First Run**: Monitor closely for the first 30 minutes to catch any issues
2. **Checkpoints**: The notebook saves checkpoints every 500 steps - you won't lose progress
3. **Multi-GPU**: The notebook should automatically use both GPUs with `device_map='auto'`
4. **Wandb**: Optional - you can skip wandb initialization if not needed
5. **Backup**: Copy important files before starting long training runs

---

## Support

If you encounter issues:
1. Check the notebook's troubleshooting section
2. Review PyCharm's console output
3. Check `nvidia-smi` for GPU status
4. Verify all dependencies are installed correctly

Good luck with your training! 🚀

