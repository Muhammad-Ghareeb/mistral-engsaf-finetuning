# Setup Guide Summary

## 📋 Quick Answer to Your Questions

### ✅ Can you leave it running for 9 hours?
**YES!** Training takes 3-5 hours, so 9 hours is more than enough. The notebook:
- Automatically saves checkpoints every 500 steps
- Saves the final model when training completes
- Will continue running even if you disconnect (as long as PyCharm stays open)

### ✅ Will it use both Quadro RTX 5000 GPUs?
**YES!** The notebook uses `device_map='auto'` which automatically distributes the model across both GPUs.

### ✅ What to do step-by-step?
**See the detailed guide below or `PYCHARM_SETUP_GUIDE.md`**

---

## 🚀 Step-by-Step Process

### Phase 1: Initial Setup (30-45 minutes)

1. **Connect via Remote Desktop**
   - Connect to your university station
   - Verify GPUs: Open PowerShell → Run `nvidia-smi` → Should see 2 GPUs

2. **Set Up Python Environment**
   ```powershell
   # Navigate to project
   cd "C:\Studying\9th Semester\GP 2\QA\mistral - quadro"
   
   # Create virtual environment
   python -m venv venv
   
   # Activate it
   .\venv\Scripts\Activate.ps1
   
   # Install PyTorch with CUDA
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Install other packages
   pip install transformers peft bitsandbytes datasets accelerate wandb scikit-learn nltk rouge-score bert-score jupyter ipykernel
   ```

3. **Verify Setup**
   ```powershell
   python verify_setup.py
   ```
   Should show all checks passing ✓

### Phase 2: PyCharm Configuration (15 minutes)

1. **Open Project in PyCharm**
   - File → Open → Select your project folder
   - Wait for indexing

2. **Configure Python Interpreter**
   - File → Settings → Project → Python Interpreter
   - Add → Existing Environment → Select `venv\Scripts\python.exe`

3. **Install Jupyter Support**
   - In interpreter settings, install: `jupyter`, `ipykernel`, `ipywidgets`

4. **Open Notebook**
   - Double-click `mistral_engsaf_finetuning.ipynb`

### Phase 3: Run Training (5 minutes setup + 3-5 hours training)

1. **Run Setup Cells (0-22)**
   - Click "Run All" or run cells sequentially
   - Verify GPU detection shows both GPUs
   - Verify dataset loads correctly

2. **Uncomment Training Code (Cell 23)**
   - Find Cell 23 (training cell)
   - Uncomment the training code:
     ```python
     print("Starting training...")
     trainer.train()
     
     # Save final model
     final_model_path = os.path.join(OUTPUT_DIR, 'final_model')
     trainer.save_model(final_model_path)
     tokenizer.save_pretrained(final_model_path)
     ```

3. **Start Training**
   - Run Cell 23
   - Monitor for first 10-15 minutes:
     - Check `nvidia-smi` shows GPU usage
     - Verify checkpoints folder created: `./output/checkpoints/`
     - Check training loss appears in output

4. **Leave It Running**
   - Training will complete automatically in 3-5 hours
   - Checkpoints saved every 500 steps
   - Final model saved when complete

### Phase 4: Prevent Disconnection (Important!)

**Before leaving:**

1. **Run Keep-Alive Script** (in separate PowerShell):
   ```powershell
   .\keep_session_alive.ps1
   ```

2. **Configure Windows Power Settings**:
   - Control Panel → Power Options
   - Set "Turn off display": Never
   - Set "Put computer to sleep": Never

3. **Verify PyCharm Settings**:
   - Ensure PyCharm won't close on disconnect
   - Check "Keep session alive" if available

---

## 📊 What to Expect

### Timeline
- **0-5 min**: Model loading, dataset preparation
- **5-10 min**: First training steps (verify everything works)
- **10 min - 3 hours**: Training runs (checkpoints every 500 steps)
- **3-5 hours**: Training completes, final model saved
- **After**: You can run evaluation cells

### Output Files
```
./output/
├── checkpoints/
│   ├── checkpoint-500/    (saved every 500 steps)
│   ├── checkpoint-1000/
│   └── ...
└── final_model/            (saved when training completes)
```

### GPU Usage
- Each GPU: ~12-14GB memory usage
- GPU utilization: 80-100%
- Both GPUs will be used automatically

---

## 🔍 Monitoring Commands

While training is running, you can check status:

```powershell
# Check GPU usage (run in separate terminal)
nvidia-smi -l 5

# Check if Python process is running
Get-Process python | Where-Object {$_.CPU -gt 0}

# Check disk space
Get-PSDrive C | Select-Object Used,Free

# Check checkpoint files
Get-ChildItem ".\output\checkpoints\" -Recurse
```

---

## ⚠️ Troubleshooting

### Remote Desktop Disconnects
- **Solution**: Training continues if PyCharm stays open. Reconnect and check notebook output.

### GPU Not Detected
- **Solution**: Run `nvidia-smi` to verify drivers. Reinstall PyTorch with correct CUDA version.

### Out of Memory
- **Solution**: Reduce `per_device_train_batch_size` to 2 in CONFIG (Cell 7)

### Training Stops Unexpectedly
- **Solution**: Check PyCharm console for errors. Verify disk space and power settings.

---

## 📁 Files Created

This setup creates these helpful files:

1. **PYCHARM_SETUP_GUIDE.md** - Detailed step-by-step guide
2. **QUICK_START.md** - Quick reference
3. **verify_setup.py** - Setup verification script
4. **keep_session_alive.ps1** - Prevents remote desktop disconnection

---

## ✅ Final Checklist Before Leaving

- [ ] Training started successfully (no errors in first 10 minutes)
- [ ] GPU usage visible (`nvidia-smi` shows activity)
- [ ] Checkpoints folder created (`./output/checkpoints/`)
- [ ] Keep-alive script running (or Windows configured to not sleep)
- [ ] PyCharm is open and notebook is running
- [ ] At least 20GB free disk space

---

## 🎯 Your Hardware

- **GPUs**: 2x Quadro RTX 5000 (16GB each) ✅ Perfect!
- **CPU**: Intel Xeon Gold 6230R ✅ Excellent!
- **Expected Time**: 3-5 hours
- **You can safely leave for 9 hours** ✅

---

## 📚 Additional Resources

- **Detailed Guide**: See `PYCHARM_SETUP_GUIDE.md`
- **Quick Reference**: See `QUICK_START.md`
- **Notebook Documentation**: See `README.md`

---

**Good luck with your training! 🚀**

If you encounter any issues, check the troubleshooting sections in the guides above.

