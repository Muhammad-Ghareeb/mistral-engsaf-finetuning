# Quick Start: Running Training for 9 Hours

## Direct Answers to Your Questions

### Q: Can I leave it running for 9 hours?
**A: YES!** Training takes 3-5 hours, so 9 hours is plenty. The notebook:
- ✅ Saves checkpoints every 500 steps automatically
- ✅ Saves final model when training completes
- ✅ Continues running even if you disconnect (as long as PyCharm stays open)
- ✅ Has early stopping to prevent overfitting

### Q: What should I do step-by-step?
**See detailed guide in `PYCHARM_SETUP_GUIDE.md`**

**Quick version:**
1. Connect via Remote Desktop
2. Open PyCharm → Open your project
3. Set Python interpreter to your venv
4. Install packages (see guide)
5. Run notebook cells 0-22
6. **Uncomment Cell 23** (training code)
7. Run Cell 23 → Training starts
8. Monitor for 10-15 minutes
9. **Leave it running** - it will complete automatically

### Q: Will it use both GPUs?
**A: YES!** The notebook uses `device_map='auto'` which automatically distributes the model across both Quadro RTX 5000 GPUs.

---

## 5-Minute Setup Checklist

```
[ ] 1. Remote Desktop connected
[ ] 2. Run: nvidia-smi (verify 2 GPUs visible)
[ ] 3. Create venv: python -m venv venv
[ ] 4. Activate: .\venv\Scripts\Activate.ps1
[ ] 5. Install PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
[ ] 6. Install other packages: pip install transformers peft bitsandbytes datasets accelerate wandb scikit-learn nltk rouge-score bert-score jupyter ipykernel
[ ] 7. Verify: python verify_setup.py
[ ] 8. Open notebook in PyCharm
[ ] 9. Uncomment training code (Cell 23)
[ ] 10. Run all cells
[ ] 11. Monitor for 15 minutes
[ ] 12. Leave it running!
```

---

## Critical Steps Before Leaving

### 1. Prevent Remote Desktop Disconnection
```powershell
# Run this in a separate PowerShell window:
.\keep_session_alive.ps1
```

Or configure Windows:
- Control Panel → Power Options → Never sleep/never turn off display

### 2. Verify Training Started Successfully
Check these in the first 10 minutes:
- [ ] GPU memory usage: `nvidia-smi` shows ~12-14GB per GPU
- [ ] Training loss appears in notebook output
- [ ] Checkpoints folder created: `./output/checkpoints/`
- [ ] No errors in PyCharm console

### 3. What to Expect

**Timeline:**
- **0-5 min**: Model loading, dataset preparation
- **5-10 min**: First training steps, verify GPU usage
- **10 min - 3 hours**: Training runs (checkpoints every 500 steps)
- **3-5 hours**: Training completes, model saved
- **After**: You can run evaluation cells

**Output Files:**
- `./output/checkpoints/checkpoint-500/` (every 500 steps)
- `./output/checkpoints/checkpoint-1000/`
- `./output/final_model/` (when complete)

---

## If Something Goes Wrong

### Training Stops Early
- Check PyCharm console for errors
- Check `nvidia-smi` - GPU might have crashed
- Check disk space: `Get-PSDrive C`
- Resume from checkpoint (modify notebook to load checkpoint)

### Remote Desktop Disconnects
- Training continues if PyCharm stays open
- Reconnect and check notebook output
- Verify checkpoints are still being saved

### Out of Memory
- Reduce batch size in CONFIG: `per_device_train_batch_size: 2`
- Increase gradient accumulation: `gradient_accumulation_steps: 8`

---

## Monitoring Commands

```powershell
# Check GPU usage (run in separate terminal)
nvidia-smi -l 5

# Check if training process is running
Get-Process python | Where-Object {$_.CPU -gt 0}

# Check disk space
Get-PSDrive C | Select-Object Used,Free
```

---

## After Training Completes

1. **Check final model**: `./output/final_model/` should exist
2. **Review training logs**: Check loss values in notebook
3. **Run evaluation**: Uncomment Cell 28 and run
4. **Save results**: Export notebook to HTML/PDF

---

## Your Hardware Specs

- **GPUs**: 2x Quadro RTX 5000 (16GB each) ✅ Perfect for this task
- **CPU**: Intel Xeon Gold 6230R ✅ Excellent
- **Expected Training Time**: 3-5 hours
- **Memory Usage**: ~12-14GB per GPU (with 4-bit quantization)

**You have excellent hardware - training should run smoothly!**

---

## Need Help?

1. Run `python verify_setup.py` to check everything
2. See `PYCHARM_SETUP_GUIDE.md` for detailed steps
3. Check notebook's troubleshooting section
4. Verify GPU: `nvidia-smi`

**Good luck! 🚀**

