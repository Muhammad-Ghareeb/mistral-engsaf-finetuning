# GitHub Repository Setup Guide

This guide will help you create a GitHub repository and push your Mistral fine-tuning project to it.

## Prerequisites

- [ ] GitHub account created
- [ ] Git installed on your system
- [ ] GitHub CLI (`gh`) installed (optional, but recommended)

---

## Step 1: Create GitHub Repository

### Option A: Using GitHub Website (Recommended for beginners)

1. **Go to GitHub**
   - Visit https://github.com
   - Sign in to your account

2. **Create New Repository**
   - Click the **"+"** icon in the top right
   - Select **"New repository"**

3. **Repository Settings**
   - **Repository name**: `mistral-engsaf-finetuning` (or your preferred name)
   - **Description**: `Fine-tuning Mistral-7B for Automatic Short Answer Grading on EngSAF dataset`
   - **Visibility**: Choose **Public** or **Private**
   - **DO NOT** initialize with README, .gitignore, or license (we already have files)
   - Click **"Create repository"**

4. **Copy Repository URL**
   - After creation, GitHub will show you the repository URL
   - It will look like: `https://github.com/yourusername/mistral-engsaf-finetuning.git`
   - **Save this URL** - you'll need it in Step 3

### Option B: Using GitHub CLI (Faster)

```powershell
# Login to GitHub (first time only)
gh auth login

# Create repository
gh repo create mistral-engsaf-finetuning --public --description "Fine-tuning Mistral-7B for Automatic Short Answer Grading"
```

---

## Step 2: Initialize Git in Your Local Directory

Open PowerShell or Command Prompt in your project directory:

```powershell
# Navigate to your project directory
cd "C:\Studying\9th Semester\GP 2\QA\mistral - quadro"

# Initialize git repository
git init

# Configure git (if not already done)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## Step 3: Add Files and Make First Commit

```powershell
# Check what files will be added (files not in .gitignore)
git status

# Add all files (respects .gitignore)
git add .

# Check what will be committed
git status

# Make your first commit
git commit -m "Initial commit: Mistral-7B fine-tuning for EngSAF dataset

- Added fine-tuning notebook with LoRA/PEFT
- Added setup guides for PyCharm and remote execution
- Added verification scripts and utilities
- Configured for 2x Quadro RTX 5000 GPUs"
```

---

## Step 4: Connect to GitHub Repository

### If you used GitHub Website (Option A):

```powershell
# Add remote repository (replace with your actual URL)
git remote add origin https://github.com/yourusername/mistral-engsaf-finetuning.git

# Verify remote was added
git remote -v
```

### If you used GitHub CLI (Option B):

The remote is automatically added. Verify with:
```powershell
git remote -v
```

---

## Step 5: Push to GitHub

```powershell
# Push to GitHub (first time)
git push -u origin main

# If you get an error about branch name, try:
git branch -M main
git push -u origin main

# Or if your default branch is 'master':
git push -u origin master
```

**Note**: You may be prompted for GitHub credentials:
- **Username**: Your GitHub username
- **Password**: Use a **Personal Access Token** (not your GitHub password)
  - Create one at: https://github.com/settings/tokens
  - Select scopes: `repo` (full control of private repositories)

---

## Step 6: Verify Upload

1. **Check GitHub Website**
   - Go to your repository: `https://github.com/yourusername/mistral-engsaf-finetuning`
   - Verify all files are uploaded

2. **Verify via Command Line**
   ```powershell
   git log --oneline
   git remote show origin
   ```

---

## Future Updates

After making changes to your code:

```powershell
# Check what changed
git status

# Add changed files
git add .

# Commit changes
git commit -m "Description of your changes"

# Push to GitHub
git push
```

---

## Troubleshooting

### Issue: "Repository not found" or Authentication Failed

**Solution**:
1. Verify repository URL: `git remote -v`
2. Use Personal Access Token instead of password
3. Or use SSH instead of HTTPS:
   ```powershell
   # Change remote to SSH
   git remote set-url origin git@github.com:yourusername/mistral-engsaf-finetuning.git
   ```

### Issue: "Branch 'main' does not exist"

**Solution**:
```powershell
# Create and switch to main branch
git checkout -b main
git push -u origin main
```

### Issue: Large Files (Dataset Files)

**Solution**: 
- Dataset files are already in `.gitignore`
- If you accidentally added them:
  ```powershell
  git rm --cached "EngSAF dataset/*"
  git commit -m "Remove dataset files from tracking"
  ```

### Issue: Want to Exclude More Files

**Solution**: Edit `.gitignore` and add patterns:
```powershell
# Example: exclude all .zip files
echo "*.zip" >> .gitignore
git add .gitignore
git commit -m "Update .gitignore"
```

---

## Quick Reference Commands

```powershell
# Initialize repository
git init

# Check status
git status

# Add files
git add .

# Commit changes
git commit -m "Your commit message"

# Add remote
git remote add origin https://github.com/yourusername/repo-name.git

# Push to GitHub
git push -u origin main

# Pull latest changes
git pull

# View commit history
git log --oneline

# Check remote
git remote -v
```

---

## Recommended: Add a LICENSE File

Consider adding a license to your repository:

1. Go to your GitHub repository
2. Click **"Add file"** → **"Create new file"**
3. Name it `LICENSE`
4. GitHub will suggest templates (MIT, Apache 2.0, etc.)
5. Choose one and commit

---

## Recommended: Update README.md

Your `README.md` already exists, but you might want to add:
- Installation instructions
- Usage examples
- Citation information
- License information

---

## Security Note

**Important**: The `.gitignore` file excludes:
- Dataset files (large CSV files)
- Model checkpoints (large binary files)
- Virtual environment files
- API keys and secrets

**Never commit**:
- API keys
- Passwords
- Personal access tokens
- Large dataset files
- Model weights (unless using Git LFS)

---

## Next Steps

1. ✅ Repository created
2. ✅ Code pushed to GitHub
3. 📝 Consider adding:
   - License file
   - GitHub Actions for CI/CD (optional)
   - Issues template
   - Pull request template
   - Contributing guidelines

---

**Your repository is now live on GitHub! 🎉**

Share it with: `https://github.com/yourusername/mistral-engsaf-finetuning`

