# Quick Instructions: Create GitHub Repo and Push

## ✅ Step 1: Create GitHub Repository

1. **Go to**: https://github.com/new
2. **Repository name**: `mistral-engsaf-finetuning` (or your preferred name)
3. **Description**: `Fine-tuning Mistral-7B for Automatic Short Answer Grading`
4. **Visibility**: Choose Public or Private
5. **IMPORTANT**: Do NOT check "Initialize with README" (we already have files)
6. Click **"Create repository"**

## ✅ Step 2: Copy Repository URL

After creating, GitHub will show you a page with setup instructions. **Copy the repository URL**:
- It looks like: `https://github.com/YOUR_USERNAME/mistral-engsaf-finetuning.git`
- Or: `git@github.com:YOUR_USERNAME/mistral-engsaf-finetuning.git`

## ✅ Step 3: Run Push Script

I've already prepared your local repository with:
- ✅ Git initialized
- ✅ All files added
- ✅ Initial commit created

Now run this PowerShell script:

```powershell
.\push_to_github.ps1
```

When prompted, paste your repository URL.

## Alternative: Manual Push

If you prefer to do it manually:

```powershell
# Add remote (replace with YOUR repository URL)
git remote add origin https://github.com/YOUR_USERNAME/mistral-engsaf-finetuning.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Authentication

When prompted for credentials:
- **Username**: Your GitHub username
- **Password**: Use a **Personal Access Token** (NOT your GitHub password)
  - Create token: https://github.com/settings/tokens
  - Click "Generate new token" → "Generate new token (classic)"
  - Name: `mistral-repo-push`
  - Expiration: Choose your preference
  - Scopes: Check `repo` (full control of private repositories)
  - Click "Generate token"
  - **Copy the token immediately** (you won't see it again!)
  - Use this token as your password when pushing

## Done! 🎉

Your repository will be available at:
`https://github.com/YOUR_USERNAME/mistral-engsaf-finetuning`

