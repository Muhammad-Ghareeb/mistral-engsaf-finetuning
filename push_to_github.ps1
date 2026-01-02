# Script to push code to GitHub repository
# Run this AFTER creating the repository on GitHub

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GitHub Push Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get repository URL from user
$repoUrl = Read-Host "Enter your GitHub repository URL (e.g., https://github.com/username/repo-name.git)"

if ([string]::IsNullOrWhiteSpace($repoUrl)) {
    Write-Host "Error: Repository URL is required!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Adding remote repository..." -ForegroundColor Yellow
git remote add origin $repoUrl

if ($LASTEXITCODE -ne 0) {
    Write-Host "Remote might already exist. Updating..." -ForegroundColor Yellow
    git remote set-url origin $repoUrl
}

Write-Host "Checking current branch..." -ForegroundColor Yellow
$currentBranch = git branch --show-current

if ($currentBranch -eq "master") {
    Write-Host "Renaming branch to 'main'..." -ForegroundColor Yellow
    git branch -M main
    $branchName = "main"
} else {
    $branchName = $currentBranch
}

Write-Host ""
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
Write-Host "You may be prompted for GitHub credentials:" -ForegroundColor Cyan
Write-Host "  - Username: Your GitHub username" -ForegroundColor Cyan
Write-Host "  - Password: Use a Personal Access Token (not your password)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Create token at: https://github.com/settings/tokens" -ForegroundColor Cyan
Write-Host ""

git push -u origin $branchName

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Success! Code pushed to GitHub!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Repository URL: $repoUrl" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Push failed. Common issues:" -ForegroundColor Red
    Write-Host "1. Repository doesn't exist yet - create it at https://github.com/new" -ForegroundColor Yellow
    Write-Host "2. Authentication failed - use Personal Access Token instead of password" -ForegroundColor Yellow
    Write-Host "3. Wrong repository URL - check the URL format" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Red
}

