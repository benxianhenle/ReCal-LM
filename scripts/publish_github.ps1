$ErrorActionPreference = "Stop"

function Read-Defaulted {
    param(
        [Parameter(Mandatory = $true)][string]$Prompt,
        [Parameter(Mandatory = $true)][string]$Default
    )
    $value = Read-Host "$Prompt [$Default]"
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $Default
    }
    return $value.Trim()
}

function Convert-SecureStringToPlainText {
    param([Parameter(Mandatory = $true)][securestring]$SecureString)
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureString)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$status = git status --short
if ($LASTEXITCODE -ne 0) {
    throw "git status failed"
}
if (-not [string]::IsNullOrWhiteSpace($status)) {
    throw "Working tree is not clean. Commit or discard changes before publishing."
}

$repoName = Read-Defaulted -Prompt "GitHub repository name" -Default "ReCal-LM"
$visibility = Read-Defaulted -Prompt "Visibility: public or private" -Default "public"
if ($visibility -notin @("public", "private")) {
    throw "Visibility must be public or private."
}
$description = Read-Defaulted -Prompt "Repository description" -Default "ReCal-LM demo with Router/Executor and DriftEstimator"

Write-Host ""
Write-Host "Paste a GitHub token with permission to create repositories and push contents."
Write-Host "Classic token: public_repo for public repos, repo for private repos."
Write-Host "Fine-grained token: repository creation plus Contents read/write."
$secureToken = Read-Host "GitHub token" -AsSecureString
$token = Convert-SecureStringToPlainText $secureToken
if ([string]::IsNullOrWhiteSpace($token)) {
    throw "GitHub token is empty."
}

$headers = @{
    Authorization = "Bearer $token"
    Accept = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
}

try {
    $user = Invoke-RestMethod -Method Get -Uri "https://api.github.com/user" -Headers $headers
    $private = $visibility -eq "private"
    $body = @{
        name = $repoName
        description = $description
        private = $private
        auto_init = $false
    } | ConvertTo-Json

    Write-Host ""
    Write-Host "Creating https://github.com/$($user.login)/$repoName ..."
    try {
        $repo = Invoke-RestMethod -Method Post -Uri "https://api.github.com/user/repos" -Headers $headers -Body $body -ContentType "application/json"
    }
    catch {
        if ($_.Exception.Response -and [int]$_.Exception.Response.StatusCode -eq 422) {
            Write-Host "Repository already exists; reusing it."
            $repo = Invoke-RestMethod -Method Get -Uri "https://api.github.com/repos/$($user.login)/$repoName" -Headers $headers
        }
        else {
            throw
        }
    }

    $remoteUrl = $repo.clone_url
    $remoteNames = @(git remote)
    if ($remoteNames -contains "origin") {
        git remote set-url origin $remoteUrl
    }
    else {
        git remote add origin $remoteUrl
    }

    git branch -M main
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to rename current branch to main."
    }

    Write-Host "Pushing local main branch ..."
    $basicToken = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("x-access-token:$token"))
    git -c "http.extraheader=AUTHORIZATION: basic $basicToken" push -u origin main
    if ($LASTEXITCODE -ne 0) {
        throw "git push failed."
    }

    Write-Host ""
    Write-Host "Published: $($repo.html_url)"
}
finally {
    Remove-Variable basicToken -ErrorAction SilentlyContinue
    Remove-Variable token -ErrorAction SilentlyContinue
}
