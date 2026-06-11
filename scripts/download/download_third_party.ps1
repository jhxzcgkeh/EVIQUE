$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force third_party/external | Out-Null
Get-Content third_party/versions.lock
Write-Host "Review licenses and pin commits before cloning."
