# Downloads the 24 Kodak test images (768x512 PNG) into test-images\kodak.
$ErrorActionPreference = "Stop"
$dir = Join-Path (Split-Path $PSScriptRoot -Parent) "test-images\kodak"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
1..24 | ForEach-Object {
    $n = "{0:D2}" -f $_
    $f = Join-Path $dir "kodim$n.png"
    if (-not (Test-Path $f)) {
        Write-Host "fetching kodim$n.png"
        Invoke-WebRequest -Uri "https://r0k.us/graphics/kodak/kodak/kodim$n.png" -OutFile $f
    }
}
Write-Host "done: $((Get-ChildItem $dir).Count) images in $dir"
