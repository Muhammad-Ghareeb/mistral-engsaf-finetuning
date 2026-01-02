# PowerShell script to keep remote desktop session alive
# Run this in a separate PowerShell window while training

Write-Host "Starting session keep-alive script..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow

# Function to simulate activity
function Keep-Alive {
    # Move mouse cursor slightly (doesn't interfere with other work)
    Add-Type -AssemblyName System.Windows.Forms
    $pos = [System.Windows.Forms.Cursor]::Position
    [System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point(($pos.X + 1), $pos.Y)
    Start-Sleep -Milliseconds 100
    [System.Windows.Forms.Cursor]::Position = $pos
}

# Main loop - runs every 60 seconds
$counter = 0
while ($true) {
    try {
        Keep-Alive
        $counter++
        if ($counter % 10 -eq 0) {
            Write-Host "Session kept alive - $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
        }
        Start-Sleep -Seconds 60
    }
    catch {
        Write-Host "Error: $_" -ForegroundColor Red
        Start-Sleep -Seconds 60
    }
}

