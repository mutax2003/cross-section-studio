# Run N sequential E2E verify cycles with orchestration reports.
param(
    [int]$Count = 10,
    [int]$Start = 1,
    [string]$ReportDir = "orchestration_reports"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null

$summary = @()
for ($i = $Start; $i -lt ($Start + $Count); $i++) {
    $pad = "{0:D2}" -f $i
    $report = Join-Path $ReportDir "batch_$pad.md"
    $task = "Batch verify run $i of 10 — critical QA gate"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Host "`n========== BATCH RUN $i / $($Start + $Count - 1) ==========" -ForegroundColor Cyan
    python scripts/agent_supervisor.py verify --task $task --report $report
    $code = $LASTEXITCODE
    $sw.Stop()
    $entry = [ordered]@{
        run     = $i
        exit    = $code
        seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        report  = $report
    }
    $summary += $entry
    Write-Host "RUN $i exit=$code elapsed=$($entry.seconds)s report=$report"
    if ($code -ne 0) {
        Write-Host "BATCH STOPPED on failure at run $i" -ForegroundColor Red
        break
    }
}

$summaryPath = Join-Path $ReportDir "batch_summary.json"
$summary | ConvertTo-Json | Set-Content -Encoding utf8 $summaryPath
Write-Host "`nWrote $summaryPath"
if (($summary | Where-Object { $_.exit -ne 0 }).Count -gt 0) { exit 1 }
exit 0
