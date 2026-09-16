param([Parameter(Mandatory=$true)][string]$AuditPath)
$ErrorActionPreference = 'Stop'
$qwopRuntime = Get-Content -LiteralPath (Join-Path $PSScriptRoot '../.runtime/runtime.json') -Raw | ConvertFrom-Json
$qwopProcesses = Get-CimInstance Win32_Process
$qwopById = @{}
foreach ($qwopProcess in $qwopProcesses) { $qwopById[[int]$qwopProcess.ProcessId] = $qwopProcess }
$qwopOrphans = @($qwopProcesses | Where-Object {
    $_.Name -eq 'chromedriver.exe' -and
    $_.ExecutablePath -eq $qwopRuntime.files.driver.path -and
    -not $qwopById.ContainsKey([int]$_.ParentProcessId)
})
$qwopCleanup = [System.Collections.Generic.HashSet[int]]::new()
foreach ($qwopDriver in $qwopOrphans) {
    $qwopChildren = @($qwopProcesses | Where-Object {
        $_.ParentProcessId -eq $qwopDriver.ProcessId -and $_.Name -eq 'chrome.exe' -and
        $_.ExecutablePath -eq $qwopRuntime.files.browser.path -and
        $_.CommandLine -like '*--window-size=800,650*' -and
        $_.CommandLine -like '*--headless=new*' -and
        $_.CommandLine -like '*--enable-unsafe-swiftshader*' -and
        $_.CommandLine -like '*--incognito*'
    })
    if ($qwopChildren.Count -gt 0) {
        [void]$qwopCleanup.Add([int]$qwopDriver.ProcessId)
        foreach ($qwopChild in $qwopChildren) { [void]$qwopCleanup.Add([int]$qwopChild.ProcessId) }
    }
}
do {
    $qwopCountBefore = $qwopCleanup.Count
    foreach ($qwopProcess in $qwopProcesses) {
        if ($qwopCleanup.Contains([int]$qwopProcess.ParentProcessId) -and
            $qwopProcess.Name -in @('chrome.exe','chrome_crashpad_handler.exe')) {
            [void]$qwopCleanup.Add([int]$qwopProcess.ProcessId)
        }
    }
} while ($qwopCleanup.Count -gt $qwopCountBefore)
$qwopAudit = @($qwopProcesses | Where-Object { $qwopCleanup.Contains([int]$_.ProcessId) } |
    Select-Object ProcessId,ParentProcessId,Name,CreationDate)
if (Test-Path -LiteralPath $AuditPath) { throw "Audit already exists: $AuditPath" }
ConvertTo-Json -InputObject $qwopAudit | Set-Content -LiteralPath $AuditPath
$qwopStopped = 0
foreach ($qwopIdToStop in $qwopCleanup) {
    $qwopCurrent = Get-Process -Id $qwopIdToStop -ErrorAction SilentlyContinue
    $qwopOriginal = $qwopById[$qwopIdToStop]
    if ($qwopCurrent -and
        $qwopCurrent.ProcessName -eq [System.IO.Path]::GetFileNameWithoutExtension($qwopOriginal.Name) -and
        [math]::Abs(($qwopCurrent.StartTime - $qwopOriginal.CreationDate).TotalSeconds) -lt 1) {
        Stop-Process -Id $qwopIdToStop -Force -ErrorAction SilentlyContinue
        $qwopStopped++
    }
}
[pscustomobject]@{SelectedOrphanProcesses=$qwopCleanup.Count; StoppedProcesses=$qwopStopped}
