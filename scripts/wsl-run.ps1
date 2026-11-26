<#
.SYNOPSIS
Starts a long job inside the clash-ai WSL instance from Windows, and keeps the PC awake until it ends.

.DESCRIPTION
A process inside WSL cannot stop Windows from sleeping, so the wake lock is held on the Windows
side: this script starts a hidden copy of itself that asks Windows to stay awake, runs the command
in the instance, and releases the lock when the command ends. It returns at once.

The command runs in a login shell in the repository (~/clash-ai), so the project's environment
(CLASHAI_DATA, the Python environment) is set. Its output goes to
/data/clash-ai/runs/wsl-<Name>.log inside the instance.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\wsl-run.ps1 -Name generalist-v4 -Command "python -m clashai.train.seq --name generalist-v4 --resume"

.EXAMPLE
wsl -d clash-ai -- tail -f /data/clash-ai/runs/wsl-generalist-v4.log

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\wsl-run.ps1 -Name generalist-v4 -Stop
#>
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$Command,
    [string]$Distro = "clash-ai",
    [string]$Repo = "~/clash-ai",
    [string]$Encoded,
    [switch]$Inner,
    [switch]$Stop
)

function ConvertTo-Base64([string]$text) {
    # Base64, so no quoting survives or fails on the way through PowerShell, wsl.exe and bash.
    [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($text))
}

if ($Name -notmatch '^[A-Za-z0-9._-]+$') { throw "Name may hold only letters, digits, '.', '_' and '-'" }
$log = "/data/clash-ai/runs/wsl-$Name.log"

if ($Stop) {
    # The job runs in its own session, so ending that session's process group ends the command
    # and everything it started. Both the job's shell and the setsid waiting on it carry the
    # label; each one's group is ended.
    $kill = "n=0; for p in `$(pgrep -f 'wsl-job $Name`$'); do kill -TERM -- -`$p 2>/dev/null; n=1; done; " +
            "if [ `$n = 1 ]; then echo stopped; else echo 'not running'; fi"
    & wsl.exe -d $Distro -- bash -c "echo $(ConvertTo-Base64 $kill) | base64 -d | bash"
    return
}

if (-not $Inner) {
    if (-not $Command) { throw "give the command to run with -Command" }
    $Encoded = ConvertTo-Base64 $Command
    $arguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"",
                   "-Name", $Name, "-Distro", $Distro, "-Repo", $Repo, "-Encoded", $Encoded, "-Inner")
    $process = Start-Process -WindowStyle Hidden -PassThru -FilePath "powershell.exe" -ArgumentList $arguments
    Write-Output "started (Windows process $($process.Id)); follow it with:"
    Write-Output "  wsl -d $Distro -- tail -f $log"
    Write-Output "stop it with:"
    Write-Output "  powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Name $Name -Stop"
    return
}

Add-Type -Namespace ClashAi -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
'@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
[ClashAi.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
try {
    # "wsl-job <Name>" is only a label in the process's command line, for -Stop to find. setsid
    # gives the job its own session and process group.
    $script = "cd $Repo && mkdir -p /data/clash-ai/runs && echo $Encoded | base64 -d > /tmp/wsl-job-$Name.sh && " +
              "exec setsid -w bash -l /tmp/wsl-job-$Name.sh 'wsl-job $Name' >> $log 2>&1"
    & wsl.exe -d $Distro -- bash -lc $script
}
finally {
    [ClashAi.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
}
