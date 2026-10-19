# run_gated.ps1 : run ONE headless Blender job only when the PC is free, and pause it while the user is using the PC.
#   - waits until no other background Blender (-b) is running (never two renders at once)
#   - waits until the user has not touched mouse or keyboard for $idleStart seconds
#   - runs at BelowNormal priority; while it runs, the job is suspended as soon as the user is active
#     and resumed once they have been idle for $idleResume seconds
# Example:
#   powershell -File run_gated.ps1 -script killdroid_v02_build.py -a "concept A" -log anim_cache5\kd2_A.txt
param([string]$script, [string]$blend = "", [string]$a = "", [string]$log = "anim_cache5\gated.txt",
      [int]$idleStart = 60, [int]$idleResume = 45, [int]$maxWaitMin = 240, [switch]$factory)
Set-Location C:\ai\unedited
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class GateNative {
  [StructLayout(LayoutKind.Sequential)] public struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }
  [DllImport("user32.dll")] static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);
  [DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h);
  [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);
  public static double IdleSeconds() { var l = new LASTINPUTINFO(); l.cbSize = (uint)Marshal.SizeOf(l); GetLastInputInfo(ref l); return ((uint)Environment.TickCount - l.dwTime) / 1000.0; }
}
'@
function Other-Render {
  $procs = Get-CimInstance Win32_Process -Filter "Name='blender.exe'" -ErrorAction SilentlyContinue
  foreach ($p in $procs) { if ($p.CommandLine -match '\s-b(\s|$)') { return $true } }
  return $false
}
$t0 = Get-Date
while ($true) {
  $busy = Other-Render
  $idle = [GateNative]::IdleSeconds()
  if (-not $busy -and $idle -ge $idleStart) { break }
  if (((Get-Date) - $t0).TotalMinutes -gt $maxWaitMin) { "GATE TIMEOUT (other render: $busy, idle $([int]$idle) s)"; exit 3 }
  Start-Sleep -Seconds 10
}
"GATE OPEN after $([int]((Get-Date) - $t0).TotalSeconds) s"
$args2 = @("-b")
if ($blend -ne "") { $args2 += $blend }
if ($factory) { $args2 += "--factory-startup" }
$args2 += @("--python", $script, "--") + ($a -split ' ' | Where-Object { $_ -ne '' })
$p = Start-Process -FilePath "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -ArgumentList $args2 -RedirectStandardOutput $log -RedirectStandardError ($log + ".err") -PassThru -WindowStyle Hidden
try { $p.PriorityClass = 'BelowNormal' } catch {}
$h = $p.Handle
$paused = $false; $pauses = 0; $pausedSec = 0.0
while (-not $p.HasExited) {
  $idle = [GateNative]::IdleSeconds()
  if (-not $paused -and $idle -lt 3) { [void][GateNative]::NtSuspendProcess($h); $paused = $true; $pauses++; $ps = Get-Date }
  elseif ($paused -and $idle -ge $idleResume) { [void][GateNative]::NtResumeProcess($h); $paused = $false; $pausedSec += ((Get-Date) - $ps).TotalSeconds }
  Start-Sleep -Milliseconds 1500
}
"JOB EXIT $($p.ExitCode) after $([int]((Get-Date) - $t0).TotalSeconds) s, paused $pauses times ($([int]$pausedSec) s)"
