param([string]$blend="fight_v03.blend",[string]$script="pipe_test.py",[string]$log="anim_cache5\run_py.txt",[string]$a="")
# run any Blender script headless at BelowNormal priority (one at a time); extra args go after --, space separated in -a
Set-Location C:\ai\unedited
$args2 = @("-b",$blend,"--python",$script,"--") + ($a -split ' ' | Where-Object { $_ -ne '' })
$p = Start-Process -FilePath "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -ArgumentList $args2 -RedirectStandardOutput $log -PassThru -WindowStyle Hidden
$p.PriorityClass = 'BelowNormal'
$p.WaitForExit()
