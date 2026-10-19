param([string]$blend="mc_test_relic.blend",[string]$tag="mc_relic",[string]$kind="relic")
Set-Location C:\ai\unedited
$p = Start-Process -FilePath "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -ArgumentList @("-b",$blend,"--python","fm_vid.py","--",$tag,$kind) -RedirectStandardOutput anim_cache5\vid_stdout.txt -PassThru -WindowStyle Hidden
$p.PriorityClass = 'BelowNormal'
$p.WaitForExit()
