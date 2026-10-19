param([string]$who="relic")
Set-Location C:\ai\unedited
$p = Start-Process -FilePath "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -ArgumentList @("-b","fm_test_$who.blend","--python","fm_vid.py","--",$who) -RedirectStandardOutput anim_cache5\fm_vid_stdout.txt -PassThru -WindowStyle Hidden
$p.PriorityClass = 'BelowNormal'
$p.WaitForExit()
