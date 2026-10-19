param([int]$s=1,[int]$e=430)
Set-Location C:\ai\unedited
$p = Start-Process -FilePath "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -ArgumentList @("-b","fight_v02.blend","--python","fa_rset.py","-S","Street","-s","$s","-e","$e","-a") -RedirectStandardOutput anim_cache5\fa_render_stdout.txt -PassThru -WindowStyle Hidden
$p.PriorityClass = 'BelowNormal'
$p.WaitForExit()
