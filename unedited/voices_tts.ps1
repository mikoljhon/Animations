# Placeholder voices with the free Windows speech voices. Raw lines go to audio\voices\raw\<id>.wav (22 kHz mono).
# Mender and Relic use David (processed later in different ways), Tally uses Zira.
Add-Type -AssemblyName System.Speech
$out = "C:\ai\unedited\audio\voices\raw"
New-Item -ItemType Directory -Force $out | Out-Null
$lines = @(
    @{ id = "M1"; voice = "Microsoft David Desktop"; rate = 1;  text = "Come on. What have you got for me?" },
    @{ id = "M3"; voice = "Microsoft David Desktop"; rate = 2;  text = "Dead as the rock. You beautiful thing. You are the jackpot." },
    @{ id = "M5"; voice = "Microsoft David Desktop"; rate = 1;  text = "No. No, no, no. The scanner said..." },
    @{ id = "R1"; voice = "Microsoft David Desktop"; rate = 1;  text = "Tally? Tally! What happened?" },
    @{ id = "T1"; voice = "Microsoft Zira Desktop";  rate = 0;  text = "I'm here. I'm here. Systems online. Vitals stable. Log... empty." }
)
foreach ($l in $lines) {
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $s.SelectVoice($l.voice)
    $s.Rate = $l.rate
    $s.Volume = 100
    $p = Join-Path $out ($l.id + ".wav")
    $s.SetOutputToWaveFile($p)
    $s.Speak($l.text)
    $s.Dispose()
    "{0}: {1}" -f $l.id, (Get-Item $p).Length
}
