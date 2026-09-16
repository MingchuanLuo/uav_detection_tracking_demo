# Source video

The demo expects the original source file at:

```text
video/Quadcopter_(drone).webm
```

Download it from the Wikimedia Commons page for
[Quadcopter (drone)](https://commons.wikimedia.org/wiki/File%3AQuadcopter_%28drone%29.webm),
or run this command from the repository root in PowerShell:

```powershell
Invoke-WebRequest `
  -Uri "https://commons.wikimedia.org/wiki/Special:Redirect/file/Quadcopter_(drone).webm" `
  -OutFile "video\Quadcopter_(drone).webm"
```

Verify the downloaded original:

```powershell
Get-FileHash -Algorithm SHA1 "video\Quadcopter_(drone).webm"
```

Expected SHA-1:

```text
AB8E308532ED3483AB8D120AE23282EE5715E9EC
```

The file is credited to Sounds of Changes / Työväenmuseo Werstas, recorded by
Mikael Maffei, and licensed under
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). See the main README
for the complete attribution and modification notice.
