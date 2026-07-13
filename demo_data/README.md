# Demo Data

Large TIFF demo movies are hosted as GitHub Release assets instead of normal
git files because they exceed GitHub's regular file-size limit.

Release page:

```text
https://github.com/LuoYuhong123/NeuroPilot/releases/tag/v0.1.0
```

## Direct Downloads

| Dataset | Save as | Download asset |
| --- | --- | --- |
| 24h | `demo_data/24h/demo_data_24h.tiff` | `https://github.com/LuoYuhong123/NeuroPilot/releases/download/v0.1.0/demo_data_24h.tiff` |
| spine | `demo_data/spine/demo_data_spine.tiff` | `https://github.com/LuoYuhong123/NeuroPilot/releases/download/v0.1.0/demo_data_spine.tiff` |

## PowerShell

```powershell
New-Item -ItemType Directory -Force demo_data\24h, demo_data\spine | Out-Null

Invoke-WebRequest `
  -Uri "https://github.com/LuoYuhong123/NeuroPilot/releases/download/v0.1.0/demo_data_24h.tiff" `
  -OutFile "demo_data\24h\demo_data_24h.tiff"

Invoke-WebRequest `
  -Uri "https://github.com/LuoYuhong123/NeuroPilot/releases/download/v0.1.0/demo_data_spine.tiff" `
  -OutFile "demo_data\spine\demo_data_spine.tiff"
```

## Bash

```bash
mkdir -p demo_data/24h demo_data/spine

curl -L \
  -o "demo_data/24h/demo_data_24h.tiff" \
  "https://github.com/LuoYuhong123/NeuroPilot/releases/download/v0.1.0/demo_data_24h.tiff"

curl -L \
  -o "demo_data/spine/demo_data_spine.tiff" \
  "https://github.com/LuoYuhong123/NeuroPilot/releases/download/v0.1.0/demo_data_spine.tiff"
```

## Checksums

```text
demo_data_24h.tiff
SHA256 430EADA89B33B68141184A2EEB12AFD0E609FD8D40168248986846CD540D1365

demo_data_spine.tiff
SHA256 D24E972435413243693549478A775617EC993E88E10D01293BCEF39D5A7D4714
```
