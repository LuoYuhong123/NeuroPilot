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
| 24h | `demo_data/24h/CellVideo 01-1.tif` | `https://github.com/LuoYuhong123/NeuroPilot/releases/download/v0.1.0/CellVideo.01-1.tif` |
| spine | `demo_data/spine/ju2df_5day_freemoving-male1-5day-image-pain 0.tif` | `https://github.com/LuoYuhong123/NeuroPilot/releases/download/v0.1.0/ju2df_5day_freemoving-male1-5day-image-pain.0.tif` |

GitHub normalizes spaces in Release asset names to dots, so save each download
with the local filename shown in the `Save as` column.

## PowerShell

```powershell
New-Item -ItemType Directory -Force demo_data\24h, demo_data\spine | Out-Null

Invoke-WebRequest `
  -Uri "https://github.com/LuoYuhong123/NeuroPilot/releases/download/v0.1.0/CellVideo.01-1.tif" `
  -OutFile "demo_data\24h\CellVideo 01-1.tif"

Invoke-WebRequest `
  -Uri "https://github.com/LuoYuhong123/NeuroPilot/releases/download/v0.1.0/ju2df_5day_freemoving-male1-5day-image-pain.0.tif" `
  -OutFile "demo_data\spine\ju2df_5day_freemoving-male1-5day-image-pain 0.tif"
```

## Bash

```bash
mkdir -p demo_data/24h demo_data/spine

curl -L \
  -o "demo_data/24h/CellVideo 01-1.tif" \
  "https://github.com/LuoYuhong123/NeuroPilot/releases/download/v0.1.0/CellVideo.01-1.tif"

curl -L \
  -o "demo_data/spine/ju2df_5day_freemoving-male1-5day-image-pain 0.tif" \
  "https://github.com/LuoYuhong123/NeuroPilot/releases/download/v0.1.0/ju2df_5day_freemoving-male1-5day-image-pain.0.tif"
```

## Checksums

```text
CellVideo 01-1.tif
SHA256 430EADA89B33B68141184A2EEB12AFD0E609FD8D40168248986846CD540D1365

ju2df_5day_freemoving-male1-5day-image-pain 0.tif
SHA256 D24E972435413243693549478A775617EC993E88E10D01293BCEF39D5A7D4714
```
