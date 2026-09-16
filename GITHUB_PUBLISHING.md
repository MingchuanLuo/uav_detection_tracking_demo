# GitHub Publishing Guide

This guide packages the project as a portfolio demo without filling normal Git
history with large generated binaries.

## 1. Recommended publication layout

Commit these items to the normal repository:

- `README.md` and `README.zh-CN.md`;
- `GITHUB_PUBLISHING.md`;
- `.gitignore` and `requirements.txt`;
- `src/` and `configs/`;
- `data/frame_manifest.csv` and `data/labels/`;
- `video/README.md`; and
- the single whitelisted tracking preview used by the README.

Keep these generated or large items out of normal Git:

- the original WebM file;
- extracted/selected/rejected frame images;
- the generated YOLO dataset;
- model checkpoints and training-run directories;
- full detection and tracking MP4 files; and
- raw plots, metrics, and temporary previews.

The current `.gitignore` implements that policy.

Recommended GitHub Release assets:

| Asset | Local size | Recommendation |
| --- | ---: | --- |
| `outputs/models/uav_detector_best.pt` | 5.23 MiB | Attach to the release |
| `outputs/detections/Quadcopter_detection.mp4` | 82.80 MiB | Attach to the release |
| `outputs/tracking/Quadcopter_tracking.mp4` | 87.34 MiB | Attach to the release |
| Evaluation/detection/tracking CSV files | Small | Optional release attachments |
| `video/Quadcopter_(drone).webm` | 35.93 MiB | Prefer the Wikimedia source link |

GitHub warns about normal Git files larger than 50 MiB and blocks files larger
than 100 MiB. Browser uploads to a repository are limited to 25 MiB. By
contrast, a GitHub Release can contain up to 1,000 assets, each smaller than
2 GiB. The two generated MP4 files therefore belong in a Release, not in normal
repository history.

Official references:

- [About large files on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [About releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
- [Git Large File Storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage)

## 2. Resolve the project license before publishing

The source video and the software/model stack have different licenses:

1. The video is CC BY 3.0. Keep the attribution and modification notice from the
   README anywhere derived video or annotated frames are distributed.
2. Ultralytics documents its open-source framework and models under AGPL-3.0,
   with an Enterprise option for uses that cannot comply with AGPL.
3. The repository currently has no root `LICENSE` file. Add one before making
   the project public.

For this public, source-available portfolio demo, the direct choice is GNU
AGPL-3.0. Download the unmodified official license text to the repository root:

```powershell
Invoke-WebRequest `
  -Uri "https://www.gnu.org/licenses/agpl-3.0.txt" `
  -OutFile "LICENSE"
```

Then add a short copyright line to the README if desired, for example:

```text
Copyright (c) 2026 <your name>. Project code licensed under AGPL-3.0.
```

Do not describe the CC BY video as being relicensed under AGPL. The media keeps
its own CC BY 3.0 license. If the project will be used in a closed-source or
commercial product, review Ultralytics Enterprise licensing instead of relying
on this portfolio guidance.

## 3. Create the local Git repository

At the time this guide was prepared, this directory was not yet a Git
repository. From the project root:

```powershell
git init
git branch -M main
git add .
git status
```

Before committing, check that the large local assets are ignored:

```powershell
git check-ignore -v "video/Quadcopter_(drone).webm"
git check-ignore -v "outputs/detections/Quadcopter_detection.mp4"
git check-ignore -v "outputs/tracking/Quadcopter_tracking.mp4"
git check-ignore -v "outputs/models/uav_detector_best.pt"
```

Each command should print the matching `.gitignore` rule. Also confirm that the
README preview is staged:

```powershell
git status --short
```

You should see this representative image among the files to be committed:

```text
outputs/previews/tracking_video_check/preview_08_frame_001031_time_041.200s.jpg
```

Commit the repository:

```powershell
git commit -m "Add end-to-end UAV detection and tracking demo"
```

## 4. Create and push the GitHub repository

On GitHub, create a new empty repository, for example `uav-detection-tracking-demo`.
Because the local project already has its own README and `.gitignore`, do not
initialize the remote repository with another README or `.gitignore`.

Then run, replacing both placeholders:

```powershell
git remote add origin https://github.com/<github-user>/<repository-name>.git
git push -u origin main
```

Suggested repository description:

```text
End-to-end YOLO11n UAV detection and ByteTrack demo with chronological evaluation, trajectory export, and CPU/GPU benchmarking.
```

Suggested topics:

```text
computer-vision  object-detection  yolo  yolo11  drone-detection
uav  object-tracking  bytetrack  pytorch  opencv
```

## 5. Create the demo Release

Open the repository's **Releases** page, choose **Draft a new release**, and use:

```text
Tag: v1.0.0-demo
Title: UAV Detection and Tracking Demo v1.0
```

Attach:

```text
outputs/models/uav_detector_best.pt
outputs/detections/Quadcopter_detection.mp4
outputs/tracking/Quadcopter_tracking.mp4
outputs/metrics/evaluation_metrics.csv       optional
outputs/metrics/benchmark.csv                optional
outputs/detections/detections.csv            optional
outputs/tracking/trajectory.csv               optional
```

The JSON summary files currently contain absolute local paths such as the local
Windows user directory. Do not upload those JSON files unchanged. The listed
CSV files do not contain those absolute paths.

Suggested release notes:

```markdown
## UAV detection and tracking demo

This release contains the best YOLO11n checkpoint and the complete detection
and ByteTrack result videos described in the repository README.

- Model: YOLO11n, one class (`drone`), best checkpoint at epoch 33
- Detection: 1,619 decoded frames, confidence threshold 0.30
- Tracking: ByteTrack with image-plane trail and motion export

### Source-video attribution

“Quadcopter (drone)” by Sounds of Changes / Työväenmuseo Werstas;
sound/video recorder and photographer: Mikael Maffei. Source: Wikimedia
Commons. License: CC BY 3.0. Changes: detection boxes, confidence scores,
tracking IDs, trails and HUD were overlaid, and the generated MP4 files omit
the original audio. No endorsement by the original creators is implied.
```

After publishing the release, optionally add direct release-asset links near the
top of both READMEs.

## 6. Final public-repository checklist

- The English README opens by default and links to the Chinese version.
- The tracking preview and Mermaid pipeline render on GitHub.
- The root `LICENSE` is present and GitHub recognizes it.
- The Wikimedia source and CC BY 3.0 links work.
- The release has the best model and both result videos.
- Release notes include the source-media attribution and modification notice.
- No `.venv`, raw dataset image, checkpoint, MP4, or absolute-path JSON file is
  in normal Git history.
- A fresh clone can install the dependencies and shows clear instructions for
  downloading the source video and release checkpoint.

## Optional: Git LFS instead of Releases

Git LFS is an alternative if the model or videos must behave like versioned
repository files. Install Git LFS first, then track the binary extensions before
adding them:

```powershell
git lfs install
git lfs track "*.pt"
git lfs track "*.mp4"
git lfs track "*.webm"
git add .gitattributes
```

For a portfolio demo, Releases are usually clearer: clones stay small, versioned
source remains easy to inspect, and users can download only the binary artifact
they want. Do not use both approaches for the same file unless there is a
specific reason.
