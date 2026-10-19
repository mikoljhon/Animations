# Animations

Working files for the UNEDITED animated series and the Versus tournament shorts. This repo holds the Blender scenes, the Python scripts that build, animate and check them, the Motion Assist add-on, and the finished renders. It is a work in progress.

Large binary files (`.blend` scenes, images, video, audio and models) are stored with Git LFS. Install Git LFS before cloning.

## Layout

- **`unedited/`** is the project folder (`C:\ai\unedited`). It holds:
  - `.blend` scenes.
  - Build, animation and QA scripts (`*.py`, `*.sh`, `*.ps1`).
  - Notes: `FIGHT_ANIMATION_PROCESS.md` and `fight_reference.md`.
  - References: `refs/` and `mocap_ref/`.
  - The AI source mesh in `ai/`, plus `audio/`.
  - `renders/`: encoded videos, boards and stills.
- **`addons/`** is the Motion Assist Blender add-on. Its source is in `motion_assist/` and the installable zips are in `dist/`.
- **`mocap/`** has the CMU mocap prep and AutoPose training scripts, plus the trained models.

## Not in the repo

| Left out | Size | How to get it back |
|---|---|---|
| Frame sequences in `unedited/renders/*/` | about 8.6 GB | Re-render them from the `.blend` files and scripts. |
| Blender backup files (`*.blend1`) | 1.4 GB | Blender writes them on each save. |
| CMU mocap zips and the prepared training data in `mocap/data/` | about 2 GB | `mocap/cmu_bvh/download_cmu.sh` and `mocap/prep_cmu.py` rebuild them. |
| Third-party tools (Chatterbox, MSFlow, Kokoro) | about 6.4 GB | Install them separately. |
| Third-party reference images: the original KillDroid concept art, the boards that show it, and Meshy's animation-library previews | small | Not ours to publish. They stay on the owner's PC. |

## Credits

- Motion capture: the data used in this project was obtained from [mocap.cs.cmu.edu](http://mocap.cs.cmu.edu). The database was created with funding from NSF EIA-0196217.
- Test voices were made with ElevenLabs, Kokoro and Chatterbox.
