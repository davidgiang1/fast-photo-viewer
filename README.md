# Fast Photo Viewer

An extremely fast, lightweight photo and video viewer built in **Rust**. Designed to handle directories with tens of thousands of media files efficiently using GPU acceleration.

## Download

Download the latest installer from [GitHub Releases](https://github.com/davidgiang1/fast-photo-viewer/releases).

Run `FastPhotoViewer-x.x.x-setup.exe` and follow the prompts. The installer will:
- Install Fast Photo Viewer to Program Files
- Optionally associate image and video file types
- Create Start Menu shortcuts

To set as your default viewer, go to **Windows Settings > Default Apps** and search for **Fast Photo Viewer**.

## Features

- **Instant Startup:** Compiled native binary.
- **Parallel Scanning:** Uses background threads to scan directories.
- **Image Support:** PNG, JPEG, WEBP, GIF, BMP, TIFF, ICO, SVG. Handles wrong extensions and truncated files.
- **Video Support:** MP4, MKV, AVI, MOV, WMV, FLV, WEBM, M4V, MPG, MPEG, 3GP, OGV, VOB with full playback controls.
- **GPU Rendering:** Hardware-accelerated display.
- **File Associations:** Can be set as the default Windows viewer for images and videos.
- **Random Slideshow:** Press `Space` or `Right Arrow`.
- **Navigation History:** Use `Left Arrow`.
- **Zoom & Pan:** Scroll to zoom, drag to pan. Works on both images and videos.
- **Media Filter:** Cycle between All / Images Only / Videos Only with `M`.
- **Fullscreen:** Toggle with `F11` or the fullscreen button.
- **Audio Detection:** Automatically detects whether a video has audio; shows mute indicator if not.
- **Click to Pause:** Click anywhere on a video to toggle play/pause.
- **File Explorer:** Open current file location.

## Usage

Open a file directly from the command line or by double-clicking an associated file:
```bash
fast-photo-viewer.exe "C:\Photos\image.jpg"
```

Or launch the application and use the keyboard shortcuts:

### Image Mode
| Key | Action |
|-----|--------|
| `O` | Open folder |
| `F` | Open file |
| `Space` / `Right Arrow` | Next image |
| `Left Arrow` | Previous image |
| `+` / `-` | Zoom in / out |
| Scroll wheel | Zoom |
| Click + drag | Pan |
| `M` | Cycle media filter (All / Images / Videos) |
| `F11` | Toggle fullscreen |
| `Esc` × 2 | Close |

### Video Mode
| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `Left` / `Right Arrow` | Seek ±3 seconds |
| `Ctrl+Left` / `Ctrl+Right` | Previous / Next video |
| `Up` / `Down Arrow` | Volume +/- 2% |
| `+` / `-` | Zoom in / out |
| Scroll wheel | Zoom |
| Click | Play / Pause |
| Click + drag | Pan |
| `M` | Cycle media filter (All / Images / Videos) |
| `F11` | Toggle fullscreen |
| `Esc` × 2 | Close |

## Building from Source

1. Install [Rust](https://rustup.rs/)
2. Download [FFmpeg shared libraries](https://github.com/BtbN/FFmpeg-Builds/releases) (win64-gpl-shared) and set `FFMPEG_DIR` to the extracted folder
3. Build:
    ```bash
    cargo build --release
    ```
4. The binary will be at `target/release/fast-photo-viewer.exe` (copy FFmpeg DLLs alongside it)
