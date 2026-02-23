# Fast Photo Viewer

An extremely fast, lightweight photo viewer built in **Rust**. Designed to handle directories with tens of thousands of images efficiently using GPU acceleration.

## Features

- **Instant Startup:** Compiled native binary.
- **Parallel Scanning:** Uses background threads to scan directories.
- **Robust Format Support:** Supports PNG, JPEG, WEBP, GIF, BMP, TIFF. Handles wrong extensions and truncated files (like unfinished downloads).
- **GPU Rendering:** Hardware-accelerated image display.
- **Open File:** Press `F` to open a specific image (and scan its folder).
- **Random Slideshow:** Press `Space` or `Right Arrow`.
- **Navigation History:** Use `Left Arrow`.
- **Zoom & Pan:** Scroll to zoom, drag to pan.
- **File Explorer:** Open current file location.
- **Error Logging:** Prints loading errors to console.

## Installation & Build

1.  Ensure you have [Rust](https://rustup.rs/) installed.
2.  Build the project:
    ```bash
    cargo build --release
    ```
3.  The binary will be located at `target/release/fast-photo-viewer.exe`.

## Usage

1.  Run the application:
    ```bash
    cargo run --release
    ```
2.  **Open Folder:** Press **'O'**.
3.  **Open File:** Press **'F'**.
4.  **Next Photo:** Press **Space** or **Right Arrow**.
5.  **Previous Photo:** Press **Left Arrow**.
6.  **Zoom:** Scroll wheel.
7.  **Pan:** Click and drag.
