#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod video_player;

use eframe::egui;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::process::Command;
use std::fs;
use std::collections::HashSet;
use walkdir::WalkDir;
use rand::seq::SliceRandom;
use image::DynamicImage;
use crate::video_player::{Player, PlayerState};
use ffmpeg_the_third as ffmpeg;

const IMAGE_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "bmp", "webp", "gif", "tiff", "ico", "svg",
    // HEIF family (decoded via ffmpeg)
    "heic", "heif", "avif",
    // Camera raw (decoded via rawloader + imagepipe)
    "nef", "nrw", "cr2", "arw", "srf", "sr2", "dng", "raf",
    "rw2", "orf", "pef", "srw", "3fr", "mrw", "iiq", "kdc",
    "dcr", "rwl", "x3f", "mef", "mos",
];

const RAW_EXTENSIONS: &[&str] = &[
    "nef", "nrw", "cr2", "arw", "srf", "sr2", "dng", "raf",
    "rw2", "orf", "pef", "srw", "3fr", "mrw", "iiq", "kdc",
    "dcr", "rwl", "x3f", "mef", "mos",
];

const HEIF_EXTENSIONS: &[&str] = &["heic", "heif", "avif"];

fn has_ext(path: &Path, exts: &[&str]) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| exts.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn is_raw_file(path: &Path) -> bool { has_ext(path, RAW_EXTENSIONS) }
fn is_heif_file(path: &Path) -> bool { has_ext(path, HEIF_EXTENSIONS) }

const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mkv", "avi", "mov", "wmv", "flv", "webm", "m4v",
    "mpg", "mpeg", "3gp", "ogv", "ts", "vob",
];

#[derive(Clone, Copy, PartialEq)]
enum MediaFilter {
    All,
    ImagesOnly,
    VideosOnly,
}

impl MediaFilter {
    fn label(self) -> &'static str {
        match self {
            MediaFilter::All => "All Media",
            MediaFilter::ImagesOnly => "Images Only",
            MediaFilter::VideosOnly => "Videos Only",
        }
    }

    fn cycle(self) -> Self {
        match self {
            MediaFilter::All => MediaFilter::ImagesOnly,
            MediaFilter::ImagesOnly => MediaFilter::VideosOnly,
            MediaFilter::VideosOnly => MediaFilter::All,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ViewOrder {
    Random,
    Ordered,
}

impl ViewOrder {
    fn label(self) -> &'static str {
        match self {
            ViewOrder::Random => "Random",
            ViewOrder::Ordered => "Ordered",
        }
    }

    fn toggle(self) -> Self {
        match self {
            ViewOrder::Random => ViewOrder::Ordered,
            ViewOrder::Ordered => ViewOrder::Random,
        }
    }
}

fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn is_video_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| VIDEO_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn is_supported_file(path: &Path) -> bool {
    is_image_file(path) || is_video_file(path)
}

fn matches_filter(path: &Path, filter: MediaFilter) -> bool {
    match filter {
        MediaFilter::All => true,
        MediaFilter::ImagesOnly => is_image_file(path),
        MediaFilter::VideosOnly => is_video_file(path),
    }
}

// `file_has_audio_stream` was used by the old egui-video path; the in-
// house player now owns audio probing internally.

/// Scan a byte slice for the largest embedded JPEG (SOI..EOI). Works
/// because within a valid JPEG payload, any `0xFF` byte is followed by
/// `0x00` (stuff byte), so `FF D9` only appears as a real end-of-image.
fn find_largest_embedded_jpeg(data: &[u8]) -> Option<&[u8]> {
    let mut best: Option<&[u8]> = None;
    let mut i = 0;
    while i + 2 < data.len() {
        if data[i] == 0xFF && data[i + 1] == 0xD8 && data[i + 2] == 0xFF {
            let mut j = i + 2;
            while j + 1 < data.len() {
                if data[j] == 0xFF && data[j + 1] == 0xD9 {
                    let slice = &data[i..j + 2];
                    if best.map_or(true, |b| slice.len() > b.len()) {
                        best = Some(slice);
                    }
                    i = j + 2;
                    break;
                }
                j += 1;
            }
            if j + 1 >= data.len() {
                break;
            }
        } else {
            i += 1;
        }
    }
    best
}

/// Decode a camera RAW file (NEF, CR2, ARW, DNG, etc.) to an sRGB image.
/// Fast path: extract the full-resolution JPEG preview every modern
/// camera embeds. Fallback: full rawloader + imagepipe demosaic (slow,
/// and only works for camera models in rawloader's database).
fn decode_raw_image(path: &Path) -> Result<DynamicImage, String> {
    if let Ok(bytes) = fs::read(path) {
        if let Some(jpeg) = find_largest_embedded_jpeg(&bytes) {
            // Require at least 64 KB so we don't pick up a tiny thumbnail
            // when a larger preview exists further in the file.
            if jpeg.len() >= 64 * 1024 {
                if let Ok(img) = image::load_from_memory(jpeg) {
                    return Ok(img);
                }
            }
        }
    }

    // Fallback: full raw decode via rawloader + imagepipe.
    let mut pipeline = imagepipe::Pipeline::new_from_file(path)
        .map_err(|e| format!("raw open: {:?}", e))?;
    let decoded = pipeline
        .output_8bit(None)
        .map_err(|e| format!("raw pipeline: {:?}", e))?;
    let buf = image::RgbImage::from_raw(
        decoded.width as u32,
        decoded.height as u32,
        decoded.data,
    )
    .ok_or_else(|| "raw: buffer size mismatch".to_string())?;
    Ok(DynamicImage::ImageRgb8(buf))
}

/// Extract `count` evenly-spaced thumbnail frames from a video for use
/// as a seek preview strip. Runs on a background thread; writes each
/// thumbnail into `out[i]` as an egui::ColorImage and requests a repaint
/// so the UI can lazy-upload new textures.
fn extract_seek_thumbnails(
    path: PathBuf,
    count: usize,
    out: Arc<Mutex<Vec<Option<egui::ColorImage>>>>,
    ctx: egui::Context,
) {
    let mut ictx = match ffmpeg::format::input(&path) {
        Ok(c) => c,
        Err(_) => return,
    };
    let (stream_index, src_w, src_h, src_format, params) = {
        let Some(stream) = ictx.streams().best(ffmpeg::media::Type::Video) else {
            return;
        };
        let params = stream.parameters();
        let decoder_ctx = match ffmpeg::codec::context::Context::from_parameters(params.clone()) {
            Ok(c) => c,
            Err(_) => return,
        };
        let decoder = match decoder_ctx.decoder().video() {
            Ok(d) => d,
            Err(_) => return,
        };
        (stream.index(), decoder.width(), decoder.height(), decoder.format(), params)
    };

    let decoder_ctx = match ffmpeg::codec::context::Context::from_parameters(params) {
        Ok(c) => c,
        Err(_) => return,
    };
    let mut decoder = match decoder_ctx.decoder().video() {
        Ok(d) => d,
        Err(_) => return,
    };

    let thumb_w: u32 = 192;
    let thumb_h: u32 = if src_w > 0 {
        ((thumb_w as u64 * src_h as u64) / src_w as u64).max(1) as u32
    } else {
        108
    };

    let mut scaler = match ffmpeg::software::scaling::context::Context::get(
        src_format,
        src_w,
        src_h,
        ffmpeg::format::Pixel::RGBA,
        thumb_w,
        thumb_h,
        ffmpeg::software::scaling::flag::Flags::BILINEAR,
    ) {
        Ok(s) => s,
        Err(_) => return,
    };

    let duration = ictx.duration(); // AV_TIME_BASE units (microseconds)
    if duration <= 0 {
        return;
    }

    let mut frame = ffmpeg::frame::Video::empty();
    let mut rgb = ffmpeg::frame::Video::empty();

    for i in 0..count {
        // Aim slightly past the start of each segment so we don't land
        // exactly on a non-keyframe boundary.
        let target = (duration as i128 * i as i128 / count as i128) as i64
            + duration / (count as i64 * 4);
        let _ = ictx.seek(target, ..target + duration / count as i64);
        let _ = decoder.flush();

        let mut got_frame = false;
        let mut tries = 0usize;
        for item in ictx.packets() {
            tries += 1;
            if tries > 200 {
                break;
            }
            let (s, packet) = match item {
                Ok(v) => v,
                Err(_) => break,
            };
            if s.index() != stream_index {
                continue;
            }
            if decoder.send_packet(&packet).is_err() {
                continue;
            }
            if decoder.receive_frame(&mut frame).is_ok() {
                got_frame = true;
                break;
            }
        }
        if !got_frame {
            continue;
        }

        if scaler.run(&frame, &mut rgb).is_err() {
            continue;
        }

        let w = rgb.width() as usize;
        let h = rgb.height() as usize;
        let stride = rgb.stride(0);
        let src = rgb.data(0);
        let row_bytes = w * 4;
        let mut buf = Vec::with_capacity(row_bytes * h);
        for y in 0..h {
            let start = y * stride;
            buf.extend_from_slice(&src[start..start + row_bytes]);
        }
        let color_image = egui::ColorImage::from_rgba_unmultiplied([w, h], &buf);

        {
            let mut o = out.lock().unwrap();
            if i < o.len() {
                o[i] = Some(color_image);
            }
        }
        ctx.request_repaint();
    }
}

/// Decode a single-frame HEIF/HEIC/AVIF image through ffmpeg. We treat
/// the file as a one-frame video, decode the first frame, and convert
/// to RGB24 via swscale.
fn decode_heif_image(path: &Path) -> Result<DynamicImage, String> {
    let mut ictx = ffmpeg::format::input(&path)
        .map_err(|e| format!("heif open: {}", e))?;

    let stream_index = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or_else(|| "heif: no video stream".to_string())?
        .index();

    let params = ictx
        .stream(stream_index)
        .ok_or_else(|| "heif: missing stream".to_string())?
        .parameters();
    let decoder_ctx = ffmpeg::codec::context::Context::from_parameters(params)
        .map_err(|e| format!("heif codec ctx: {}", e))?;
    let mut decoder = decoder_ctx
        .decoder()
        .video()
        .map_err(|e| format!("heif decoder: {}", e))?;

    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        ffmpeg::format::Pixel::RGB24,
        decoder.width(),
        decoder.height(),
        ffmpeg::software::scaling::flag::Flags::BILINEAR,
    )
    .map_err(|e| format!("heif scaler: {}", e))?;

    let extract = |scaler: &mut ffmpeg::software::scaling::context::Context,
                   frame: &ffmpeg::frame::Video|
     -> Result<DynamicImage, String> {
        let mut rgb = ffmpeg::frame::Video::empty();
        scaler
            .run(frame, &mut rgb)
            .map_err(|e| format!("heif scale: {}", e))?;
        let w = rgb.width();
        let h = rgb.height();
        let stride = rgb.stride(0);
        let src = rgb.data(0);
        let row_bytes = w as usize * 3;
        let mut buf = Vec::with_capacity(row_bytes * h as usize);
        for y in 0..h as usize {
            let start = y * stride;
            buf.extend_from_slice(&src[start..start + row_bytes]);
        }
        let img = image::RgbImage::from_raw(w, h, buf)
            .ok_or_else(|| "heif: buffer size mismatch".to_string())?;
        Ok(DynamicImage::ImageRgb8(img))
    };

    let mut frame = ffmpeg::frame::Video::empty();
    for item in ictx.packets() {
        let (stream, packet) = item.map_err(|e| format!("heif packet: {}", e))?;
        if stream.index() != stream_index {
            continue;
        }
        decoder
            .send_packet(&packet)
            .map_err(|e| format!("heif send: {}", e))?;
        if decoder.receive_frame(&mut frame).is_ok() {
            return extract(&mut scaler, &frame);
        }
    }

    decoder
        .send_eof()
        .map_err(|e| format!("heif eof: {}", e))?;
    if decoder.receive_frame(&mut frame).is_ok() {
        return extract(&mut scaler, &frame);
    }

    Err("heif: no frame decoded".to_string())
}

fn main() -> eframe::Result<()> {
    let initial_file: Option<PathBuf> = std::env::args().nth(1).map(PathBuf::from);

    let icon_bytes = include_bytes!("../assets/icon.ico");
    let icon_image = image::load_from_memory(icon_bytes).expect("Failed to load app icon");
    let icon_rgba = icon_image.into_rgba8();
    let icon = egui::IconData {
        width: icon_rgba.width(),
        height: icon_rgba.height(),
        rgba: icon_rgba.into_raw(),
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_title("Fast Photo Viewer")
            .with_icon(std::sync::Arc::new(icon)),
        // Request the wgpu renderer so we can obtain a wgpu::Device in
        // CreationContext and manage our own video textures / custom
        // render callbacks downstream.
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            // Override eframe's default device descriptor so we can
            // request optional features. `TEXTURE_FORMAT_16BIT_NORM`
            // is needed for the 10-bit HEVC playback path to upload
            // directly to an `Rgba16Unorm` texture.
            device_descriptor: std::sync::Arc::new(|adapter| {
                let wanted = eframe::wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
                let required_features = if adapter.features().contains(wanted) {
                    wanted
                } else {
                    eframe::wgpu::Features::empty()
                };
                eframe::wgpu::DeviceDescriptor {
                    label: Some("fast-photo-viewer"),
                    required_features,
                    required_limits: eframe::wgpu::Limits::default(),
                    memory_hints: eframe::wgpu::MemoryHints::default(),
                }
            }),
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "Fast Photo Viewer",
        options,
        Box::new(move |cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::new(PhotoViewer::new(cc, initial_file)) as Box<dyn eframe::App>)
        }),
    )
}

struct PhotoViewer {
    media_paths: Arc<Mutex<Vec<PathBuf>>>,
    current_media_path: Option<PathBuf>,
    current_image: Option<DynamicImage>,

    // Video
    video_player: Option<Player>,
    is_video: bool,
    video_looping: bool,
    video_volume: f32,
    video_has_audio: bool,
    seek_frac: f32,
    video_rotation: u16,

    // Scrubbing: while the user is dragging on the seek bar, we hold
    // the target position here and only commit it to the player on
    // release so the decoder isn't hammered every frame.
    scrubbing: Option<f32>,

    // YouTube-style preview thumbnails for the current video: the bg
    // thread fills `seek_thumbs`, and the main thread lazily uploads
    // each one into `seek_thumb_textures` the first time it's needed.
    seek_thumbs: Arc<Mutex<Vec<Option<egui::ColorImage>>>>,
    seek_thumb_textures: Vec<Option<egui::TextureHandle>>,

    // UI state
    last_esc_press: Option<std::time::Instant>,

    // History
    history: Vec<PathBuf>,
    history_index: Option<usize>,

    // View State
    zoom: f32,
    pan: egui::Vec2,

    // Filter
    media_filter: MediaFilter,
    view_order: ViewOrder,

    is_scanning: Arc<Mutex<bool>>,
    scan_count: Arc<Mutex<usize>>,
    texture: Option<egui::TextureHandle>,
    error_msg: Option<String>,
    pending_initial_file: Option<PathBuf>,
    wgpu_backend: Option<video_player::WgpuBackend>,
}

impl PhotoViewer {
    fn new(cc: &eframe::CreationContext<'_>, initial_file: Option<PathBuf>) -> Self {
        let wgpu_backend = cc.wgpu_render_state.as_ref().map(|rs| {
            let supports_16bit_norm = rs
                .device
                .features()
                .contains(eframe::wgpu::Features::TEXTURE_FORMAT_16BIT_NORM);
            video_player::WgpuBackend {
                device: rs.device.clone(),
                queue: rs.queue.clone(),
                renderer: rs.renderer.clone(),
                supports_16bit_norm,
            }
        });
        Self {
            media_paths: Arc::new(Mutex::new(Vec::new())),
            current_media_path: None,
            current_image: None,
            video_player: None,
            is_video: false,
            video_looping: true,
            video_volume: 0.10,
            video_has_audio: true,
            seek_frac: 0.0,
            video_rotation: 0,
            scrubbing: None,
            seek_thumbs: Arc::new(Mutex::new(Vec::new())),
            seek_thumb_textures: Vec::new(),
            last_esc_press: None,
            history: Vec::new(),
            history_index: None,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            media_filter: MediaFilter::All,
            view_order: ViewOrder::Random,
            wgpu_backend,
            is_scanning: Arc::new(Mutex::new(false)),
            scan_count: Arc::new(Mutex::new(0)),
            texture: None,
            error_msg: None,
            pending_initial_file: initial_file,
        }
    }

    fn open_directory(&mut self) {
        if let Some(path) = rfd::FileDialog::new().pick_folder() {
            self.start_scan(path);
        }
    }

    fn open_file_dialog(&mut self, ctx: &egui::Context) {
        let all_extensions: Vec<&str> = IMAGE_EXTENSIONS.iter()
            .chain(VIDEO_EXTENSIONS.iter())
            .copied()
            .collect();

        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Media", &all_extensions)
            .add_filter("Images", IMAGE_EXTENSIONS)
            .add_filter("Videos", VIDEO_EXTENSIONS)
            .pick_file()
        {
            if let Some(parent) = path.parent() {
                self.start_scan(parent.to_path_buf());
            }

            self.current_media_path = Some(path.clone());
            self.history.push(path.clone());
            self.history_index = Some(0);
            self.load_media(path, ctx);
        }
    }

    fn start_scan(&mut self, directory: PathBuf) {
        {
            let mut paths = self.media_paths.lock().unwrap();
            paths.clear();
        }
        self.history.clear();
        self.history_index = None;
        self.current_media_path = None;
        self.texture = None;
        self.video_player = None;
        self.is_video = false;
        self.reset_view();

        *self.scan_count.lock().unwrap() = 0;
        *self.is_scanning.lock().unwrap() = true;

        let paths_clone = self.media_paths.clone();
        let scanning_clone = self.is_scanning.clone();
        let count_clone = self.scan_count.clone();

        thread::spawn(move || {
            for entry in WalkDir::new(directory).into_iter().filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.is_file() && is_supported_file(path) {
                    let mut p = paths_clone.lock().unwrap();
                    p.push(path.to_path_buf());
                    *count_clone.lock().unwrap() += 1;
                }
            }
            // Sort for deterministic ordered navigation
            paths_clone.lock().unwrap().sort();
            *scanning_clone.lock().unwrap() = false;
        });
    }

    fn reset_view(&mut self) {
        self.zoom = 1.0;
        self.pan = egui::Vec2::ZERO;
        self.video_rotation = 0;
    }

    fn go_next(&mut self, ctx: &egui::Context) {
        if self.view_order == ViewOrder::Ordered {
            self.ordered_step(ctx, 1);
            return;
        }
        if let Some(idx) = self.history_index {
            // Look forward through history for a matching entry
            let mut next_idx = idx + 1;
            while next_idx < self.history.len() {
                if matches_filter(&self.history[next_idx], self.media_filter) {
                    self.history_index = Some(next_idx);
                    let path = self.history[next_idx].clone();
                    self.current_media_path = Some(path.clone());
                    self.reset_view();
                    self.load_media(path, ctx);
                    return;
                }
                next_idx += 1;
            }
        }
        self.next_random_media(ctx);
    }

    fn go_prev(&mut self, ctx: &egui::Context) {
        if self.view_order == ViewOrder::Ordered {
            self.ordered_step(ctx, -1);
            return;
        }
        if let Some(idx) = self.history_index {
            let mut prev_idx = idx;
            while prev_idx > 0 {
                prev_idx -= 1;
                if matches_filter(&self.history[prev_idx], self.media_filter) {
                    self.history_index = Some(prev_idx);
                    let path = self.history[prev_idx].clone();
                    self.current_media_path = Some(path.clone());
                    self.reset_view();
                    self.load_media(path, ctx);
                    return;
                }
            }
        }
    }

    /// Step through `media_paths` in sorted order by `delta` (+1 / -1),
    /// skipping files that don't match the current media filter. Wraps
    /// around at both ends.
    fn ordered_step(&mut self, ctx: &egui::Context, delta: isize) {
        let next_path = {
            let paths = self.media_paths.lock().unwrap();
            if paths.is_empty() {
                return;
            }
            let n = paths.len();
            let start = self
                .current_media_path
                .as_ref()
                .and_then(|p| paths.iter().position(|mp| mp == p))
                .map(|i| i as isize)
                .unwrap_or(if delta > 0 { -1 } else { n as isize });

            let mut found = None;
            for step in 1..=n {
                let idx = ((start + delta * step as isize).rem_euclid(n as isize)) as usize;
                if matches_filter(&paths[idx], self.media_filter) {
                    found = Some(paths[idx].clone());
                    break;
                }
            }
            found
        };

        if let Some(p) = next_path {
            self.current_media_path = Some(p.clone());
            self.reset_view();
            self.load_media(p, ctx);
        }
    }

    fn next_random_media(&mut self, ctx: &egui::Context) {
        let path = {
            let paths = self.media_paths.lock().unwrap();
            if paths.is_empty() {
                return;
            }

            let history_set: HashSet<&PathBuf> = self.history.iter().collect();

            // Filter by media type AND exclude already-seen files
            let available: Vec<&PathBuf> = paths.iter()
                .filter(|p| matches_filter(p, self.media_filter) && !history_set.contains(p))
                .collect();

            let mut rng = rand::thread_rng();

            if !available.is_empty() {
                available.choose(&mut rng).map(|p| (*p).clone())
            } else {
                // All matching files seen — pick any matching file except the current one
                let matching: Vec<&PathBuf> = paths.iter()
                    .filter(|p| matches_filter(p, self.media_filter) && Some(*p) != self.current_media_path.as_ref())
                    .collect();
                if !matching.is_empty() {
                    matching.choose(&mut rng).map(|p| (*p).clone())
                } else {
                    // Only one matching file (or none)
                    paths.iter()
                        .find(|p| matches_filter(p, self.media_filter))
                        .cloned()
                }
            }
        };

        if let Some(p) = path {
            self.history.push(p.clone());
            self.history_index = Some(self.history.len() - 1);

            self.current_media_path = Some(p.clone());
            self.reset_view();
            self.load_media(p, ctx);
        }
    }

    fn open_in_explorer(&self) {
        if let Some(path) = &self.current_media_path {
            #[cfg(target_os = "windows")]
            {
                Command::new("explorer")
                    .args(["/select,", &path.to_string_lossy()])
                    .spawn()
                    .ok();
            }
            #[cfg(not(target_os = "windows"))]
            {
                let _ = path;
            }
        }
    }

    fn load_media(&mut self, path: PathBuf, ctx: &egui::Context) {
        if is_video_file(&path) {
            self.load_video(path, ctx);
        } else {
            self.load_image(path, ctx);
        }
    }

    fn load_video(&mut self, path: PathBuf, ctx: &egui::Context) {
        // Clear image state
        self.current_image = None;
        self.texture = None;

        // Reset any in-flight scrub and preview thumbnails.
        self.scrubbing = None;
        self.seek_thumb_textures.clear();
        const THUMB_COUNT: usize = 20;
        {
            let mut thumbs = self.seek_thumbs.lock().unwrap();
            *thumbs = vec![None; THUMB_COUNT];
        }
        self.seek_thumb_textures.resize_with(THUMB_COUNT, || None);
        {
            let path_clone = path.clone();
            let thumbs_clone = self.seek_thumbs.clone();
            let ctx_clone = ctx.clone();
            thread::spawn(move || {
                extract_seek_thumbnails(path_clone, THUMB_COUNT, thumbs_clone, ctx_clone);
            });
        }

        match Player::open_with_backend(ctx, &path, self.wgpu_backend.clone()) {
            Ok(mut player) => {
                player.set_looping(self.video_looping);
                player.set_volume(self.video_volume);
                player.play();
                self.video_has_audio = player.has_audio();
                self.video_player = Some(player);
                self.is_video = true;
                self.seek_frac = 0.0;
                self.error_msg = None;
            }
            Err(e) => {
                let msg = format!("Error loading video {}: {}", path.display(), e);
                println!("{}", msg);
                self.error_msg = Some(msg);
                self.video_player = None;
                self.is_video = false;
            }
        }
    }

    fn load_image(&mut self, path: PathBuf, ctx: &egui::Context) {
        // Clear video state
        self.video_player = None;
        self.is_video = false;

        let result = if is_raw_file(&path) {
            // Raw decoding can panic inside rawloader/imagepipe on malformed
            // files; catch and report as error.
            let p = path.clone();
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| decode_raw_image(&p))) {
                Ok(r) => r,
                Err(_) => Err("raw decoder panicked".to_string()),
            }
        } else if is_heif_file(&path) {
            decode_heif_image(&path)
        } else {
            match fs::read(&path).map_err(|e| e.to_string()) {
                Ok(mut bytes) => match image::load_from_memory(&bytes) {
                    Ok(img) => Ok(img),
                    Err(e) => {
                        if bytes.len() > 2 && bytes[0] == 0xFF && bytes[1] == 0xD8 {
                            println!("Attempting to repair truncated JPEG: {}", path.display());
                            bytes.push(0xFF);
                            bytes.push(0xD9);
                            bytes.extend(std::iter::repeat(0).take(1024));

                            image::load_from_memory(&bytes).map_err(|retry_err| {
                                format!("Original: {}, Retry: {}", e, retry_err)
                            })
                        } else {
                            Err(e.to_string())
                        }
                    }
                },
                Err(e) => Err(e),
            }
        };

        match result {
            Ok(image) => {
                self.current_image = Some(image);
                self.regenerate_texture(ctx);
                self.error_msg = None;
            }
            Err(e) => {
                let msg = format!("Error loading {}: {}", path.display(), e);
                println!("{}", msg);
                self.error_msg = Some(msg);
                self.current_image = None;
                self.texture = None;
            }
        }
    }

    fn regenerate_texture(&mut self, ctx: &egui::Context) {
        if let Some(image) = &self.current_image {
            let size = [image.width() as usize, image.height() as usize];
            let image_buffer = image.to_rgba8();
            let pixels = image_buffer.as_flat_samples();

            let color_image = egui::ColorImage::from_rgba_unmultiplied(
                size,
                pixels.as_slice(),
            );

            self.texture = Some(ctx.load_texture(
                "current_image",
                color_image,
                egui::TextureOptions::LINEAR,
            ));
        }
    }

    fn rotate_cw(&mut self, ctx: &egui::Context) {
        if let Some(img) = &self.current_image {
            self.current_image = Some(img.rotate90());
            self.regenerate_texture(ctx);
        }
    }

    fn rotate_ccw(&mut self, ctx: &egui::Context) {
        if let Some(img) = &self.current_image {
            self.current_image = Some(img.rotate270());
            self.regenerate_texture(ctx);
        }
    }

    fn format_time(ms: i64) -> String {
        let total_secs = ms / 1000;
        let hours = total_secs / 3600;
        let mins = (total_secs % 3600) / 60;
        let secs = total_secs % 60;
        if hours > 0 {
            format!("{:02}:{:02}:{:02}", hours, mins, secs)
        } else {
            format!("{:02}:{:02}", mins, secs)
        }
    }

    /// Render a video frame with rotation using a custom textured mesh.
    fn render_rotated_video(ui: &mut egui::Ui, texture_id: egui::TextureId, rect: egui::Rect, rotation: u16) {
        // UV coordinates for each rotation (top-left, top-right, bottom-right, bottom-left)
        let uvs: [(f32, f32); 4] = match rotation {
            90  => [(0.0, 1.0), (0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
            180 => [(1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)],
            270 => [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)],
            _   => [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        };

        let white = egui::Color32::WHITE;
        let mut mesh = egui::Mesh::with_texture(texture_id);
        mesh.vertices.push(egui::epaint::Vertex { pos: rect.left_top(),     uv: egui::pos2(uvs[0].0, uvs[0].1), color: white });
        mesh.vertices.push(egui::epaint::Vertex { pos: rect.right_top(),    uv: egui::pos2(uvs[1].0, uvs[1].1), color: white });
        mesh.vertices.push(egui::epaint::Vertex { pos: rect.right_bottom(), uv: egui::pos2(uvs[2].0, uvs[2].1), color: white });
        mesh.vertices.push(egui::epaint::Vertex { pos: rect.left_bottom(),  uv: egui::pos2(uvs[3].0, uvs[3].1), color: white });
        mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
        ui.painter().add(egui::Shape::mesh(mesh));
    }
}

impl eframe::App for PhotoViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle deferred initial file loading (first frame only)
        if let Some(path) = self.pending_initial_file.take() {
            if path.exists() && path.is_file() {
                if let Some(parent) = path.parent() {
                    self.start_scan(parent.to_path_buf());
                }
                self.current_media_path = Some(path.clone());
                self.history.push(path.clone());
                self.history_index = Some(0);
                self.load_media(path, ctx);
            }
        }

        // Upload any new decoded video frames.
        if let Some(player) = &mut self.video_player {
            player.tick();
        }

        // Handle input differently for video vs image mode
        let ctrl_held = ctx.input(|i| i.modifiers.ctrl);

        if self.is_video {
            // Video mode: arrows seek, ctrl+arrows navigate
            if ctrl_held {
                if ctx.input(|i| i.key_pressed(egui::Key::ArrowRight)) {
                    self.go_next(ctx);
                }
                if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
                    self.go_prev(ctx);
                }
            } else {
                // Seek ±3 seconds
                if ctx.input(|i| i.key_pressed(egui::Key::ArrowRight)) {
                    if let Some(player) = &mut self.video_player {
                        let duration = player.duration_ms();
                        if duration > 0 {
                            let step = 3000.0 / duration as f32;
                            let current = player.elapsed_ms() as f32 / duration as f32;
                            let target = (current + step).clamp(0.0, 1.0);
                            player.seek(target);
                        }
                    }
                }
                if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
                    if let Some(player) = &mut self.video_player {
                        let duration = player.duration_ms();
                        if duration > 0 {
                            let step = 3000.0 / duration as f32;
                            let current = player.elapsed_ms() as f32 / duration as f32;
                            let target = (current - step).clamp(0.0, 1.0);
                            player.seek(target);
                        }
                    }
                }
            }
            // Volume: Up/Down arrows ±2%
            if ctx.input(|i| i.key_pressed(egui::Key::ArrowUp)) {
                self.video_volume = (self.video_volume + 0.02).clamp(0.0, 1.0);
                if let Some(player) = &mut self.video_player {
                    player.set_volume(self.video_volume);
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
                self.video_volume = (self.video_volume - 0.02).clamp(0.0, 1.0);
                if let Some(player) = &mut self.video_player {
                    player.set_volume(self.video_volume);
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
                // Space toggles play/pause for video
                if let Some(player) = &mut self.video_player {
                    match player.state() {
                        PlayerState::Playing => player.pause(),
                        PlayerState::Paused => player.play(),
                        PlayerState::EndOfFile => {
                            player.seek(0.0);
                            player.play();
                        }
                        _ => {}
                    }
                }
            }
        } else {
            // Image mode: original behavior
            if ctx.input(|i| i.key_pressed(egui::Key::Space) || i.key_pressed(egui::Key::ArrowRight)) {
                self.go_next(ctx);
            }
            if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
                self.go_prev(ctx);
            }
        }

        if ctx.input(|i| i.key_pressed(egui::Key::O)) {
            self.open_directory();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F)) {
            self.open_file_dialog(ctx);
        }

        // F11: toggle fullscreen (both modes)
        if ctx.input(|i| i.key_pressed(egui::Key::F11)) {
            let is_fullscreen = ctx.input(|i| i.viewport().fullscreen.unwrap_or(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(!is_fullscreen));
        }

        // Double-tap Esc within 500ms to close the app
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            let now = std::time::Instant::now();
            if let Some(last) = self.last_esc_press {
                if now.duration_since(last) < std::time::Duration::from_millis(500) {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
            self.last_esc_press = Some(now);
        }

        // M: cycle media filter
        if ctx.input(|i| i.key_pressed(egui::Key::M)) {
            self.media_filter = self.media_filter.cycle();
        }

        // R: toggle random vs ordered viewing
        if ctx.input(|i| i.key_pressed(egui::Key::R)) {
            self.view_order = self.view_order.toggle();
        }

        // 0: reset zoom + pan to defaults
        if ctx.input(|i| i.key_pressed(egui::Key::Num0)) {
            self.zoom = 1.0;
            self.pan = egui::Vec2::ZERO;
        }

        // Handle Zoom (Mouse Wheel + keyboard) - images and videos
        {
            let scroll = ctx.input(|i| i.raw_scroll_delta);
            if scroll.y != 0.0 {
                let zoom_factor = if scroll.y > 0.0 { 1.1 } else { 0.9 };
                self.zoom *= zoom_factor;
                self.zoom = self.zoom.clamp(0.1, 50.0);
            }
            // +/= to zoom in, - to zoom out
            if ctx.input(|i| i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals)) {
                self.zoom *= 1.15;
                self.zoom = self.zoom.clamp(0.1, 50.0);
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Minus)) {
                self.zoom *= 0.87;
                self.zoom = self.zoom.clamp(0.1, 50.0);
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let rect = ui.available_rect_before_wrap();
            ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);

            if self.is_video {
                // Video rendering
                if let Some(player) = &mut self.video_player {
                    let available = rect.size();
                    // Reserve space for the floating controls panel at the
                    // bottom so panning doesn't sit under them.
                    let controls_height = 120.0;
                    let video_area = egui::vec2(available.x, available.y - controls_height);

                    // Click-to-pause and drag-to-pan on the video area (above controls)
                    let interact_rect = egui::Rect::from_min_size(rect.min, video_area);
                    let video_interact = ui.interact(interact_rect, ui.id().with("video_interact"), egui::Sense::click_and_drag());
                    if video_interact.dragged() {
                        self.pan += video_interact.drag_delta();
                    }
                    if video_interact.clicked() {
                        match player.state() {
                            PlayerState::Playing => player.pause(),
                            PlayerState::Paused => player.play(),
                            PlayerState::EndOfFile => {
                                player.seek(0.0);
                                player.play();
                            }
                            _ => {}
                        }
                    }

                    // Scale video to fit while maintaining aspect ratio
                    // For 90/270 rotation, swap the video dimensions for aspect ratio calc
                    let (src_w, src_h) = player.size();
                    let video_size = egui::vec2(src_w as f32, src_h as f32);
                    let (effective_w, effective_h) = if self.video_rotation == 90 || self.video_rotation == 270 {
                        (video_size.y, video_size.x)
                    } else {
                        (video_size.x, video_size.y)
                    };
                    let scale = if effective_w > 0.0 && effective_h > 0.0 {
                        (video_area.x / effective_w).min(video_area.y / effective_h)
                    } else {
                        1.0
                    };
                    // Apply zoom to display size
                    let display_size = egui::vec2(effective_w * scale * self.zoom, effective_h * scale * self.zoom);

                    // Center the video in the available area, with pan offset
                    let video_rect = egui::Rect::from_center_size(
                        egui::pos2(
                            rect.min.x + available.x / 2.0,
                            rect.min.y + video_area.y / 2.0,
                        ) + self.pan,
                        display_size,
                    );

                    // Render via painter directly (not render_frame_at) so our
                    // click interaction isn't consumed by an internal widget.
                    if let Some(texture_id) = player.texture_id() {
                        if self.video_rotation == 0 {
                            ui.painter().image(
                                texture_id,
                                video_rect,
                                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                egui::Color32::WHITE,
                            );
                        } else {
                            Self::render_rotated_video(ui, texture_id, video_rect, self.video_rotation);
                        }
                    }

                    // Floating controls panel: centered, rounded, not full
                    // width. Two rows — a wide seek slider for precise
                    // scrubbing on top, buttons/volume/fullscreen below.
                    let panel_w = (available.x * 0.80).clamp(420.0, 1100.0);
                    let panel_h = 88.0;
                    let panel_bottom_margin = 0.0;
                    let panel_rect = egui::Rect::from_min_size(
                        egui::pos2(
                            rect.min.x + (available.x - panel_w) / 2.0,
                            rect.max.y - panel_h - panel_bottom_margin,
                        ),
                        egui::vec2(panel_w, panel_h),
                    );

                    egui::Area::new(egui::Id::new("VideoControls"))
                        .fixed_pos(panel_rect.min)
                        .order(egui::Order::Foreground)
                        .show(ctx, |ui| {
                            ui.set_width(panel_rect.width());
                            ui.set_height(panel_rect.height());
                            egui::Frame::none()
                                .fill(egui::Color32::from_black_alpha(200))
                                .rounding(14.0)
                                .inner_margin(egui::Margin::symmetric(14.0, 10.0))
                                .shadow(egui::epaint::Shadow {
                                    offset: egui::vec2(0.0, 3.0),
                                    blur: 12.0,
                                    spread: 0.0,
                                    color: egui::Color32::from_black_alpha(140),
                                })
                                .show(ui, |ui| {
                                    ui.set_width(panel_rect.width() - 28.0);
                            let elapsed = player.elapsed_ms();
                            let duration = player.duration_ms();

                            ui.vertical(|ui| {
                                // ---- Row 1: custom-drawn seek bar ----
                                let bar_height = 12.0;
                                let (bar_rect, bar_response) = ui.allocate_exact_size(
                                    egui::vec2(ui.available_width(), bar_height),
                                    egui::Sense::click_and_drag(),
                                );

                                let pointer_frac = bar_response
                                    .interact_pointer_pos()
                                    .or_else(|| ui.input(|i| i.pointer.hover_pos()))
                                    .map(|pos| {
                                        ((pos.x - bar_rect.left()) / bar_rect.width())
                                            .clamp(0.0, 1.0)
                                    });

                                // While dragging, buffer the target position and
                                // commit it on release — calling player.seek() on
                                // every drag frame causes noticeable lag.
                                if bar_response.drag_started() {
                                    if let Some(f) = pointer_frac {
                                        self.scrubbing = Some(f);
                                    }
                                }
                                if bar_response.dragged() {
                                    if let Some(f) = pointer_frac {
                                        self.scrubbing = Some(f);
                                    }
                                }
                                if bar_response.drag_stopped() {
                                    if let Some(f) = self.scrubbing.take() {
                                        if duration > 0 {
                                            player.seek(f);
                                        }
                                    }
                                }
                                if bar_response.clicked() && duration > 0 {
                                    if let Some(f) = pointer_frac {
                                        player.seek(f);
                                    }
                                }

                                let played_frac = if duration > 0 {
                                    (elapsed as f32 / duration as f32).clamp(0.0, 1.0)
                                } else {
                                    0.0
                                };
                                let display_frac = self.scrubbing.unwrap_or(played_frac);

                                let painter = ui.painter();
                                let rounding = egui::Rounding::same(bar_height * 0.5);
                                painter.rect_filled(
                                    bar_rect,
                                    rounding,
                                    egui::Color32::from_rgb(55, 55, 60),
                                );
                                if display_frac > 0.0 {
                                    let mut filled = bar_rect;
                                    filled.max.x =
                                        filled.min.x + bar_rect.width() * display_frac;
                                    painter.rect_filled(
                                        filled,
                                        rounding,
                                        egui::Color32::from_rgb(100, 200, 255),
                                    );
                                }
                                let handle_x =
                                    bar_rect.left() + bar_rect.width() * display_frac;
                                painter.circle_filled(
                                    egui::pos2(handle_x, bar_rect.center().y),
                                    bar_height * 0.8,
                                    egui::Color32::WHITE,
                                );

                                // ---- Preview thumbnail (YouTube-style) ----
                                let show_preview = (bar_response.hovered()
                                    || bar_response.dragged())
                                    && duration > 0;
                                if show_preview {
                                    if let Some(frac) = pointer_frac {
                                        // Lazily upload any newly-decoded
                                        // thumbnails into GPU textures.
                                        {
                                            let thumbs = self.seek_thumbs.lock().unwrap();
                                            for (i, slot) in thumbs.iter().enumerate() {
                                                if let Some(img) = slot {
                                                    if self
                                                        .seek_thumb_textures
                                                        .get(i)
                                                        .and_then(|t| t.as_ref())
                                                        .is_none()
                                                    {
                                                        let handle = ctx.load_texture(
                                                            format!("seek_thumb_{}", i),
                                                            img.clone(),
                                                            egui::TextureOptions::LINEAR,
                                                        );
                                                        if i < self.seek_thumb_textures.len() {
                                                            self.seek_thumb_textures[i] =
                                                                Some(handle);
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        let count = self.seek_thumb_textures.len().max(1);
                                        let idx = ((frac * count as f32) as usize)
                                            .min(count - 1);
                                        // Find the nearest available thumb
                                        // (search outward from idx).
                                        let mut found: Option<&egui::TextureHandle> = None;
                                        for step in 0..count {
                                            let candidates = [
                                                idx.saturating_sub(step),
                                                (idx + step).min(count - 1),
                                            ];
                                            for c in candidates {
                                                if let Some(Some(tex)) =
                                                    self.seek_thumb_textures.get(c)
                                                {
                                                    found = Some(tex);
                                                    break;
                                                }
                                            }
                                            if found.is_some() {
                                                break;
                                            }
                                        }

                                        if let Some(tex) = found {
                                            let tex_size = tex.size_vec2();
                                            let preview_w = 192.0_f32;
                                            let preview_h = if tex_size.x > 0.0 {
                                                preview_w * tex_size.y / tex_size.x
                                            } else {
                                                108.0
                                            };
                                            let caption_h = 18.0;
                                            let total_h = preview_h + caption_h;
                                            let padding = 4.0;

                                            let cursor_x = bar_rect.left()
                                                + bar_rect.width() * frac;
                                            let preview_bottom = bar_rect.top() - 10.0;
                                            let mut preview_left = cursor_x - preview_w / 2.0;
                                            // Clamp within the panel so it doesn't
                                            // slip off either edge.
                                            let clamp_left = panel_rect.left() + 6.0;
                                            let clamp_right = panel_rect.right() - 6.0;
                                            if preview_left < clamp_left {
                                                preview_left = clamp_left;
                                            }
                                            if preview_left + preview_w > clamp_right {
                                                preview_left = clamp_right - preview_w;
                                            }
                                            let preview_rect = egui::Rect::from_min_size(
                                                egui::pos2(
                                                    preview_left,
                                                    preview_bottom - total_h,
                                                ),
                                                egui::vec2(preview_w, total_h),
                                            );

                                            // Draw on a top-layer painter so
                                            // the preview sits above the panel.
                                            let layer = egui::LayerId::new(
                                                egui::Order::Foreground,
                                                egui::Id::new("seek_preview"),
                                            );
                                            let top = ctx.layer_painter(layer);
                                            let bg_rect = preview_rect.expand(padding);
                                            top.rect_filled(
                                                bg_rect,
                                                egui::Rounding::same(6.0),
                                                egui::Color32::from_black_alpha(220),
                                            );
                                            let img_rect = egui::Rect::from_min_size(
                                                preview_rect.min,
                                                egui::vec2(preview_w, preview_h),
                                            );
                                            top.image(
                                                tex.id(),
                                                img_rect,
                                                egui::Rect::from_min_max(
                                                    egui::pos2(0.0, 0.0),
                                                    egui::pos2(1.0, 1.0),
                                                ),
                                                egui::Color32::WHITE,
                                            );
                                            let hover_ms = (frac * duration as f32) as i64;
                                            top.text(
                                                egui::pos2(
                                                    preview_rect.center().x,
                                                    preview_rect.min.y + preview_h + 2.0,
                                                ),
                                                egui::Align2::CENTER_TOP,
                                                Self::format_time(hover_ms),
                                                egui::FontId::monospace(12.0),
                                                egui::Color32::WHITE,
                                            );
                                        } else {
                                            // Thumbs not ready yet — at least
                                            // show a time tooltip.
                                            let hover_ms = (frac * duration as f32) as i64;
                                            bar_response.clone().on_hover_text(
                                                Self::format_time(hover_ms),
                                            );
                                        }
                                    }
                                }

                                ui.add_space(6.0);

                                // ---- Row 2: buttons + time + volume + fullscreen ----
                                ui.horizontal(|ui| {
                                    let state = player.state();
                                    let btn_text = match state {
                                        PlayerState::Playing => "⏸",
                                        _ => "▶",
                                    };
                                    if ui.button(egui::RichText::new(btn_text).size(16.0)).clicked() {
                                        match state {
                                            PlayerState::Playing => player.pause(),
                                            PlayerState::Paused => player.play(),
                                            PlayerState::EndOfFile => {
                                                player.seek(0.0);
                                                player.play();
                                            }
                                            _ => player.play(),
                                        }
                                    }

                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{} / {}",
                                            Self::format_time(elapsed),
                                            Self::format_time(duration),
                                        ))
                                        .color(egui::Color32::WHITE)
                                        .monospace(),
                                    );

                                    // Loop toggle
                                    let loop_btn = ui.button(
                                        egui::RichText::new("🔁").color(
                                            if self.video_looping {
                                                egui::Color32::from_rgb(100, 200, 255)
                                            } else {
                                                egui::Color32::GRAY
                                            },
                                        ),
                                    );
                                    if loop_btn.on_hover_text("Toggle loop").clicked() {
                                        self.video_looping = !self.video_looping;
                                        player.set_looping(self.video_looping);
                                    }

                                    // Right-side cluster: fullscreen on the far right,
                                    // volume immediately left of it. `with_layout`
                                    // lays children out right-to-left so the items
                                    // read left-to-right in the final result.
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            let is_fullscreen = ctx.input(|i| {
                                                i.viewport().fullscreen.unwrap_or(false)
                                            });
                                            let fs_icon = if is_fullscreen { "⊡" } else { "⊞" };
                                            if ui
                                                .button(
                                                    egui::RichText::new(fs_icon)
                                                        .color(egui::Color32::WHITE),
                                                )
                                                .on_hover_text("Toggle fullscreen (F11)")
                                                .clicked()
                                            {
                                                ctx.send_viewport_cmd(
                                                    egui::ViewportCommand::Fullscreen(!is_fullscreen),
                                                );
                                            }

                                            if self.video_has_audio {
                                                let vol_slider = egui::Slider::new(
                                                    &mut self.video_volume,
                                                    0.0..=1.0,
                                                )
                                                .show_value(false)
                                                .trailing_fill(true);
                                                let vol_response =
                                                    ui.add_sized([110.0, 20.0], vol_slider);
                                                if vol_response.changed() {
                                                    player.set_volume(self.video_volume);
                                                }
                                                vol_response.on_hover_text(format!(
                                                    "{}%",
                                                    (self.video_volume * 100.0).round() as i32
                                                ));

                                                let vol_icon = if self.video_volume == 0.0 {
                                                    "🔇"
                                                } else {
                                                    "🔊"
                                                };
                                                ui.label(
                                                    egui::RichText::new(vol_icon)
                                                        .color(egui::Color32::WHITE),
                                                );
                                            } else {
                                                ui.label(
                                                    egui::RichText::new("🔇 No Audio").color(
                                                        egui::Color32::from_rgb(180, 180, 180),
                                                    ),
                                                );
                                            }
                                        },
                                    );
                                });
                            });
                                });
                        });

                    // Request continuous repaint for video playback
                    ctx.request_repaint();
                }
            } else {
                // Image rendering (original logic)
                let response = ui.interact(rect, ui.id().with("pan_drag"), egui::Sense::drag());
                if response.dragged() {
                    self.pan += response.drag_delta();
                }

                if let Some(texture) = &self.texture {
                    let available_size = rect.size();
                    let original_size = texture.size_vec2();

                    if original_size.x > 0.0 && original_size.y > 0.0 {
                        let width_ratio = available_size.x / original_size.x;
                        let height_ratio = available_size.y / original_size.y;
                        let base_scale = width_ratio.min(height_ratio);

                        let display_size = original_size * base_scale * self.zoom;

                        let center_x = rect.min.x + available_size.x / 2.0;
                        let center_y = rect.min.y + available_size.y / 2.0;

                        let image_rect = egui::Rect::from_center_size(
                            egui::pos2(center_x, center_y) + self.pan,
                            display_size,
                        );

                        ui.painter().image(
                            texture.id(),
                            image_rect,
                            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                            egui::Color32::WHITE,
                        );
                    }
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.label(
                            egui::RichText::new(
                                "Press 'O' to Open Folder  |  'F' to Open File\n\n\
                                 Images: Space / Right Arrow = Next  |  Left Arrow = Prev  |  +/- Zoom\n\
                                 Videos: Left/Right Arrow = Seek 3s  |  Ctrl + Left/Right = Prev/Next\n\
                                 Space: Play/Pause (video)  |  Next (image)\n\
                                 Scroll to Zoom  |  Drag to Pan  |  0: Reset View\n\
                                 F11: Fullscreen  |  M: Filter  |  R: Random / Ordered\n\
                                 Double-tap Esc: Close",
                            )
                            .color(egui::Color32::WHITE)
                            .size(18.0),
                        );
                    });
                }
            }

            // Overlays (Status)
            let scanning = *self.is_scanning.lock().unwrap();
            let count = *self.scan_count.lock().unwrap();

            if scanning || count > 0 {
                egui::Window::new("Status")
                    .anchor(egui::Align2::LEFT_TOP, [10.0, 10.0])
                    .title_bar(false)
                    .resizable(false)
                    .auto_sized()
                    .frame(egui::Frame::popup(ui.style()).multiply_with_opacity(0.8))
                    .show(ctx, |ui| {
                        if scanning {
                            ui.horizontal(|ui| {
                                ui.spinner();
                                ui.label(format!("Scanning... Found {} media files", count));
                            });
                            ctx.request_repaint();
                        } else {
                            ui.label(format!("Total: {} media files", count));
                            if let Some(path) = &self.current_media_path {
                                ui.label(
                                    path.file_name()
                                        .unwrap_or_default()
                                        .to_string_lossy(),
                                );
                                if self.is_video {
                                    ui.label("(Video)");
                                }
                                if let Some(idx) = self.history_index {
                                    ui.label(format!(
                                        "History: {}/{}",
                                        idx + 1,
                                        self.history.len()
                                    ));
                                }
                            }
                        }
                        // Show active filter
                        if self.media_filter != MediaFilter::All {
                            ui.colored_label(
                                egui::Color32::from_rgb(100, 200, 255),
                                format!("Filter: {}", self.media_filter.label()),
                            );
                        }
                        ui.colored_label(
                            egui::Color32::from_rgb(100, 200, 255),
                            format!("Order: {}", self.view_order.label()),
                        );
                        if let Some(err) = &self.error_msg {
                            ui.colored_label(egui::Color32::RED, err);
                        }
                    });
            }

            // Controls overlay — rotation for both images and videos
            if self.current_media_path.is_some() {
                egui::Window::new("Controls")
                    .anchor(egui::Align2::RIGHT_BOTTOM, [-10.0, -10.0])
                    .title_bar(false)
                    .resizable(false)
                    .auto_sized()
                    .frame(egui::Frame::popup(ui.style()).multiply_with_opacity(0.8))
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            if ui
                                .button("⟲")
                                .on_hover_text("Rotate Left")
                                .clicked()
                            {
                                if self.is_video {
                                    self.video_rotation = (self.video_rotation + 270) % 360;
                                } else {
                                    self.rotate_ccw(ctx);
                                }
                            }
                            if ui
                                .button("⟳")
                                .on_hover_text("Rotate Right")
                                .clicked()
                            {
                                if self.is_video {
                                    self.video_rotation = (self.video_rotation + 90) % 360;
                                } else {
                                    self.rotate_cw(ctx);
                                }
                            }
                        });
                        if ui.button("📂 Show in Explorer").clicked() {
                            self.open_in_explorer();
                        }
                    });
            }
        });
    }
}
