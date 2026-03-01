#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

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
use egui_video::{AudioDevice, Player, PlayerState};
use ffmpeg_the_third as ffmpeg;

const IMAGE_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "bmp", "webp", "gif", "tiff", "ico", "svg",
];

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

/// Check if a video file actually contains an audio stream using ffmpeg.
fn file_has_audio_stream(path: &Path) -> bool {
    match ffmpeg::format::input(&path) {
        Ok(ctx) => ctx.streams().any(|s| s.parameters().medium() == ffmpeg::media::Type::Audio),
        Err(_) => true, // Assume audio if probe fails
    }
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
    audio_device: AudioDevice,
    is_video: bool,
    video_looping: bool,
    video_volume: f32,
    video_has_audio: bool,
    seek_frac: f32,
    video_rotation: u16,

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

    is_scanning: Arc<Mutex<bool>>,
    scan_count: Arc<Mutex<usize>>,
    texture: Option<egui::TextureHandle>,
    error_msg: Option<String>,
    pending_initial_file: Option<PathBuf>,
}

impl PhotoViewer {
    fn new(_cc: &eframe::CreationContext<'_>, initial_file: Option<PathBuf>) -> Self {
        let audio_device = AudioDevice::new().expect("Failed to initialize audio device");
        Self {
            media_paths: Arc::new(Mutex::new(Vec::new())),
            current_media_path: None,
            current_image: None,
            video_player: None,
            audio_device,
            is_video: false,
            video_looping: true,
            video_volume: 0.10,
            video_has_audio: true,
            seek_frac: 0.0,
            video_rotation: 0,
            last_esc_press: None,
            history: Vec::new(),
            history_index: None,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            media_filter: MediaFilter::All,
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
            *scanning_clone.lock().unwrap() = false;
        });
    }

    fn reset_view(&mut self) {
        self.zoom = 1.0;
        self.pan = egui::Vec2::ZERO;
        self.video_rotation = 0;
    }

    fn go_next(&mut self, ctx: &egui::Context) {
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

        // Probe file for audio streams before opening player
        self.video_has_audio = file_has_audio_stream(&path);

        let path_str = path.to_string_lossy().to_string();
        match Player::new(ctx, &path_str) {
            Ok(player) => {
                match player.with_audio(&mut self.audio_device) {
                    Ok(mut player) => {
                        player.options.looping = self.video_looping;
                        if self.video_has_audio {
                            player.options.audio_volume.set(self.video_volume);
                        }
                        player.start();
                        self.video_player = Some(player);
                        self.is_video = true;
                        self.seek_frac = 0.0;
                        self.error_msg = None;
                    }
                    Err(e) => {
                        // Try without audio
                        println!("Audio init failed ({}), playing without audio", e);
                        let mut player = Player::new(ctx, &path_str).unwrap();
                        player.options.looping = self.video_looping;
                        player.start();
                        self.video_player = Some(player);
                        self.is_video = true;
                        self.video_has_audio = false;
                        self.seek_frac = 0.0;
                        self.error_msg = None;
                    }
                }
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

        let read_result = fs::read(&path).map_err(|e| e.to_string());

        let result = match read_result {
            Ok(mut bytes) => {
                match image::load_from_memory(&bytes) {
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
                }
            }
            Err(e) => Err(e),
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

        // Process video state
        if let Some(player) = &mut self.video_player {
            player.process_state();
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
                        if player.duration_ms > 0 {
                            let step = 3000.0 / player.duration_ms as f32;
                            let current = player.elapsed_ms() as f32 / player.duration_ms as f32;
                            let target = (current + step).clamp(0.0, 1.0);
                            player.seek(target);
                        }
                    }
                }
                if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
                    if let Some(player) = &mut self.video_player {
                        if player.duration_ms > 0 {
                            let step = 3000.0 / player.duration_ms as f32;
                            let current = player.elapsed_ms() as f32 / player.duration_ms as f32;
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
                    player.options.audio_volume.set(self.video_volume);
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
                self.video_volume = (self.video_volume - 0.02).clamp(0.0, 1.0);
                if let Some(player) = &mut self.video_player {
                    player.options.audio_volume.set(self.video_volume);
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
                // Space toggles play/pause for video
                if let Some(player) = &mut self.video_player {
                    match player.player_state.get() {
                        PlayerState::Playing => player.pause(),
                        PlayerState::Paused => player.resume(),
                        PlayerState::EndOfFile => {
                            player.seek(0.0);
                            player.resume();
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
                    // Reserve space for controls at bottom
                    let controls_height = 40.0;
                    let video_area = egui::vec2(available.x, available.y - controls_height);

                    // Click-to-pause and drag-to-pan on the video area (above controls)
                    let interact_rect = egui::Rect::from_min_size(rect.min, video_area);
                    let video_interact = ui.interact(interact_rect, ui.id().with("video_interact"), egui::Sense::click_and_drag());
                    if video_interact.dragged() {
                        self.pan += video_interact.drag_delta();
                    }
                    if video_interact.clicked() {
                        match player.player_state.get() {
                            PlayerState::Playing => player.pause(),
                            PlayerState::Paused => player.resume(),
                            PlayerState::EndOfFile => {
                                player.seek(0.0);
                                player.resume();
                            }
                            _ => {}
                        }
                    }

                    // Scale video to fit while maintaining aspect ratio
                    // For 90/270 rotation, swap the video dimensions for aspect ratio calc
                    let video_size = player.size;
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
                    let texture_id = player.texture_handle.id();
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

                    // Controls bar at bottom
                    let controls_rect = egui::Rect::from_min_size(
                        egui::pos2(rect.min.x, rect.max.y - controls_height),
                        egui::vec2(available.x, controls_height),
                    );

                    egui::Window::new("VideoControls")
                        .fixed_rect(controls_rect)
                        .title_bar(false)
                        .resizable(false)
                        .frame(egui::Frame::none().fill(egui::Color32::from_black_alpha(180)).inner_margin(4.0))
                        .show(ctx, |ui| {
                            ui.horizontal(|ui| {
                                // Play/Pause button
                                let state = player.player_state.get();
                                let btn_text = match state {
                                    PlayerState::Playing => "⏸",
                                    _ => "▶",
                                };
                                if ui.button(btn_text).clicked() {
                                    match state {
                                        PlayerState::Playing => player.pause(),
                                        PlayerState::Paused => player.resume(),
                                        PlayerState::EndOfFile => {
                                            player.seek(0.0);
                                            player.resume();
                                        }
                                        _ => player.start(),
                                    }
                                }

                                // Current time
                                let elapsed = player.elapsed_ms();
                                let duration = player.duration_ms;
                                ui.label(
                                    egui::RichText::new(Self::format_time(elapsed))
                                        .color(egui::Color32::WHITE)
                                        .monospace(),
                                );

                                // Seek bar
                                let mut seek_pos = if duration > 0 {
                                    elapsed as f32 / duration as f32
                                } else {
                                    0.0
                                };
                                let slider = egui::Slider::new(&mut seek_pos, 0.0..=1.0)
                                    .show_value(false)
                                    .trailing_fill(true);
                                let response = ui.add_sized(
                                    [ui.available_width() - 160.0, 20.0],
                                    slider,
                                );
                                if response.changed() {
                                    player.seek(seek_pos);
                                }
                                // Hover tooltip: show time at cursor position
                                if response.hovered() {
                                    if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                                        let frac = ((pos.x - response.rect.left()) / response.rect.width()).clamp(0.0, 1.0);
                                        let hover_ms = (frac * duration as f32) as i64;
                                        response.on_hover_text(Self::format_time(hover_ms));
                                    }
                                }

                                // Total time
                                ui.label(
                                    egui::RichText::new(Self::format_time(duration))
                                        .color(egui::Color32::WHITE)
                                        .monospace(),
                                );

                                // Loop toggle
                                let loop_text = if player.options.looping { "🔁" } else { "🔁" };
                                let loop_btn = ui.button(
                                    egui::RichText::new(loop_text).color(
                                        if player.options.looping {
                                            egui::Color32::from_rgb(100, 200, 255)
                                        } else {
                                            egui::Color32::GRAY
                                        }
                                    )
                                );
                                if loop_btn.on_hover_text("Toggle loop").clicked() {
                                    self.video_looping = !self.video_looping;
                                    player.options.looping = self.video_looping;
                                }

                                // Volume control or no-audio indicator
                                if self.video_has_audio {
                                    let vol_icon = if self.video_volume == 0.0 { "🔇" } else { "🔊" };
                                    ui.label(egui::RichText::new(vol_icon).color(egui::Color32::WHITE));
                                    let vol_slider = egui::Slider::new(&mut self.video_volume, 0.0..=1.0)
                                        .show_value(false)
                                        .trailing_fill(true);
                                    let vol_response = ui.add_sized([100.0, 20.0], vol_slider);
                                    if vol_response.changed() {
                                        player.options.audio_volume.set(self.video_volume);
                                    }
                                    vol_response.on_hover_text(format!("{}%", (self.video_volume * 100.0).round() as i32));
                                } else {
                                    ui.label(
                                        egui::RichText::new("🔇 No Audio")
                                            .color(egui::Color32::from_rgb(180, 180, 180)),
                                    );
                                }

                                // Fullscreen toggle
                                let is_fullscreen = ctx.input(|i| {
                                    i.viewport().fullscreen.unwrap_or(false)
                                });
                                let fs_icon = if is_fullscreen { "⊡" } else { "⊞" };
                                if ui.button(egui::RichText::new(fs_icon).color(egui::Color32::WHITE))
                                    .on_hover_text("Toggle fullscreen (F11)")
                                    .clicked()
                                {
                                    ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(!is_fullscreen));
                                }
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
                                 Scroll to Zoom  |  Drag to Pan\n\
                                 F11: Fullscreen  |  M: Filter (All / Images / Videos)\n\
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
