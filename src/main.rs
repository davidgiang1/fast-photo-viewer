use eframe::egui;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::process::Command;
use std::fs;
use std::collections::HashSet;
use walkdir::WalkDir;
use rand::seq::SliceRandom;
use image::DynamicImage;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_title("Fast Photo Viewer"),
        ..Default::default()
    };
    
    eframe::run_native(
        "Fast Photo Viewer",
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Box::new(PhotoViewer::new(cc)) as Box<dyn eframe::App>
        }),
    )
}

struct PhotoViewer {
    image_paths: Arc<Mutex<Vec<PathBuf>>>,
    current_image_path: Option<PathBuf>,
    current_image: Option<DynamicImage>,
    
    // History
    history: Vec<PathBuf>,
    history_index: Option<usize>,

    // View State
    zoom: f32,
    pan: egui::Vec2,

    is_scanning: Arc<Mutex<bool>>,
    scan_count: Arc<Mutex<usize>>,
    texture: Option<egui::TextureHandle>,
    error_msg: Option<String>,
}

impl PhotoViewer {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            image_paths: Arc::new(Mutex::new(Vec::new())),
            current_image_path: None,
            current_image: None,
            history: Vec::new(),
            history_index: None,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            is_scanning: Arc::new(Mutex::new(false)),
            scan_count: Arc::new(Mutex::new(0)),
            texture: None,
            error_msg: None,
        }
    }

    fn open_directory(&mut self) {
        if let Some(path) = rfd::FileDialog::new().pick_folder() {
            self.start_scan(path);
        }
    }

    fn open_file_dialog(&mut self, ctx: &egui::Context) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Images", &["jpg", "jpeg", "png", "bmp", "webp", "gif", "tiff", "ico", "svg"])
            .pick_file() 
        {
            if let Some(parent) = path.parent() {
                self.start_scan(parent.to_path_buf());
            }
            
            // Set as current and load immediately (after start_scan clears everything)
            self.current_image_path = Some(path.clone());
            self.history.push(path.clone());
            self.history_index = Some(0);
            self.load_image(path, ctx);
        }
    }

    fn start_scan(&mut self, directory: PathBuf) {
        {
            let mut paths = self.image_paths.lock().unwrap();
            paths.clear();
        }
        self.history.clear();
        self.history_index = None;
        self.current_image_path = None;
        self.texture = None;
        self.reset_view();

        *self.scan_count.lock().unwrap() = 0;
        *self.is_scanning.lock().unwrap() = true;

        let paths_clone = self.image_paths.clone();
        let scanning_clone = self.is_scanning.clone();
        let count_clone = self.scan_count.clone();

        thread::spawn(move || {
            let valid_extensions = ["jpg", "jpeg", "png", "bmp", "webp", "gif"];
            
            for entry in WalkDir::new(directory).into_iter().filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                        if valid_extensions.contains(&ext.to_lowercase().as_str()) {
                            let mut p = paths_clone.lock().unwrap();
                            p.push(path.to_path_buf());
                            *count_clone.lock().unwrap() += 1;
                        }
                    }
                }
            }
            *scanning_clone.lock().unwrap() = false;
        });
    }

    fn reset_view(&mut self) {
        self.zoom = 1.0;
        self.pan = egui::Vec2::ZERO;
    }

    fn go_next(&mut self, ctx: &egui::Context) {
        if let Some(idx) = self.history_index {
            if idx + 1 < self.history.len() {
                self.history_index = Some(idx + 1);
                let path = self.history[idx + 1].clone();
                self.current_image_path = Some(path.clone());
                self.reset_view();
                self.load_image(path, ctx);
                return;
            }
        }
        self.next_random_image(ctx);
    }

    fn go_prev(&mut self, ctx: &egui::Context) {
         if let Some(idx) = self.history_index {
            if idx > 0 {
                self.history_index = Some(idx - 1);
                let path = self.history[idx - 1].clone();
                self.current_image_path = Some(path.clone());
                self.reset_view();
                self.load_image(path, ctx);
            }
        }
    }

    fn next_random_image(&mut self, ctx: &egui::Context) {
        let path = {
            let paths = self.image_paths.lock().unwrap();
            if paths.is_empty() {
                return;
            }
            
            // Optimization: Use HashSet for fast lookups
            let history_set: HashSet<&PathBuf> = self.history.iter().collect();
            
            let available: Vec<&PathBuf> = paths.iter()
                .filter(|p| !history_set.contains(p))
                .collect();
            
            let mut rng = rand::thread_rng();
            
            if !available.is_empty() {
                 available.choose(&mut rng).map(|p| (*p).clone())
            } else {
                 // All images have been shown. Fall back to picking any random image.
                 // We try to pick one that isn't the current image to avoid immediate repetition.
                 if paths.len() > 1 {
                    paths.iter()
                         .filter(|p| Some(*p) != self.current_image_path.as_ref())
                         .collect::<Vec<_>>()
                         .choose(&mut rng)
                         .map(|p| (*p).clone())
                 } else {
                    paths.choose(&mut rng).cloned()
                }
            }
        };

        if let Some(p) = path {
            self.history.push(p.clone());
            self.history_index = Some(self.history.len() - 1);
            
            self.current_image_path = Some(p.clone());
            self.reset_view();
            self.load_image(p, ctx);
        }
    }

    fn open_in_explorer(&self) {
        if let Some(path) = &self.current_image_path {
            #[cfg(target_os = "windows")]
            {
                Command::new("explorer")
                    .args(["/select,", &path.to_string_lossy()])
                    .spawn()
                    .ok();
            }
            #[cfg(not(target_os = "windows"))]
            {
                 // Fallback
            }
        }
    }

    fn load_image(&mut self, path: PathBuf, ctx: &egui::Context) {
        // Read file bytes first to handle potential format mismatches
        let read_result = fs::read(&path).map_err(|e| e.to_string());

        let result = match read_result {
            Ok(mut bytes) => {
                // Try loading normally
                match image::load_from_memory(&bytes) {
                    Ok(img) => Ok(img),
                    Err(e) => {
                        // Check if it's a likely JPEG (starts with FF D8)
                        if bytes.len() > 2 && bytes[0] == 0xFF && bytes[1] == 0xD8 {
                             println!("Attempting to repair truncated JPEG: {}", path.display());
                             // Append standard JPEG EOI marker
                             bytes.push(0xFF);
                             bytes.push(0xD9);
                             // Also add some padding for safety
                             bytes.extend(std::iter::repeat(0).take(1024));
                             
                             image::load_from_memory(&bytes).map_err(|retry_err| {
                                 format!("Original: {}, Retry: {}", e, retry_err)
                             })
                        } else {
                             // Not a JPEG or repair not applicable, return original error
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
                println!("{}", msg); // Log to console for debugging
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
}

impl eframe::App for PhotoViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle Global Input
        if ctx.input(|i| i.key_pressed(egui::Key::Space) || i.key_pressed(egui::Key::ArrowRight)) {
            self.go_next(ctx);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
            self.go_prev(ctx);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::O)) {
            self.open_directory();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F)) {
            self.open_file_dialog(ctx);
        }

        // Handle Zoom (Mouse Wheel)
        let scroll = ctx.input(|i| i.raw_scroll_delta);
        if scroll.y != 0.0 {
            let zoom_factor = if scroll.y > 0.0 { 1.1 } else { 0.9 };
            self.zoom *= zoom_factor;
            self.zoom = self.zoom.clamp(0.1, 50.0);
        }

        // Handle Pan (Mouse Drag)
        // We check this generally; if the user drags on the image, we pan.
        // We'll capture this in the central panel response.

        egui::CentralPanel::default().show(ctx, |ui| {
            let rect = ui.available_rect_before_wrap();
            ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);

            // Handle panning input on the whole panel
            let response = ui.interact(rect, ui.id().with("pan_drag"), egui::Sense::drag());
            if response.dragged() {
                self.pan += response.drag_delta();
            }

            if let Some(texture) = &self.texture {
                let available_size = rect.size();
                let original_size = texture.size_vec2();
                
                if original_size.x > 0.0 && original_size.y > 0.0 {
                    // 1. Calculate base scale to "fit" the window
                    let width_ratio = available_size.x / original_size.x;
                    let height_ratio = available_size.y / original_size.y;
                    let base_scale = width_ratio.min(height_ratio);
                    
                    // 2. Apply user zoom
                    let display_size = original_size * base_scale * self.zoom;
                    
                    // 3. Calculate centered position + user pan
                    let center_x = rect.min.x + available_size.x / 2.0;
                    let center_y = rect.min.y + available_size.y / 2.0;
                    
                    let image_rect = egui::Rect::from_center_size(
                        egui::pos2(center_x, center_y) + self.pan, 
                        display_size
                    );

                    // 4. Draw
                    ui.painter().image(
                        texture.id(),
                        image_rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE
                    );
                }
            } else {
                 ui.centered_and_justified(|ui| {
                    ui.label(egui::RichText::new("Press 'O' to Open Folder\nPress 'F' to Open File\nSpace/Right: Next | Left: Previous\nScroll to Zoom | Drag to Pan")
                        .color(egui::Color32::WHITE)
                        .size(20.0));
                });
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
                                ui.label(format!("Scanning... Found {} images", count));
                            });
                            ctx.request_repaint(); 
                        } else {
                            ui.label(format!("Total: {} images", count));
                            if let Some(path) = &self.current_image_path {
                                ui.label(path.file_name().unwrap_or_default().to_string_lossy());
                                if let Some(idx) = self.history_index {
                                    ui.label(format!("History: {}/{}", idx + 1, self.history.len()));
                                }
                            }
                        }
                        if let Some(err) = &self.error_msg {
                            ui.colored_label(egui::Color32::RED, err);
                        }
                    });
            }
            
            // Controls (Explorer Button)
            if self.current_image_path.is_some() {
                 egui::Window::new("Controls")
                    .anchor(egui::Align2::RIGHT_BOTTOM, [-10.0, -10.0])
                    .title_bar(false)
                    .resizable(false)
                    .auto_sized()
                    .frame(egui::Frame::popup(ui.style()).multiply_with_opacity(0.8)) 
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            if ui.button("⟲").on_hover_text("Rotate Left").clicked() {
                                self.rotate_ccw(ctx);
                            }
                            if ui.button("⟳").on_hover_text("Rotate Right").clicked() {
                                self.rotate_cw(ctx);
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
