//! Headless integration test for the video_player module.
//!
//! Opens a video file, plays it for N seconds, and reports:
//!   - whether Player::open succeeded
//!   - duration / size / has_audio the module reports
//!   - how many ticks happened
//!   - clock sample progression (looking for monotonic advance and resets)
//!   - whether any video frames actually landed in the queue
//!
//! No egui UI, no interaction. Run it like:
//!   cargo run --release --bin video_test -- "C:/path/to/video.mp4"
//!
//! With no argument, it searches common locations for a .mp4/.mov/.mkv.

#[path = "../video_player.rs"]
mod video_player;

use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use eframe::egui;

use video_player::Player;

fn find_test_video() -> Option<PathBuf> {
    if let Some(p) = env::args().nth(1) {
        let path = PathBuf::from(p);
        if path.exists() && path.is_file() {
            return Some(path);
        }
        eprintln!("video_test: arg path does not exist: {}", path.display());
    }
    let candidates = [
        "C:/Users/david/Downloads",
        "C:/Users/david/Videos",
        "C:/Users/david/Pictures",
    ];
    let exts = ["mp4", "mov", "mkv", "webm", "m4v"];
    for root in candidates {
        if let Ok(entries) = std::fs::read_dir(root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if exts.contains(&ext.to_lowercase().as_str()) {
                        return Some(path);
                    }
                }
            }
        }
    }
    None
}

fn main() {
    let path = match find_test_video() {
        Some(p) => p,
        None => {
            eprintln!("video_test: no test video found. Pass a path as the first argument.");
            std::process::exit(2);
        }
    };
    println!("video_test: using {}", path.display());

    // A default egui Context is enough — we never actually paint.
    let ctx = egui::Context::default();

    let open_start = Instant::now();
    let mut player = match Player::open(&ctx, &path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("video_test: Player::open failed: {}", e);
            std::process::exit(1);
        }
    };
    let open_dur = open_start.elapsed();
    let (w, h) = player.size();
    println!(
        "video_test: opened in {:?} — {}x{}, duration_ms={}, has_audio={}",
        open_dur,
        w,
        h,
        player.duration_ms(),
        player.has_audio()
    );

    player.play();
    println!("video_test: play() sent");

    let total_wall = Duration::from_secs(8);
    let start = Instant::now();
    let mut tick_count: u64 = 0;
    let mut frames_seen: u64 = 0;
    let mut last_uploaded_pts: i64 = i64::MIN;
    let mut clock_samples: Vec<(u128, i64)> = Vec::new();

    while start.elapsed() < total_wall {
        player.tick();
        tick_count += 1;
        let clock = player.elapsed_ms();
        let uploaded = player.uploaded_pts_us_for_test();
        if uploaded != last_uploaded_pts && uploaded != i64::MIN {
            frames_seen += 1;
            last_uploaded_pts = uploaded;
        }
        if tick_count.is_multiple_of(100) {
            clock_samples.push((start.elapsed().as_millis(), clock));
        }
        // Tight loop — no sleep — so `frames_seen / wall_secs` reflects
        // the actual pipeline throughput, not our tick cadence.
    }

    let wall_secs = start.elapsed().as_secs_f64();
    let effective_fps = frames_seen as f64 / wall_secs;

    println!();
    println!("=== RESULTS ===");
    println!("tick_count: {}", tick_count);
    println!("frames uploaded to texture: {}", frames_seen);
    println!("effective fps: {:.1}", effective_fps);
    println!("final elapsed_ms: {}", player.elapsed_ms());
    println!("final state: {:?}", player.state());
    println!("clock samples (wall_ms, clock_ms):");
    for (wall, clock) in clock_samples.iter().take(30) {
        println!("  wall={:>5}  clock={:>6}", wall, clock);
    }
    let mut resets = 0usize;
    let mut max_seen = i64::MIN;
    for (_, c) in &clock_samples {
        if *c < max_seen - 100 {
            resets += 1;
        }
        if *c > max_seen {
            max_seen = *c;
        }
    }
    println!("monotonic resets observed: {}", resets);
    println!("max clock reached: {} ms", max_seen);

    let duration_ms = player.duration_ms();
    let verdict_clock = max_seen >= 2000;
    let verdict_frames = frames_seen >= 30;
    let verdict_no_resets = resets <= 1;
    println!();
    println!(
        "clock_advanced_past_2s: {}{}",
        verdict_clock,
        if verdict_clock { " ✓" } else { " ✗" }
    );
    println!(
        "uploaded_at_least_30_frames: {}{}",
        verdict_frames,
        if verdict_frames { " ✓" } else { " ✗" }
    );
    println!(
        "no_monotonic_regressions: {}{}",
        verdict_no_resets,
        if verdict_no_resets { " ✓" } else { " ✗" }
    );

    if verdict_clock && verdict_frames && verdict_no_resets {
        println!("\nPASS");
        std::process::exit(0);
    } else {
        println!(
            "\nFAIL (duration was {} ms, wall {}s)",
            duration_ms,
            total_wall.as_secs()
        );
        std::process::exit(1);
    }
}
