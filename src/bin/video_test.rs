//! Headless integration tests for `video_player::Player`.
//!
//! This binary spins up a **real wgpu device** (without a window or
//! display surface) and a minimal `egui_wgpu::Renderer` so the player's
//! direct-wgpu NV12 / RGBA8 / RGBA16 upload paths are actually
//! exercised. The earlier version of this test used the fallback
//! `egui::ColorImage` path, which meant bugs in the NV12/wgpu pipeline
//! slipped through.
//!
//! Run with:
//!
//!     cargo run --release --bin video_test -- "C:/path/to/video.mp4"
//!
//! If no argument is given, the binary searches common locations
//! (Downloads, Videos, Pictures) for the first .mp4/.mov/.mkv it can
//! find.
//!
//! The binary runs a series of scenarios against the same file and
//! exits with a non-zero code if any of them fail.

#[path = "../video_player.rs"]
mod video_player;

use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui;
use eframe::egui_wgpu;
use eframe::wgpu;

use video_player::{Player, PlayerState, WgpuBackend};

const PTS_TOLERANCE_US: i64 = 150_000; // ±150 ms tolerance for seeks.

fn find_test_video() -> Option<PathBuf> {
    if let Some(p) = env::args().nth(1) {
        let path = PathBuf::from(p);
        if path.exists() && path.is_file() {
            return Some(path);
        }
        eprintln!("video_test: arg path does not exist: {}", path.display());
    }
    for root in [
        "C:/Users/david/Downloads",
        "C:/Users/david/Videos",
        "C:/Users/david/Pictures",
    ] {
        if let Ok(entries) = std::fs::read_dir(root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if matches!(
                        ext.to_lowercase().as_str(),
                        "mp4" | "mov" | "mkv" | "webm" | "m4v"
                    ) {
                        return Some(path);
                    }
                }
            }
        }
    }
    None
}

/// Create a headless wgpu device + queue + renderer that mimics what
/// eframe's wgpu backend gives us at runtime, so the Player's NV12
/// path is actually exercised during tests.
fn make_headless_backend() -> (egui::Context, WgpuBackend) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("no wgpu adapter available");

    let supported_features = adapter.features();
    let want_16bit = wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
    let required_features = if supported_features.contains(want_16bit) {
        want_16bit
    } else {
        wgpu::Features::empty()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("video_test"),
            required_features,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .expect("wgpu device creation failed");
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let target_format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let renderer = egui_wgpu::Renderer::new(&device, target_format, None, 1, false);
    let renderer = Arc::new(egui::mutex::RwLock::new(renderer));
    let supports_16bit_norm = device.features().contains(want_16bit);
    let backend = WgpuBackend {
        device,
        queue,
        renderer,
        target_format,
        supports_16bit_norm,
    };
    (egui::Context::default(), backend)
}

#[derive(Clone)]
struct ScenarioResult {
    name: &'static str,
    pass: bool,
    details: String,
}

impl ScenarioResult {
    fn ok(name: &'static str, details: impl Into<String>) -> Self {
        Self {
            name,
            pass: true,
            details: details.into(),
        }
    }
    fn fail(name: &'static str, details: impl Into<String>) -> Self {
        Self {
            name,
            pass: false,
            details: details.into(),
        }
    }
}

/// Mimics what `eframe` actually does: only runs an "update cycle"
/// (tick() + a synthetic paint/run) when egui says it has a pending
/// repaint. This catches bugs where the player's worker or
/// `Player::seek` fails to request a repaint, which would leave the
/// GUI staring at a stale frame even though the test harness's own
/// `spin_and_collect` (which unconditionally ticks 60×/sec) would
/// be happy.
fn gated_spin_and_collect(
    ctx: &egui::Context,
    player: &mut Player,
    wall: Duration,
) -> Vec<i64> {
    let start = Instant::now();
    let mut seen: Vec<i64> = Vec::new();
    let mut last = i64::MIN;
    // Always run one initial tick so the first poll isn't empty.
    player.tick();
    {
        let pts = player.uploaded_pts_us_for_test();
        if pts != i64::MIN {
            seen.push(pts);
            last = pts;
        }
    }
    while start.elapsed() < wall {
        // eframe only runs `update()` again if egui has requested a
        // repaint (from last update, from an event, or from another
        // thread). If neither Player::seek nor the decoder worker
        // request one, the loop stalls and no new frame is shown.
        if ctx.has_requested_repaint() {
            player.tick();
            let pts = player.uploaded_pts_us_for_test();
            if pts != last && pts != i64::MIN {
                seen.push(pts);
                last = pts;
            }
        }
        std::thread::sleep(Duration::from_millis(4));
    }
    seen
}

/// Run `tick()` at ~60 Hz for `wall` duration, collecting every unique
/// uploaded pts the Player reports.
fn spin_and_collect(player: &mut Player, wall: Duration) -> Vec<i64> {
    let start = Instant::now();
    let mut seen: Vec<i64> = Vec::new();
    let mut last = i64::MIN;
    let mut ticks = 0u32;
    while start.elapsed() < wall {
        player.tick();
        ticks += 1;
        let pts = player.uploaded_pts_us_for_test();
        if pts != last && pts != i64::MIN {
            seen.push(pts);
            last = pts;
        }
        std::thread::sleep(Duration::from_millis(16));
    }
    eprintln!(
        "spin_and_collect: wall={}ms ticks={} seen={}",
        wall.as_millis(),
        ticks,
        seen.len()
    );
    seen
}

// ----- Scenarios -----

fn scenario_basic_playback(ctx: &egui::Context, backend: &WgpuBackend, path: &Path) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => return ScenarioResult::fail("basic_playback", format!("open: {}", e)),
        };
    player.set_volume(0.0);
    player.play();
    let seen = spin_and_collect(&mut player, Duration::from_secs(3));
    let elapsed = player.elapsed_ms();
    let distinct = seen.len();
    let final_pts_us = seen.last().copied().unwrap_or(i64::MIN);
    let details = format!(
        "distinct={} elapsed={}ms final_pts_us={} state={:?}",
        distinct,
        elapsed,
        final_pts_us,
        player.state()
    );
    if distinct >= 30 && final_pts_us >= 1_500_000 {
        ScenarioResult::ok("basic_playback", details)
    } else {
        ScenarioResult::fail("basic_playback", details)
    }
}

fn scenario_seek_while_playing(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => return ScenarioResult::fail("seek_while_playing", format!("open: {}", e)),
        };
    player.set_volume(0.0);
    player.play();
    let pre = spin_and_collect(&mut player, Duration::from_millis(600));
    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("seek_while_playing", "zero duration");
    }
    let target_frac = 0.5_f32;
    let target_us = (duration_us as f32 * target_frac) as i64;
    player.seek(target_frac);
    let seen = spin_and_collect(&mut player, Duration::from_millis(2500));
    let first_post = seen
        .iter()
        .find(|p| (**p - target_us).abs() < duration_us / 3)
        .copied();
    let sample: Vec<i64> = seen.iter().step_by((seen.len() / 10).max(1)).copied().collect();
    let details = format!(
        "target_us={} first_post_seek_pts={:?} pre_seen={} post_seen={} post_sample={:?}",
        target_us,
        first_post,
        pre.len(),
        seen.len(),
        sample
    );
    match first_post {
        Some(p) if (p - target_us).abs() <= PTS_TOLERANCE_US => {
            ScenarioResult::ok("seek_while_playing", details)
        }
        _ => ScenarioResult::fail("seek_while_playing", details),
    }
}

fn scenario_seek_while_paused(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => return ScenarioResult::fail("seek_while_paused", format!("open: {}", e)),
        };
    player.set_volume(0.0);
    // Do NOT call play(). Player is in Paused state after open.
    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("seek_while_paused", "zero duration");
    }
    let target_frac = 0.3_f32;
    let target_us = (duration_us as f32 * target_frac) as i64;
    player.seek(target_frac);
    // Give the priming phase enough wall time.
    let seen = spin_and_collect(&mut player, Duration::from_millis(2000));
    let final_pts = seen.last().copied().unwrap_or(i64::MIN);
    let details = format!(
        "target_us={} final_pts={} seen={}",
        target_us,
        final_pts,
        seen.len()
    );
    if final_pts != i64::MIN && (final_pts - target_us).abs() <= PTS_TOLERANCE_US {
        ScenarioResult::ok("seek_while_paused", details)
    } else {
        ScenarioResult::fail("seek_while_paused", details)
    }
}

/// Regression test: play for a moment, pause, seek, then resume. Real
/// GUI flow. Must land on the seek target without fast-forwarding,
/// and playback must actually resume (state transitions from Paused
/// to Playing, uploaded pts continues past the seek target).
fn scenario_play_pause_seek_resume(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail("play_pause_seek_resume", format!("open: {}", e))
            }
        };
    player.set_volume(0.0);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(500));
    player.pause();
    // Give the pause a moment to actually quiesce the audio stream.
    let _ = spin_and_collect(&mut player, Duration::from_millis(200));
    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("play_pause_seek_resume", "zero duration");
    }
    let target_frac = 0.5_f32;
    let target_us = (duration_us as f32 * target_frac) as i64;
    player.seek(target_frac);
    // Before resuming, verify the seek landed on a frame at/near the
    // target (not a fast-forward from before).
    let paused_post_seek = spin_and_collect(&mut player, Duration::from_millis(800));
    let landed = paused_post_seek
        .iter()
        .find(|p| (**p - target_us).abs() <= PTS_TOLERANCE_US)
        .copied();
    // Now resume.
    player.play();
    let resumed = spin_and_collect(&mut player, Duration::from_millis(1200));
    let final_pts = resumed.last().copied().unwrap_or(i64::MIN);
    let paused_sample: Vec<i64> = paused_post_seek.iter().copied().collect();
    let resumed_sample: Vec<i64> = resumed
        .iter()
        .step_by((resumed.len() / 10).max(1))
        .copied()
        .collect();
    let details = format!(
        "target_us={} landed={:?} paused_pts={:?} resumed_count={} resumed_sample={:?} final_pts={} state={:?}",
        target_us, landed, paused_sample, resumed.len(), resumed_sample, final_pts, player.state()
    );
    let landed_ok =
        landed.map(|p| (p - target_us).abs() <= PTS_TOLERANCE_US).unwrap_or(false);
    let resumed_ok = resumed.len() >= 10
        && final_pts > target_us + 200_000
        && matches!(player.state(), PlayerState::Playing);
    if landed_ok && resumed_ok {
        ScenarioResult::ok("play_pause_seek_resume", details)
    } else {
        ScenarioResult::fail("play_pause_seek_resume", details)
    }
}

/// Regression test for the frame-step keybinds (comma / period).
/// Pause, then call `step_frames(1)` repeatedly — each call should
/// advance `uploaded_pts` by roughly one frame interval, not stay
/// stuck on the starting frame.
fn scenario_paused_frame_step(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail("paused_frame_step", format!("open: {}", e))
            }
        };
    player.set_volume(0.0);
    player.play();
    // Warm up so the player is past the first-frame state.
    let _ = spin_and_collect(&mut player, Duration::from_millis(400));
    player.pause();
    let _ = spin_and_collect(&mut player, Duration::from_millis(100));

    // Seek to a known anchor so successive step-forward results
    // are predictable.
    player.seek(0.50);
    let _ = spin_and_collect(&mut player, Duration::from_millis(1500));
    let start_pts = player.uploaded_pts_us_for_test();

    // Hammer step-forward a lot — on long-GOP HEVC the first few
    // steps drain the buffer the decoder filled during priming, so
    // we also need the demuxer to keep topping up the queue during
    // pause for this to keep working past ~16 steps.
    let mut landed: Vec<i64> = vec![start_pts];
    for _ in 0..20 {
        player.step_frames(1);
        let _ = spin_and_collect(&mut player, Duration::from_millis(400));
        landed.push(player.uploaded_pts_us_for_test());
    }
    let unique: std::collections::BTreeSet<i64> = landed.iter().copied().collect();
    let details = format!("pts_sequence={:?}", landed);
    let monotonic = landed.windows(2).all(|w| w[1] >= w[0]);
    let all_distinct = unique.len() == landed.len();
    if monotonic && all_distinct {
        ScenarioResult::ok("paused_frame_step", details)
    } else {
        ScenarioResult::fail("paused_frame_step", details)
    }
}

/// Gated version of `paused_frame_step`: same check but only ticks
/// when egui has requested a repaint.
fn scenario_paused_frame_step_gated(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail(
                    "paused_frame_step_gated",
                    format!("open: {}", e),
                )
            }
        };
    player.set_volume(0.0);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(400));
    player.pause();
    let _ = spin_and_collect(&mut player, Duration::from_millis(100));

    player.seek(0.50);
    let _ = gated_spin_and_collect(ctx, &mut player, Duration::from_millis(1500));
    let start_pts = player.uploaded_pts_us_for_test();

    let mut landed: Vec<i64> = vec![start_pts];
    for _ in 0..20 {
        player.step_frames(1);
        let _ = gated_spin_and_collect(ctx, &mut player, Duration::from_millis(400));
        landed.push(player.uploaded_pts_us_for_test());
    }
    let unique: std::collections::BTreeSet<i64> = landed.iter().copied().collect();
    let details = format!("pts_sequence={:?}", landed);
    let monotonic = landed.windows(2).all(|w| w[1] >= w[0]);
    let all_distinct = unique.len() == landed.len();
    if monotonic && all_distinct {
        ScenarioResult::ok("paused_frame_step_gated", details)
    } else {
        ScenarioResult::fail("paused_frame_step_gated", details)
    }
}

fn scenario_rapid_seek(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => return ScenarioResult::fail("rapid_seek", format!("open: {}", e)),
        };
    player.set_volume(0.0);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(400));
    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("rapid_seek", "zero duration");
    }
    // Blast many seek commands back-to-back. The final one should win.
    for frac in [0.1, 0.2, 0.3, 0.5, 0.7, 0.4, 0.6] {
        player.seek(frac);
    }
    let final_target_us = (duration_us as f32 * 0.6_f32) as i64;
    let seen = spin_and_collect(&mut player, Duration::from_millis(2500));
    // The FIRST frame that lands in the final-target window is what
    // proves the seek coalesced correctly and landed accurately.
    // Later frames just reflect subsequent playback drift.
    let first_near_target = seen
        .iter()
        .find(|p| (**p - final_target_us).abs() < duration_us / 5)
        .copied();
    let sample: Vec<i64> = seen
        .iter()
        .step_by((seen.len() / 12).max(1))
        .copied()
        .collect();
    let first = seen.first().copied().unwrap_or(i64::MIN);
    let last = seen.last().copied().unwrap_or(i64::MIN);
    let details = format!(
        "final_target_us={} first_near_target={:?} seen={} first={} last={} sample={:?}",
        final_target_us, first_near_target, seen.len(), first, last, sample
    );
    match first_near_target {
        Some(p) if (p - final_target_us).abs() <= PTS_TOLERANCE_US => {
            ScenarioResult::ok("rapid_seek", details)
        }
        _ => ScenarioResult::fail("rapid_seek", details),
    }
}

/// Regression test for the GUI path: looping=true (as the GUI sets it
/// by default). On a short, fast-decoding file the demuxer races to
/// EOF in ~200 ms, and without the "wait for playback to catch up"
/// gate in the EOF handler the clock keeps getting reset to 0 and
/// playback appears frozen on the first frame.
fn scenario_looping_short_file(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => return ScenarioResult::fail("looping_short_file", format!("open: {}", e)),
        };
    player.set_volume(0.0);
    player.set_looping(true);
    player.play();
    let seen = spin_and_collect(&mut player, Duration::from_secs(3));
    let elapsed = player.elapsed_ms();
    let distinct = seen.len();
    let final_pts_us = seen.last().copied().unwrap_or(i64::MIN);
    let details = format!(
        "distinct={} elapsed={}ms final_pts_us={} state={:?}",
        distinct, elapsed, final_pts_us, player.state()
    );
    // Must make real forward progress even with looping enabled.
    if distinct >= 30 && final_pts_us >= 1_500_000 {
        ScenarioResult::ok("looping_short_file", details)
    } else {
        ScenarioResult::fail("looping_short_file", details)
    }
}

fn scenario_no_first_frame_stuck(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => return ScenarioResult::fail("no_first_frame_stuck", format!("open: {}", e)),
        };
    player.set_volume(0.0);
    player.play();
    let seen = spin_and_collect(&mut player, Duration::from_secs(2));
    let distinct = seen.len();
    let first = seen.first().copied().unwrap_or(i64::MIN);
    let last = seen.last().copied().unwrap_or(i64::MIN);
    let details = format!(
        "distinct={} first={} last={} span_ms={}",
        distinct,
        first,
        last,
        (last - first) / 1000
    );
    // Specifically a regression check for "plays first frame and then
    // nothing more": we must observe many distinct frames AND the span
    // between first and last uploaded pts must cover a meaningful
    // fraction of wall time.
    if distinct >= 30 && (last - first) >= 1_000_000 {
        ScenarioResult::ok("no_first_frame_stuck", details)
    } else {
        ScenarioResult::fail("no_first_frame_stuck", details)
    }
}

// ============================================================================
// Aggressive "VLC-parity" scenarios
// ============================================================================
//
// These scenarios simulate the user-reported bug: the display frame
// does not update when seeking (or frame-stepping) multiple times in
// rapid succession at random positions in the video. Unlike the
// earlier scenarios which seek to 3-5 known targets with generous
// settle windows, these fire 30+ operations in quick succession and
// verify that EVERY one of them resolves to a new, distinct frame at
// or near the requested target — exactly like VLC does.

/// Deterministic LCG so random-seek scenarios are reproducible across
/// runs. Mixes the 64-bit state with a standard MCG64 constant.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform float in [0.05, 0.95] so we stay inside the file.
    fn next_frac(&mut self) -> f32 {
        let r = self.next_u32() as f32 / (u32::MAX / 2) as f32;
        0.05 + (r * 0.45).clamp(0.0, 0.9)
    }
}

/// Wait up to `timeout` for the player's uploaded pts to land within
/// `PTS_TOLERANCE_US` of `target_us`. Returns (landed_pts, elapsed_ms).
/// Ticks the player ~every 4 ms so the check runs at ~60 Hz (the
/// actual tick budget egui would give the GUI).
fn wait_for_seek_target(
    player: &mut Player,
    target_us: i64,
    timeout: Duration,
) -> (i64, u64) {
    let start = Instant::now();
    while start.elapsed() < timeout {
        player.tick();
        let pts = player.uploaded_pts_us_for_test();
        if pts != i64::MIN && (pts - target_us).abs() <= PTS_TOLERANCE_US {
            return (pts, start.elapsed().as_millis() as u64);
        }
        std::thread::sleep(Duration::from_millis(4));
    }
    (player.uploaded_pts_us_for_test(), timeout.as_millis() as u64)
}

/// Wait up to `timeout` for `uploaded_pts` to change from `prev`. Used
/// for frame-step tests where we don't know the exact target pts.
/// Returns the new pts (or `prev` if nothing changed).
fn wait_for_new_pts(player: &mut Player, prev: i64, timeout: Duration) -> i64 {
    let start = Instant::now();
    while start.elapsed() < timeout {
        player.tick();
        let pts = player.uploaded_pts_us_for_test();
        if pts != prev && pts != i64::MIN {
            return pts;
        }
        std::thread::sleep(Duration::from_millis(4));
    }
    player.uploaded_pts_us_for_test()
}

/// VLC-parity: pause, then spam 40 random-position seeks with only a
/// short settle window between each. Every single seek must resolve
/// to a distinct frame at or near its target — if any one of them
/// leaves the display stuck, the test fails with a detailed report.
fn scenario_paused_random_spam_seek(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail(
                    "paused_random_spam_seek",
                    format!("open: {}", e),
                )
            }
        };
    player.set_volume(0.0);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(300));
    player.pause();
    let _ = spin_and_collect(&mut player, Duration::from_millis(100));

    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("paused_random_spam_seek", "zero duration");
    }

    let mut rng = Lcg::new(0xCAFEBABE);
    let mut failures: Vec<(usize, f32, i64, i64, u64)> = Vec::new();
    let mut landed: Vec<i64> = Vec::new();
    const N: usize = 40;
    for i in 0..N {
        let frac = rng.next_frac();
        let target_us = (duration_us as f32 * frac) as i64;
        player.seek(frac);
        let (pts, ms) = wait_for_seek_target(
            &mut player,
            target_us,
            Duration::from_millis(1500),
        );
        if pts == i64::MIN || (pts - target_us).abs() > PTS_TOLERANCE_US {
            failures.push((i, frac, target_us, pts, ms));
        } else {
            landed.push(pts);
        }
    }

    // Count how often a pts value repeats. Two random targets can
    // legitimately land on the same displayed frame when the
    // video's GOP structure puts multiple targets in the same
    // reachable-keyframe bucket — `landed == target ± tolerance`
    // is what the bug check cares about, not strict uniqueness.
    // A large dup cluster (one pts repeated many times) is what
    // indicates the "stuck frame" bug; a handful of coincidental
    // dups on random targets is fine.
    let unique: std::collections::BTreeSet<i64> = landed.iter().copied().collect();
    let mut max_repeat = 0usize;
    for u in &unique {
        let c = landed.iter().filter(|&&p| p == *u).count();
        if c > max_repeat {
            max_repeat = c;
        }
    }
    let details = format!(
        "landed_ok={}/{} unique_landed={} max_repeat={} first_failures={:?}",
        landed.len(),
        N,
        unique.len(),
        max_repeat,
        failures.iter().take(5).collect::<Vec<_>>()
    );
    // Accept up to a 3-way tie on random targets. A stuck-frame
    // bug would have max_repeat close to N (every seek lands on
    // the same stale pts).
    if failures.is_empty() && max_repeat <= 3 {
        ScenarioResult::ok("paused_random_spam_seek", details)
    } else {
        ScenarioResult::fail("paused_random_spam_seek", details)
    }
}

/// Same as `paused_random_spam_seek` but while playing. VLC would
/// resume playback from each new seek target without any frame
/// getting stuck.
fn scenario_playing_random_spam_seek(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail(
                    "playing_random_spam_seek",
                    format!("open: {}", e),
                )
            }
        };
    player.set_volume(0.0);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(300));

    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("playing_random_spam_seek", "zero duration");
    }

    let mut rng = Lcg::new(0xDEADBEEF);
    let mut failures: Vec<(usize, f32, i64, i64, u64)> = Vec::new();
    const N: usize = 30;
    for i in 0..N {
        let frac = rng.next_frac();
        let target_us = (duration_us as f32 * frac) as i64;
        player.seek(frac);
        let (pts, ms) = wait_for_seek_target(
            &mut player,
            target_us,
            Duration::from_millis(1500),
        );
        if pts == i64::MIN || (pts - target_us).abs() > PTS_TOLERANCE_US {
            failures.push((i, frac, target_us, pts, ms));
        }
    }

    let details = format!(
        "failures={}/{} first_failures={:?}",
        failures.len(),
        N,
        failures.iter().take(5).collect::<Vec<_>>()
    );
    if failures.is_empty() {
        ScenarioResult::ok("playing_random_spam_seek", details)
    } else {
        ScenarioResult::fail("playing_random_spam_seek", details)
    }
}

/// Burst seeks with zero wall time between them: simulates a user
/// mashing the left/right arrow keys before any decode catches up.
/// The LAST seek must always land — coalescing is allowed and
/// expected (VLC does the same), but the display must eventually
/// show the last requested frame.
fn scenario_burst_coalesce_seek(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail(
                    "burst_coalesce_seek",
                    format!("open: {}", e),
                )
            }
        };
    player.set_volume(0.0);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(300));
    player.pause();
    let _ = spin_and_collect(&mut player, Duration::from_millis(100));

    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("burst_coalesce_seek", "zero duration");
    }

    // Run a bunch of independent bursts. Each burst fires M seeks with
    // no wall time in between, then waits for the LAST target to land.
    // Fail loudly if any burst fails to surface its last target.
    let mut rng = Lcg::new(0x1234_5678_9ABC_DEF0);
    let bursts: [usize; 6] = [3, 5, 8, 2, 10, 4];
    let mut failed: Vec<(usize, f32, i64, i64)> = Vec::new();
    for (burst_idx, &m) in bursts.iter().enumerate() {
        let mut last_frac = 0.5f32;
        for _ in 0..m {
            last_frac = rng.next_frac();
            player.seek(last_frac);
            // NO sleep / NO tick between seeks — simulate key mashing.
        }
        let last_target = (duration_us as f32 * last_frac) as i64;
        let (pts, _ms) = wait_for_seek_target(
            &mut player,
            last_target,
            Duration::from_millis(2000),
        );
        if (pts - last_target).abs() > PTS_TOLERANCE_US {
            failed.push((burst_idx, last_frac, last_target, pts));
        }
    }

    let details = format!(
        "failed_bursts={}/{} detail={:?}",
        failed.len(),
        bursts.len(),
        failed
    );
    if failed.is_empty() {
        ScenarioResult::ok("burst_coalesce_seek", details)
    } else {
        ScenarioResult::fail("burst_coalesce_seek", details)
    }
}

/// Long run of frame-forward steps. Step 60 times and verify each
/// step produced a distinct, monotonically-advancing frame. If any
/// step silently repeats the previous frame, we reproduce the user's
/// "display doesn't update when I hold period" complaint.
fn scenario_frame_step_forward_long_run(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail(
                    "frame_step_forward_long_run",
                    format!("open: {}", e),
                )
            }
        };
    player.set_volume(0.0);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(300));
    player.pause();
    let _ = spin_and_collect(&mut player, Duration::from_millis(100));

    player.seek(0.30);
    let _ = spin_and_collect(&mut player, Duration::from_millis(1500));
    let start_pts = player.uploaded_pts_us_for_test();

    let mut pts_seq: Vec<i64> = vec![start_pts];
    let mut stuck: Vec<(usize, i64)> = Vec::new();
    const N: usize = 60;
    for i in 0..N {
        let prev = *pts_seq.last().unwrap();
        player.step_frames(1);
        let new = wait_for_new_pts(&mut player, prev, Duration::from_millis(1500));
        if new == prev {
            stuck.push((i, prev));
        }
        pts_seq.push(new);
    }

    let monotonic = pts_seq.windows(2).all(|w| w[1] >= w[0]);
    let unique: std::collections::BTreeSet<i64> = pts_seq.iter().copied().collect();
    let details = format!(
        "unique={}/{} monotonic={} stuck_at={:?} head={:?} tail={:?}",
        unique.len(),
        pts_seq.len(),
        monotonic,
        stuck.iter().take(5).collect::<Vec<_>>(),
        &pts_seq[..pts_seq.len().min(6)],
        &pts_seq[pts_seq.len().saturating_sub(6)..],
    );
    if stuck.is_empty() && monotonic && unique.len() == pts_seq.len() {
        ScenarioResult::ok("frame_step_forward_long_run", details)
    } else {
        ScenarioResult::fail("frame_step_forward_long_run", details)
    }
}

/// Long run of frame-backward steps. Exercises the
/// `step_frames(-1)` path, which always falls back to a full seek
/// because the decoded queue only grows forward.
fn scenario_frame_step_backward_long_run(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail(
                    "frame_step_backward_long_run",
                    format!("open: {}", e),
                )
            }
        };
    player.set_volume(0.0);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(300));
    player.pause();
    let _ = spin_and_collect(&mut player, Duration::from_millis(100));

    player.seek(0.70);
    let _ = spin_and_collect(&mut player, Duration::from_millis(1500));
    let start_pts = player.uploaded_pts_us_for_test();

    let mut pts_seq: Vec<i64> = vec![start_pts];
    let mut stuck: Vec<(usize, i64)> = Vec::new();
    const N: usize = 40;
    for i in 0..N {
        let prev = *pts_seq.last().unwrap();
        player.step_frames(-1);
        let new = wait_for_new_pts(&mut player, prev, Duration::from_millis(1500));
        if new == prev {
            stuck.push((i, prev));
        }
        pts_seq.push(new);
    }

    let monotonic_back = pts_seq.windows(2).all(|w| w[1] <= w[0]);
    let unique: std::collections::BTreeSet<i64> = pts_seq.iter().copied().collect();
    let details = format!(
        "unique={}/{} monotonic_back={} stuck_at={:?} head={:?} tail={:?}",
        unique.len(),
        pts_seq.len(),
        monotonic_back,
        stuck.iter().take(5).collect::<Vec<_>>(),
        &pts_seq[..pts_seq.len().min(6)],
        &pts_seq[pts_seq.len().saturating_sub(6)..],
    );
    if stuck.is_empty() && monotonic_back && unique.len() == pts_seq.len() {
        ScenarioResult::ok("frame_step_backward_long_run", details)
    } else {
        ScenarioResult::fail("frame_step_backward_long_run", details)
    }
}

/// Interleaved random seeks + frame steps. Simulates a user scrubbing
/// around the file and then fine-tuning with the comma/period keys.
fn scenario_mixed_random_seek_and_step(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail(
                    "mixed_random_seek_and_step",
                    format!("open: {}", e),
                )
            }
        };
    player.set_volume(0.0);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(300));
    player.pause();
    let _ = spin_and_collect(&mut player, Duration::from_millis(100));

    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("mixed_random_seek_and_step", "zero duration");
    }

    let mut rng = Lcg::new(0xF00D_FACE_CAFE_0001);
    #[derive(Debug)]
    enum Op {
        Seek(f32),
        StepF(i32),
        StepB(i32),
    }
    let mut ops: Vec<Op> = Vec::new();
    for _ in 0..25 {
        match rng.next_u32() % 3 {
            0 => ops.push(Op::Seek(rng.next_frac())),
            1 => ops.push(Op::StepF(1 + (rng.next_u32() % 3) as i32)),
            _ => ops.push(Op::StepB(1 + (rng.next_u32() % 3) as i32)),
        }
    }

    let mut last_pts = player.uploaded_pts_us_for_test();
    let mut failures: Vec<(usize, String, i64, i64)> = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        let prev = last_pts;
        match op {
            Op::Seek(f) => {
                let target = (duration_us as f32 * f) as i64;
                player.seek(*f);
                let (got, _) = wait_for_seek_target(
                    &mut player,
                    target,
                    Duration::from_millis(1500),
                );
                if (got - target).abs() > PTS_TOLERANCE_US {
                    failures.push((i, format!("{:?}", op), target, got));
                }
                last_pts = got;
            }
            Op::StepF(n) => {
                for _ in 0..*n {
                    player.step_frames(1);
                }
                let got = wait_for_new_pts(
                    &mut player,
                    prev,
                    Duration::from_millis(1500),
                );
                if got <= prev {
                    failures.push((i, format!("{:?}", op), prev, got));
                }
                last_pts = got;
            }
            Op::StepB(n) => {
                for _ in 0..*n {
                    player.step_frames(-1);
                }
                let got = wait_for_new_pts(
                    &mut player,
                    prev,
                    Duration::from_millis(1500),
                );
                if got >= prev {
                    failures.push((i, format!("{:?}", op), prev, got));
                }
                last_pts = got;
            }
        }
    }

    let details = format!(
        "failures={}/{} first_failures={:?}",
        failures.len(),
        ops.len(),
        failures.iter().take(5).collect::<Vec<_>>()
    );
    if failures.is_empty() {
        ScenarioResult::ok("mixed_random_seek_and_step", details)
    } else {
        ScenarioResult::fail("mixed_random_seek_and_step", details)
    }
}

/// Eframe-faithful simulation: drive the player through an explicit
/// `ctx.begin_pass()` + `ctx.end_pass()` cycle on every "frame", just
/// like real eframe does. Between passes, sleep exactly the amount
/// eframe would sleep based on egui's repaint-delay request. If the
/// player forgets to request a repaint at any point during a seek
/// storm, this scenario hangs waiting for the next pass — and we
/// fail the test.
fn scenario_eframe_gui_seek_storm(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail(
                    "eframe_gui_seek_storm",
                    format!("open: {}", e),
                )
            }
        };
    player.set_volume(0.0);

    // Helper: runs one "frame" of eframe: begin_pass → tick (GUI
    // would also process input and draw) → end_pass. Returns whether
    // the player bumped last_uploaded_pts_us during this pass.
    fn run_pass(ctx: &egui::Context, player: &mut Player) -> i64 {
        let raw_input = egui::RawInput::default();
        ctx.begin_pass(raw_input);
        player.tick();
        let pts = player.uploaded_pts_us_for_test();
        let _ = ctx.end_pass();
        pts
    }

    // Warm up: run a few "playing" passes so the decoder is
    // producing frames.
    player.play();
    for _ in 0..30 {
        let _ = run_pass(ctx, &mut player);
        std::thread::sleep(Duration::from_millis(16));
    }
    player.pause();
    for _ in 0..6 {
        let _ = run_pass(ctx, &mut player);
        std::thread::sleep(Duration::from_millis(16));
    }

    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("eframe_gui_seek_storm", "zero duration");
    }

    // Now simulate a user mashing seek keys: each "keypress" is its
    // own pass, where input handling (the seek) runs INSIDE the pass
    // (after tick, same as real GUI). After each seek pass we run up
    // to 40 more passes waiting for the target frame to land,
    // sleeping a fraction of a frame between each.
    let mut rng = Lcg::new(0xABCD_1234_5678_9ABC);
    let mut failures: Vec<(usize, f32, i64, i64)> = Vec::new();
    const N: usize = 30;
    for i in 0..N {
        let frac = rng.next_frac();
        let target = (duration_us as f32 * frac) as i64;

        // Pass 0: the "keypress" pass — tick first (simulating the
        // `player.tick()` call at the top of the GUI update()), then
        // seek (simulating the input handler), then end_pass.
        let raw_input = egui::RawInput::default();
        ctx.begin_pass(raw_input);
        player.tick();
        player.seek(frac);
        let _ = ctx.end_pass();

        // Passes 1..=120: wait for the seek to land. We only run a
        // pass when egui says it wants one (simulating eframe's own
        // schedule). Long-GOP codecs on HW decode can take ~1 s to
        // walk from a keyframe to a random target on some GPUs,
        // so the budget is ~2 s of wall time (120 × ~16 ms).
        let mut landed = player.uploaded_pts_us_for_test();
        for _ in 0..120 {
            std::thread::sleep(Duration::from_millis(16));
            if !ctx.has_requested_repaint() {
                continue;
            }
            let pts = run_pass(ctx, &mut player);
            if (pts - target).abs() <= PTS_TOLERANCE_US {
                landed = pts;
                break;
            }
            landed = pts;
        }
        if (landed - target).abs() > PTS_TOLERANCE_US {
            failures.push((i, frac, target, landed));
        }
    }

    let details = format!(
        "failures={}/{} first_failures={:?}",
        failures.len(),
        N,
        failures.iter().take(5).collect::<Vec<_>>()
    );
    if failures.is_empty() {
        ScenarioResult::ok("eframe_gui_seek_storm", details)
    } else {
        ScenarioResult::fail("eframe_gui_seek_storm", details)
    }
}

fn main() {
    let path = match find_test_video() {
        Some(p) => p,
        None => {
            eprintln!("video_test: no test video found. Pass a path as the first argument.");
            std::process::exit(2);
        }
    };
    eprintln!("video_test: using {}", path.display());

    let (ctx, backend) = make_headless_backend();
    eprintln!(
        "video_test: wgpu ready, supports_16bit_norm={}",
        backend.supports_16bit_norm
    );

    let scenarios: &[(
        &'static str,
        fn(&egui::Context, &WgpuBackend, &Path) -> ScenarioResult,
    )] = &[
        ("basic_playback", scenario_basic_playback),
        ("looping_short_file", scenario_looping_short_file),
        ("no_first_frame_stuck", scenario_no_first_frame_stuck),
        ("seek_while_playing", scenario_seek_while_playing),
        ("seek_while_paused", scenario_seek_while_paused),
        ("play_pause_seek_resume", scenario_play_pause_seek_resume),
        ("paused_frame_step", scenario_paused_frame_step),
        ("paused_frame_step_gated", scenario_paused_frame_step_gated),
        ("rapid_seek", scenario_rapid_seek),
        // VLC-parity aggressive scenarios — these exercise the
        // user-reported "display doesn't update after rapid seeks"
        // bug pattern.
        ("paused_random_spam_seek", scenario_paused_random_spam_seek),
        ("playing_random_spam_seek", scenario_playing_random_spam_seek),
        ("burst_coalesce_seek", scenario_burst_coalesce_seek),
        ("frame_step_forward_long_run", scenario_frame_step_forward_long_run),
        ("frame_step_backward_long_run", scenario_frame_step_backward_long_run),
        ("mixed_random_seek_and_step", scenario_mixed_random_seek_and_step),
        ("eframe_gui_seek_storm", scenario_eframe_gui_seek_storm),
    ];

    let mut results: Vec<ScenarioResult> = Vec::new();
    for (name, f) in scenarios {
        eprintln!("\n---- {} ----", name);
        let r = f(&ctx, &backend, &path);
        eprintln!("  [{}] {}", if r.pass { "PASS" } else { "FAIL" }, r.details);
        results.push(r);
    }

    eprintln!("\n===== SUMMARY =====");
    let mut pass_count = 0;
    let mut fail_count = 0;
    for r in &results {
        let tag = if r.pass { "PASS" } else { "FAIL" };
        eprintln!("  [{}] {}: {}", tag, r.name, r.details);
        if r.pass {
            pass_count += 1;
        } else {
            fail_count += 1;
        }
    }
    eprintln!(
        "\n{} scenario(s) passed, {} failed",
        pass_count, fail_count
    );
    if fail_count > 0 {
        std::process::exit(1);
    }
}

// Keep this alias around so `cargo` doesn't warn about unused imports
// from the shared `video_player` module that the scenarios don't all
// touch directly.
#[allow(dead_code)]
fn _type_touch(_s: PlayerState) {}
