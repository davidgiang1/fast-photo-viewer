//! In-house video player built directly on ffmpeg-the-third and cpal.
//!
//! Replaces the egui-video dependency so we own the decode pipeline, AV
//! sync, and can later plug in hardware acceleration.
//!
//! Architecture:
//!   - `open()` spawns a decoder thread that opens the file, sets up
//!     video+audio decoders, and loops on `Packet::read`. Each decoded
//!     video frame is converted to RGBA and pushed into a bounded
//!     PTS-ordered queue. Each decoded audio frame is resampled to
//!     interleaved stereo f32 @ 48 kHz and pushed into a ring buffer.
//!   - A cpal output stream consumes the audio ring buffer. The stream
//!     advances a shared `audio_clock_us` atomic — this is the master
//!     clock against which video frames are scheduled. Videos without
//!     audio advance the clock via wall time.
//!   - `tick()` runs on the egui thread every frame, picks the newest
//!     queued video frame whose PTS ≤ audio_clock, drops older ones,
//!     and uploads to an egui texture.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;
use ffmpeg_the_third as ffmpeg;
use ffmpeg::ffi::{
    av_buffer_ref, av_buffer_unref, av_frame_unref, av_hwdevice_ctx_create,
    av_hwframe_transfer_data, av_seek_frame, avformat_flush, avio_seek, AVBufferRef,
    AVCodecContext, AVHWDeviceType, AVPixelFormat, AVSEEK_FLAG_ANY, AVSEEK_FLAG_BACKWARD,
};
use ringbuf::{
    traits::{Consumer, Producer, Split},
    HeapRb,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlayerState {
    Loading,
    Playing,
    Paused,
    EndOfFile,
    Error,
}

enum VideoFramePayload {
    /// 8-bit sRGB RGBA, stored as a pre-built ColorImage so the main
    /// thread can hand it directly to egui in the fallback path.
    Rgba8Srgb {
        width: u32,
        height: u32,
        image: egui::ColorImage,
    },
    /// 16-bit linear RGBA (u16 per channel little-endian), matching
    /// wgpu's `Rgba16Unorm` byte layout on x86. Used only on the
    /// direct-wgpu path and only when the source is 10-bit.
    Rgba16Unorm {
        width: u32,
        height: u32,
        bytes: Vec<u8>,
    },
    /// NV12: planar Y (full resolution) + interleaved UV (half
    /// resolution). Produced when HW-decoded frames arrive as NV12
    /// after `av_hwframe_transfer_data`. Skipping swscale saves a
    /// CPU YUV→RGB pass; GPU does the color conversion in a shader.
    Nv12 {
        width: u32,
        height: u32,
        y_plane: Vec<u8>,
        uv_plane: Vec<u8>,
    },
}

struct VideoFrame {
    pts_us: i64,
    payload: VideoFramePayload,
}

enum Command {
    Play,
    Pause,
    Seek(f32),
    SetLooping(bool),
    Stop,
}

/// Message type routed from the demuxer into the video or audio
/// decoder threads. `Flush` marks a discontinuity (seek or loop wrap)
/// — the receiving thread should flush its decoder and discard any
/// queued output.
enum DecodeMsg {
    /// Encoded packet, tagged with the demuxer's `flush_seq` at the
    /// moment it was routed. Workers compare the tag against the
    /// current `Shared::flush_seq` and discard messages whose tag
    /// is stale — this lets a mid-seek worker skip decoding the
    /// entire backlog of pre-seek packets almost instantly instead
    /// of paying one full decode per stale packet.
    Packet(ffmpeg::Packet, u64),
    Flush,
}

struct SharedState {
    player_state: PlayerState,
    duration_us: i64,
    width: u32,
    height: u32,
    has_audio: bool,
    /// Average frame interval in microseconds, computed from the
    /// video stream's reported frame rate. Used by the GUI for
    /// frame-step keybinds (comma / period) and as a pacing hint.
    frame_interval_us: i64,
    error: Option<String>,
}

struct Shared {
    state: Mutex<SharedState>,
    video_queue: Mutex<VecDeque<VideoFrame>>,
    /// Master playback clock in microseconds. Driven by the audio output
    /// callback when audio is present, otherwise advanced by the decoder
    /// thread from wall time.
    audio_clock_us: AtomicI64,
    /// Volume as f32 bits (0.0..=1.0) applied by the cpal callback.
    volume_bits: AtomicU32,
    /// Set to true briefly after a seek so the audio callback can decide
    /// not to advance the clock past what's been flushed.
    clock_frozen: std::sync::atomic::AtomicBool,
    /// Signals the cpal callback to discard everything currently in
    /// the audio ring buffer and start fresh. Set by the demuxer on
    /// seek or loop-wraparound so stale buffered audio stops
    /// immediately instead of continuing until the ring drains.
    audio_drain_flag: std::sync::atomic::AtomicBool,
    /// Raised by the demuxer on seek and cleared by the video worker
    /// when it processes the subsequent `DecodeMsg::Flush`. While
    /// raised, the worker drops any frames it decodes instead of
    /// pushing them into the shared queue — this prevents frames
    /// that were buffered in the ffmpeg decoder from before the seek
    /// from flickering through as "fast-forward" before the new
    /// position takes effect.
    flush_pending: std::sync::atomic::AtomicBool,
    /// Minimum pts (microseconds) that the video worker is allowed
    /// to push into the display queue. Set by the demuxer to the
    /// seek target so the codec's pre-target warmup frames (from the
    /// keyframe backward-seek) get decoded but not displayed.
    min_display_pts_us: AtomicI64,
    /// Incremented every time the video worker successfully pushes a
    /// post-seek frame (i.e., a frame that passed the
    /// `min_display_pts_us` gate). The demuxer uses this to know
    /// when it can stop priming packets after a seek and return to
    /// its normal paused state.
    post_seek_frames_pushed: std::sync::atomic::AtomicU32,
    /// Set on Player drop. Worker threads check this in their
    /// back-pressure wait so `decode_thread.join()` can return
    /// even when the shared video queue is full.
    stopping: std::sync::atomic::AtomicBool,
    /// Monotonic flush counter. The demuxer increments this when it
    /// initiates a seek; the workers compare it against the last
    /// value they saw and, on mismatch, call `decoder.flush()` plus
    /// clear the display queue. Replaces the in-channel `Flush`
    /// message so seek handling never blocks on a full channel.
    flush_seq: std::sync::atomic::AtomicU64,
}

/// Render backend passed to `Player::open`. When `Wgpu` is supplied,
/// the player manages its own `wgpu::Texture` and uploads decoded
/// frames directly via `queue.write_texture`, bypassing egui's
/// `ColorImage` / `TextureHandle` path. When `None`, the player falls
/// back to egui's texture manager (used by the headless test binary
/// which doesn't run a real wgpu backend).
#[derive(Clone)]
pub struct WgpuBackend {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub renderer: Arc<egui::mutex::RwLock<egui_wgpu::Renderer>>,
    pub target_format: wgpu::TextureFormat,
    /// True when the wgpu device was created with the
    /// `TEXTURE_FORMAT_16BIT_NORM` feature, which gates use of
    /// `Rgba16Unorm` textures for our 10-bit source path.
    pub supports_16bit_norm: bool,
}

pub struct Player {
    shared: Arc<Shared>,
    cmd_tx: Sender<Command>,
    decode_thread: Option<thread::JoinHandle<()>>,
    _audio_stream: Option<cpal::Stream>,
    // Egui-managed fallback path (test binary uses this).
    texture: Option<egui::TextureHandle>,
    // Direct wgpu path — preferred when available.
    wgpu: Option<WgpuBackend>,
    wgpu_texture: Option<wgpu::Texture>,
    wgpu_texture_id: Option<egui::TextureId>,
    wgpu_texture_size: (u32, u32),
    wgpu_texture_format: Option<wgpu::TextureFormat>,
    // NV12 direct-upload pipeline (partial Phase E)
    yuv_renderer: Option<YuvRenderer>,
    nv12_state: Option<Nv12GpuState>,
    // Caches the intrinsic video frame size so the GUI can lay out
    // the video area correctly even when the current GPU texture is
    // a planar YUV pair without an associated `TextureId`.
    intrinsic_size: (u32, u32),
    /// When set, the most recent uploaded frame went through the
    /// NV12 path — the main thread should render via
    /// `draw_into_painter` instead of `painter.image(texture_id())`.
    last_upload_was_nv12: bool,
    egui_ctx: egui::Context,
    last_uploaded_pts_us: i64,
    /// Wall-clock fallback: when the file has no audio, we advance the
    /// shared clock ourselves based on elapsed real time since playback
    /// started.
    no_audio_clock_origin: Option<Instant>,
    no_audio_clock_base_us: i64,
    playing_cached: bool,
}

#[allow(dead_code)]
impl Player {
    pub fn open(ctx: &egui::Context, path: &Path) -> Result<Self, String> {
        Self::open_with_backend(ctx, path, None)
    }

    pub fn open_with_backend(
        ctx: &egui::Context,
        path: &Path,
        wgpu: Option<WgpuBackend>,
    ) -> Result<Self, String> {
        // Query cpal for the output device's native rate/channels up
        // front. We resample the decoded audio directly to that rate so
        // the playback stream doesn't need any further conversion.
        let host = cpal::default_host();
        let (target_rate, target_channels) = match host.default_output_device() {
            Some(dev) => match dev.default_output_config() {
                Ok(cfg) => (cfg.sample_rate().0, cfg.channels() as usize),
                Err(_) => (48_000u32, 2usize),
            },
            None => (48_000u32, 2usize),
        };

        let shared = Arc::new(Shared {
            state: Mutex::new(SharedState {
                player_state: PlayerState::Loading,
                duration_us: 0,
                width: 0,
                height: 0,
                has_audio: false,
                frame_interval_us: 33_333,
                error: None,
            }),
            video_queue: Mutex::new(VecDeque::new()),
            audio_clock_us: AtomicI64::new(0),
            volume_bits: AtomicU32::new(1.0f32.to_bits()),
            clock_frozen: std::sync::atomic::AtomicBool::new(false),
            audio_drain_flag: std::sync::atomic::AtomicBool::new(false),
            flush_pending: std::sync::atomic::AtomicBool::new(false),
            min_display_pts_us: AtomicI64::new(i64::MIN),
            post_seek_frames_pushed: std::sync::atomic::AtomicU32::new(0),
            stopping: std::sync::atomic::AtomicBool::new(false),
            flush_seq: std::sync::atomic::AtomicU64::new(0),
        });

        // Audio ring buffer: 4 seconds worth of stereo f32 at the
        // output device's native rate. Generous headroom so a brief
        // scheduling hiccup doesn't drain it.
        let rb = HeapRb::<f32>::new(target_rate as usize * 2 * 4);
        let (audio_producer, audio_consumer) = rb.split();

        let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();

        let high_bit_depth_enabled = wgpu
            .as_ref()
            .map(|b| b.supports_16bit_norm)
            .unwrap_or(false);
        let shared_for_thread = shared.clone();
        let ctx_for_thread = ctx.clone();
        let path_for_thread = path.to_path_buf();
        let decode_thread = thread::spawn(move || {
            if let Err(e) = run_decode_pipeline(
                path_for_thread,
                cmd_rx,
                shared_for_thread.clone(),
                audio_producer,
                ctx_for_thread,
                target_rate,
                high_bit_depth_enabled,
            ) {
                let mut state = shared_for_thread.state.lock().unwrap();
                state.player_state = PlayerState::Error;
                state.error = Some(e);
            }
        });

        // Wait up to 5 s for the decoder to finish init (duration/size
        // populated and state transitioned out of Loading).
        let start = Instant::now();
        loop {
            {
                let state = shared.state.lock().unwrap();
                if state.player_state != PlayerState::Loading {
                    if let Some(e) = &state.error {
                        return Err(e.clone());
                    }
                    break;
                }
            }
            if start.elapsed() > Duration::from_secs(5) {
                return Err("timeout waiting for decoder initialization".into());
            }
            thread::sleep(Duration::from_millis(5));
        }

        let has_audio = shared.state.lock().unwrap().has_audio;
        let audio_stream = if has_audio {
            setup_audio_stream(audio_consumer, shared.clone(), target_rate, target_channels).ok()
        } else {
            None
        };

        Ok(Self {
            shared,
            cmd_tx,
            decode_thread: Some(decode_thread),
            _audio_stream: audio_stream,
            texture: None,
            yuv_renderer: wgpu
                .as_ref()
                .map(|b| YuvRenderer::new(&b.device, b.target_format)),
            nv12_state: None,
            intrinsic_size: (0, 0),
            last_upload_was_nv12: false,
            wgpu,
            wgpu_texture: None,
            wgpu_texture_id: None,
            wgpu_texture_size: (0, 0),
            wgpu_texture_format: None,
            egui_ctx: ctx.clone(),
            last_uploaded_pts_us: i64::MIN,
            no_audio_clock_origin: None,
            no_audio_clock_base_us: 0,
            playing_cached: false,
        })
    }

    pub fn play(&mut self) {
        let _ = self.cmd_tx.send(Command::Play);
        self.playing_cached = true;
        if let Some(stream) = &self._audio_stream {
            let _ = stream.play();
        }
        if !self.has_audio() {
            self.no_audio_clock_origin = Some(Instant::now());
            self.no_audio_clock_base_us = self.shared.audio_clock_us.load(Ordering::Relaxed);
        }
    }

    pub fn pause(&mut self) {
        let _ = self.cmd_tx.send(Command::Pause);
        self.playing_cached = false;
        if let Some(stream) = &self._audio_stream {
            let _ = stream.pause();
        }
        if !self.has_audio() {
            // Freeze the clock at its current value.
            if let Some(origin) = self.no_audio_clock_origin.take() {
                let now_us = self.no_audio_clock_base_us
                    + origin.elapsed().as_micros() as i64;
                self.shared.audio_clock_us.store(now_us, Ordering::Relaxed);
            }
        }
    }

    pub fn seek(&mut self, fraction: f32) {
        // Lock the display pipeline synchronously before the demuxer
        // even sees the command: clear the queue so any pre-seek
        // frames still buffered there vanish, set `min_display_pts_us`
        // to MAX so tick() can't accidentally display a stale frame
        // while the demuxer is still processing, and raise
        // `clock_frozen` so the priming path is the only one that
        // can surface the next rendered frame. The demuxer's seek
        // handler will overwrite `min_display_pts_us` with the real
        // target shortly afterwards.
        self.shared
            .min_display_pts_us
            .store(i64::MAX, Ordering::Relaxed);
        self.shared.video_queue.lock().unwrap().clear();
        self.shared
            .clock_frozen
            .store(true, Ordering::Relaxed);
        let _ = self.cmd_tx.send(Command::Seek(fraction.clamp(0.0, 1.0)));
        // Briefly wait for the demuxer to actually pick up the new
        // seek target (it overwrites `min_display_pts_us` from MAX
        // to the real target inside its seek handler). Without this
        // short handoff, subsequent tick() calls on the main thread
        // can race with the demuxer and see stale state, which has
        // been observed as the GUI not updating after back-to-back
        // seeks / frame-step keypresses.
        let deadline = Instant::now() + Duration::from_millis(30);
        while Instant::now() < deadline {
            if self
                .shared
                .min_display_pts_us
                .load(Ordering::Relaxed)
                != i64::MAX
            {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        // Make sure egui schedules another update cycle so that
        // when the decoder worker pushes the post-seek frame shortly
        // afterwards, tick() actually runs and surfaces it.
        self.egui_ctx.request_repaint();
        if !self.has_audio() {
            let duration = self.duration_ms() * 1000;
            let target = (duration as f32 * fraction.clamp(0.0, 1.0)) as i64;
            self.shared.audio_clock_us.store(target, Ordering::Relaxed);
            self.no_audio_clock_origin = if self.playing_cached {
                Some(Instant::now())
            } else {
                None
            };
            self.no_audio_clock_base_us = target;
        }
        // clock_frozen is cleared by the decoder thread after the seek.
    }

    pub fn set_volume(&mut self, v: f32) {
        self.shared
            .volume_bits
            .store(v.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }

    pub fn set_looping(&mut self, looping: bool) {
        let _ = self.cmd_tx.send(Command::SetLooping(looping));
    }

    pub fn state(&self) -> PlayerState {
        self.shared.state.lock().unwrap().player_state
    }

    pub fn duration_ms(&self) -> i64 {
        self.shared.state.lock().unwrap().duration_us / 1000
    }

    pub fn elapsed_ms(&self) -> i64 {
        self.current_clock_us() / 1000
    }

    /// Average frame interval in microseconds, derived from the
    /// source video's `avg_frame_rate`. Used by the GUI to implement
    /// frame-step seeking (comma / period keys).
    pub fn frame_interval_us(&self) -> i64 {
        self.shared.state.lock().unwrap().frame_interval_us.max(1_000)
    }

    /// Step by `delta` frames. Forward steps preferentially consume
    /// frames the worker has already decoded and buffered past the
    /// current display position — this avoids the cost of calling
    /// `av_seek_frame` back to the enclosing keyframe and
    /// re-decoding ~a GOP's worth of content, which for long-GOP
    /// HEVC can take hundreds of ms per step and feel frozen.
    /// Backward steps always fall back to a real seek.
    pub fn step_frames(&mut self, delta: i32) {
        let duration_us = self.shared.state.lock().unwrap().duration_us;
        if duration_us <= 0 {
            return;
        }

        if delta > 0 {
            // Try to advance through the already-decoded queue
            // first. If there's a buffered frame immediately after
            // the currently displayed one, just show it. No seek,
            // no decode, no stall.
            let mut queue = self.shared.video_queue.lock().unwrap();
            let cutoff = self.last_uploaded_pts_us;
            // Drop anything at/before the current display pts so
            // the next pop yields the following frame.
            while let Some(front) = queue.front() {
                if front.pts_us <= cutoff {
                    queue.pop_front();
                } else {
                    break;
                }
            }
            if let Some(frame) = queue.pop_front() {
                drop(queue);
                self.display_frame(frame);
                return;
            }
            // Queue didn't have anything buffered — fall through
            // to a real seek below.
            drop(queue);
        }

        let interval = self.frame_interval_us();
        let current_us = self.current_clock_us();
        if delta < 0 {
            // Backward step: "first frame ≥ target" semantics can
            // round back to `current` when the `target_us =
            // current - interval` value falls between actual
            // frames (variable frame rate / avg_frame_rate
            // mismatch), producing a visible oscillation. Seek to
            // a point safely several frames back, then pick the
            // LATEST decoded frame strictly before `current_us`.
            let back = (interval.max(16_000)) * (-delta as i64) * 4;
            let target_us = (current_us - back).max(0);
            let frac = (target_us as f32 / duration_us as f32).clamp(0.0, 1.0);
            self.seek(frac);
            // Wait briefly for the worker to fill the queue with
            // post-seek frames, then pop the latest frame whose
            // pts is strictly less than `current_us`. The 800 ms
            // deadline covers long-GOP HW decode from a keyframe
            // to the seek target.
            let cutoff = current_us;
            let deadline = Instant::now() + Duration::from_millis(800);
            while Instant::now() < deadline {
                let mut queue = self.shared.video_queue.lock().unwrap();
                let mut latest_back: Option<VideoFrame> = None;
                while let Some(front) = queue.front() {
                    if front.pts_us < cutoff {
                        latest_back = queue.pop_front();
                    } else {
                        break;
                    }
                }
                if let Some(frame) = latest_back {
                    drop(queue);
                    self.display_frame(frame);
                    return;
                }
                drop(queue);
                thread::sleep(Duration::from_millis(8));
            }
            return;
        }

        let target_us =
            (current_us + interval * delta as i64).clamp(0, duration_us - interval);
        let frac = (target_us as f32 / duration_us as f32).clamp(0.0, 1.0);
        self.seek(frac);
    }

    /// Upload `frame` to the active texture, update the bookkeeping
    /// atoms so subsequent ticks/seeks see a consistent post-step
    /// state, and request a repaint. Shared between the forward- and
    /// backward-step paths.
    fn display_frame(&mut self, frame: VideoFrame) {
        let (w, h) = match &frame.payload {
            VideoFramePayload::Rgba8Srgb { width, height, .. }
            | VideoFramePayload::Rgba16Unorm { width, height, .. }
            | VideoFramePayload::Nv12 { width, height, .. } => (*width, *height),
        };
        self.intrinsic_size = (w, h);
        let new_pts = frame.pts_us;
        if self.wgpu.is_some() {
            self.upload_wgpu(frame.payload);
        } else if let VideoFramePayload::Rgba8Srgb { image, .. } = frame.payload {
            self.upload_egui(image);
            self.last_upload_was_nv12 = false;
        }
        self.last_uploaded_pts_us = new_pts;
        self.shared
            .audio_clock_us
            .store(new_pts, Ordering::Relaxed);
        self.shared
            .clock_frozen
            .store(false, Ordering::Relaxed);
        self.egui_ctx.request_repaint();
    }

    pub fn size(&self) -> (u32, u32) {
        let s = self.shared.state.lock().unwrap();
        (s.width, s.height)
    }

    pub fn has_audio(&self) -> bool {
        self.shared.state.lock().unwrap().has_audio
    }

    /// True while the player is in post-seek priming, i.e. the
    /// display clock is frozen waiting for the first post-seek frame
    /// to land. The GUI uses this to keep requesting repaints while
    /// paused so the target frame actually renders.
    pub fn is_seeking(&self) -> bool {
        self.shared
            .clock_frozen
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn texture(&self) -> Option<&egui::TextureHandle> {
        self.texture.as_ref()
    }

    /// Returns the egui `TextureId` that should be drawn for the
    /// current video frame, regardless of whether it's backed by an
    /// egui `TextureHandle` or a directly-managed `wgpu::Texture`.
    pub fn texture_id(&self) -> Option<egui::TextureId> {
        if let Some(id) = self.wgpu_texture_id {
            return Some(id);
        }
        self.texture.as_ref().map(|t| t.id())
    }

    pub fn error(&self) -> Option<String> {
        self.shared.state.lock().unwrap().error.clone()
    }

    /// Test hook: last video frame PTS uploaded to the texture. Used
    /// by the `video_test` binary to detect progression without having
    /// to inspect the texture handle.
    pub fn uploaded_pts_us_for_test(&self) -> i64 {
        self.last_uploaded_pts_us
    }

    /// Called from the egui update loop every frame. Advances the
    /// no-audio clock, picks the newest video frame whose PTS ≤ the
    /// current clock, and uploads it to a texture.
    pub fn tick(&mut self) {
        // No-audio wall-clock advance.
        if !self.has_audio() && self.playing_cached {
            if let Some(origin) = self.no_audio_clock_origin {
                let now_us =
                    self.no_audio_clock_base_us + origin.elapsed().as_micros() as i64;
                self.shared.audio_clock_us.store(now_us, Ordering::Relaxed);
            }
        }

        let clock_frozen = self
            .shared
            .clock_frozen
            .load(Ordering::Relaxed);
        let clock_us = self.current_clock_us();

        let min_pts = self
            .shared
            .min_display_pts_us
            .load(Ordering::Relaxed);

        // While paused (and not in post-seek warmup), the display
        // has already been chosen by the last seek / step_frames
        // call. Don't drain the demuxer-filled lookahead queue
        // here: that lookahead exists so the next step-forward
        // can reuse it, and popping it eagerly would "scroll
        // forward" through buffered frames — which on high-fps
        // content makes step_frames(-1) appear to jump forward
        // because tick()'s ±15 ms tolerance sweeps up the next
        // frame every time the clock unfreezes.
        if !self.playing_cached && !clock_frozen {
            return;
        }
        let best = {
            let mut queue = self.shared.video_queue.lock().unwrap();
            if clock_frozen {
                // Post-seek warmup: the clock is intentionally frozen
                // at the seek target until the first decoded frame
                // arrives. Pop the first frame at or past the seek
                // target, discarding any stale frames that leaked
                // through the flush-race window.
                let mut chosen: Option<VideoFrame> = None;
                while let Some(front) = queue.front() {
                    if front.pts_us >= min_pts {
                        chosen = queue.pop_front();
                        break;
                    } else {
                        let _ = queue.pop_front();
                    }
                }
                chosen
            } else {
                let mut best: Option<VideoFrame> = None;
                while let Some(front) = queue.front() {
                    // Also honour min_pts here in case a stale frame
                    // leaked through after a seek but before the
                    // worker cleared the queue on its Flush handler.
                    if front.pts_us < min_pts {
                        let _ = queue.pop_front();
                        continue;
                    }
                    if front.pts_us <= clock_us + 15_000 {
                        best = queue.pop_front();
                    } else {
                        break;
                    }
                }
                best
            }
        };

        if let Some(frame) = best {
            if frame.pts_us != self.last_uploaded_pts_us {
                // If the clock is frozen (post-seek warmup), unfreeze
                // it and re-anchor to this frame's pts so playback
                // resumes from exactly this frame instead of jumping
                // forward by the decoder-warmup latency.
                if self
                    .shared
                    .clock_frozen
                    .load(Ordering::Relaxed)
                {
                    self.shared
                        .audio_clock_us
                        .store(frame.pts_us, Ordering::Relaxed);
                    self.shared
                        .clock_frozen
                        .store(false, Ordering::Relaxed);
                    // For the no-audio wall-clock path, also reset
                    // the wall-clock origin so it advances from here.
                    if !self.has_audio() && self.playing_cached {
                        self.no_audio_clock_origin = Some(Instant::now());
                        self.no_audio_clock_base_us = frame.pts_us;
                    }
                }
                let (w, h) = match &frame.payload {
                    VideoFramePayload::Rgba8Srgb { width, height, .. }
                    | VideoFramePayload::Rgba16Unorm { width, height, .. }
                    | VideoFramePayload::Nv12 { width, height, .. } => (*width, *height),
                };
                self.intrinsic_size = (w, h);
                if self.wgpu.is_some() {
                    self.upload_wgpu(frame.payload);
                } else {
                    match frame.payload {
                        VideoFramePayload::Rgba8Srgb { image, .. } => {
                            self.upload_egui(image);
                            self.last_upload_was_nv12 = false;
                        }
                        VideoFramePayload::Rgba16Unorm { .. }
                        | VideoFramePayload::Nv12 { .. } => {
                            // These variants never reach us without a
                            // wgpu backend — the decode thread gates
                            // on features that require one.
                        }
                    }
                }
                self.last_uploaded_pts_us = frame.pts_us;
            }
        }
    }

    /// True when the most recent frame was uploaded via the NV12
    /// direct path; the renderer must draw it with a custom
    /// PaintCallback because egui's image path can't sample planar
    /// YUV textures.
    pub fn is_nv12_path(&self) -> bool {
        self.last_upload_was_nv12
    }

    pub fn intrinsic_size(&self) -> (u32, u32) {
        self.intrinsic_size
    }

    fn upload_egui(&mut self, image: egui::ColorImage) {
        let same_size = self
            .texture
            .as_ref()
            .map(|t| t.size() == image.size)
            .unwrap_or(false);
        if same_size {
            if let Some(t) = self.texture.as_mut() {
                t.set(image, egui::TextureOptions::LINEAR);
            }
        } else {
            self.texture = Some(self.egui_ctx.load_texture(
                "video_frame",
                image,
                egui::TextureOptions::LINEAR,
            ));
        }
    }

    fn upload_wgpu(&mut self, payload: VideoFramePayload) {
        let backend = match self.wgpu.as_ref() {
            Some(b) => b.clone(),
            None => return,
        };

        // NV12 path is entirely separate: two textures, bind group,
        // custom render pipeline.
        if let VideoFramePayload::Nv12 {
            width,
            height,
            y_plane,
            uv_plane,
        } = payload
        {
            self.upload_wgpu_nv12(&backend, width, height, &y_plane, &uv_plane);
            self.last_upload_was_nv12 = true;
            return;
        }
        self.last_upload_was_nv12 = false;
        self.nv12_state = None;

        let (width, height, wgpu_format, bytes_per_pixel): (u32, u32, wgpu::TextureFormat, u32) =
            match &payload {
                VideoFramePayload::Rgba8Srgb { width, height, .. } => (
                    *width,
                    *height,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    4,
                ),
                VideoFramePayload::Rgba16Unorm { width, height, .. } => (
                    *width,
                    *height,
                    wgpu::TextureFormat::Rgba16Unorm,
                    8,
                ),
                VideoFramePayload::Nv12 { .. } => unreachable!("handled above"),
            };

        // (Re)create the wgpu texture on the first frame, when the
        // aspect ratio changes, or when the pixel format changes
        // (e.g. switching from an 8-bit to a 10-bit video).
        let needs_new_texture = self.wgpu_texture.is_none()
            || self.wgpu_texture_size != (width, height)
            || self.wgpu_texture_format != Some(wgpu_format);
        if needs_new_texture {
            let tex = backend.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("video_player_frame"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

            let mut renderer = backend.renderer.write();
            let new_id = if let Some(existing_id) = self.wgpu_texture_id {
                renderer.update_egui_texture_from_wgpu_texture(
                    &backend.device,
                    &view,
                    wgpu::FilterMode::Linear,
                    existing_id,
                );
                existing_id
            } else {
                renderer.register_native_texture(
                    &backend.device,
                    &view,
                    wgpu::FilterMode::Linear,
                )
            };
            drop(renderer);
            self.wgpu_texture = Some(tex);
            self.wgpu_texture_id = Some(new_id);
            self.wgpu_texture_size = (width, height);
            self.wgpu_texture_format = Some(wgpu_format);
        }

        // Reinterpret the frame payload as a flat byte slice and
        // write it straight into the wgpu texture.
        let bytes: &[u8] = match &payload {
            VideoFramePayload::Rgba8Srgb { image, .. } => unsafe {
                std::slice::from_raw_parts(
                    image.pixels.as_ptr() as *const u8,
                    image.pixels.len() * 4,
                )
            },
            VideoFramePayload::Rgba16Unorm { bytes, .. } => bytes.as_slice(),
            VideoFramePayload::Nv12 { .. } => unreachable!("handled above"),
        };
        if let Some(tex) = self.wgpu_texture.as_ref() {
            backend.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytes,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(width * bytes_per_pixel),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    fn current_clock_us(&self) -> i64 {
        self.shared.audio_clock_us.load(Ordering::Relaxed)
    }

    fn upload_wgpu_nv12(
        &mut self,
        backend: &WgpuBackend,
        width: u32,
        height: u32,
        y_plane: &[u8],
        uv_plane: &[u8],
    ) {
        let renderer = match self.yuv_renderer.as_ref() {
            Some(r) => r.clone(),
            None => return,
        };
        let uv_h = (height + 1) / 2;

        // (Re)create state when the size changes.
        let needs_new = match &self.nv12_state {
            Some(s) => s.width != width || s.height != height,
            None => true,
        };
        if needs_new {
            let y_texture = backend.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("nv12_y_texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let uv_texture = backend.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("nv12_uv_texture"),
                size: wgpu::Extent3d {
                    width: width / 2,
                    height: uv_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rg8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let y_view = y_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let uv_view = uv_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Uniform buffer holding the NDC rect (left, top, right,
            // bottom) for the vertex shader. Initialised to a
            // full-NDC quad so the first frame draws even before
            // `set_nv12_ndc_rect` is called.
            let ndc_buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nv12_ndc_uniform"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let initial_ndc: [f32; 4] = [-1.0, 1.0, 1.0, -1.0];
            let initial_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    initial_ndc.as_ptr() as *const u8,
                    std::mem::size_of::<[f32; 4]>(),
                )
            };
            backend.queue.write_buffer(&ndc_buffer, 0, initial_bytes);

            let bind_group = Arc::new(backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("nv12_bind_group"),
                layout: &renderer.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&y_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&uv_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&renderer.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: ndc_buffer.as_entire_binding(),
                    },
                ],
            }));

            self.nv12_state = Some(Nv12GpuState {
                y_texture,
                uv_texture,
                ndc_buffer,
                bind_group,
                width,
                height,
            });
        }
        let state = match self.nv12_state.as_ref() {
            Some(s) => s,
            None => return,
        };

        // Upload Y plane.
        backend.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &state.y_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            y_plane,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        // Upload UV plane.
        backend.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &state.uv_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            uv_plane,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width), // Rg8 × (width/2) = width bytes
                rows_per_image: Some(uv_h),
            },
            wgpu::Extent3d {
                width: width / 2,
                height: uv_h,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Emit an `egui_wgpu::Callback` that draws the current NV12
    /// video into `video_rect`. egui_wgpu sets the wgpu viewport to
    /// this rect (in pixels) before the callback runs, so our shader
    /// just emits a full NDC quad.
    pub fn nv12_paint_callback(
        &self,
        video_rect: egui::Rect,
    ) -> Option<egui::epaint::PaintCallback> {
        let renderer = self.yuv_renderer.as_ref()?;
        let state = self.nv12_state.as_ref()?;
        let cb = Nv12PaintCallback {
            pipeline: renderer.pipeline.clone(),
            bind_group: state.bind_group.clone(),
        };
        Some(egui_wgpu::Callback::new_paint_callback(video_rect, cb))
    }

    /// Update the NV12 vertex shader's quad rect for the next frame.
    /// Caller passes the unclamped video rect (in egui points) and the
    /// caller's own viewport metrics. We compute the equivalent
    /// quad-in-NDC inside the egui_wgpu-clamped viewport so the video
    /// translates correctly when panning pushes the rect off-screen
    /// (the rasterizer clips the over-extended quad via the scissor).
    pub fn set_nv12_render_rect(
        &self,
        video_rect: egui::Rect,
        screen_size_points: egui::Vec2,
        pixels_per_point: f32,
    ) {
        let backend = match self.wgpu.as_ref() {
            Some(b) => b,
            None => return,
        };
        let state = match self.nv12_state.as_ref() {
            Some(s) => s,
            None => return,
        };
        let scr_w = screen_size_points.x * pixels_per_point;
        let scr_h = screen_size_points.y * pixels_per_point;
        let rect_l = video_rect.min.x * pixels_per_point;
        let rect_t = video_rect.min.y * pixels_per_point;
        let rect_r = video_rect.max.x * pixels_per_point;
        let rect_b = video_rect.max.y * pixels_per_point;
        // Replicate epaint::ViewportInPixels::from_points clamping.
        let vp_l = rect_l.max(0.0).min(scr_w);
        let vp_t = rect_t.max(0.0).min(scr_h);
        let vp_r = rect_r.max(vp_l).min(scr_w);
        let vp_b = rect_b.max(vp_t).min(scr_h);
        let vp_w = (vp_r - vp_l).max(1.0);
        let vp_h = (vp_b - vp_t).max(1.0);
        // NDC X: -1 at viewport left, +1 at viewport right.
        let ndc_l = (rect_l - vp_l) / vp_w * 2.0 - 1.0;
        let ndc_r = (rect_r - vp_l) / vp_w * 2.0 - 1.0;
        // NDC Y: +1 at top, -1 at bottom (wgpu).
        let ndc_t = 1.0 - (rect_t - vp_t) / vp_h * 2.0;
        let ndc_b = 1.0 - (rect_b - vp_t) / vp_h * 2.0;
        let data: [f32; 4] = [ndc_l, ndc_t, ndc_r, ndc_b];
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                std::mem::size_of::<[f32; 4]>(),
            )
        };
        backend.queue.write_buffer(&state.ndc_buffer, 0, bytes);
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        self.shared.stopping.store(true, Ordering::Relaxed);
        let _ = self.cmd_tx.send(Command::Stop);
        if let Some(stream) = &self._audio_stream {
            let _ = stream.pause();
        }
        // Wait briefly for the decode thread to exit cleanly, but
        // don't block forever: if it's wedged (e.g. in ffmpeg I/O),
        // we'd rather leak the thread than freeze the GUI shutdown.
        // The OS will clean up when the process exits.
        if let Some(h) = self.decode_thread.take() {
            let deadline = Instant::now() + Duration::from_millis(300);
            while Instant::now() < deadline {
                if h.is_finished() {
                    let _ = h.join();
                    return;
                }
                thread::sleep(Duration::from_millis(5));
            }
            // Timed out — detach. Handle drop leaves the thread
            // running, but since Player::drop is only called when
            // the owning window is closing, that's fine.
            std::mem::drop(h);
        }
    }
}

// =========================================================================
// Audio output (cpal)
// =========================================================================

type AudioConsumer = <HeapRb<f32> as Split>::Cons;

fn setup_audio_stream(
    mut consumer: AudioConsumer,
    shared: Arc<Shared>,
    target_rate: u32,
    target_channels: usize,
) -> Result<cpal::Stream, String> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| "no default audio output device".to_string())?;
    let config = device
        .default_output_config()
        .map_err(|e| format!("audio default config: {}", e))?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    // Sanity: the caller queried the same device just before us, so
    // these should match. If they don't (device changed mid-init), we
    // trust cpal's current numbers and hope resampling is close.
    let _ = target_rate;
    let _ = target_channels;

    let us_per_frame = 1_000_000.0_f64 / sample_rate as f64;

    let shared_cb = shared.clone();
    let err_fn = |err| eprintln!("cpal stream error: {}", err);

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device
            .build_output_stream(
                &config.into(),
                move |data: &mut [f32], _| {
                    let volume = f32::from_bits(shared_cb.volume_bits.load(Ordering::Relaxed));
                    let frozen = shared_cb.clock_frozen.load(Ordering::Relaxed);
                    // Drain the ring buffer if a seek just happened,
                    // so we don't continue playing stale audio from
                    // before the seek target.
                    if shared_cb.audio_drain_flag.swap(false, Ordering::Relaxed) {
                        while consumer.try_pop().is_some() {}
                    }
                    let mut played_frames = 0usize;
                    // data is interleaved `channels` samples per frame.
                    let frames = data.len() / channels.max(1);
                    for frame in 0..frames {
                        // Our resampler outputs stereo; if cpal wants
                        // more channels, duplicate; if mono, take L.
                        let l = consumer.try_pop().unwrap_or(0.0);
                        let r = consumer.try_pop().unwrap_or(l);
                        for ch in 0..channels {
                            let sample = if ch == 0 { l } else if ch == 1 { r } else { (l + r) * 0.5 };
                            data[frame * channels + ch] = sample * volume;
                        }
                        played_frames += 1;
                    }
                    if !frozen && played_frames > 0 {
                        let advance_us = (played_frames as f64 * us_per_frame) as i64;
                        shared_cb
                            .audio_clock_us
                            .fetch_add(advance_us, Ordering::Relaxed);
                    }
                },
                err_fn,
                None,
            )
            .map_err(|e| format!("cpal build stream: {}", e))?,
        other => return Err(format!("unsupported cpal sample format: {:?}", other)),
    };

    // Do NOT call stream.play() here — the stream starts paused so
    // the Player sits silent until Player::play() explicitly resumes
    // it. Without this, cpal immediately starts advancing the audio
    // clock even though the Player is logically in the Paused state.
    Ok(stream)
}

// =========================================================================
// Decode pipeline: demuxer thread + video decode thread + audio decode thread
// =========================================================================
//
// The demuxer owns the format input and is responsible for reading
// packets and routing them to the appropriate decoder thread via
// unbounded channels. Each decoder thread runs independently, so a
// slow video frame cannot starve audio playback. On seek or loop
// wraparound, the demuxer calls ictx.seek, clears shared state, and
// sends a Flush marker into each channel so the decoder threads flush
// their internal state and resume cleanly.

fn run_decode_pipeline(
    path: PathBuf,
    cmd_rx: Receiver<Command>,
    shared: Arc<Shared>,
    audio_producer: <HeapRb<f32> as Split>::Prod,
    egui_ctx: egui::Context,
    target_rate: u32,
    high_bit_depth_enabled: bool,
) -> Result<(), String> {
    let mut ictx =
        ffmpeg::format::input(&path).map_err(|e| format!("open: {}", e))?;
    let duration_us = ictx.duration();

    // --- Video stream ---
    let (video_stream_index, video_time_base, video_frame_interval_us) = {
        let stream = ictx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| "no video stream".to_string())?;
        let rate = stream.avg_frame_rate();
        let frame_us = if rate.denominator() > 0 && rate.numerator() > 0 {
            1_000_000_i64 * rate.denominator() as i64 / rate.numerator() as i64
        } else {
            33_333
        };
        (stream.index(), stream.time_base(), frame_us)
    };

    // Re-enabled now that decode threads are split: HW transfer lives
    // on the video thread and can no longer starve the audio path.
    const HW_ACCEL: bool = true;

    let (video_decoder, hw_enabled) = {
        let stream = ictx
            .stream(video_stream_index)
            .ok_or_else(|| "stream disappeared".to_string())?;
        let mut ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| format!("video codec ctx: {}", e))?;
        // Enable multi-threaded decoding — 0 means "use one thread per
        // available core". Big win for software H.264/HEVC decode on
        // multi-core machines. Harmless when HW decode is active.
        ctx.set_threading(ffmpeg::codec::threading::Config {
            kind: ffmpeg::codec::threading::Type::Frame,
            count: 0,
        });
        let hw = if HW_ACCEL {
            unsafe { try_enable_hw_accel(ctx.as_mut_ptr()) }
        } else {
            false
        };
        let dec = ctx
            .decoder()
            .video()
            .map_err(|e| format!("video decoder: {}", e))?;
        (dec, hw)
    };
    let _ = hw_enabled;
    let video_src_w = video_decoder.width();
    let video_src_h = video_decoder.height();

    // --- Audio stream (optional) ---
    let audio_info = ictx
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .map(|s| s.index());

    let (audio_decoder, audio_resampler, audio_stream_index): (
        Option<ffmpeg::decoder::Audio>,
        Option<ffmpeg::software::resampling::context::Context>,
        Option<usize>,
    ) = if let Some(idx) = audio_info {
        let stream = ictx.stream(idx).unwrap();
        let ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| format!("audio codec ctx: {}", e))?;
        let dec = ctx
            .decoder()
            .audio()
            .map_err(|e| format!("audio decoder: {}", e))?;
        let in_format = dec.format();
        let in_rate = dec.rate();
        let in_layout = dec.ch_layout();
        let resampler = ffmpeg::software::resampling::context::Context::get2(
            in_format,
            in_layout,
            in_rate,
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Planar),
            ffmpeg::ChannelLayout::STEREO,
            target_rate,
        )
        .map_err(|e| format!("resampler: {}", e))?;
        (Some(dec), Some(resampler), Some(idx))
    } else {
        (None, None, None)
    };

    // Populate initial shared state and hand off to Paused.
    {
        let mut state = shared.state.lock().unwrap();
        state.duration_us = duration_us;
        state.width = video_src_w;
        state.height = video_src_h;
        state.has_audio = audio_stream_index.is_some();
        state.frame_interval_us = video_frame_interval_us;
        state.player_state = PlayerState::Paused;
    }
    egui_ctx.request_repaint();

    // --- Spawn the two decode workers ---
    // Unbounded channels so the demuxer never blocks on `send` while
    // waiting for the workers to catch up. Blocking here would
    // prevent the demuxer from reaching its `cmd_rx.try_recv()` at
    // the top of the loop, which is what causes rapid-seek commands
    // to be queued but never processed until playback resumes.
    // Runtime back-pressure is still applied via the `video_queue`
    // length check above (demuxer sleeps when the decoded frame
    // queue is full), so in normal playback the channels don't
    // grow unbounded — they only accumulate during the brief
    // priming window after a seek, where we *want* the demuxer to
    // run ahead.
    // Bounded channels cap how far the demuxer can race ahead of
    // the workers, keeping memory bounded. Flushes are signalled
    // out-of-band via `shared.flush_seq`, so send-side blocking
    // here can't deadlock seeks even when the worker is paused.
    let (video_tx, video_rx) = mpsc::sync_channel::<DecodeMsg>(8);
    let (audio_tx_opt, audio_rx_opt) = if audio_stream_index.is_some() {
        let (tx, rx) = mpsc::sync_channel::<DecodeMsg>(16);
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    let video_handle = {
        let shared_c = shared.clone();
        let ctx_c = egui_ctx.clone();
        thread::spawn(move || {
            video_decode_loop(
                video_decoder,
                video_time_base,
                video_rx,
                shared_c,
                ctx_c,
                high_bit_depth_enabled,
            );
        })
    };
    let audio_handle = match (audio_decoder, audio_resampler, audio_rx_opt) {
        (Some(dec), Some(res), Some(rx)) => {
            let shared_a = shared.clone();
            Some(thread::spawn(move || {
                audio_decode_loop(dec, res, rx, shared_a, audio_producer);
            }))
        }
        _ => None,
    };

    // --- Demuxer loop ---
    let mut paused = true;
    let mut looping = false;
    let mut eof = false;
    // When true, the demuxer reads packets regardless of `paused`
    // state until the video worker has successfully pushed at least
    // one post-seek frame. This ensures the user sees the new frame
    // at the seek target even while playback is paused.
    let mut priming_after_seek = false;
    // Safety cap so we don't spin forever on a broken file.
    let mut prime_packets_read: u32 = 0;
    const MAX_PRIME_PACKETS: u32 = 600;

    // Carried across iterations: the packet-routing back-pressure
    // helper can pull commands out of cmd_rx when it has to wait for
    // channel space, and it stashes them here so the next iteration
    // of the main loop processes them instead of losing them.
    let mut pending_seek: Option<f32> = None;
    let mut pending_play: Option<bool> = None;

    'main: loop {
        // Drain all pending commands in one go and coalesce them.
        // Multiple queued `Seek`s collapse into the latest — so a
        // rapid drag on the scrub bar never processes a backlog.
        let mut should_stop = false;
        loop {
            match cmd_rx.try_recv() {
                Ok(Command::Play) => pending_play = Some(true),
                Ok(Command::Pause) => pending_play = Some(false),
                Ok(Command::Seek(frac)) => pending_seek = Some(frac),
                Ok(Command::SetLooping(l)) => looping = l,
                Ok(Command::Stop) => {
                    should_stop = true;
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    should_stop = true;
                    break;
                }
            }
        }
        if should_stop {
            break 'main;
        }
        if let Some(play) = pending_play.take() {
            paused = !play;
            eof = false;
            let mut state = shared.state.lock().unwrap();
            if state.player_state != PlayerState::Error {
                state.player_state = if play {
                    PlayerState::Playing
                } else {
                    PlayerState::Paused
                };
            }
        }
        if let Some(frac) = pending_seek.take() {
            let target_us =
                (duration_us as f32 * frac.clamp(0.0, 1.0)) as i64;
            // av_seek_frame(..., BACKWARD) lands on a keyframe at or
            // before the target. Combined with the `min_display_pts_us`
            // drop on the worker, that gives frame-accurate seeking —
            // decoder gets the keyframe, walks forward until the
            // target, then the first displayed frame is the first
            // one at or past the target.
            let seek_rc = unsafe {
                av_seek_frame(
                    ictx.as_mut_ptr(),
                    -1,
                    target_us,
                    AVSEEK_FLAG_BACKWARD,
                )
            };
            let _ = seek_rc;
            unsafe {
                let fctx = ictx.as_mut_ptr();
                if !fctx.is_null() && !(*fctx).pb.is_null() {
                    (*(*fctx).pb).eof_reached = 0;
                    (*(*fctx).pb).error = 0;
                }
                if !fctx.is_null() {
                    avformat_flush(fctx);
                }
            }
            // Set every gate BEFORE we send the Flush message,
            // because the worker may process the Flush + subsequent
            // packets before the demuxer finishes this handler. If
            // the gates aren't all set first, post-Flush frames can
            // slip through with stale min_display_pts_us (and end
            // up being displayed at completely wrong timestamps).
            shared
                .min_display_pts_us
                .store(target_us, Ordering::Relaxed);
            shared
                .post_seek_frames_pushed
                .store(0, Ordering::Relaxed);
            shared
                .flush_pending
                .store(true, Ordering::Relaxed);
            shared.audio_clock_us.store(target_us, Ordering::Relaxed);
            shared.clock_frozen.store(true, Ordering::Relaxed);
            shared
                .audio_drain_flag
                .store(true, Ordering::Relaxed);
            // Bump the flush counter so workers notice on their
            // next decode iteration. Sending a Flush *message* via
            // the bounded channel can deadlock if the channel is
            // full and the worker is parked in back-pressure (which
            // happens on pause-then-seek).
            shared.flush_seq.fetch_add(1, Ordering::Relaxed);
            shared.video_queue.lock().unwrap().clear();
            eof = false;
            priming_after_seek = true;
            prime_packets_read = 0;
        }

        if priming_after_seek {
            let pushed = shared
                .post_seek_frames_pushed
                .load(Ordering::Relaxed);
            if pushed > 0 || prime_packets_read >= MAX_PRIME_PACKETS {
                priming_after_seek = false;
            }
        }
        // While paused, keep the demuxer routing until the display
        // queue is well-populated. This lets `Player::step_frames`
        // and the seek bar consume many buffered frames without
        // falling back into the expensive full-seek-and-decode
        // path. `PAUSED_FILL_TARGET` is the depth we aim for; once
        // the queue has that many buffered frames we sleep until
        // tick() drains some.
        if paused && !priming_after_seek {
            const PAUSED_FILL_TARGET: usize = 12;
            let qlen = shared.video_queue.lock().unwrap().len();
            if qlen >= PAUSED_FILL_TARGET {
                thread::sleep(Duration::from_millis(10));
                continue;
            }
        }
        if eof {
            thread::sleep(Duration::from_millis(30));
            continue;
        }

        // Back-pressure: cap the decoded-video queue so it doesn't
        // grow without bound. CRITICAL: only stall once the clock has
        // actually started advancing. During startup the audio decoder
        // needs its first packets to arrive through the demuxer before
        // it can produce samples that move the clock. Stalling on
        // depth alone here deadlocks high-fps content (e.g. HEVC
        // 60fps) where ~16 decoded frames span less than one audio
        // decode cycle — the video queue hits the cap before any
        // audio packet has been routed, so audio never starts, the
        // clock never ticks, tick() never pops, and the demuxer
        // stays stalled forever.
        let clock_us = shared.audio_clock_us.load(Ordering::Relaxed);
        let video_queue_len = shared.video_queue.lock().unwrap().len();
        if video_queue_len > 48 && clock_us > 0 {
            thread::sleep(Duration::from_millis(3));
            continue;
        }
        if video_queue_len > 240 {
            // Hard safety cap regardless of clock state.
            thread::sleep(Duration::from_millis(3));
            continue;
        }

        // Read one packet.
        let mut packet = ffmpeg::Packet::empty();
        match packet.read(&mut ictx) {
            Ok(()) => {}
            Err(ffmpeg::Error::Eof) => {
                if looping {
                    // Wait for playback to actually catch up to the
                    // end of the file before looping. Without this,
                    // fast-decoding short files have the demuxer
                    // reaching EOF in ~200 ms while audio has only
                    // played ~50 ms — we'd then reset the clock to
                    // 0, race through again, and never make progress.
                    let duration_us =
                        shared.state.lock().unwrap().duration_us;
                    loop {
                        let clock_us =
                            shared.audio_clock_us.load(Ordering::Relaxed);
                        let video_queue_len =
                            shared.video_queue.lock().unwrap().len();
                        if duration_us <= 0
                            || (clock_us >= duration_us - 250_000
                                && video_queue_len == 0)
                        {
                            break;
                        }
                        // Respond to incoming commands while waiting.
                        if let Ok(cmd) = cmd_rx.try_recv() {
                            match cmd {
                                Command::Play | Command::Pause => {}
                                _ => break,
                            }
                        }
                        thread::sleep(Duration::from_millis(10));
                    }
                    // Full reset to start using av_seek_frame with
                    // BACKWARD|ANY flags, an explicit byte-zero
                    // avio_seek fallback, clear eof_reached, and
                    // avformat_flush. Then verify the next read
                    // actually succeeds — if it doesn't, give up on
                    // looping rather than spin forever.
                    let mut seek_ok = false;
                    unsafe {
                        let fctx = ictx.as_mut_ptr();
                        if !fctx.is_null() {
                            let r = av_seek_frame(
                                fctx,
                                -1,
                                0,
                                AVSEEK_FLAG_BACKWARD | AVSEEK_FLAG_ANY,
                            );
                            if !(*fctx).pb.is_null() {
                                avio_seek((*fctx).pb, 0, 0 /* SEEK_SET */);
                                (*(*fctx).pb).eof_reached = 0;
                                (*(*fctx).pb).error = 0;
                            }
                            avformat_flush(fctx);
                            seek_ok = r >= 0;
                        }
                    }
                    // Verify: one test read. If it returns something
                    // other than EOF, the seek recovered and we can
                    // resume; otherwise stop looping for this file.
                    let mut verify_pkt = ffmpeg::Packet::empty();
                    let verify_res = verify_pkt.read(&mut ictx);
                    if matches!(verify_res, Err(ffmpeg::Error::Eof)) || !seek_ok {
                        eprintln!(
                            "demuxer: loop seek did not recover (seek_ok={}, verify={:?}); stopping loop",
                            seek_ok, verify_res
                        );
                        looping = false;
                        eof = true;
                        let mut state = shared.state.lock().unwrap();
                        state.player_state = PlayerState::EndOfFile;
                        continue;
                    }
                    // Feed the verify packet through the normal path.
                    shared.flush_seq.fetch_add(1, Ordering::Relaxed);
                    shared.video_queue.lock().unwrap().clear();
                    shared
                        .audio_drain_flag
                        .store(true, Ordering::Relaxed);
                    shared.audio_clock_us.store(0, Ordering::Relaxed);
                    shared
                        .min_display_pts_us
                        .store(0, Ordering::Relaxed);
                    // Dispatch the verification packet so we don't
                    // waste it.
                    let vi = verify_pkt.stream();
                    let tag = shared.flush_seq.load(Ordering::Relaxed);
                    if vi == video_stream_index {
                        let _ = video_tx.send(DecodeMsg::Packet(verify_pkt, tag));
                    } else if Some(vi) == audio_stream_index {
                        if let Some(ref tx) = audio_tx_opt {
                            let _ = tx.send(DecodeMsg::Packet(verify_pkt, tag));
                        }
                    }
                    continue;
                }
                eof = true;
                let mut state = shared.state.lock().unwrap();
                state.player_state = PlayerState::EndOfFile;
                continue;
            }
            Err(e) => {
                eprintln!("packet read: {}", e);
                thread::sleep(Duration::from_millis(10));
                continue;
            }
        }

        let packet_idx = packet.stream();
        if packet_idx == video_stream_index {
            let tag = shared.flush_seq.load(Ordering::Relaxed);
            let mut msg = Some(DecodeMsg::Packet(packet, tag));
            loop {
                match video_tx.try_send(msg.take().unwrap()) {
                    Ok(()) => break,
                    Err(mpsc::TrySendError::Full(m)) => {
                        msg = Some(m);
                        if shared.stopping.load(Ordering::Relaxed) {
                            break 'main;
                        }
                        // Peek for a pending seek/stop so we don't
                        // stall here while the user is waiting.
                        if let Ok(cmd) = cmd_rx.try_recv() {
                            match cmd {
                                Command::Play => pending_play = Some(true),
                                Command::Pause => pending_play = Some(false),
                                Command::Seek(f) => {
                                    pending_seek = Some(f);
                                    // Drop the in-flight stale
                                    // packet; the seek path
                                    // will flush anyway.
                                    break;
                                }
                                Command::SetLooping(l) => looping = l,
                                Command::Stop => break 'main,
                            }
                        }
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(mpsc::TrySendError::Disconnected(_)) => break 'main,
                }
            }
        } else if Some(packet_idx) == audio_stream_index {
            if let Some(ref tx) = audio_tx_opt {
                let tag = shared.flush_seq.load(Ordering::Relaxed);
                let mut msg = Some(DecodeMsg::Packet(packet, tag));
                loop {
                    match tx.try_send(msg.take().unwrap()) {
                        Ok(()) => break,
                        Err(mpsc::TrySendError::Full(m)) => {
                            msg = Some(m);
                            if shared.stopping.load(Ordering::Relaxed) {
                                break 'main;
                            }
                            if let Ok(cmd) = cmd_rx.try_recv() {
                                match cmd {
                                    Command::Play => pending_play = Some(true),
                                    Command::Pause => pending_play = Some(false),
                                    Command::Seek(f) => {
                                        pending_seek = Some(f);
                                        break;
                                    }
                                    Command::SetLooping(l) => looping = l,
                                    Command::Stop => break 'main,
                                }
                            }
                            thread::sleep(Duration::from_millis(2));
                        }
                        Err(mpsc::TrySendError::Disconnected(_)) => break 'main,
                    }
                }
            }
        }
        if priming_after_seek {
            prime_packets_read += 1;
        }
    }

    // Drop the send halves so the workers exit their recv loops.
    drop(video_tx);
    drop(audio_tx_opt);
    let _ = video_handle.join();
    if let Some(h) = audio_handle {
        let _ = h.join();
    }

    Ok(())
}

const WORKER_QUEUE_CAP: usize = 16;

/// Push a decoded video frame into the shared display queue, then
/// (after pushing) park the worker if the queue is now at capacity.
/// This is post-push back-pressure: we always make at least one
/// frame available — important during post-seek priming when tick()
/// needs a frame to land before it can clear `clock_frozen` — and
/// only stall once that frame is buffered. Exits on the shared
/// `stopping` flag so `decode_thread.join()` can return promptly.
/// Returns true when the frame actually landed in the shared queue.
/// Callers must consult this result before bumping
/// `post_seek_frames_pushed`, otherwise the demuxer can incorrectly
/// conclude that seek priming is done and stop routing packets.
fn push_with_cap(shared: &Arc<Shared>, frame: VideoFrame, last_flush_seq: u64) -> bool {
    let mut slot = Some(frame);
    for _ in 0..400 {
        if shared.stopping.load(Ordering::Relaxed) {
            return false;
        }
        if shared.flush_seq.load(Ordering::Relaxed) != last_flush_seq {
            return false;
        }
        {
            let mut q = shared.video_queue.lock().unwrap();
            if q.len() < WORKER_QUEUE_CAP {
                q.push_back(slot.take().expect("frame present"));
                return true;
            }
        }
        thread::sleep(Duration::from_millis(5));
    }
    false
}

/// Video decode worker: pulls DecodeMsg from the channel, decodes,
/// scales to RGBA, pushes finished frames into shared.video_queue.
fn video_decode_loop(
    mut decoder: ffmpeg::decoder::Video,
    time_base: ffmpeg::Rational,
    rx: mpsc::Receiver<DecodeMsg>,
    shared: Arc<Shared>,
    egui_ctx: egui::Context,
    high_bit_depth_enabled: bool,
) {
    let mut video_frame = ffmpeg::frame::Video::empty();
    let mut rgba_frame = ffmpeg::frame::Video::empty();
    let mut sw_frame = ffmpeg::frame::Video::empty();
    let mut video_scaler: Option<ffmpeg::software::scaling::context::Context> = None;
    // cfg is (src_fmt, src_w, src_h, output_is_16bit) so a transition
    // between 8- and 16-bit output forces a scaler rebuild.
    let mut scaler_cfg: Option<(ffmpeg::format::Pixel, u32, u32, bool)> = None;
    let mut last_flush_seq: u64 = 0;
    loop {
        let msg = match rx.recv() {
            Ok(m) => m,
            Err(_) => return,
        };
        let cur_seq = shared.flush_seq.load(Ordering::Relaxed);
        if cur_seq != last_flush_seq {
            last_flush_seq = cur_seq;
            decoder.flush();
            shared.video_queue.lock().unwrap().clear();
            shared.flush_pending.store(false, Ordering::Relaxed);
        }
        let packet = match msg {
            DecodeMsg::Flush => {
                decoder.flush();
                shared.video_queue.lock().unwrap().clear();
                shared
                    .flush_pending
                    .store(false, Ordering::Relaxed);
                continue;
            }
            DecodeMsg::Packet(p, tag) => {
                if tag != last_flush_seq {
                    continue;
                }
                p
            }
        };

        if decoder.send_packet(&packet).is_err() {
            continue;
        }
        while decoder.receive_frame(&mut video_frame).is_ok() {
            // If a seek arrived mid-decode, drop the frame in hand
            // and stop draining the decoder — the next outer loop
            // iteration will service the flush.
            let s = shared.flush_seq.load(Ordering::Relaxed);
            if s != last_flush_seq {
                break;
            }
            // Drop any frames the decoder is draining from pre-seek
            // state — they show as "fast-forward" blur otherwise.
            if shared.flush_pending.load(Ordering::Relaxed) {
                continue;
            }
            let raw_format = unsafe { (*video_frame.as_ptr()).format };
            let is_hw = raw_format == AVPixelFormat::AV_PIX_FMT_D3D11 as i32;

            let pts_raw = video_frame.pts().unwrap_or(0);
            let pts_us = ts_to_us(pts_raw, time_base);

            // After a seek, av_seek_frame lands on the keyframe
            // before the target. Frames between that keyframe and
            // the seek target must be decoded (they feed the codec
            // state) but should not be displayed, so we drop any
            // frame whose pts is strictly before the target.
            let min_pts = shared
                .min_display_pts_us
                .load(Ordering::Relaxed);
            if pts_us < min_pts {
                unsafe {
                    av_frame_unref(video_frame.as_mut_ptr());
                }
                continue;
            }


            let source_frame: &ffmpeg::frame::Video = if is_hw {
                let ret = unsafe {
                    av_hwframe_transfer_data(
                        sw_frame.as_mut_ptr(),
                        video_frame.as_ptr(),
                        0,
                    )
                };
                if ret < 0 {
                    eprintln!("hwframe transfer failed: {}", ret);
                    continue;
                }
                &sw_frame
            } else {
                &video_frame
            };

            let src_fmt = source_frame.format();
            let src_w = source_frame.width();
            let src_h = source_frame.height();

            // NV12 fast path: skip swscale and pass Y + UV planes
            // straight through. The GPU does YUV→RGB conversion in
            // the fragment shader. Typically hit after HW decode
            // because `av_hwframe_transfer_data` produces NV12 on
            // CPU.
            if src_fmt == ffmpeg::format::Pixel::NV12 {
                let y_stride = source_frame.stride(0);
                let uv_stride = source_frame.stride(1);
                let y_src = source_frame.data(0);
                let uv_src = source_frame.data(1);
                let y_row = src_w as usize;
                let uv_row = src_w as usize; // interleaved UV: 2 bytes per pixel at half width = src_w
                let uv_h = (src_h as usize + 1) / 2;
                let mut y_plane: Vec<u8> = Vec::with_capacity(y_row * src_h as usize);
                let mut uv_plane: Vec<u8> = Vec::with_capacity(uv_row * uv_h);
                if y_stride == y_row {
                    y_plane.extend_from_slice(&y_src[..y_row * src_h as usize]);
                } else {
                    for y in 0..src_h as usize {
                        let s = y * y_stride;
                        y_plane.extend_from_slice(&y_src[s..s + y_row]);
                    }
                }
                if uv_stride == uv_row {
                    uv_plane.extend_from_slice(&uv_src[..uv_row * uv_h]);
                } else {
                    for y in 0..uv_h {
                        let s = y * uv_stride;
                        uv_plane.extend_from_slice(&uv_src[s..s + uv_row]);
                    }
                }
                let pushed = push_with_cap(
                    &shared,
                    VideoFrame {
                        pts_us,
                        payload: VideoFramePayload::Nv12 {
                            width: src_w,
                            height: src_h,
                            y_plane,
                            uv_plane,
                        },
                    },
                    last_flush_seq,
                );
                if pushed {
                    shared
                        .post_seek_frames_pushed
                        .fetch_add(1, Ordering::Relaxed);
                    egui_ctx.request_repaint();
                }
                continue;
            }

            // Decide whether this frame should go through the 16-bit
            // path: feature enabled AND the source is actually a
            // 10-bit-or-higher pixel format worth preserving.
            let use_16bit = high_bit_depth_enabled && is_high_bit_depth_format(src_fmt);
            let dst_fmt = if use_16bit {
                ffmpeg::format::Pixel::RGBA64LE
            } else {
                ffmpeg::format::Pixel::RGBA
            };
            let bytes_per_pixel: usize = if use_16bit { 8 } else { 4 };

            // Cap output width to keep the RGBA buffer small and
            // texture uploads cheap. 1920-wide is enough for most
            // displays — scaling up to fit on the egui side uses GPU
            // bilinear at essentially zero cost, whereas CPU-side
            // upload of a 3440-wide frame chews bandwidth.
            const MAX_OUT_WIDTH: u32 = 1920;
            let (dst_w, dst_h) = if src_w > MAX_OUT_WIDTH {
                let ratio = MAX_OUT_WIDTH as f64 / src_w as f64;
                let h = ((src_h as f64 * ratio).round() as u32).max(1);
                (MAX_OUT_WIDTH, h)
            } else {
                (src_w, src_h)
            };

            let cfg = (src_fmt, src_w, src_h, use_16bit);
            if scaler_cfg != Some(cfg) {
                let flags = ffmpeg::software::scaling::flag::Flags::BILINEAR
                    | ffmpeg::software::scaling::flag::Flags::ACCURATE_RND
                    | ffmpeg::software::scaling::flag::Flags::FULL_CHR_H_INT
                    | ffmpeg::software::scaling::flag::Flags::FULL_CHR_H_INP;
                match ffmpeg::software::scaling::context::Context::get(
                    src_fmt,
                    src_w,
                    src_h,
                    dst_fmt,
                    dst_w,
                    dst_h,
                    flags,
                ) {
                    Ok(s) => {
                        video_scaler = Some(s);
                        scaler_cfg = Some(cfg);
                    }
                    Err(e) => {
                        eprintln!("scaler rebuild: {}", e);
                        continue;
                    }
                }
            }
            let scaler = match video_scaler.as_mut() {
                Some(s) => s,
                None => continue,
            };
            if scaler.run(source_frame, &mut rgba_frame).is_err() {
                continue;
            }

            let w = rgba_frame.width();
            let h = rgba_frame.height();
            let stride = rgba_frame.stride(0);
            let src = rgba_frame.data(0);
            let row = w as usize * bytes_per_pixel;
            let total = row * h as usize;
            let mut pixels: Vec<u8> = Vec::with_capacity(total);
            if stride == row {
                pixels.extend_from_slice(&src[..total]);
            } else {
                for y in 0..h as usize {
                    let s = y * stride;
                    pixels.extend_from_slice(&src[s..s + row]);
                }
            }

            let payload = if use_16bit {
                VideoFramePayload::Rgba16Unorm {
                    width: w,
                    height: h,
                    bytes: pixels,
                }
            } else {
                // Transmute Vec<u8> -> Vec<Color32> without touching
                // the pixel bytes. Safe: Color32 is #[repr(C)] [u8; 4]
                // with alignment 1, same as u8, byte count multiple
                // of 4.
                let color_pixels: Vec<egui::Color32> = unsafe {
                    debug_assert!(pixels.len() % 4 == 0);
                    debug_assert!(pixels.capacity() % 4 == 0);
                    let len = pixels.len() / 4;
                    let cap = pixels.capacity() / 4;
                    let ptr = pixels.as_mut_ptr() as *mut egui::Color32;
                    std::mem::forget(pixels);
                    Vec::from_raw_parts(ptr, len, cap)
                };
                let image = egui::ColorImage {
                    size: [w as usize, h as usize],
                    pixels: color_pixels,
                };
                VideoFramePayload::Rgba8Srgb {
                    width: w,
                    height: h,
                    image,
                }
            };

            let pushed = push_with_cap(
                &shared,
                VideoFrame {
                    pts_us,
                    payload,
                },
                last_flush_seq,
            );
            if pushed {
                shared
                    .post_seek_frames_pushed
                    .fetch_add(1, Ordering::Relaxed);
                egui_ctx.request_repaint();
            }
        }
    }
}

// =========================================================================
// NV12 YUV → RGB custom render pipeline (partial Phase E)
// =========================================================================

/// WGSL shader for NV12 sampling: full-screen-quad vertex shader with
/// a user-specified NDC rect, fragment shader that samples Y + UV,
/// applies BT.709 limited-range YUV→R'G'B', decodes sRGB gamma, and
/// outputs linear RGB (the swapchain's sRGB-encoded target will
/// re-encode on store).
const NV12_SHADER_SRC: &str = r#"
@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var uv_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

// NDC rect (left, top, right, bottom) in the current render-pass
// viewport's coordinate space. Set from the CPU side every frame
// based on the unclamped video_rect relative to the clamped
// egui_wgpu viewport, so the quad is positioned correctly even when
// panning pushes the video partially off-screen.
struct NdcRect {
    rect: vec4<f32>,
};
@group(0) @binding(3) var<uniform> ndc: NdcRect;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    let l = ndc.rect.x;
    let t = ndc.rect.y;
    let r = ndc.rect.z;
    let b = ndc.rect.w;
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(l, t),
        vec2<f32>(r, t),
        vec2<f32>(r, b),
        vec2<f32>(l, t),
        vec2<f32>(r, b),
        vec2<f32>(l, b),
    );
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
    );
    var out: VsOut;
    out.pos = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}

fn srgb_to_linear(c: f32) -> f32 {
    if (c <= 0.04045) {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let y  = textureSample(y_tex,  samp, in.uv).r;
    let uv = textureSample(uv_tex, samp, in.uv).rg;

    // BT.709 limited-range YUV (all channels normalized to [0,1]).
    // Output is gamma-encoded R'G'B'.
    let y_lin = 1.164 * (y - 0.0627);
    let u_off = uv.r - 0.502;
    let v_off = uv.g - 0.502;
    let r_g = clamp(y_lin + 1.793 * v_off, 0.0, 1.0);
    let g_g = clamp(y_lin - 0.213 * u_off - 0.533 * v_off, 0.0, 1.0);
    let b_g = clamp(y_lin + 2.112 * u_off, 0.0, 1.0);

    // FRAGMENT_RETURN
}
"#;

struct YuvRenderer {
    pipeline: Arc<wgpu::RenderPipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
    sampler: Arc<wgpu::Sampler>,
}

impl Clone for YuvRenderer {
    fn clone(&self) -> Self {
        Self {
            pipeline: self.pipeline.clone(),
            bind_group_layout: self.bind_group_layout.clone(),
            sampler: self.sampler.clone(),
        }
    }
}

impl YuvRenderer {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let is_srgb_target = matches!(
            target_format,
            wgpu::TextureFormat::Rgba8UnormSrgb
                | wgpu::TextureFormat::Bgra8UnormSrgb
                | wgpu::TextureFormat::Bc1RgbaUnormSrgb
                | wgpu::TextureFormat::Bc2RgbaUnormSrgb
                | wgpu::TextureFormat::Bc3RgbaUnormSrgb
                | wgpu::TextureFormat::Bc7RgbaUnormSrgb
        );
        // Build the fragment output expression based on target format.
        // - sRGB target: hardware encodes linear→sRGB on store, so the
        //   shader must output LINEAR. We apply the BT.709/sRGB inverse
        //   transfer function to the YUV→R'G'B' result.
        // - Non-sRGB target: hardware stores bytes as-is, so the shader
        //   must output gamma-encoded R'G'B' directly.
        let source = if is_srgb_target {
            NV12_SHADER_SRC.replace(
                "// FRAGMENT_RETURN",
                "return vec4<f32>(\
                    srgb_to_linear(r_g),\
                    srgb_to_linear(g_g),\
                    srgb_to_linear(b_g),\
                    1.0,\
                );",
            )
        } else {
            NV12_SHADER_SRC.replace(
                "// FRAGMENT_RETURN",
                "return vec4<f32>(r_g, g_g, b_g, 1.0);",
            )
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nv12_shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("nv12_bind_group_layout"),
                entries: &[
                    // Y plane texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // UV plane texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // NDC-rect uniform (vec4<f32>): left, top, right,
                    // bottom in the clamped-viewport's NDC space. Lets
                    // us draw the correct un-squished slice of the
                    // video when the video rect extends past the
                    // visible area (egui clamps the viewport, so a
                    // hardcoded full-NDC quad would otherwise look
                    // like a resize on pan).
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nv12_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("nv12_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nv12_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        Self {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bind_group_layout),
            sampler: Arc::new(sampler),
        }
    }
}

/// Per-video NV12 texture state: the Y plane texture, the UV plane
/// texture, and the bind group that binds them to the `YuvRenderer`
/// pipeline.
struct Nv12GpuState {
    y_texture: wgpu::Texture,
    uv_texture: wgpu::Texture,
    ndc_buffer: wgpu::Buffer,
    bind_group: Arc<wgpu::BindGroup>,
    width: u32,
    height: u32,
}

/// Per-frame PaintCallback that runs inside egui_wgpu's render pass
/// to draw the NV12 video. Holds cheap Arc-clones of the pipeline and
/// bind group plus the NDC quad coordinates for the video area.
struct Nv12PaintCallback {
    pipeline: Arc<wgpu::RenderPipeline>,
    bind_group: Arc<wgpu::BindGroup>,
}

impl egui_wgpu::CallbackTrait for Nv12PaintCallback {
    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        _callback_resources: &egui_wgpu::CallbackResources,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..6, 0..1);
    }
}

fn is_high_bit_depth_format(fmt: ffmpeg::format::Pixel) -> bool {
    use ffmpeg::format::Pixel::*;
    matches!(
        fmt,
        YUV420P10LE | YUV420P10BE
        | YUV422P10LE | YUV422P10BE
        | YUV444P10LE | YUV444P10BE
        | YUV420P12LE | YUV420P12BE
        | YUV422P12LE | YUV422P12BE
        | YUV444P12LE | YUV444P12BE
        | YUV420P16LE | YUV420P16BE
        | YUV422P16LE | YUV422P16BE
        | YUV444P16LE | YUV444P16BE
        | P010LE | P010BE
        | P016LE | P016BE
    )
}

/// Audio decode worker: pulls DecodeMsg, decodes, resamples, pushes
/// f32 samples into the cpal ring buffer.
fn audio_decode_loop(
    mut decoder: ffmpeg::decoder::Audio,
    mut resampler: ffmpeg::software::resampling::context::Context,
    rx: mpsc::Receiver<DecodeMsg>,
    shared: Arc<Shared>,
    mut audio_producer: <HeapRb<f32> as Split>::Prod,
) {
    let mut audio_frame = ffmpeg::frame::Audio::empty();
    let mut resampled = ffmpeg::frame::Audio::empty();
    let mut last_flush_seq: u64 = 0;

    loop {
        let msg = match rx.recv() {
            Ok(m) => m,
            Err(_) => return,
        };
        let cur_seq = shared.flush_seq.load(Ordering::Relaxed);
        if cur_seq != last_flush_seq {
            last_flush_seq = cur_seq;
            decoder.flush();
        }
        let packet = match msg {
            DecodeMsg::Flush => {
                decoder.flush();
                continue;
            }
            DecodeMsg::Packet(p, tag) => {
                if tag != last_flush_seq {
                    continue;
                }
                p
            }
        };

        if decoder.send_packet(&packet).is_err() {
            continue;
        }
        while decoder.receive_frame(&mut audio_frame).is_ok() {
            if resampler.run(&audio_frame, &mut resampled).is_err() {
                continue;
            }
            push_planar_stereo(&resampled, &mut audio_producer);
            while resampler.delay().is_some() {
                if resampler.flush(&mut resampled).is_err() {
                    break;
                }
                if resampled.samples() == 0 {
                    break;
                }
                push_planar_stereo(&resampled, &mut audio_producer);
            }
        }
    }
}

/// Interleave planar stereo f32 samples and bulk-push into the
/// ring buffer. Bulk pushes are cheaper than per-sample atomic ops
/// and avoid the per-sample sleep that on Windows stalls for the
/// ~15 ms timer resolution when the ring is full.
///
/// Drops the tail of the frame if the ring stays full for longer
/// than `PUSH_DEADLINE_MS`. The ring only stays full when the cpal
/// consumer isn't draining — typically because playback is paused
/// — and in that state blocking here would back-pressure the audio
/// channel, which back-pressures the demuxer, which stops routing
/// video packets, which leaves the UI stuck on the last-displayed
/// frame after any seek. Dropping audio in that state matches
/// VLC/mpv behavior: favour video liveness over stale audio.
fn push_planar_stereo(
    frame: &ffmpeg::frame::Audio,
    producer: &mut <HeapRb<f32> as Split>::Prod,
) {
    const PUSH_DEADLINE_MS: u128 = 4;
    let nb = frame.samples();
    if nb == 0 {
        return;
    }
    let planes = frame.planes();
    let left = frame.plane::<f32>(0);
    let right: &[f32] = if planes > 1 {
        frame.plane::<f32>(1)
    } else {
        left
    };
    let mut interleaved: Vec<f32> = Vec::with_capacity(nb * 2);
    for i in 0..nb {
        interleaved.push(*left.get(i).unwrap_or(&0.0));
        interleaved.push(*right.get(i).unwrap_or(&0.0));
    }
    let mut offset = 0usize;
    let start = Instant::now();
    while offset < interleaved.len() {
        let pushed = producer.push_slice(&interleaved[offset..]);
        offset += pushed;
        if offset >= interleaved.len() {
            break;
        }
        if start.elapsed().as_millis() > PUSH_DEADLINE_MS {
            break;
        }
        std::thread::yield_now();
    }
}

fn ts_to_us(ts: i64, tb: ffmpeg::Rational) -> i64 {
    // PTS (in stream time_base units) → microseconds.
    (ts as f64 * tb.numerator() as f64 / tb.denominator() as f64 * 1_000_000.0) as i64
}

// =========================================================================
// D3D11VA hardware acceleration (Phase 3)
// =========================================================================

/// Picked by avcodec_open2 / get_format when negotiating the decoder's
/// output pixel format. Prefer D3D11 if it's in the offered list so the
/// decoder will produce GPU surfaces; otherwise accept the first listed
/// (software) format and decode on the CPU.
unsafe extern "C" fn get_hw_format(
    _ctx: *mut AVCodecContext,
    mut pix_fmts: *const AVPixelFormat,
) -> AVPixelFormat {
    let mut fallback = AVPixelFormat::AV_PIX_FMT_NONE;
    while !pix_fmts.is_null() && *pix_fmts != AVPixelFormat::AV_PIX_FMT_NONE {
        if fallback == AVPixelFormat::AV_PIX_FMT_NONE {
            fallback = *pix_fmts;
        }
        if *pix_fmts == AVPixelFormat::AV_PIX_FMT_D3D11 {
            return AVPixelFormat::AV_PIX_FMT_D3D11;
        }
        pix_fmts = pix_fmts.add(1);
    }
    fallback
}

/// Set up a D3D11VA device context and attach it to the given codec
/// context. Must be called BEFORE avcodec_open2 (before
/// `.decoder().video()` in ffmpeg-the-third terms).
unsafe fn try_enable_hw_accel(cctx: *mut AVCodecContext) -> bool {
    let mut hw_dev: *mut AVBufferRef = std::ptr::null_mut();
    let ret = av_hwdevice_ctx_create(
        &mut hw_dev,
        AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA,
        std::ptr::null(),
        std::ptr::null_mut(),
        0,
    );
    if ret < 0 || hw_dev.is_null() {
        return false;
    }
    // The codec context takes its own ref; we unref our local handle
    // after the assignment, leaving only the decoder's ownership.
    (*cctx).hw_device_ctx = av_buffer_ref(hw_dev);
    (*cctx).get_format = Some(get_hw_format);
    av_buffer_unref(&mut hw_dev);
    true
}
