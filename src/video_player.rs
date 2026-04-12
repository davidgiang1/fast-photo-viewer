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
    av_buffer_ref, av_buffer_unref, av_hwdevice_ctx_create, av_hwframe_transfer_data,
    av_seek_frame, avformat_flush, avio_seek, AVBufferRef, AVCodecContext, AVHWDeviceType,
    AVPixelFormat, AVSEEK_FLAG_ANY, AVSEEK_FLAG_BACKWARD,
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

struct VideoFrame {
    pts_us: i64,
    /// Pre-built ColorImage ready for direct handoff to egui. Built on
    /// the decode worker thread so the main thread's `tick()` only has
    /// to move ownership — no per-pixel work happens on the UI path.
    image: egui::ColorImage,
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
    Packet(ffmpeg::Packet),
    Flush,
}

struct SharedState {
    player_state: PlayerState,
    duration_us: i64,
    width: u32,
    height: u32,
    has_audio: bool,
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
}

pub struct Player {
    shared: Arc<Shared>,
    cmd_tx: Sender<Command>,
    _audio_stream: Option<cpal::Stream>,
    // Egui-managed fallback path (test binary uses this).
    texture: Option<egui::TextureHandle>,
    // Direct wgpu path — preferred when available.
    wgpu: Option<WgpuBackend>,
    wgpu_texture: Option<wgpu::Texture>,
    wgpu_texture_id: Option<egui::TextureId>,
    wgpu_texture_size: (u32, u32),
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
                error: None,
            }),
            video_queue: Mutex::new(VecDeque::new()),
            audio_clock_us: AtomicI64::new(0),
            volume_bits: AtomicU32::new(1.0f32.to_bits()),
            clock_frozen: std::sync::atomic::AtomicBool::new(false),
        });

        // Audio ring buffer: 4 seconds worth of stereo f32 at the
        // output device's native rate. Generous headroom so a brief
        // scheduling hiccup doesn't drain it.
        let rb = HeapRb::<f32>::new(target_rate as usize * 2 * 4);
        let (audio_producer, audio_consumer) = rb.split();

        let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();

        let shared_for_thread = shared.clone();
        let ctx_for_thread = ctx.clone();
        let path_for_thread = path.to_path_buf();
        thread::spawn(move || {
            if let Err(e) = run_decode_pipeline(
                path_for_thread,
                cmd_rx,
                shared_for_thread.clone(),
                audio_producer,
                ctx_for_thread,
                target_rate,
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
            _audio_stream: audio_stream,
            texture: None,
            wgpu,
            wgpu_texture: None,
            wgpu_texture_id: None,
            wgpu_texture_size: (0, 0),
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
        if !self.has_audio() {
            self.no_audio_clock_origin = Some(Instant::now());
            self.no_audio_clock_base_us = self.shared.audio_clock_us.load(Ordering::Relaxed);
        }
    }

    pub fn pause(&mut self) {
        let _ = self.cmd_tx.send(Command::Pause);
        self.playing_cached = false;
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
        self.shared
            .clock_frozen
            .store(true, Ordering::Relaxed);
        let _ = self.cmd_tx.send(Command::Seek(fraction.clamp(0.0, 1.0)));
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

    pub fn size(&self) -> (u32, u32) {
        let s = self.shared.state.lock().unwrap();
        (s.width, s.height)
    }

    pub fn has_audio(&self) -> bool {
        self.shared.state.lock().unwrap().has_audio
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

        let clock_us = self.current_clock_us();

        let best = {
            let mut queue = self.shared.video_queue.lock().unwrap();
            let mut best: Option<VideoFrame> = None;
            while let Some(front) = queue.front() {
                if front.pts_us <= clock_us + 15_000 {
                    best = queue.pop_front();
                } else {
                    break;
                }
            }
            best
        };

        if let Some(frame) = best {
            if frame.pts_us != self.last_uploaded_pts_us {
                if self.wgpu.is_some() {
                    self.upload_wgpu(&frame);
                } else {
                    self.upload_egui(frame.image);
                }
                self.last_uploaded_pts_us = frame.pts_us;
            }
        }
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

    fn upload_wgpu(&mut self, frame: &VideoFrame) {
        let backend = match self.wgpu.as_ref() {
            Some(b) => b.clone(),
            None => return,
        };
        let width = frame.image.size[0] as u32;
        let height = frame.image.size[1] as u32;

        // (Re)create the wgpu texture when the size changes (first
        // frame or aspect ratio flip between videos).
        if self.wgpu_texture.is_none() || self.wgpu_texture_size != (width, height) {
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
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
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
        }

        // Upload the RGBA bytes directly into the texture. ColorImage
        // stores pixels as Vec<Color32>, which is repr(C) [u8; 4] —
        // safe to reinterpret as a flat &[u8].
        let pixels = &frame.image.pixels;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 4)
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
                    bytes_per_row: Some(width * 4),
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
}

impl Drop for Player {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(Command::Stop);
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

    stream
        .play()
        .map_err(|e| format!("cpal stream play: {}", e))?;
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
) -> Result<(), String> {
    let mut ictx =
        ffmpeg::format::input(&path).map_err(|e| format!("open: {}", e))?;
    let duration_us = ictx.duration();

    // --- Video stream ---
    let (video_stream_index, video_time_base) = {
        let stream = ictx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| "no video stream".to_string())?;
        (stream.index(), stream.time_base())
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
        state.player_state = PlayerState::Paused;
    }
    egui_ctx.request_repaint();

    // --- Spawn the two decode workers ---
    // Bounded channels: demuxer will block when the decoder can't keep
    // up, so we don't race ahead and buffer the entire file in memory.
    // These sizes give ~1 second of 60fps video lead time and ~1 second
    // of 48kHz audio, which is plenty for smooth playback.
    let (video_tx, video_rx) = mpsc::sync_channel::<DecodeMsg>(64);
    let (audio_tx_opt, audio_rx_opt) = if audio_stream_index.is_some() {
        let (tx, rx) = mpsc::sync_channel::<DecodeMsg>(64);
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    let video_handle = {
        let shared_c = shared.clone();
        let ctx_c = egui_ctx.clone();
        thread::spawn(move || {
            video_decode_loop(video_decoder, video_time_base, video_rx, shared_c, ctx_c);
        })
    };
    let audio_handle = match (audio_decoder, audio_resampler, audio_rx_opt) {
        (Some(dec), Some(res), Some(rx)) => Some(thread::spawn(move || {
            audio_decode_loop(dec, res, rx, audio_producer);
        })),
        _ => None,
    };

    // --- Demuxer loop ---
    let mut paused = true;
    let mut looping = true;
    let mut eof = false;

    'main: loop {
        // Drain any pending commands.
        loop {
            match cmd_rx.try_recv() {
                Ok(Command::Play) => {
                    paused = false;
                    eof = false;
                    let mut state = shared.state.lock().unwrap();
                    if state.player_state != PlayerState::Error {
                        state.player_state = PlayerState::Playing;
                    }
                }
                Ok(Command::Pause) => {
                    paused = true;
                    let mut state = shared.state.lock().unwrap();
                    if state.player_state != PlayerState::Error {
                        state.player_state = PlayerState::Paused;
                    }
                }
                Ok(Command::Seek(frac)) => {
                    let target_us =
                        (duration_us as f32 * frac.clamp(0.0, 1.0)) as i64;
                    let _ = ictx.seek(target_us, ..target_us + duration_us / 20);
                    unsafe {
                        let fctx = ictx.as_mut_ptr();
                        if !fctx.is_null() && !(*fctx).pb.is_null() {
                            (*(*fctx).pb).eof_reached = 0;
                            (*(*fctx).pb).error = 0;
                            avformat_flush(fctx);
                        }
                    }
                    let _ = video_tx.send(DecodeMsg::Flush);
                    if let Some(ref tx) = audio_tx_opt {
                        let _ = tx.send(DecodeMsg::Flush);
                    }
                    shared.video_queue.lock().unwrap().clear();
                    shared.audio_clock_us.store(target_us, Ordering::Relaxed);
                    shared.clock_frozen.store(false, Ordering::Relaxed);
                    eof = false;
                }
                Ok(Command::SetLooping(l)) => looping = l,
                Ok(Command::Stop) => break 'main,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break 'main,
            }
        }

        if paused {
            thread::sleep(Duration::from_millis(10));
            continue;
        }
        if eof {
            thread::sleep(Duration::from_millis(30));
            continue;
        }

        // Light back-pressure: don't pile up more than ~1 second of
        // decoded video ahead of playback. The video decoder is much
        // cheaper than display at this queue depth.
        let video_queue_len = shared.video_queue.lock().unwrap().len();
        if video_queue_len > 16 {
            thread::sleep(Duration::from_millis(3));
            continue;
        }

        // Read one packet.
        let mut packet = ffmpeg::Packet::empty();
        match packet.read(&mut ictx) {
            Ok(()) => {}
            Err(ffmpeg::Error::Eof) => {
                if looping {
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
                    let _ = video_tx.send(DecodeMsg::Flush);
                    if let Some(ref tx) = audio_tx_opt {
                        let _ = tx.send(DecodeMsg::Flush);
                    }
                    shared.video_queue.lock().unwrap().clear();
                    shared.audio_clock_us.store(0, Ordering::Relaxed);
                    // Dispatch the verification packet so we don't
                    // waste it.
                    let vi = verify_pkt.stream();
                    if vi == video_stream_index {
                        let _ = video_tx.send(DecodeMsg::Packet(verify_pkt));
                    } else if Some(vi) == audio_stream_index {
                        if let Some(ref tx) = audio_tx_opt {
                            let _ = tx.send(DecodeMsg::Packet(verify_pkt));
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
            if video_tx.send(DecodeMsg::Packet(packet)).is_err() {
                break 'main;
            }
        } else if Some(packet_idx) == audio_stream_index {
            if let Some(ref tx) = audio_tx_opt {
                if tx.send(DecodeMsg::Packet(packet)).is_err() {
                    break 'main;
                }
            }
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

/// Video decode worker: pulls DecodeMsg from the channel, decodes,
/// scales to RGBA, pushes finished frames into shared.video_queue.
fn video_decode_loop(
    mut decoder: ffmpeg::decoder::Video,
    time_base: ffmpeg::Rational,
    rx: Receiver<DecodeMsg>,
    shared: Arc<Shared>,
    egui_ctx: egui::Context,
) {
    let mut video_frame = ffmpeg::frame::Video::empty();
    let mut rgba_frame = ffmpeg::frame::Video::empty();
    let mut sw_frame = ffmpeg::frame::Video::empty();
    let mut video_scaler: Option<ffmpeg::software::scaling::context::Context> = None;
    let mut scaler_cfg: Option<(ffmpeg::format::Pixel, u32, u32)> = None;
    loop {
        let msg = match rx.recv() {
            Ok(m) => m,
            Err(_) => return,
        };
        let packet = match msg {
            DecodeMsg::Flush => {
                decoder.flush();
                shared.video_queue.lock().unwrap().clear();
                continue;
            }
            DecodeMsg::Packet(p) => p,
        };

        if decoder.send_packet(&packet).is_err() {
            continue;
        }
        while decoder.receive_frame(&mut video_frame).is_ok() {
            let raw_format = unsafe { (*video_frame.as_ptr()).format };
            let is_hw = raw_format == AVPixelFormat::AV_PIX_FMT_D3D11 as i32;

            let pts_raw = video_frame.pts().unwrap_or(0);
            let pts_us = ts_to_us(pts_raw, time_base);

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

            let cfg = (src_fmt, src_w, src_h);
            if scaler_cfg != Some(cfg) {
                let flags = ffmpeg::software::scaling::flag::Flags::BILINEAR
                    | ffmpeg::software::scaling::flag::Flags::ACCURATE_RND
                    | ffmpeg::software::scaling::flag::Flags::FULL_CHR_H_INT
                    | ffmpeg::software::scaling::flag::Flags::FULL_CHR_H_INP;
                match ffmpeg::software::scaling::context::Context::get(
                    src_fmt,
                    src_w,
                    src_h,
                    ffmpeg::format::Pixel::RGBA,
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
            let row = w as usize * 4;
            let total = row * h as usize;
            // Single memcpy fast path when swscale produced a tight
            // buffer (stride == row), else row-by-row.
            let mut pixels: Vec<u8> = Vec::with_capacity(total);
            if stride == row {
                pixels.extend_from_slice(&src[..total]);
            } else {
                for y in 0..h as usize {
                    let s = y * stride;
                    pixels.extend_from_slice(&src[s..s + row]);
                }
            }

            // Transmute Vec<u8> -> Vec<Color32> without touching the
            // pixel bytes. Safe because Color32 is #[repr(C)] [u8; 4]
            // with the same alignment (1) as u8, the byte count is a
            // multiple of 4, and we immediately forget the original.
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

            shared.video_queue.lock().unwrap().push_back(VideoFrame {
                pts_us,
                image,
            });
            egui_ctx.request_repaint();
        }
    }
}

/// Audio decode worker: pulls DecodeMsg, decodes, resamples, pushes
/// f32 samples into the cpal ring buffer.
fn audio_decode_loop(
    mut decoder: ffmpeg::decoder::Audio,
    mut resampler: ffmpeg::software::resampling::context::Context,
    rx: Receiver<DecodeMsg>,
    mut audio_producer: <HeapRb<f32> as Split>::Prod,
) {
    let mut audio_frame = ffmpeg::frame::Audio::empty();
    let mut resampled = ffmpeg::frame::Audio::empty();

    loop {
        let msg = match rx.recv() {
            Ok(m) => m,
            Err(_) => return,
        };
        let packet = match msg {
            DecodeMsg::Flush => {
                decoder.flush();
                continue;
            }
            DecodeMsg::Packet(p) => p,
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
fn push_planar_stereo(
    frame: &ffmpeg::frame::Audio,
    producer: &mut <HeapRb<f32> as Split>::Prod,
) {
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
    while offset < interleaved.len() {
        let pushed = producer.push_slice(&interleaved[offset..]);
        offset += pushed;
        if offset < interleaved.len() {
            // Ring is full — yield to let the cpal callback drain some.
            std::thread::yield_now();
        }
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
