//! GPU-accelerated terminal renderer using wgpu + cosmic-text.
//!
//! Architecture
//! ─────────────
//! • **Three render pipelines**
//!   1. *Background pipeline* – solid-colour quads for cell backgrounds.
//!      Alpha blending enabled so `cfg.window.opacity` is honoured.
//!   2. *Glyph pipeline* – textured quads sourced from the glyph atlas with
//!      alpha blending so glyph coverage masks blend over the background.
//!   3. *Cursor pipeline* – same vertex shader as background, but always uses
//!      alpha blending so the semi-transparent block cursor blends correctly.
//!
//! • **Glyph atlas** – a single R8Unorm 2048×2048 texture.  Glyphs are
//!   rasterised by cosmic-text / swash and packed using a simple shelf
//!   algorithm.  The atlas is invalidated and rebuilt on font-size changes.
//!
//! • **Text shaping** – one `cosmic_text::Buffer` per visible terminal row,
//!   shaped with HarfBuzz (via rustybuzz) for full ligature and Unicode
//!   support when `config.font.ligatures` is true.
//!
//! Performance notes
//! ─────────────────
//! • Per-row shaping results (glyph positions / cache keys) are cached by row
//!   text content and reused across frames when the row has not changed, so
//!   HarfBuzz / cosmic-text work is O(changed rows) instead of O(all rows).
//! • Vertex buffers are rebuilt from scratch each frame (terminal grids are
//!   small; typical 80×24 = 1920 cells ≈ 11 520 vertices – negligible GPU
//!   upload cost).  TODO: persist vertex buffers and use `queue.write_buffer`.
//! • `PresentMode::AutoNoVsync` is selected so frames are presented as fast
//!   as possible (Mailbox → Immediate fallback).

use std::collections::HashMap;
use std::sync::Arc;
use bytemuck::{Pod, Zeroable};
use cosmic_text::{Attrs, Buffer, Family, FontSystem, Metrics, Shaping, SwashCache, SwashContent};
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::config::Config;
use crate::terminal::{CellFlags, TermColor, TerminalState};

// ── Atlas constants ──────────────────────────────────────────────────────────

const ATLAS_SIZE: u32 = 2048;

// ── Vertex layout ────────────────────────────────────────────────────────────

/// One vertex of a screen-space quad (background or glyph).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    /// Pixel-space position [x, y]
    pos:   [f32; 2],
    /// Atlas UV (only meaningful for glyph quads)
    uv:    [f32; 2],
    /// RGBA colour (background colour for bg quads, fg colour for glyph quads)
    color: [f32; 4],
}

/// Uniform block shared by both pipelines.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    viewport: [f32; 2],
    _pad:     [f32; 2],
}

// ── WGSL shader (single module, two fragment entry points) ───────────────────

const SHADER_SRC: &str = r#"
struct Globals {
    viewport : vec2<f32>,
    _pad     : vec2<f32>,
};

@group(0) @binding(0) var<uniform> globals : Globals;

struct VertIn {
    @location(0) pos   : vec2<f32>,
    @location(1) uv    : vec2<f32>,
    @location(2) color : vec4<f32>,
};

struct VertOut {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0)       uv       : vec2<f32>,
    @location(1)       color    : vec4<f32>,
};

@vertex
fn vs_main(in : VertIn) -> VertOut {
    var out : VertOut;
    // Convert pixel-space → NDC  (Y-axis flipped: pixel 0 is at the top)
    let ndc_x =  (in.pos.x / globals.viewport.x) * 2.0 - 1.0;
    let ndc_y = -(in.pos.y / globals.viewport.y) * 2.0 + 1.0;
    out.clip_pos = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv       = in.uv;
    out.color    = in.color;
    return out;
}

// ── Background (solid colour) ────────────────────────────────────────────────
@fragment
fn fs_bg(in : VertOut) -> @location(0) vec4<f32> {
    return in.color;
}

// ── Glyph (alpha-mask from atlas) ────────────────────────────────────────────
@group(1) @binding(0) var t_atlas   : texture_2d<f32>;
@group(1) @binding(1) var s_atlas   : sampler;

@fragment
fn fs_glyph(in : VertOut) -> @location(0) vec4<f32> {
    let coverage = textureSample(t_atlas, s_atlas, in.uv).r;
    return vec4<f32>(in.color.rgb, coverage);
}
"#;

// ── Glyph atlas ──────────────────────────────────────────────────────────────

struct AtlasEntry {
    /// Pixel coordinates within the atlas texture
    x: u32, y: u32, w: u32, h: u32,
    /// Bearing from baseline (swash placement)
    left: i32, top: i32,
}

struct GlyphAtlas {
    texture: wgpu::Texture,
    view:    wgpu::TextureView,
    cache:   HashMap<cosmic_text::CacheKey, Option<AtlasEntry>>,
    /// Current shelf-packer cursor
    shelf_x: u32,
    shelf_y: u32,
    shelf_h: u32,
}

impl GlyphAtlas {
    fn new(device: &wgpu::Device) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("glyph_atlas"),
            size:  wgpu::Extent3d { width: ATLAS_SIZE, height: ATLAS_SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::R8Unorm,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[],
        });
        let view = texture.create_view(&Default::default());
        Self {
            texture, view,
            cache:   HashMap::new(),
            shelf_x: 0, shelf_y: 0, shelf_h: 0,
        }
    }

    /// Return a reference to the atlas entry for `key`, rasterising and
    /// uploading the glyph if it has not been seen before.
    /// Returns `None` for invisible (zero-size) glyphs such as spaces.
    fn get_or_insert(
        &mut self,
        key:         cosmic_text::CacheKey,
        font_system: &mut FontSystem,
        swash:       &mut SwashCache,
        queue:       &wgpu::Queue,
    ) -> Option<&AtlasEntry> {
        // Already cached (including "no pixels" sentinel)?
        if self.cache.contains_key(&key) {
            return self.cache[&key].as_ref();
        }

        let image = swash.get_image_uncached(font_system, key)?;
        let (w, h) = (image.placement.width, image.placement.height);

        if w == 0 || h == 0 {
            self.cache.insert(key, None);
            return None;
        }

        // Shelf packing
        if self.shelf_x + w > ATLAS_SIZE {
            self.shelf_y += self.shelf_h + 1;
            self.shelf_x  = 0;
            self.shelf_h  = 0;
        }
        if self.shelf_y + h > ATLAS_SIZE {
            log::warn!("Glyph atlas full – glyph will not be rendered");
            self.cache.insert(key, None);
            return None;
        }

        let (ax, ay) = (self.shelf_x, self.shelf_y);
        self.shelf_x += w + 1;
        self.shelf_h  = self.shelf_h.max(h);

        // Convert swash image content to a greyscale mask
        let mask: Vec<u8> = match image.content {
            SwashContent::Mask => image.data.clone(),
            SwashContent::SubpixelMask => {
                // Downsample RGB subpixel to single alpha channel
                image.data.chunks(3).map(|rgb| rgb[0]).collect()
            }
            SwashContent::Color => {
                // Use the alpha channel of colour glyphs (emoji, etc.)
                image.data.chunks(4).map(|rgba| rgba[3]).collect()
            }
        };

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture:   &self.texture,
                mip_level: 0,
                origin:    wgpu::Origin3d { x: ax, y: ay, z: 0 },
                aspect:    wgpu::TextureAspect::All,
            },
            &mask,
            wgpu::ImageDataLayout {
                offset:           0,
                bytes_per_row:    Some(w),
                rows_per_image:   None,
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );

        self.cache.insert(key, Some(AtlasEntry {
            x: ax, y: ay, w, h,
            left: image.placement.left,
            top:  image.placement.top,
        }));

        self.cache[&key].as_ref()
    }
}

// ── Colour helpers ───────────────────────────────────────────────────────────

/// Standard xterm 256-colour palette (the 6×6×6 colour cube + greyscale ramp).
fn xterm256(idx: u8) -> [f32; 4] {
    let (r, g, b) = match idx {
        // System colours 0-15: let the theme override these; return a
        // recognisable mid-grey so callers know to use the theme instead.
        0..=15  => return [0.5, 0.5, 0.5, 1.0],
        16..=231 => {
            let i = idx as u32 - 16;
            let r = (i / 36) % 6;
            let g = (i / 6)  % 6;
            let b =  i       % 6;
            let cv = |v: u32| if v == 0 { 0u8 } else { (55 + v * 40) as u8 };
            (cv(r), cv(g), cv(b))
        }
        232..=255 => {
            let grey = (idx as u32 - 232) * 10 + 8;
            (grey as u8, grey as u8, grey as u8)
        }
    };
    [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]
}

fn resolve_color(color: TermColor, theme: &crate::config::ThemeConfig, is_fg: bool) -> [f32; 4] {
    match color {
        TermColor::Default => {
            let hex = if is_fg { &theme.foreground } else { &theme.background };
            Config::parse_hex_color(hex)
        }
        TermColor::Named(idx) => Config::parse_hex_color(theme.ansi_color(idx)),
        TermColor::Indexed(idx @ 0..=15) => Config::parse_hex_color(theme.ansi_color(idx)),
        TermColor::Indexed(idx)          => xterm256(idx),
        TermColor::Rgb(r, g, b)          => {
            [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]
        }
    }
}

// ── Vertex helpers ───────────────────────────────────────────────────────────

/// Push two triangles (a quad) into `buf`.
#[inline]
fn push_quad(buf: &mut Vec<Vertex>, x: f32, y: f32, w: f32, h: f32, color: [f32; 4], uv0: [f32; 2], uv1: [f32; 2]) {
    let (x1, y1) = (x + w, y + h);
    let tl = Vertex { pos: [x,  y ], uv: [uv0[0], uv0[1]], color };
    let tr = Vertex { pos: [x1, y ], uv: [uv1[0], uv0[1]], color };
    let bl = Vertex { pos: [x,  y1], uv: [uv0[0], uv1[1]], color };
    let br = Vertex { pos: [x1, y1], uv: [uv1[0], uv1[1]], color };
    buf.extend_from_slice(&[tl, tr, bl,  tr, br, bl]);
}

// ── Row shaping cache ────────────────────────────────────────────────────────

/// Cached shaping result for a single terminal row.
/// Keyed by the row's text content; invalidated whenever the text changes.
struct RowShapeCache {
    /// The text that was used to produce these glyphs.
    text:   String,
    /// Shaped glyph data: (cache_key, x_pixel_offset, baseline_y, byte_offset_in_text)
    glyphs: Vec<(cosmic_text::CacheKey, f32, f32, usize)>,
}

// ── Renderer ─────────────────────────────────────────────────────────────────

pub struct Renderer {
    surface:            wgpu::Surface<'static>,
    device:             wgpu::Device,
    queue:              wgpu::Queue,
    surface_cfg:        wgpu::SurfaceConfiguration,
    bg_pipeline:        wgpu::RenderPipeline,
    glyph_pipeline:     wgpu::RenderPipeline,
    /// Dedicated pipeline for the cursor block; uses alpha blending so the
    /// cursor's alpha (0.6) actually blends over the cell underneath.
    cursor_pipeline:    wgpu::RenderPipeline,
    globals_buf:        wgpu::Buffer,
    globals_bg:         wgpu::BindGroup,
    atlas:              GlyphAtlas,
    /// wgpu sampler kept alive here (referenced through atlas_bg bind group)
    #[allow(dead_code)]
    atlas_sampler:      wgpu::Sampler,
    atlas_bg:           wgpu::BindGroup,
    /// BindGroupLayout kept for potential atlas rebuild
    #[allow(dead_code)]
    atlas_bgl:          wgpu::BindGroupLayout,
    pub font_system:    FontSystem,
    pub swash_cache:    SwashCache,
    pub cell_width:     f32,
    pub cell_height:    f32,
    pub width:          u32,
    pub height:         u32,
    /// Per-row shaping cache. Entry `i` is valid while `entry.text` matches
    /// the current row `i` content; cleared on resize.
    shape_cache:        Vec<Option<RowShapeCache>>,
}

impl Renderer {
    pub async fn new(window: Arc<Window>, cfg: &Config) -> Result<Self, Box<dyn std::error::Error>> {
        let size = window.inner_size();

        // ── wgpu instance / surface ──────────────────────────────────────────
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends:            wgpu::Backends::all(),
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
            flags:               wgpu::InstanceFlags::default(),
            gles_minor_version:  wgpu::Gles3MinorVersion::default(),
        });

        let surface = instance.create_surface(window)?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("No suitable wgpu adapter found")?;

        log::info!("wgpu adapter: {} ({:?})", adapter.get_info().name, adapter.get_info().backend);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label:             Some("weighterm-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits:   wgpu::Limits::default(),
                },
                None,
            )
            .await?;

        // ── Surface configuration ────────────────────────────────────────────
        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let surface_cfg = wgpu::SurfaceConfiguration {
            usage:                        wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width:                        size.width.max(1),
            height:                       size.height.max(1),
            present_mode:                 wgpu::PresentMode::AutoNoVsync,
            alpha_mode:                   caps.alpha_modes[0],
            view_formats:                 vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_cfg);

        // ── Shader module ────────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("terminal_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        // ── Bind group layouts ───────────────────────────────────────────────
        let globals_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("globals_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty:         wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let atlas_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("atlas_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Texture {
                        multisampled:  false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        // filterable:true for broader driver/backend compatibility;
                        // we still use Nearest filtering in the sampler for
                        // pixel-perfect glyph rendering.
                        sample_type:   wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // ── Globals uniform buffer ───────────────────────────────────────────
        let globals_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("globals_buf"),
            contents: bytemuck::bytes_of(&Globals {
                viewport: [size.width as f32, size.height as f32],
                _pad:     [0.0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let globals_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("globals_bg"),
            layout:  &globals_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: globals_buf.as_entire_binding(),
            }],
        });

        // ── Glyph atlas ──────────────────────────────────────────────────────
        let atlas = GlyphAtlas::new(&device);

        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:           Some("atlas_sampler"),
            address_mode_u:  wgpu::AddressMode::ClampToEdge,
            address_mode_v:  wgpu::AddressMode::ClampToEdge,
            mag_filter:      wgpu::FilterMode::Nearest,
            min_filter:      wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let atlas_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("atlas_bg"),
            layout:  &atlas_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::TextureView(&atlas.view),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                },
            ],
        });

        // ── Vertex buffer layout ─────────────────────────────────────────────
        let vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &wgpu::vertex_attr_array![
                0 => Float32x2,   // pos
                1 => Float32x2,   // uv
                2 => Float32x4,   // color
            ],
        };

        // ── Pipeline layouts ─────────────────────────────────────────────────
        let bg_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("bg_layout"),
            bind_group_layouts:   &[&globals_bgl],
            push_constant_ranges: &[],
        });

        let glyph_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("glyph_layout"),
            bind_group_layouts:   &[&globals_bgl, &atlas_bgl],
            push_constant_ranges: &[],
        });

        // ── Background pipeline (alpha blending so opacity < 1.0 shows through) ──
        let bg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("bg_pipeline"),
            layout: Some(&bg_layout),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: "vs_main",
                buffers:     &[vbl.clone()],
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: "fs_bg",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive:    wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample:   wgpu::MultisampleState::default(),
            multiview:     None,
        });

        // ── Cursor pipeline (same geometry as bg, always alpha-blended) ──────
        let cursor_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("cursor_pipeline"),
            layout: Some(&bg_layout),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: "vs_main",
                buffers:     &[vbl.clone()],
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: "fs_bg",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive:    wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample:   wgpu::MultisampleState::default(),
            multiview:     None,
        });

        // ── Glyph pipeline (alpha blending) ──────────────────────────────────
        let glyph_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("glyph_pipeline"),
            layout: Some(&glyph_layout),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: "vs_main",
                buffers:     &[vbl],
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: "fs_glyph",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive:    wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample:   wgpu::MultisampleState::default(),
            multiview:     None,
        });

        // ── Font system ──────────────────────────────────────────────────────
        let mut font_system = FontSystem::new();
        let swash_cache     = SwashCache::new();

        let (cell_width, cell_height) =
            Self::measure_cell(&mut font_system, cfg);

        log::info!("Cell size: {cell_width:.1}×{cell_height:.1} px");

        Ok(Self {
            surface, device, queue, surface_cfg,
            bg_pipeline, glyph_pipeline, cursor_pipeline,
            globals_buf, globals_bg,
            atlas, atlas_sampler, atlas_bg, atlas_bgl,
            font_system, swash_cache,
            cell_width, cell_height,
            width:  size.width,
            height: size.height,
            shape_cache: Vec::new(),
        })
    }

    // ── Cell size measurement ─────────────────────────────────────────────────

    fn measure_cell(font_system: &mut FontSystem, cfg: &Config) -> (f32, f32) {
        let line_h = cfg.font.size * 1.4;
        let metrics = Metrics::new(cfg.font.size, line_h);
        let mut buf = Buffer::new(font_system, metrics);
        buf.set_size(font_system, 1000.0, line_h);
        buf.set_text(
            font_system,
            "M",
            Attrs::new().family(Family::Name(&cfg.font.family)),
            Shaping::Basic,
        );
        buf.shape_until_scroll(font_system, false);

        let cell_w = buf
            .layout_runs()
            .next()
            .and_then(|run| run.glyphs.first())
            .map(|g| g.w)
            .unwrap_or(cfg.font.size * 0.6);

        (cell_w.ceil(), line_h.ceil())
    }

    // ── Resize ────────────────────────────────────────────────────────────────

    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        if new_width == 0 || new_height == 0 { return; }
        self.width  = new_width;
        self.height = new_height;
        self.surface_cfg.width  = new_width;
        self.surface_cfg.height = new_height;
        self.surface.configure(&self.device, &self.surface_cfg);

        // Update globals uniform
        self.queue.write_buffer(
            &self.globals_buf,
            0,
            bytemuck::bytes_of(&Globals {
                viewport: [new_width as f32, new_height as f32],
                _pad: [0.0; 2],
            }),
        );

        // Invalidate the row shaping cache; row count will have changed.
        self.shape_cache.clear();
    }

    /// Terminal grid dimensions in (cols, rows) given current cell size.
    pub fn grid_size(&self) -> (usize, usize) {
        let cols = (self.width  as f32 / self.cell_width ).floor() as usize;
        let rows = (self.height as f32 / self.cell_height).floor() as usize;
        (cols.max(1), rows.max(1))
    }

    // ── Render ────────────────────────────────────────────────────────────────

    pub fn render(
        &mut self,
        terminal: &TerminalState,
        cfg:      &Config,
    ) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view   = output.texture.create_view(&Default::default());

        let clear_color = Config::parse_hex_color(&cfg.theme.background);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame_encoder"),
        });

        // Build vertex data (needs mutable access to atlas/font_system/swash_cache)
        let (bg_verts, glyph_verts) = self.build_vertices(terminal, cfg);

        let bg_buf = if !bg_verts.is_empty() {
            Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some("bg_vbuf"),
                contents: bytemuck::cast_slice(&bg_verts),
                usage:    wgpu::BufferUsages::VERTEX,
            }))
        } else { None };

        let glyph_buf = if !glyph_verts.is_empty() {
            Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some("glyph_vbuf"),
                contents: bytemuck::cast_slice(&glyph_verts),
                usage:    wgpu::BufferUsages::VERTEX,
            }))
        } else { None };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear_color[0] as f64,
                            g: clear_color[1] as f64,
                            b: clear_color[2] as f64,
                            a: cfg.window.opacity as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes:         None,
                occlusion_query_set:      None,
            });

            // 1) Background quads
            if let Some(ref buf) = bg_buf {
                pass.set_pipeline(&self.bg_pipeline);
                pass.set_bind_group(0, &self.globals_bg, &[]);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..bg_verts.len() as u32, 0..1);
            }

            // 2) Glyph quads
            if let Some(ref buf) = glyph_buf {
                pass.set_pipeline(&self.glyph_pipeline);
                pass.set_bind_group(0, &self.globals_bg, &[]);
                pass.set_bind_group(1, &self.atlas_bg, &[]);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..glyph_verts.len() as u32, 0..1);
            }
        }

        // Cursor overlay
        let cursor_verts = self.build_cursor(terminal, cfg);
        if !cursor_verts.is_empty() {
            let cursor_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some("cursor_vbuf"),
                contents: bytemuck::cast_slice(&cursor_verts),
                usage:    wgpu::BufferUsages::VERTEX,
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cursor_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes:         None,
                occlusion_query_set:      None,
            });
            pass.set_pipeline(&self.cursor_pipeline);
            pass.set_bind_group(0, &self.globals_bg, &[]);
            pass.set_vertex_buffer(0, cursor_buf.slice(..));
            pass.draw(0..cursor_verts.len() as u32, 0..1);
        }

        self.queue.submit([encoder.finish()]);
        output.present();
        Ok(())
    }

    // ── Vertex builders ───────────────────────────────────────────────────────

    fn build_vertices(
        &mut self,
        terminal: &TerminalState,
        cfg:      &Config,
    ) -> (Vec<Vertex>, Vec<Vertex>) {
        let mut bg_verts    = Vec::with_capacity(terminal.cols * terminal.rows * 6);
        let mut glyph_verts = Vec::with_capacity(terminal.cols * terminal.rows * 6);

        let cw = self.cell_width;
        let ch = self.cell_height;

        // We need to borrow atlas, font_system, swash_cache, queue, and
        // shape_cache simultaneously. Rust allows multiple distinct fields to be
        // borrowed at once through explicit field paths within one scope.
        let atlas        = &mut self.atlas;
        let font_system  = &mut self.font_system;
        let swash_cache  = &mut self.swash_cache;
        let queue        = &self.queue;
        let shape_cache  = &mut self.shape_cache;
        let opacity      = cfg.window.opacity;

        // Ensure the cache vector is large enough.
        if shape_cache.len() < terminal.rows {
            shape_cache.resize_with(terminal.rows, || None);
        }

        for row in 0..terminal.rows {
            // ── Background quads ─────────────────────────────────────────────
            for col in 0..terminal.cols {
                let cell  = terminal.cell(col, row);
                let mut bg = resolve_color(cell.bg, &cfg.theme, false);
                if cell.flags.contains(CellFlags::REVERSE) {
                    bg = resolve_color(cell.fg, &cfg.theme, true);
                }
                // Propagate window opacity into each background quad's alpha so
                // that `cfg.window.opacity < 1.0` makes the terminal translucent.
                // The bg_pipeline uses alpha blending, so this value is honoured.
                bg[3] = opacity;
                let x = col as f32 * cw;
                let y = row as f32 * ch;
                push_quad(&mut bg_verts, x, y, cw, ch, bg, [0.0; 2], [0.0; 2]);
            }

            // ── Glyph quads (with optional ligature shaping) ─────────────────
            let line_text: String = (0..terminal.cols)
                .map(|c| {
                    let ch = terminal.cell(c, row).c;
                    if ch == '\0' { ' ' } else { ch }
                })
                .collect();

            // Per-row shaping cache: only re-invoke HarfBuzz / cosmic-text when
            // the row's text content has actually changed.
            let cache_hit = shape_cache
                .get(row)
                .and_then(|e| e.as_ref())
                .map(|e| e.text == line_text)
                .unwrap_or(false);

            let glyph_data: Vec<(cosmic_text::CacheKey, f32, f32, usize)> = if cache_hit {
                shape_cache[row].as_ref().unwrap().glyphs.clone()
            } else {
                let shaping = if cfg.font.ligatures { Shaping::Advanced } else { Shaping::Basic };
                let metrics = Metrics::new(cfg.font.size, ch);

                let mut buf = Buffer::new(font_system, metrics);
                buf.set_size(font_system, terminal.cols as f32 * cw, ch);
                buf.set_text(
                    font_system,
                    &line_text,
                    Attrs::new().family(Family::Name(&cfg.font.family)),
                    shaping,
                );
                buf.shape_until_scroll(font_system, false);

                let mut data: Vec<(cosmic_text::CacheKey, f32, f32, usize)> = Vec::new();
                for run in buf.layout_runs() {
                    for g in run.glyphs.iter() {
                        let phys = g.physical((0.0, 0.0), 1.0);
                        data.push((phys.cache_key, g.x, run.line_y, g.start));
                    }
                }
                // `buf` (and its font_system borrow) is dropped here
                drop(buf);

                // Store in cache
                shape_cache[row] = Some(RowShapeCache {
                    text:   line_text.clone(),
                    glyphs: data.clone(),
                });

                data
            };

            // ── Upload glyphs to atlas and emit quads ────────────────────────
            for (cache_key, gx, baseline, byte_off) in glyph_data {
                // Convert the UTF-8 byte offset back to a column index.
                // `line_text` has exactly one char per terminal column (wide
                // chars are stored in one cell; the next cell stores '\0'
                // which we mapped to ' ').  Because cells may contain multi-
                // byte UTF-8 characters we must count chars, not bytes.
                let safe_off = byte_off.min(line_text.len());
                let col = line_text[..safe_off].chars().count()
                    .min(terminal.cols.saturating_sub(1));
                let cell = terminal.cell(col, row);

                let mut fg = resolve_color(cell.fg, &cfg.theme, true);
                if cell.flags.contains(CellFlags::REVERSE) {
                    fg = resolve_color(cell.bg, &cfg.theme, false);
                }

                // Skip invisible characters
                if cell.flags.contains(CellFlags::HIDDEN) { continue; }

                let entry = atlas.get_or_insert(cache_key, font_system, swash_cache, queue);

                if let Some(e) = entry {
                    let qx = (gx + e.left as f32).round();
                    let qy = (row as f32 * ch + baseline - e.top as f32).round();
                    let qw = e.w as f32;
                    let qh = e.h as f32;

                    let u0 = e.x as f32         / ATLAS_SIZE as f32;
                    let v0 = e.y as f32         / ATLAS_SIZE as f32;
                    let u1 = (e.x + e.w) as f32 / ATLAS_SIZE as f32;
                    let v1 = (e.y + e.h) as f32 / ATLAS_SIZE as f32;

                    push_quad(&mut glyph_verts, qx, qy, qw, qh, fg, [u0, v0], [u1, v1]);
                }
            }
        }

        (bg_verts, glyph_verts)
    }

    fn build_cursor(&self, terminal: &TerminalState, cfg: &Config) -> Vec<Vertex> {
        let col = terminal.cursor.col.min(terminal.cols.saturating_sub(1));
        let row = terminal.cursor.row.min(terminal.rows.saturating_sub(1));
        let x   = col as f32 * self.cell_width;
        let y   = row as f32 * self.cell_height;
        // Draw a simple block cursor with a slight transparency
        let color = Config::parse_hex_color(&cfg.theme.cursor);
        let color = [color[0], color[1], color[2], 0.6];
        let mut v = Vec::new();
        push_quad(&mut v, x, y, self.cell_width, self.cell_height, color, [0.0; 2], [0.0; 2]);
        v
    }
}
