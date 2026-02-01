//! Shared utility module for examples.
//!
//! Provides common functionality like FPS display that can be used across examples.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;

/// Plugin that adds FPS display functionality.
///
/// Usage:
/// ```rust
/// App::new()
///     .add_plugins(DefaultPlugins)
///     .add_plugins(FpsPlugin)
///     // ...
/// ```
pub struct FpsPlugin;

impl Plugin for FpsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(FpsPrevState::default())
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_systems(Startup, spawn_fps_text)
            .add_systems(Update, update_fps_text);
    }
}

/// Marker component for the FPS parent node.
#[derive(Component)]
struct FpsText;

/// Marker components for individual text lines
#[derive(Component)]
struct FpsValue;

#[derive(Component)]
struct Below50Count;

#[derive(Component)]
struct Below30Count;

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Static atomic counters accessible after `app.run()` completes.
/// Use the methods on `FpsStats` to update/read counts and statistics.
pub struct FpsStats;

static GLOBAL_BELOW_50: AtomicU32 = AtomicU32::new(0);
static GLOBAL_BELOW_30: AtomicU32 = AtomicU32::new(0);
static GLOBAL_LAST_FPS: AtomicU32 = AtomicU32::new(0);
static GLOBAL_MIN_FPS: AtomicU32 = AtomicU32::new(u32::MAX);
static GLOBAL_FPS_SAMPLES: AtomicU32 = AtomicU32::new(0);
static GLOBAL_FPS_SUM: AtomicU64 = AtomicU64::new(0);
const MIN_SAMPLE_FRAMES: u32 = 5; // require this many frames before considering min

impl FpsStats {
    pub fn increment_below_50() {
        GLOBAL_BELOW_50.fetch_add(1, Ordering::Relaxed);
    }
    pub fn increment_below_30() {
        GLOBAL_BELOW_30.fetch_add(1, Ordering::Relaxed);
    }
    pub fn set_fps(v: u32) {
        // store last FPS and bump sample counter
        GLOBAL_LAST_FPS.store(v, Ordering::Relaxed);
        let samples = GLOBAL_FPS_SAMPLES.fetch_add(1, Ordering::Relaxed) + 1;

        // accumulate sum for average calculation
        GLOBAL_FPS_SUM.fetch_add(v as u64, Ordering::Relaxed);

        // don't update min too early (ignore the first few frames)
        if samples < MIN_SAMPLE_FRAMES {
            return;
        }

        // update min (atomic compare-exchange loop)
        let mut cur = GLOBAL_MIN_FPS.load(Ordering::Relaxed);
        while v < cur {
            match GLOBAL_MIN_FPS.compare_exchange(cur, v, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
    }
    pub fn below_50() -> u32 {
        GLOBAL_BELOW_50.load(Ordering::Relaxed)
    }
    pub fn below_30() -> u32 {
        GLOBAL_BELOW_30.load(Ordering::Relaxed)
    }
    pub fn last_fps() -> u32 {
        GLOBAL_LAST_FPS.load(Ordering::Relaxed)
    }
    pub fn avg_fps() -> Option<f64> {
        let samples = GLOBAL_FPS_SAMPLES.load(Ordering::Relaxed) as u64;
        if samples == 0 {
            return None;
        }
        let sum = GLOBAL_FPS_SUM.load(Ordering::Relaxed) as f64;
        Some(sum / (samples as f64))
    }
    pub fn min_fps() -> Option<u32> {
        let v = GLOBAL_MIN_FPS.load(Ordering::Relaxed);
        if v == u32::MAX { None } else { Some(v) }
    }
}

/// Resource that remembers previous below-threshold states to detect transitions.
#[derive(Resource, Default)]
struct FpsPrevState {
    was_below_50: bool,
    was_below_30: bool,
}

/// Spawns the FPS text and counters in the top-left corner.
fn spawn_fps_text(mut commands: Commands) {
    // Insert prev state resource (counters are static atomic globals)
    commands.insert_resource(FpsPrevState::default());

    commands
        .spawn((
            FpsText,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                left: Val::Px(10.0), // top-left
                padding: UiRect::all(Val::Px(5.0)),
                border: UiRect::all(Val::Px(5.0)),
                ..default()
            },
        ))
        .with_children(|parent| {
            // FPS value (large)
            parent.spawn((
                FpsValue,
                Text::new("FPS: --"),
                TextFont {
                    font_size: 18.0,
                    ..default()
                },
                TextColor(Color::srgb(0.0, 1.0, 0.0)),
            ));

            // Below 50 counter (yellow)
            parent.spawn((
                Below50Count,
                Text::new("Below 50: 0"),
                TextFont {
                    font_size: 18.0,
                    ..default()
                },
                TextColor(Color::srgb(1.0, 0.8, 0.0)),
            ));

            // Below 30 counter (red)
            parent.spawn((
                Below30Count,
                Text::new("Below 30: 0"),
                TextFont {
                    font_size: 14.0,
                    ..default()
                },
                TextColor(Color::srgb(1.0, 0.0, 0.0)),
            ));
        });
}

/// Updates the FPS text each frame and updates counters when thresholds are crossed downward.
fn update_fps_text(
    diagnostics: Res<DiagnosticsStore>,
    mut prev: ResMut<FpsPrevState>,
    mut set: ParamSet<(
        Query<(&mut Text, &mut TextColor), With<FpsValue>>,
        Query<(&mut Text, &mut TextColor), With<Below50Count>>,
        Query<(&mut Text, &mut TextColor), With<Below30Count>>,
    )>,
) {
    if let Some(fps_diag) = diagnostics.get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FPS) {
        if let Some(value) = fps_diag.smoothed() {
            let fps_value = value;

            // Update main FPS text and color (param set partition 0)
            if let Ok((mut text, mut text_color)) = set.p0().single_mut() {
                **text = format!("FPS: {fps_value:.0}");

                let fps_color = if fps_value >= 50.0 {
                    Color::srgb(0.0, 1.0, 0.0)
                } else if fps_value >= 30.0 {
                    Color::srgb(1.0, 0.8, 0.0)
                } else {
                    Color::srgb(1.0, 0.0, 0.0)
                };

                *text_color = TextColor(fps_color);
            }

            FpsStats::set_fps(fps_value as u32);

            // Detect downward transitions and increment counters
            let now_below_50 = fps_value < 50.0;
            let now_below_30 = fps_value < 30.0;

            if now_below_50 && !prev.was_below_50 {
                FpsStats::increment_below_50();
            }
            if now_below_30 && !prev.was_below_30 {
                FpsStats::increment_below_30();
            }

            prev.was_below_50 = now_below_50;
            prev.was_below_30 = now_below_30;

            // Update counter displays (param set partitions 1 & 2)
            if let Ok((mut t50, mut c50)) = set.p1().single_mut() {
                **t50 = format!("Below 50: {}", FpsStats::below_50());
                *c50 = TextColor(Color::srgb(1.0, 0.8, 0.0));
            }
            if let Ok((mut t30, mut c30)) = set.p2().single_mut() {
                **t30 = format!("Below 30: {}", FpsStats::below_30());
                *c30 = TextColor(Color::srgb(1.0, 0.0, 0.0));
            }
        }
    }
}

/// Prints FPS counters (reads from static counters).
pub fn print_fps() {
    let last = FpsStats::last_fps();
    let avg = FpsStats::avg_fps().map_or_else(|| "N/A".to_string(), |v| format!("{v:.1}"));
    let min = FpsStats::min_fps().map_or_else(|| "N/A".to_string(), |v| v.to_string());
    info!(
        "FPS last: {}, avg: {}, min: {}, Threshold Crossings - Below 50: {}, Below 30: {}",
        last,
        avg,
        min,
        FpsStats::below_50(),
        FpsStats::below_30()
    );
}
