//! A transparent, always-on-top shell for the Jarvis orb.
//!
//! The Python core is the source of truth; this window is a viewport onto it.
//! It owns exactly three things the browser cannot do:
//!
//!   * a frameless transparent window that floats above other applications;
//!   * click-through, so the desktop keeps working around the orb;
//!   * optionally starting the core process and stopping it again on exit.
//!
//! Everything else -- what the orb looks like, what it says -- comes from
//! `http://127.0.0.1:8765`, served by `jarvis ui`.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;

use tauri::{LogicalPosition, Manager, RunEvent, Url, WebviewWindow, WebviewWindowBuilder};

/// Where the Python core is listening. Must match `[ui] port` in config.toml.
const DEFAULT_CORE_URL: &str = "http://127.0.0.1:8765";
/// How often the cursor is sampled for the click-through test.
const CURSOR_POLL: Duration = Duration::from_millis(50);
/// Fraction of the window half-width that counts as "over the orb".
const ORB_RADIUS_RATIO: f64 = 0.46;
/// Gap from the screen edge when the orb is parked in a corner.
const MARGIN: f64 = 24.0;

/// Pages this window is allowed to show: its own bundled assets, and the core.
///
/// The orb is a viewport onto one local service. Anything else -- a link in a
/// reply, a compromised page -- must not be able to navigate this frameless,
/// always-on-top window somewhere the user cannot see the address of.
fn navigation_allowed(url: &Url, core: &Url) -> bool {
    let bundled = url.scheme() == "tauri" || url.host_str() == Some("tauri.localhost");
    bundled || url.origin() == core.origin()
}

/// The core process, when this shell started one, so it can be stopped on exit.
struct CoreProcess(Mutex<Option<Child>>);

fn core_url() -> String {
    std::env::var("JARVIS_CORE_URL").unwrap_or_else(|_| DEFAULT_CORE_URL.to_string())
}

/// Start the Python core, but only when explicitly asked to.
///
/// Attaching to an already-running `jarvis ui` is the default because it keeps
/// one obvious owner of the core process. Set `JARVIS_CORE_CMD` to have the
/// shell own it instead, e.g. `JARVIS_CORE_CMD="python3 -m jarvis ui"`.
fn spawn_core() -> Option<Child> {
    let command = std::env::var("JARVIS_CORE_CMD").ok()?;
    let mut parts = command.split_whitespace();
    let program = parts.next()?;
    match Command::new(program).args(parts).spawn() {
        Ok(child) => Some(child),
        Err(error) => {
            eprintln!("jarvis-orb: could not start the core ({command}): {error}");
            None
        }
    }
}

/// Park the orb in a corner of its monitor: `JARVIS_ORB_POSITION=bottom-right`.
///
/// The window is frameless, so there is no title bar to drag; a fixed corner is
/// more predictable than remembering a position nobody can see themselves set.
fn place_window(window: &WebviewWindow) {
    let corner = std::env::var("JARVIS_ORB_POSITION").unwrap_or_else(|_| "bottom-right".into());
    let Ok(Some(monitor)) = window.current_monitor() else {
        return;
    };
    let scale = monitor.scale_factor();
    let screen = monitor.size().to_logical::<f64>(scale);
    let origin = monitor.position().to_logical::<f64>(scale);
    let Ok(size) = window.outer_size() else {
        return;
    };
    let size = size.to_logical::<f64>(scale);

    let right = origin.x + screen.width - size.width - MARGIN;
    let bottom = origin.y + screen.height - size.height - MARGIN;
    let left = origin.x + MARGIN;
    let top = origin.y + MARGIN;
    let (x, y) = match corner.as_str() {
        "top-left" => (left, top),
        "top-right" => (right, top),
        "bottom-left" => (left, bottom),
        "center" => (
            origin.x + (screen.width - size.width) / 2.0,
            origin.y + (screen.height - size.height) / 2.0,
        ),
        _ => (right, bottom),
    };
    let _ = window.set_position(LogicalPosition::new(x, y));
}

/// True when the pointer is over the orb itself rather than the transparent
/// corners of its window.
fn cursor_over_orb(window: &WebviewWindow) -> Option<bool> {
    let cursor = window.app_handle().cursor_position().ok()?;
    let position = window.outer_position().ok()?;
    let size = window.outer_size().ok()?;

    let centre_x = position.x as f64 + size.width as f64 / 2.0;
    let centre_y = position.y as f64 + size.height as f64 / 2.0;
    let radius = size.width.min(size.height) as f64 * ORB_RADIUS_RATIO;

    let dx = cursor.x - centre_x;
    let dy = cursor.y - centre_y;
    Some(dx * dx + dy * dy <= radius * radius)
}

/// Toggle click-through so only the orb's disc catches the mouse.
///
/// Tauri cannot hit-test the DOM, so the shape is reproduced here: the page is
/// loaded with `?mode=orb`, which hides everything except the centred orb.
fn watch_cursor(window: WebviewWindow) {
    std::thread::spawn(move || {
        let mut ignoring: Option<bool> = None;
        loop {
            std::thread::sleep(CURSOR_POLL);
            if window.is_closable().is_err() {
                return; // the window went away; stop polling
            }
            let over_orb = cursor_over_orb(&window).unwrap_or(false);
            if ignoring != Some(!over_orb) {
                let _ = window.set_ignore_cursor_events(!over_orb);
                ignoring = Some(!over_orb);
            }
        }
    });
}

fn main() {
    let app = tauri::Builder::default()
        .manage(CoreProcess(Mutex::new(spawn_core())))
        .setup(|app| {
            let config = app
                .config()
                .app
                .windows
                .first()
                .cloned()
                .expect("the 'main' window is declared in tauri.conf.json");
            let core = Url::parse(&core_url()).expect("JARVIS_CORE_URL must be a URL");
            let allowed = core.clone();

            let window = WebviewWindowBuilder::from_config(app, &config)?
                .on_navigation(move |url| navigation_allowed(url, &allowed))
                .build()?;

            place_window(&window);
            window.show()?;
            if std::env::var("JARVIS_ORB_CLICK_THROUGH").as_deref() != Ok("0") {
                watch_cursor(window.clone());
            }
            println!("jarvis-orb: attached to {core}");
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to start the Jarvis orb");

    app.run(|handle, event| {
        if let RunEvent::ExitRequested { .. } = event {
            // A core this shell started is a child of this shell: take it with us.
            if let Some(mut child) = handle.state::<CoreProcess>().0.lock().unwrap().take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    });
}
