# The orb, on your desktop

A [Tauri v2](https://tauri.app) shell that puts the Jarvis orb in a frameless,
transparent, always-on-top window that the rest of your desktop can be clicked
straight through.

The shell is deliberately thin. The Python core is still the source of truth —
this window just points a webview at `http://127.0.0.1:8765`. It owns only the
three things a browser tab cannot do:

| | |
|---|---|
| **Float** | frameless, transparent, always-on-top, off the taskbar |
| **Get out of the way** | click-through everywhere except the orb's disc |
| **Lifecycle** | optionally start the core and stop it again on exit |

## Run it

```bash
# terminal 1 - the core
jarvis ui

# terminal 2 - the shell
cd desktop/src-tauri
cargo run              # or: cargo tauri dev, if you have the Tauri CLI
```

The window opens on the bundled bootstrap page, polls `/api/status`, and hands
itself over to the core the moment it answers — so starting them in either
order works.

To let the shell own the core instead:

```bash
JARVIS_CORE_CMD="python3 -m jarvis ui" cargo run
```

## Settings

All environment variables, all optional:

| Variable | Default | Meaning |
|---|---|---|
| `JARVIS_CORE_URL` | `http://127.0.0.1:8765` | Where the core is listening. Must match `[ui]` in `config.toml` |
| `JARVIS_CORE_CMD` | unset | Command to start the core; unset means attach to a running one |
| `JARVIS_ORB_POSITION` | `bottom-right` | `top-left`, `top-right`, `bottom-left`, `bottom-right`, `center` |
| `JARVIS_ORB_CLICK_THROUGH` | on | Set to `0` to make the whole window catch the mouse |

## How click-through works

Tauri cannot hit-test the DOM, so the shell reproduces the orb's shape instead:
a thread samples the cursor every 50ms and turns `set_ignore_cursor_events` on
whenever the pointer is outside a circle of `ORB_RADIUS_RATIO` × the window's
short side. That only matches reality because the window loads the page with
`?mode=orb`, which hides the chat panel and footer and scales the orb to fill
the window. `tests/test_desktop_shell.py` keeps the two in step.

## Navigation is pinned

The window is built in `setup()` rather than declared with `create: true`,
because `on_navigation` can only be attached to a `WebviewWindowBuilder`. It
allows the bundled assets and the core's origin, and nothing else: a frameless
always-on-top window with no address bar should not be able to end up showing
an arbitrary page.

## Icons

```bash
python desktop/scripts/make_icons.py
```

Generates the PNG set and `icon.ico` — a glowing orb on transparency, drawn
with nothing but `zlib` and `struct`, in the same palette as `orb.js`. Regenerate
rather than hand-editing. macOS bundles also want an `.icns`; produce one with
`cargo tauri icon desktop/src-tauri/icons/icon.png`.

## Building a bundle

```bash
cargo install tauri-cli --version '^2'
cargo tauri build
```

Linux also needs the webview development packages, e.g. on Ubuntu 24.04:

```bash
sudo apt install libwebkit2gtk-4.1-dev libsoup-3.0-dev build-essential curl file
```

## What has and has not been verified

`cargo check` and `cargo clippy` pass cleanly against Tauri 2.11.5. The window
has **not** been launched here — this container has no display — so the visual
behaviour (transparency, always-on-top, the click-through radius, corner
placement) is unverified on a real desktop and may need tuning per platform.
Transparent always-on-top windows are notoriously compositor-dependent: on
Linux, transparency needs a compositing window manager, and `alwaysOnTop` is
advisory on some tiling WMs.
