# What this costs

Short version: **local voice is free, chat is cents, and the screen loop is the
only thing that can surprise you.**

## The arithmetic

Anthropic bills images by tokens, roughly:

```
tokens ≈ (width × height) / 750
```

…after scaling the image so its long edge fits the model's limit (1,568px for
Haiku 4.5 and Sonnet; 2,576px for Opus 5). A 1080p frame is therefore about
1,844 tokens on Haiku/Sonnet and 2,765 on Opus.

At one frame every 10 seconds, 8 hours a day, 30 days — 86,400 frames — with no
local gating at all:

| Model | $/image | $/month |
|---|---|---|
| Haiku 4.5 | ~$0.0018 | **~$224** |
| Sonnet 5 | ~$0.0037 | ~$448 |
| Opus 5 | ~$0.0138 | ~$1,518 |

Run `jarvis estimate --resolution 2560x1440 --interval 5` for your own numbers.

## The four levers

1. **Don't send the frame.** The change detector rejects frames whose perceptual
   hash barely moved. On real desktop use most consecutive frames are identical,
   so an escalation rate of 10–20% is typical — a 5–10× saving, and by far the
   biggest one. Tune with `vision.change_threshold` (dHash distance, 0–64;
   higher = fewer escalations).
2. **Make the frame smaller.** Cost scales with pixel *area*, so halving each
   edge quarters the bill. `vision.max_edge_px = 1024` is a good default; text
   in most apps is still legible to the model.
3. **Slow the loop.** `vision.capture_interval_s = 30` cuts the frame count by
   3× versus 10s. A companion rarely needs 6 looks a minute.
4. **Use the cheap model.** Haiku 4.5 for the always-on loop; escalate only
   *interesting* frames to Sonnet if description quality is not good enough.
   Never run the loop on Opus.

Realistically: 1024px frames, every 30s, 12% escalation rate, on Haiku →
**a few dollars a month**.

## What prompt caching does and does not do

A cache entry is a **byte-exact prefix match** over `tools` → `system` →
`messages`. Everything before the breakpoint must be identical between requests,
so Jarvis splits the system prompt in two: the persona (static, marked) and
everything that moves — the clock, your remembered facts, recalled context
(after the breakpoint). Putting the time in the cached half is the classic way
to pay the write surcharge forever and never get a hit.

**Below a model-dependent minimum, nothing is cached and nothing says so:**

| Model | Minimum cacheable prefix |
|---|---:|
| Claude Opus 5 | 512 tokens |
| Claude Sonnet 5 | 1,024 tokens |
| Claude Haiku 4.5 | 4,096 tokens |

Jarvis estimates the static prefix (tool definitions + persona) and only sends
`cache_control` when it clears the model's minimum — otherwise the marker would
buy a 1.25× write for an entry that never exists. `jarvis doctor` reports which
case you are in:

```
 + prompt cache   no-op   static prefix ~336 tokens < 1024 for claude-sonnet-5
```

That is the honest default position: the shipped persona is ~67 tokens and the
tools ~270, so **caching does nothing until your static half gets substantial** —
a long persona, house rules, a style guide. Once it does, reads cost ~0.1× and
two requests sharing the prefix already beat two uncached ones.

Verify rather than assume — `jarvis memory` prints today's counters:

```
prompt cache today: 12400 tokens read, 1100 written, over 14 calls
```

Zero reads across many calls means something is invalidating the prefix.

Caching does **not** help the screen loop at all: the screenshot changes every
frame, so it can never be a hit. Caching trims text overhead; only the change
detector trims the dominant image bill.

## The hard stop

`vision.daily_budget_usd` is checked before every escalation against spend
recorded in the `usage` table. When today's spend plus the estimated cost of the
next image would exceed it, `tick()` returns `over-budget` and no call is made.
Because the accounting lives in SQLite, a restart loop cannot spend past the cap.

Check it any time:

```bash
jarvis memory          # includes "spend today"
```

## Voice

| Component | Local (default) | Hosted |
|---|---|---|
| STT | faster-whisper / whisper.cpp — **$0** | Deepgram, AssemblyAI |
| TTS | Kokoro-82M — **$0** | ElevenLabs (~$5–$99/mo typical personal tiers) |

Keeping STT and TTS local is also the right latency call: reserve the network
round-trip for reasoning and vision.

## A caveat

Model prices and image-token caps move, and third-party summaries of them
disagree. The table in `perception/costs.py` is the single place to update, and
`jarvis estimate` prints a reminder to check the live pricing page before
trusting any monthly figure.
