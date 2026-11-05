# AI Agents

Everything the bot uses AI for is configured on one page: **`/ai-agents`**. It
replaced the old "AI APIs" block on the Settings page, which was three free-text
fields and a hard-coded model.

## The rule

**One provider per task.** A task category resolves to exactly one provider and
one model. Models from different providers are never offered together, never
combined, and never silently substituted for each other.

The flow is always:

```
Provider  →  Category  →  Model  →  Model configuration
```

Nothing below a step is shown until that step is answered.

## The two categories

| Category | Used for | Configured model drives |
|---|---|---|
| **Chat, Text & Analysis AI** | brief analysis, brief condensation, reference-image vision, design critique, art direction, proposal writing + review, submission descriptions | `bot/llm.py` |
| **Image Generation AI** | text-to-image, image-to-image, reference-driven generation, critic refinement | `bot/generator.py` + `bot/image_providers.py` |

## Models

Parameters are exactly what each provider's official docs define — nothing more.

### APIDot (`https://api.apidot.ai`)

| Model | Endpoint | Parameters |
|---|---|---|
| GPT-5.5 | `/v1/chat/completions` | temperature (0–2), top_p, max_tokens |
| GPT-5.6 | `/v1/responses` | variant (luna/terra/sol), reasoning effort (low…max), max_output_tokens |
| Claude Sonnet 5 | `/v1/messages` | max_tokens, temperature (0–1), top_p, top_k, stop_sequences |
| Claude Opus 5 | `/v1/messages` | as Sonnet 5 |
| GPT Image 2.5 | `/api/generate/submit` | variant (Flare/Sunburst), quality (low…max), resolution (1K/2K/4K), canvas (11 ratios + custom W×H), background, output format (PNG/JPEG/WEBP), reference images |
| GPT Image 2 | `/api/generate/submit` | quality (low/medium/high), resolution, canvas (11 ratios + custom W×H). **No** background, output format or mask — its page documents none |


### Kie.ai (`https://api.kie.ai`)

| Model | Endpoint | Parameters |
|---|---|---|
| GPT-5.5 | `/codex/v1/responses` | reasoning effort (low…xhigh), tools (none / web search) — vision-capable. **Currently unstable: ~70% of calls return HTTP 500** (measured 2026-09-18); it falls over to GPT-5.2 automatically |
| GPT-5.2 | `/gpt-5-2/v1/chat/completions` | reasoning effort (low/high), max tokens, temperature, tools — vision + JSON mode. Slower (8–17 s) but answered 8/8 in the same window |
| GPT Image 2.5 Flare | `/api/v1/jobs/createTask` | canvas (13 ratios), resolution, background |
| GPT Image 2.5 Sunburst | `/api/v1/jobs/createTask` | canvas (13 ratios), resolution, background |

### PoYo (`https://api.poyo.ai`) — image generation only

APIDot's contract on PoYo's host: the same submit body, status states, upload
route, balance shape and webhook body.

| Model | Endpoint | Parameters |
|---|---|---|
| GPT Image 2.5 Flare | `/api/generate/submit` | quality (low…max), resolution (1K/2K/4K), canvas (10 ratios + auto + custom W×H), background, output format (PNG/JPEG/WEBP); edits with up to 16 reference images |
| GPT Image 2.5 Sunburst | `/api/generate/submit` | same fields as Flare |

### KovAPI (`https://kovapi.com`) — image generation only

OpenAI-style asynchronous jobs: `POST /v1/images/generations` answers 202 with a
job id, `GET /v1/images/jobs/{id}` reports `queued → processing → succeeded |
failed`. Priced per tier only — 1K $0.005, 2K $0.03, 4K $0.05 per image; a
failed job is refunded.

| Model | Endpoint | Parameters |
|---|---|---|
| GPT Image 2.5 Flare | `/v1/images/generations` | resolution (1K/2K/4K), canvas (7 documented ratios + auto), quality (1K only); up to 8 reference images, sent inline |
| GPT Image 2.5 Sunburst | `/v1/images/generations` | same fields as Flare |

KovAPI also accepts `background`, `output_format` and `style` but documents
them as "not guaranteed to take effect; output is always PNG", and recommends
against `n` — so none of them is offered.

### APIMart (`https://api.apimart.ai`) — image generation only

Asynchronous tasks: `POST /v1/images/generations` returns a task id, results are
polled at `GET /v1/tasks/{id}` (or pushed to the webhook). Two documented model
families:

| Model | Parameters |
|---|---|
| GPT Image 2.5 Flare (`gpt-image-2.5-flare`) | quality (auto/low…max), resolution (`1k`/`2k`/`4k`, ratio canvases only), canvas (auto + 15 ratios incl. 2:1, 3:1, 9:21 + exact W×H), background, output format, moderation (low/auto); edits with up to 16 reference images |
| GPT Image 2.5 Sunburst (`gpt-image-2.5-sunburst`) | same fields as Flare; favours editing precision |
| GPT Image 2.5 Ext (`gpt-image-2.5-ext`) | version (Flare/Sunburst), resolution (`1K`/`2K`/`4K`), canvas (auto + 10 ratios, no exact sizes); up to 16 reference images |

- Flare and Sunburst are billed by tokens: a 1024² image is about $0.006 at low,
  $0.013 medium, $0.053 high, $0.094 xhigh and $0.21 max. **Quality "Auto"** (the
  documented default) reserves funds at the Max price until the task settles —
  choose High for a fixed, predictable cost. Ext is billed per image by version
  and resolution.
- An exact W×H follows the same limits as on APIDot/PoYo (multiples of 16, ≤ 3840,
  ≤ 3:1, 655,360–8,294,400 px; above 2560×1440 is experimental) but the
  resolution tier is simply ignored for it, and `auto` works at every tier.
- References are uploaded to `POST /v1/uploads/images` (kept 72 h) and sent as URLs.
- The webhook is a **base** URL: the bot sends `<server_host>/api/apimart` and
  APIMart posts to `/api/apimart/callback` — public http(s) only, ports 80/443
  recommended.
- **The balance shown is this key's own quota** (`GET /v1/balance`), so a key
  whose quota is used up is caught by the free Test connection and before a
  batch; an unlimited key shows the account (`GET /v1/user/balance`).

## Things worth knowing

- **You never pick the generation mode — the brief analysis does.** A contest
  that ships reference files is generated image-to-image; one without is
  generated from the prompt alone. Where a provider splits those into two model
  ids (kie.ai's Flare/Sunburst pairs, APIDot's `gpt-image-2-edit`), the bot
  switches between them per contest, so only the entry model of each family is
  offered here.
- **Canvas "Auto" follows the brief.** The analyzer picks an aspect ratio per
  contest (from the brief's stated size when it gives one); "Auto" honours it,
  and any other value forces that canvas. A ratio the selected model doesn't
  document moves to the nearest one it does, or to `auto` when none is close.
- **GPT Image 2.5's canvas rules are enforced, not guessed** (APIDot and PoYo
  document the same ones): `auto` only exists at 1K — at 2K/4K a contest with no
  ratio gets the documented 1:1 default — and a custom W×H needs 2K or 4K, sides
  in multiples of 16, a long edge ≤ 3840 (which needs 4K), at most 3:1, and
  655,360–8,294,400 pixels. Saving an impossible combination corrects it and
  says what changed; nothing invalid is ever sent.
- **KovAPI renders 1K by cropping.** At 1K it draws the nearest square / 3:2 /
  2:3 frame and centre-crops it to the canvas, so the prompt for a wide or tall
  canvas gains a "keep everything central" line. Choose 2K or 4K for native
  ratios (6× / 10× the 1K price).
- **References go where each provider can read them.** Kie.ai, APIDot, PoYo and
  APIMart get the contest's images through their own upload endpoints; KovAPI
  has none, so references travel inside the request as compressed data URIs
  (≤ 1536 px).
- **Max output tokens is a ceiling.** Stages request what they need; the
  configured value caps them. Below ~3000 the brief-analysis JSON truncates.
- **Temperature / reasoning effort apply to calls that don't pin their own.**
  Most stages pin task-specific values (0.2 for critique, 0.7 for art direction).
- **Streaming is shown but disabled** on every chat model: the pipeline consumes
  whole responses before scoring or saving them.
- **Extended thinking is shown but disabled** on the APIDot Claude models:
  APIDot names a `thinking` object without publishing its schema, and the bot
  does not guess request shapes.
- **Failover never leaves the provider.** If a route dies after its retries, the
  call is retried once on the same provider's sibling model (APIDot's chat and
  messages routes fail independently — kie.ai's Claude route was down for hours
  on 2026-08-03 while its GPT route was healthy). Kie.ai publishes one chat
  model, so there the failure surfaces. Toggle it off under Advanced.
- **Test connection is free; Test model costs credits.** See the section below —
  the two are deliberately separate actions.
- **Webhooks are optional and only sent to a reachable address.** With
  `server_host` set to a public URL, results arrive by webhook
  (`/api/kie-callback`, `/api/apidot-callback`, `/api/poyo-callback`,
  `/api/kovapi-callback`, `/api/apimart/callback`); otherwise — or while
  `server_host` is the dev default
  `http://localhost:5000` — they are read by polling. PoYo only delivers to
  HTTPS, and KovAPI rejects a job whose callback is not a public address, so the
  bot leaves the field out rather than fail the job.
- **Variations render in parallel.** All of a contest's variations are
  submitted first and collected together, so three variations take about one
  render time.

## When something goes wrong

Every AI failure is classified into a **category** and a **scope**, and the two
decide what happens next. Nothing is ever reported as a vague "LLM unavailable".

| Category | Retried? | Sibling model tried? | What you see |
|---|---|---|---|
| BILLING | no | **no** | "APIDot account has insufficient credits — add credits to resume AI requests." (a capped key says "raise this key's credit limit" instead) |
| AUTHENTICATION | no | **no** | "APIDot authentication failed — check the API key." (a KovAPI key refused by its model / IP allowlist says so) |
| RATE_LIMIT | yes, with backoff | no | "rate limit reached — requests are being slowed down" |
| MODEL_UNAVAILABLE | no | yes | "model 'X' is unavailable: …" |
| UPSTREAM / TIMEOUT / NETWORK | yes, bounded | yes | "…is having server-side trouble" |
| INVALID_REQUEST | no | no | the provider's own rejection text |
| PARSING | no | no | `APIDOT_RESPONSE_PARSE_ERROR` + which keys came back |

Two rules follow, and they matter for your credit balance:

- **Billing and authentication stop the whole provider.** Every model behind one
  key shares one balance, so trying the second model is a second wasted request.
  The provider is put out of service for the workspace, and further contests
  fail instantly and locally — no HTTP call at all — until it recovers.
- **Contests stopped this way are `blocked`, not `failed`.** The brief is fine;
  the account is not. Click **Test Connection** once credits are back and every
  blocked contest is requeued automatically.

The breaker re-checks itself after 15 minutes (30 for an auth failure), and
clears immediately when you save a new key, run a successful Test Connection, or
any AI call succeeds.

## Test Connection vs Test Model

- **Test Connection** — free. Checks authentication and reads the credit
  balance (`GET /api/user/balance`), plus the media status endpoint. Use it
  freely; it never runs a generation. It also unblocks a previously blocked
  provider and requeues blocked contests when it passes.
- **Test Model** — spends a few credits. Sends one minimal real request
  ("Reply with exactly: APIDOT_OK") to the selected chat model and reports the
  reply, latency and token usage. For PoYo, KovAPI and APIMart, which serve
  images only here, it submits one real 1K, low-quality, square test image
  instead (KovAPI: one credit, $0.005; APIMart Flare/Sunburst ≈ $0.006) and
  reports whether the job was accepted — the only check that
  can see a key's own budget or cap. It refuses to run when the provider is
  already known to be out of credits, so it cannot waste a request proving the
  obvious.

Every provider publishes a balance endpoint, so all show real credits: APIDot
and PoYo via `/api/user/balance`, kie.ai via `/api/v1/chat/credit`, KovAPI via
`/v1/usage` (`balance.credits`, 200 credits = $1), APIMart via `/v1/balance`
(the key's own `remain_credits`; `/v1/user/balance` when the key is unlimited).

Before a batch of contests is analysed, the image provider gets one real check
so a capped key cannot cost a run of paid analyses: a one-token chat ping for
kie.ai and APIDot, and for PoYo / KovAPI / APIMart one test image — skipped whenever a
real image job succeeded in the last six hours.

**KovAPI's 402 is ambiguous by design** — `insufficient_quota` covers an empty
balance, the key's daily budget, the key's total quota and a burst outrunning
the pre-charge reserve, and its wording is not stable. When the account balance
is healthy (≥ 30 credits) the bot reports it as the key's own limit and holds the
provider for a day (cleared by Test model or a new key), instead of telling you to
add credits.

## If kie.ai returns "Server exception, please try again later"

That is an HTTP 500 from kie.ai's own upstream, not a problem with your key,
your credits or your settings. The bot already handles it: it retries three
times, then switches to **GPT-5.2** (a different route on the same account) and
keeps using it for ten minutes before re-testing GPT-5.5.

If you want to avoid the retry delay altogether, select **GPT-5.2** as the chat
model on this page — it is the more stable route, at the cost of a slower reply.

## Upgrading an existing workspace

Nothing to do. A Space that only ever set the old `llm_model` / `image_gen_model`
values resolves through `LEGACY_*_MODEL_MAP` to a current model, so generation
keeps working until someone opens the page and chooses. Saving a configuration
keeps those legacy keys mirrored.

## Adding a model or provider

Edit `bot/ai_registry.py` — one dict in `MODELS` with its parameter specs. The
page, the save validation and the wire payload all follow. A genuinely new wire
shape also needs a transport: a branch in `LLMClient._build` + `_parse` (chat)
or a class in `bot/image_providers.py` (image). `tests/test_ai_registry.py` enforces the
structural invariants, including that no failover or sibling ever crosses
providers.
