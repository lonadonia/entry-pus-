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

## Things worth knowing

- **You never pick the generation mode — the brief analysis does.** A contest
  that ships reference files is generated image-to-image; one without is
  generated from the prompt alone. Where a provider splits those into two model
  ids (kie.ai's Flare/Sunburst pairs, APIDot's `gpt-image-2-edit`), the bot
  switches between them per contest, so only the entry model of each family is
  offered here.
- **Canvas "Auto" follows the brief.** The analyzer picks an aspect ratio per
  contest; "Auto" honours it, and any other value forces that canvas. A ratio the
  selected model doesn't document degrades to `auto` instead of erroring.
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

## When something goes wrong

Every AI failure is classified into a **category** and a **scope**, and the two
decide what happens next. Nothing is ever reported as a vague "LLM unavailable".

| Category | Retried? | Sibling model tried? | What you see |
|---|---|---|---|
| BILLING | no | **no** | "APIDot account has insufficient credits — add credits to resume AI requests." |
| AUTHENTICATION | no | **no** | "APIDot authentication failed — check the API key." |
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
  reply, latency and token usage. It refuses to run when the provider is already
  known to be out of credits, so it cannot waste a request proving the obvious.

Both providers publish a balance endpoint, so both show real credits: APIDot via
`/api/user/balance`, kie.ai via `/api/v1/chat/credit`.

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
