# Deployment record — 2026-09-15

Deployed by an automated agent session on EC2 `i-04973c8e403551810` (t3.small, Ubuntu 24.04.4).
Public URL: **http://13.63.20.152/**

## Method
Deployed exactly as the project intends: `docker compose up -d` from
`/home/ubuntu/freelancer-contest-bot`, using the shipped `Dockerfile`,
`docker-compose.yml` and `nginx.conf`. The nginx **container** owns host ports
80/443 and reverse-proxies to gunicorn — same as `deploy.sh` does.

## Files in the project that were CHANGED
*None.* No source file, `docker-compose.yml`, `nginx.conf`, `.env`, `Dockerfile`
or template was edited. The existing `.env` was preserved byte-for-byte.

## Files ADDED to the project directory
- `storage/`, `storage/designs/`  — bind-mount targets required by compose (were absent).
- `DEPLOYMENT_NOTES.md`           — this file.
- `app/static/temp/`              — already present in the archive; untouched.

## Server configuration CHANGED
| What | Change | Revert |
|---|---|---|
| `/swapfile` + `/etc/fstab` | Added 2 GB swap (1.9 GB RAM, 5 containers) | `swapoff /swapfile; rm /swapfile`; drop the fstab line |
| APT packages | Installed `unzip`, `docker.io`, `docker-compose-v2` | `apt remove` |
| `/etc/systemd/system/freelancer-bot.service` | **New** unit, enabled at boot | `systemctl disable --now freelancer-bot` |
| user `ubuntu` | Added to the `docker` group | `gpasswd -d ubuntu docker` |

No host nginx was installed, so **no existing nginx config was modified or backed
up — there was none**. Nothing else on the server was touched; SSH was the only
pre-existing listener and is unaffected.

## Database setting CHANGED (via the app's own settings table)
- `server_host`: `http://localhost:5000` → `http://13.63.20.152`
  The dev default is unreachable from kie.ai, so the result-callback webhook
  (`/api/kie-callback`) would never arrive and generation would fall back to
  slower polling. Editable any time on the Settings page.

## Still requires the operator (intentionally left empty — never invented)
Set these on the **Settings** page, per Space (`app/models.py::_DEFAULTS` seeds them blank):
- `kie_api_key` — powers BOTH LLM analysis and GPT-Image-2 generation. The bot cannot run without it.
- `freelancer_cookies` — Cookie-Editor JSON export; needs `GETAFREE_AUTH_HASH_V2` + `session2`.
- `imgbb_api_key` — only if `image_host_method=imgbb`.

## Operating it
```bash
cd /home/ubuntu/freelancer-contest-bot
sudo docker compose ps                      # status
sudo docker compose logs -f celery_worker   # the useful pipeline log
sudo systemctl restart freelancer-bot       # restart whole stack
sudo docker compose up -d --build web celery_worker   # after ANY code edit (code is baked into the image)
```


---

# Change record — 2026-10-30 (extraction, profile and proposal overhaul)

Public URL is now **http://13.53.151.43/** (the instance has no Elastic IP, so its public
IP changes when it is stopped and started).

- Backup taken before deleting anything, mode 600 because it holds session cookies and
  API keys: `/home/ubuntu/freelancer-backups/pre-fresh-extraction-20261030/`
  (`contestbot_full.sql.gz`, `storage_projects.tar.gz`, `source_before.tar.gz`).
  Delete it once you no longer need the old bid history.
- Added `constraints.txt` and applied it in the `Dockerfile`: every package version is now
  locked. Without it, this deployment's rebuild pulled SQLAlchemy 2.1 and took `web` down
  for about 12 minutes.
- Deploy only with `docker compose up -d --build web celery_worker`. nginx also fronts the
  unrelated growmaz sites on this host and was neither changed nor restarted.


---

# Pending change — 2026-11-04 (daily discovery, cleanup, dashboard, proposal writer)

**Not deployed yet** — written in a cloud session without access to the server. To roll out:

```bash
cd /home/ubuntu/freelancer-contest-bot
git pull
sudo docker compose stop -t 290 celery_worker        # warm shutdown: keep the task in flight
sudo docker compose up -d --build web celery_worker celery_beat
sudo docker compose logs -f celery_beat              # a tick every minute
```

- **New service `celery_beat`** (one instance). Without it nothing daily happens; Settings →
  Daily Discovery & Cleanup shows "Scheduler not running".
- **Schema** — applied automatically at boot by `migrate_discovery_and_cleanup()`: adds
  `projects.status_changed_at` (backfilled from `updated_at`) and `proposals.bid_clicked_at`,
  creates `project_discovery_runs` and `project_archive`. Additive only; nothing is dropped.
- **First cleanup** (~15 min after `celery_beat` starts) deletes Skipped/Failed projects whose
  last update is more than 24 h old — the backlog that has been accumulating. Projects with any
  trace of a bid are kept; every deleted row stays listed in `project_archive`. To review the
  backlog first, start only `web celery_worker`, set *Auto-Delete Skipped / Failed* to Disabled
  on the Settings page, and start `celery_beat` afterwards.
- **First discovery**: tonight at 23:00 Africa/Casablanca (UTC+0 since 2026-09-20 in the tz
  database). Turning the schedule on never back-fills earlier days.
- The removed 6-hourly *contest* fetch had never run (beat was not deployed); nothing contest-side
  changes.


---

# Pending change — 2026-11-04 (contests: PoYo + KovAPI, design fidelity, pipeline fixes)

**Not deployed yet.** Same roll-out as the change above (one `git pull`, then
`sudo docker compose stop -t 290 celery_worker && sudo docker compose up -d --build web celery_worker celery_beat`).
No schema change: the three new settings rows (`poyo_api_key`, `kovapi_api_key` as
secrets, `contest_skip_no_ai`) are seeded at boot by `seed_default_settings()`.

**New image providers** (AI Agents → Image Generation AI): **PoYo** and **KovAPI**, each with
GPT Image 2.5 **Flare** and **Sunburst**. Paste the key under Provider Credentials, press
*Test connection* (free: balance + status endpoint), then *Test model* (one real 1K test
image; KovAPI charges 1 credit = $0.005). Nothing changes for a Space until someone selects
one of them.

- PoYo delivers webhooks to **HTTPS only**; with the current `server_host`
  (`http://13.53.151.43`) its results are read by polling (12 s) — set an HTTPS
  `server_host` to get webhooks. KovAPI accepts the public http address.
- KovAPI renders 1K by centre-cropping a square / 3:2 / 2:3 frame; 2K ($0.03) and
  4K ($0.05) render the exact ratio. The prompt keeps content central at 1K.

**Behaviour changes worth knowing**
- **Contests whose brief forbids AI-generated work are now `skipped`** (before any paid
  analysis when the description says so; after analysis when an attachment does). Every
  entry is declared AI-generated, so these could not win. Settings → *Skip Contests That
  Forbid AI* → "Enter anyway" restores the old behaviour.
- Variations are submitted together and collected together (≈ one render time per
  contest instead of three).
- With GPT Image 2.5 (APIDot or PoYo), saving an impossible canvas (custom size at 1K,
  edges not divisible by 16, > 3840 px, transparent JPEG) is corrected on save and the
  page says what changed.
- Required text and brand names the brief never contains are no longer lettered onto
  logos (they are logged as "Grounding: …" warnings on the contest).
