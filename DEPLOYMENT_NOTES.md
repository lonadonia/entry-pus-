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
