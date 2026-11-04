FROM python:3.11-slim

# ── System deps ────────────────────────────────────────────────────────────────
# build-essential  : compiles C extensions (lxml, psycopg2 fallback, etc.)
# chromium + driver: Selenium / undetected-chromedriver
# fonts-*          : prevent blank-box rendering in headless Chrome
# libmagic1        : runtime lib required by python-magic
# libglib2.0-0 etc : Chromium runtime shared libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        wget \
        gnupg \
        unzip \
        # Chrome runtime libs (Chrome-for-Testing is installed below, not the
        # distro chromium package — see the "Chrome for Testing" block)
        libglib2.0-0 \
        libnss3 \
        libnspr4 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libdbus-1-3 \
        libxkbcommon0 \
        libx11-6 \
        libxcomposite1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libpango-1.0-0 \
        libcairo2 \
        libasound2 \
        # Fonts (prevents blank glyphs in headless Chrome screenshots)
        fonts-liberation \
        fonts-noto \
        fonts-noto-cjk \
        # python-magic runtime
        libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# ── Chrome for Testing (pinned) ────────────────────────────────────────────────
# Debian trixie's rolling `chromium` package reached 150.x, which crashes on this
# host with SIGTRAP on EVERY launch (verified 2026-07-07: `chromium --dump-dom`
# exits 133 regardless of flags/user/seccomp), breaking all scraping and
# submission. Pin a known-good Chrome-for-Testing build + its exactly-matching
# chromedriver instead of the distro package. Bump CFT_VERSION to upgrade.
ENV CFT_VERSION=131.0.6778.264
RUN mkdir -p /opt/chrome \
    && wget -q "https://storage.googleapis.com/chrome-for-testing-public/${CFT_VERSION}/linux64/chrome-linux64.zip" -O /tmp/chrome.zip \
    && wget -q "https://storage.googleapis.com/chrome-for-testing-public/${CFT_VERSION}/linux64/chromedriver-linux64.zip" -O /tmp/chromedriver.zip \
    && unzip -q /tmp/chrome.zip -d /opt/chrome \
    && unzip -q /tmp/chromedriver.zip -d /opt/chrome \
    && ln -sf /opt/chrome/chrome-linux64/chrome /usr/bin/chromium \
    && ln -sf /opt/chrome/chromedriver-linux64/chromedriver /usr/bin/chromedriver \
    && rm -f /tmp/chrome.zip /tmp/chromedriver.zip

# ── Chrome environment ─────────────────────────────────────────────────────────
ENV CHROME_BIN=/opt/chrome/chrome-linux64/chrome
ENV CHROMEDRIVER_PATH=/opt/chrome/chromedriver-linux64/chromedriver

# ── Python environment ─────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

WORKDIR /app

# ── Python deps (cached layer — only busts when requirements.txt changes) ──────
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt -c constraints.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY . .

# ── Runtime directories ────────────────────────────────────────────────────────
RUN mkdir -p \
        /app/storage/designs \
        /app/app/static/temp

EXPOSE 5000

CMD ["gunicorn", "wsgi:app", "-w", "2", "-k", "eventlet", "--bind", "0.0.0.0:5000", "--timeout", "120", "--log-level", "info"]
