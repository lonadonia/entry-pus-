#!/bin/bash

# --- 1. Check running as root or sudo ---
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root or using sudo"
  exit 1
fi

set -e

echo "--- Starting Deployment for Freelancer Contest Bot ---"

# --- 2. Update system ---
echo "Updating system packages..."
apt-get update && apt-get upgrade -y

# --- 3. Install dependencies ---
echo "Installing dependencies..."
apt-get install -y docker.io docker-compose-v2 git curl nginx certbot python3-certbot-nginx openssl

# --- 4. Create directory ---
INSTALL_DIR="/opt/freelancer-bot"
echo "Setting up directory at $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"
cp -r . "$INSTALL_DIR"
cd "$INSTALL_DIR"

# --- 5. Prompt user for values ---
read -p "Enter your Domain Name (leave empty for IP-only): " DOMAIN
read -s -p "Enter PostgreSQL Password: " POSTGRES_PASSWORD
echo ""
read -p "Enter Flask Secret Key (leave empty to generate random): " SECRET_KEY

if [ -z "$SECRET_KEY" ]; then
    SECRET_KEY=$(openssl rand -hex 32)
    echo "Generated Secret Key: $SECRET_KEY"
fi

# --- 6. Create .env file ---
echo "Configuring .env file..."
cp .env.example .env

# Use sed to replace values. Using | as delimiter to avoid issues with special characters in passwords/keys
sed -i "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$POSTGRES_PASSWORD|" .env
sed -i "s|^SECRET_KEY=.*|SECRET_KEY=$SECRET_KEY|" .env
# Also update DATABASE_URL in .env if it's there
sed -i "s|:contestbot@|:$POSTGRES_PASSWORD@|g" .env

# --- 7. Build and start Docker containers ---
echo "Stopping host nginx (if running) to free port 80..."
systemctl stop nginx || true
systemctl disable nginx || true

echo "Building and starting Docker containers..."
docker compose up --build -d

# --- 8. Wait for services healthy ---
echo "Waiting for services to become healthy..."
MAX_RETRIES=30
RETRY_COUNT=0
until [ $(docker ps --filter "health=healthy" | grep -c "healthy") -ge 2 ] || [ $RETRY_COUNT -eq $MAX_RETRIES ]; do
    echo "Waiting for healthchecks (Attempt $((RETRY_COUNT+1))/$MAX_RETRIES)..."
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Warning: Services did not report healthy in time. Continuing anyway..."
fi

# --- 9. If domain provided: setup Nginx + Let's Encrypt SSL ---
if [ -n "$DOMAIN" ]; then
    echo "Configuring Nginx and SSL for $DOMAIN..."
    
    # Create basic nginx config for certbot
    cat <<EOF > /etc/nginx/sites-available/freelancer-bot
server {
    listen 80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF
    ln -sf /etc/nginx/sites-available/freelancer-bot /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    systemctl restart nginx

    # Run certbot
    certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --register-unsafely-without-email
fi

# --- 10. Create systemd service ---
echo "Creating systemd service..."
cat <<EOF > /etc/systemd/system/freelancer-bot.service
[Unit]
Description=Freelancer Contest Bot Docker Service
After=docker.service
Requires=docker.service

[Service]
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/bin/docker compose -f $INSTALL_DIR/docker-compose.yml up
ExecStop=/usr/bin/docker compose -f $INSTALL_DIR/docker-compose.yml down
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# --- 11. Enable and start service ---
echo "Enabling and starting systemd service..."
systemctl daemon-reload
systemctl enable freelancer-bot.service
# We don't necessarily need to 'start' it if we already ran 'docker compose up -d', 
# but it's good to ensure systemd is managing it.
# Actually, since we started it with -d, systemd might get confused or try to start another instance.
# Let's stop the manual one and let systemd take over.
docker compose down
systemctl restart freelancer-bot.service

# --- 12. Create cron job for daily backup ---
echo "Creating daily backup cron job..."
mkdir -p /opt/backups
BACKUP_SCRIPT="$INSTALL_DIR/backup.sh"
cat <<EOF > "$BACKUP_SCRIPT"
#!/bin/bash
DATE=\$(date +%Y%m%d_%H%M%S)
docker exec \$(docker ps -q -f name=postgres) pg_dump -U contestbot contestbot > /opt/backups/db_backup_\$DATE.sql
# Keep only last 7 days
find /opt/backups -type f -name "*.sql" -mtime +7 -delete
EOF
chmod +x "$BACKUP_SCRIPT"

(crontab -l 2>/dev/null; echo "0 2 * * * $BACKUP_SCRIPT") | crontab -

# --- 13. Print final status ---
echo "--- Deployment Complete! ---"
echo "Container status:"
docker compose ps

if [ -n "$DOMAIN" ]; then
    echo "Access URL: https://$DOMAIN"
else
    IP_ADDR=$(curl -s https://ifconfig.me)
    echo "Access URL: http://$IP_ADDR"
fi

echo "Upload your freelancer.json in Settings to start"
