#!/bin/bash
# 🛡️ DeepFakeShield AI — VPS Initial Setup Script
# This script prepares the VPS for deployment and sets up SSL.

set -e

DOMAIN="deepshield.cloud"
EMAIL="admin@$DOMAIN"
PROJECT_DIR="/root/deepshield"

echo "🚀 Starting VPS Setup for $DOMAIN..."

# 2. Configure NATIVE NGINX as Reverse Proxy
echo "🌐 Configuring native Nginx for $DOMAIN..."

# Install Certbot for the host if not present
if ! command -v certbot &> /dev/null; then
    apt-get install -y certbot python3-certbot-nginx
fi

# Create Nginx server block for deepshield.cloud
cat > /etc/nginx/sites-available/deepshield << EOF
server {
    listen 80;
    server_name $DOMAIN www.$DOMAIN;

    location / {
        proxy_pass http://127.0.0.1:8081;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Large file uploads
        client_max_body_size 500M;
    }
}
EOF

# Enable the site
ln -sf /etc/nginx/sites-available/deepshield /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

# 3. Install Docker (if not installed)
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=\$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \$(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi

# 4. Create project structure
mkdir -p $PROJECT_DIR/deploy/nginx
mkdir -p $PROJECT_DIR/deploy/postgres
mkdir -p $PROJECT_DIR/storage/uploads $PROJECT_DIR/storage/thumbnails $PROJECT_DIR/storage/reports

# 5. SSL Setup via Host Certbot
echo "🔒 Securing $DOMAIN with SSL..."
certbot --nginx -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos --email $EMAIL --redirect

echo "✅ SSL certificates and Nginx proxy configured successfully!"

echo "======================================"
echo "🎉 VPS is ready for GitHub Actions deployment."
echo "======================================"
