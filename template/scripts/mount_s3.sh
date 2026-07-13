#!/bin/bash
# mount_s3.sh — Auto-mount S3 bucket at container startup
# Required env vars: S3_ACCESS_KEY, S3_SECRET_KEY, S3_ENDPOINT, S3_BUCKET

set -e

S3_MOUNT_POINT="${S3_MOUNT_POINT:-/workspace/s3}"

if [ -z "${S3_ACCESS_KEY:-}" ] || [ -z "${S3_SECRET_KEY:-}" ] || [ -z "${S3_BUCKET:-}" ]; then
    echo "[mount_s3] S3 credentials not configured, skipping"
    return 0 2>/dev/null || exit 0
fi

echo "[mount_s3] Mounting s3://${S3_BUCKET} -> ${S3_MOUNT_POINT}"

mkdir -p "$S3_MOUNT_POINT"

# Try s3fs first
if command -v s3fs &>/dev/null; then
    echo "${S3_ACCESS_KEY}:${S3_SECRET_KEY}" > /tmp/.s3passwd
    chmod 600 /tmp/.s3passwd
    s3fs "${S3_BUCKET}" "$S3_MOUNT_POINT" \
        -o passwd_file=/tmp/.s3passwd \
        -o url="${S3_ENDPOINT:-https://s3api-us-ne-1.runpod.io}" \
        -o use_path_request_style \
        -o allow_other \
        -o nonempty 2>/dev/null && \
        echo "[mount_s3] s3fs mount successful" || \
        echo "[mount_s3] s3fs mount failed"
    rm -f /tmp/.s3passwd
    return 0 2>/dev/null || exit 0
fi

# Try rclone as fallback
if command -v rclone &>/dev/null; then
    rclone config create s3-storage s3 \
        provider Other \
        endpoint "${S3_ENDPOINT:-https://s3api-us-ne-1.runpod.io}" \
        access_key_id "${S3_ACCESS_KEY}" \
        secret_access_key "${S3_SECRET_KEY}" \
        region us-east-1 2>/dev/null || true

    rclone mount "s3-storage:${S3_BUCKET}" "$S3_MOUNT_POINT" \
        --allow-other --daemon 2>/dev/null && \
        echo "[mount_s3] rclone mount successful" || \
        echo "[mount_s3] rclone mount failed"
    return 0 2>/dev/null || exit 0
fi

# Fallback: write S3 config for boto3-based Python access
echo "[mount_s3] No s3fs/rclone found. Writing boto3 config..."
python -c "
import os, json
conf = {
    'aws_access_key_id': os.environ['S3_ACCESS_KEY'],
    'aws_secret_access_key': os.environ['S3_SECRET_KEY'],
    'endpoint_url': os.environ.get('S3_ENDPOINT', 'https://s3api-us-ne-1.runpod.io'),
    'region_name': 'us-east-1',
}
os.makedirs('${S3_MOUNT_POINT}', exist_ok=True)
with open('/workspace/.s3_config.json', 'w') as f:
    json.dump(conf, f, indent=2)
print('[mount_s3] S3 config saved to /workspace/.s3_config.json')
" 2>/dev/null && echo "[mount_s3] Config written for boto3 access" || echo "[mount_s3] Python fallback failed"
