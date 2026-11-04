#!/bin/sh
# uso: K=... ./setup.sh   (opcional E2B_KEYS)
set -e
cd "$(dirname "$0")"
mkdir -p w
cat d/p.* | openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass env:K | tar -xz -C w
[ -n "$E2B_KEYS" ] && printf '%s\n' "$E2B_KEYS" > ~/.e2b_keys && chmod 600 ~/.e2b_keys
grep -qx 'w/' .gitignore 2>/dev/null || echo 'w/' >> .gitignore
echo ok
