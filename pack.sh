#!/bin/sh
# uso: K=... ./pack.sh
set -e
cd "$(dirname "$0")"
rm -f d/p.*
tar -C w -czf - . | openssl enc -aes-256-cbc -pbkdf2 -iter 200000 -salt -pass env:K | split -b 90m -d - d/p.
echo ok
