#!/usr/bin/env sh
set -eu
curl -fsS http://127.0.0.1:8000/version
echo
curl -fsS http://127.0.0.1:8000/v1/models
echo
