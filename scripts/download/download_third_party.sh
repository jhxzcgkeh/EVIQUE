#!/usr/bin/env bash
set -euo pipefail
mkdir -p third_party/external
cat third_party/versions.lock
printf '
Review licenses and pin commits before cloning.
'
