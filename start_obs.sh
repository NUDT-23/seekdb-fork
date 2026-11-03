#!/bin/bash
# Start seekdb without the container http proxy, so that requests to the local
# mock AI service (http://0.0.0.0:8000) are not routed through the proxy.
cd /workspace/data || exit 1
exec env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  /workspace/seekdb/build_release/src/observer/seekdb \
  --nodaemon --base-dir /workspace/data --port 2881
