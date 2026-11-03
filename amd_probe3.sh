#!/bin/bash
echo "===== cache tree ====="
du -sh /opt/seekdb-deps-cache 2>/dev/null
find /opt/seekdb-deps-cache -maxdepth 3 | head -25
echo "===== hips ====="
du -sh /opt/hips 2>/dev/null
find /opt/hips -maxdepth 2 | head -25
echo "===== big files ====="
find /opt /workspace -maxdepth 4 -type f -size +20M 2>/dev/null | head -10
echo "===== speed without proxy ====="
curl -s --noproxy '*' -o /dev/null --max-time 12 -w "direct: %{size_download} bytes %{speed_download} B/s\n" \
  "https://codeload.github.com/NUDT-23/seekdb-fork/tar.gz/refs/heads/dev-env"
echo "===== speed with proxy ====="
curl -s -o /dev/null --max-time 12 -w "proxy : %{size_download} bytes %{speed_download} B/s\n" \
  "https://codeload.github.com/NUDT-23/seekdb-fork/tar.gz/refs/heads/dev-env"
echo "===== mirrors ====="
for u in "https://ghfast.top/https://github.com/NUDT-23/seekdb-fork/archive/refs/heads/dev-env.tar.gz" \
         "https://ghproxy.net/https://github.com/NUDT-23/seekdb-fork/archive/refs/heads/dev-env.tar.gz" \
         "https://kkgithub.com/NUDT-23/seekdb-fork/archive/refs/heads/dev-env.tar.gz"; do
  printf "%-30s " "$(echo $u | cut -d/ -f3)"
  curl -s --noproxy '*' -o /dev/null --max-time 10 -w "%{http_code} %{size_download}B %{speed_download}B/s\n" "$u" || echo fail
done
echo "===== done ====="
