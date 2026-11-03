#!/bin/bash
# Look for a China-friendly mirror of github that is fast enough for the source tree.
T="NUDT-23/seekdb-fork/archive/refs/heads/dev-env.tar.gz"
try() {
  name="$1"; url="$2"
  out=$(curl -s --noproxy '*' -o /dev/null --max-time 10 -w "%{http_code} %{size_download} %{speed_download}" "$url" 2>/dev/null)
  echo "$name  $out"
}
try "gh-proxy.com  " "https://gh-proxy.com/https://github.com/$T"
try "moeyy.xyz     " "https://github.moeyy.xyz/https://github.com/$T"
try "hub.gitmirror " "https://hub.gitmirror.com/https://github.com/$T"
try "ghproxy.cc    " "https://ghproxy.cc/https://github.com/$T"
try "gh.llkk.cc    " "https://gh.llkk.cc/https://github.com/$T"
try "ghfast.top    " "https://ghfast.top/https://github.com/$T"
try "codeload(base)" "https://codeload.github.com/$T"
echo "===== gitee reachability ====="
curl -s --noproxy '*' -o /dev/null --max-time 8 -w "gitee %{http_code}\n" https://gitee.com
echo "===== done ====="
