#!/usr/bin/env bash
# Rebuild the reference reel format with the reference audio:
#   0.0–3.4 s  hook clip  + caption at the top
#   3.4–15.0 s payoff clip (hard cut on the beat)
#   audio: ref-audio.m4a, untouched, full length
#
# Usage: ./make_reel.sh HOOK.mp4 PAYOFF.mp4 "Caption line 1\nCaption line 2" out.mp4 [hook_start] [payoff_start]
#   hook_start / payoff_start = seconds into each source to begin from (default 0)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
HOOK="$1"; PAYOFF="$2"; CAPTION="$3"; OUT="$4"
HS="${5:-0}"; PS="${6:-0}"
CUT=3.4; TOTAL=14.974
PAY_LEN=$(echo "$TOTAL - $CUT" | bc)
FONT="${FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf}"
CAPFILE=$(mktemp); printf "%b" "$CAPTION" > "$CAPFILE"

FIT="scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,setsar=1,fps=30"
ffmpeg -v error -y \
  -ss "$HS" -t "$CUT" -i "$HOOK" \
  -ss "$PS" -t "$PAY_LEN" -i "$PAYOFF" \
  -i "$DIR/ref-audio.m4a" \
  -filter_complex "\
[0:v]$FIT,drawtext=fontfile=$FONT:textfile=$CAPFILE:fontsize=46:fontcolor=white:line_spacing=6:\
borderw=2:bordercolor=black@0.6:x=(w-text_w)/2:y=110[a];\
[1:v]$FIT[b];[a][b]concat=n=2:v=1:a=0[v]" \
  -map "[v]" -map 2:a -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  -c:a aac -b:a 128k -t "$TOTAL" -movflags +faststart "$OUT"
rm -f "$CAPFILE"
echo "Wrote $OUT"
