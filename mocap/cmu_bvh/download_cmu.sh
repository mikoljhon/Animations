#!/usr/bin/env bash
# Download the cgspeed Daz-friendly BVH release of the CMU motion capture database (primary, hip-corrected release).
# Each MediaFire file page carries a direct link to the zip; fetch the page, pull the link, download, test the zip.
cd "$(dirname "$0")"
mkdir -p zips
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
KEYS="01-09:jajok14comokr2p 10-14:8fe0mzq5c236vje 15-19:lfd0th6cf61fmea 20-29:k6m3ret89dbveco 30-34:rjqpfa45igk5hfa
35-39:ww4c5igmctg4uvg 40-45:9tqzon4afbu4ebd 46-56:voy0nxlneo6i58m 60-75:32d95ekvxg848sz 76-80:gi3i92b39fsf8yg
81-85:kqwb4q452o30p18 86-94:bk2x3byuc1gmhc3 102-111:8b7kbjf0e77wnz7 113-128:gefpmcbtmhk85m0 131-135:uk53ygwt16frp5n
136-140:bvq1x4ml297mugs 141-144:hzb294jh77z95c7"
for item in $KEYS; do
  range=${item%%:*}; key=${item##*:}
  out="zips/cmuconvert-daz-$range.zip"
  if [ -f "$out" ] && python -c "import zipfile,sys; sys.exit(0 if zipfile.ZipFile('$out').testzip() is None else 1)" 2>/dev/null; then
    echo "$range already here"; continue
  fi
  for attempt in 1 2 3; do
    curl -sSL -A "$UA" "https://www.mediafire.com/file/$key" -o "page_$range.html"
    url=$(grep -oE 'https://download[0-9]*\.mediafire\.com/[^"'"'"' ]+\.zip' "page_$range.html" | head -1)
    rm -f "page_$range.html"
    if [ -z "$url" ]; then echo "$range: no link (attempt $attempt)"; sleep 20; continue; fi
    echo "$range: downloading $(basename "$url")"
    if curl -sSL -A "$UA" --retry 3 --retry-delay 10 -o "$out" "$url" && \
       python -c "import zipfile,sys; sys.exit(0 if zipfile.ZipFile('$out').testzip() is None else 1)" 2>/dev/null; then
      echo "$range: ok $(du -h "$out" | cut -f1)"; break
    fi
    echo "$range: bad zip (attempt $attempt)"; rm -f "$out"; sleep 20
  done
done
echo "done: $(ls zips/*.zip 2>/dev/null | wc -l) zips, $(du -sh zips | cut -f1)"
