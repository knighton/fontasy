#!/bin/sh
#
# Font dataset preprocessing pipeline.
#
# 1. Find fonts
# 2. Dedupe fonts
# 3. Visualize font properties
# 4. Filter fonts

PY=python3
TTF_ROOT=~/dataset/fonts/
PROC_ROOT=data/pre/

FONT_PROP=regular
MIN_FONT_SIZE=1
MAX_FONT_SIZE=64
FONT_SIZE=40
IMG_HEIGHT=64
CHRS=33..127
IMG_WIDTH=32

rm -rf $PROC_ROOT
mkdir -p $PROC_ROOT

on_begin() {
    T0=`date +%s.%N`
}

on_end() {
    T1=`date +%s.%N`
    $PY -c "print('[%8.3f sec] %s' % ($T1 - $T0, '$1'))"
}

# 1. Find fonts.
#
# Traverse the given directory tree and list all font files found, with their
# hashes and font names.  Results in a JSONL file with entries:
#
# {
#     file: str,
#     size: int,
#     sha256: str,
#     name: str,
# }

on_begin
$PY -m fontasy.pre.find_fonts \
    --in $TTF_ROOT \
    --out $PROC_ROOT/found_fonts.jsonl
on_end "1. Find fonts"

# 2. Dedupe fonts.
#
# Drop the fonts with hash or name collisions, resulting in a new list (same
# format as previous step).

on_begin
$PY -m fontasy.pre.dedupe_fonts \
    --in $PROC_ROOT/found_fonts.jsonl \
    --out $PROC_ROOT/deduped_fonts.jsonl \
    --out_hash2files $PROC_ROOT/dedupe_hash2files.jsonl \
    --out_name2files $PROC_ROOT/dedupe_name2files.jsonl
on_end "2. Dedupe fonts"

# 3. Visualize font properties.
#
# Collect and dump some statistics about the properties of the fonts we have.
#
# Probably want to review and drop unusual properties for distribution sanity.

on_begin
$PY -m fontasy.pre.vis_font_props \
    --in $PROC_ROOT/deduped_fonts.jsonl \
    --out $PROC_ROOT/font_properties.txt
on_end "3. Visualize font properties"

# 4. Filter fonts.
#
# Just drop all non-regular fonts, resulting in a new list (same format).

on_begin
$PY -m fontasy.pre.filter_fonts \
    --in $PROC_ROOT/deduped_fonts.jsonl \
    --font_prop $FONT_PROP \
    --out $PROC_ROOT/fonts.jsonl
on_end "4. Restrict to regular fonts"

echo

echo Fonts:

F=$PROC_ROOT/found_fonts.jsonl
N=`cat $F | wc -l`
echo - Found $N \($F\)

F=$PROC_ROOT/deduped_fonts.jsonl
N=`cat $F | wc -l`
echo - Deduped $N \($F\)

F=$PROC_ROOT/fonts.jsonl
N=`cat $F | wc -l`
echo - Filtered $N \($F\)

echo
