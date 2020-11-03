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

# 5. List characters.
#
# Get the list of supported code points for each font.

on_begin
$PY -m fontasy.pre.list_chrs \
    --in $PROC_ROOT/fonts.jsonl \
    --out $PROC_ROOT/chrs.jsonl
on_end "5. List characters per font"

# 6. Visualize character frequencies.
#
# Show the distribution of supported characters in the selected fonts, for
# deciding what broadly-available characters to include in dataset.

on_begin
$PY -m fontasy.pre.vis_chr_freqs \
    --in $PROC_ROOT/chrs.jsonl \
    --out_by_chr $PROC_ROOT/chr_freqs_by_chr.jsonl \
    --out_by_freq $PROC_ROOT/chr_freqs_by_freq.jsonl \
    --out_ascii $PROC_ROOT/chr_freqs_ascii.txt
on_end "6. Visualize characters per font (to decide characters to use)"

# 7. Calculate heights.
#
# Calculate the heights (ascent and descent) for every font for every sane font
# size.  This is used to decide ideal font size to use (trading off font size
# and font coverage).

on_begin
$PY -m fontasy.pre.calc_heights \
    --in $PROC_ROOT/fonts.jsonl \
    --min_font_size $MIN_FONT_SIZE \
    --max_font_size $MAX_FONT_SIZE \
    --out $PROC_ROOT/heights.npy
on_end "7. Get heights for every font size"

# 8. Visualize heights.
#
# Display visualizations to help determine the optimal font size to use.

on_begin
$PY -m fontasy.pre.vis_heights \
    --in $PROC_ROOT/heights.npy \
    --min_font_size $MIN_FONT_SIZE \
    --max_font_size $MAX_FONT_SIZE \
    --img_height $IMG_HEIGHT \
    --out_coverage $PROC_ROOT/heights_coverage.txt \
    --out_best $PROC_ROOT/heights_best.csv
on_end "8. Visualize font heights (to decide font size to use)"
