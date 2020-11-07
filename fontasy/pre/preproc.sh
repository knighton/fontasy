#!/bin/sh
#
# Font dataset preprocessing pipeline.
#
# 1. Find fonts
# 2. Dedupe fonts
# 3. Visualize font properties
# 4. Filter fonts
# 5. List characters.
# 6. Visualize character frequencies.
# 7. Calculate heights.
# 8. Visualize heights.
# 9. Make dataset.
# 10. Visualize dataset.
# 11. Split dataset.

PY=python3
TTF_ROOT=~/dataset/fonts/
PROC_ROOT=data/pre/
FONT_PROP=regular
CHRS=33..127
MIN_FONT_SIZE=1
MAX_FONT_SIZE=64
FONT_SIZE=40
IMG_HEIGHT=64
MAX_ASCENT=46
MAX_DESCENT=18
IMG_WIDTH=48
MIN_FONT_OK_FRAC=0.8
VAL_FRAC=0.1

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
$PY -m fontasy.pre.list_chars \
    --in $PROC_ROOT/fonts.jsonl \
    --out $PROC_ROOT/chars.jsonl
on_end "5. List characters per font"

# 6. Visualize character frequencies.
#
# Show the distribution of supported characters in the selected fonts, for
# deciding what broadly-available characters to include in dataset.

on_begin
$PY -m fontasy.pre.vis_char_freqs \
    --in $PROC_ROOT/chars.jsonl \
    --out_by_char $PROC_ROOT/char_freqs_by_char.jsonl \
    --out_by_freq $PROC_ROOT/char_freqs_by_freq.jsonl \
    --out_table $PROC_ROOT/char_freqs_table.txt
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
    --out $PROC_ROOT/heights.i16
on_end "7. Get heights for every font size"

# 8. Visualize heights.
#
# Display visualizations to help determine the optimal font size to use.

on_begin
$PY -m fontasy.pre.vis_heights \
    --in $PROC_ROOT/heights.i16 \
    --min_font_size $MIN_FONT_SIZE \
    --max_font_size $MAX_FONT_SIZE \
    --img_height $IMG_HEIGHT \
    --out_coverage $PROC_ROOT/heights_coverage.txt \
    --out_best $PROC_ROOT/heights_best.csv
on_end "8. Visualize font heights (to decide font size to use)"

# 9. Make dataset.
#
# Draw the glyphs to a binary table.

on_begin
$PY -m fontasy.pre.make_dataset \
    --in $PROC_ROOT/fonts.jsonl \
    --chars $CHRS \
    --font_size $FONT_SIZE \
    --max_ascent $MAX_ASCENT \
    --max_descent $MAX_DESCENT \
    --img_width $IMG_WIDTH \
    --min_font_ok_frac $MIN_FONT_OK_FRAC \
    --out $PROC_ROOT/dataset/
on_end "9. Make dataset (fonts x chars)"

# 10. Visualize dataset.
#
# Show important distribution information about the dataset created.

on_begin
$PY -m fontasy.pre.vis_dataset \
    --in $PROC_ROOT/dataset/ \
    --out_font_freqs $PROC_ROOT/dataset_font_freqs.png \
    --out_char_freqs $PROC_ROOT/dataset_char_freqs.png \
    --out_char_table $PROC_ROOT/dataset_char_table.txt \
    --out_heatmap_txt $PROC_ROOT/dataset_heatmap.txt \
    --out_heatmap_img $PROC_ROOT/dataset_heatmap.png \
    --out_heatmap_img_log10 $PROC_ROOT/dataset_heatmap_log10.png
on_end "10. Analyze dataset distributions"

# 11. Split dataset.
#
# Divide samples into training and validation splits.

on_begin
$PY -m fontasy.pre.split_dataset \
    --dataset $PROC_ROOT/dataset/ \
    --val_frac $VAL_FRAC
on_end "11. Create dataset splits"
