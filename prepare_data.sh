#!/bin/bash -e

rm -rf rotated
mkdir -p rotated/train rotated/test

PREFIX=

if [ "$1" == "--print" ]; then
    PREFIX=echo
fi

for ds in datasets/train/external datasets/train/narc; do
    for img in $ds/*.png; do
        $PREFIX ./rnd_windows.py --output-dir=rotated/train $img
        $PREFIX ./rnd_windows.py --scale=.5 --output-dir=rotated/train $img
    done
done

for img in datasets/test/*.png; do
    $PREFIX ./rnd_windows.py --output-dir=rotated/test $img
    $PREFIX ./rnd_windows.py --scale=.5 --output-dir=rotated/test $img
done

