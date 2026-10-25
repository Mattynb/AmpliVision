
# Input
if [[ -z "$1" ]]; then
    echo "Error: Missing argument. Usage: $0 <PREFIX>" >&2
    exit 1
fi
PREFIX=$1

echo "cleaning up...."

# move all output and error files
mkdir -p _experiments/AUTO/${PREFIX}
mv _experiments/AUTO/*.out _experiments/AUTO/${PREFIX}
mv _experiments/AUTO/*.err _experiments/AUTO/${PREFIX}

# clean some results
rm -r AmpliVision/data/pyod/*
rm -r AmpliVision/data/ML_models/*

# clean generated images
if [[ "$2" == "-f" ]]; then
    cd /hpcstor6/scratch01/m/matheus.berbet001/
    find . -maxdepth 1 -type f -printf '%f\n' | grep -v -F -x -f <(for c in {a..z} {A..Z} {0..9}; do find . -maxdepth 1 -type f -name "$c*" -printf '%f\n' | sort | head -n 1; done) | xargs -I {} rm "{}"
    cd -
    echo deleted files
else 
    echo no files deleted
fi


echo "done cleaning up...."

