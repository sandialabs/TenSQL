#!/bin/bash

set -ex

mkdir -p benchmark/data

cd benchmark/data

if ! echo 'c3b6e5fa980c688ef783cd6afabc0c17a635e5e7ad8f3edf8396907d3f1ab451  gplus_combined.txt' | sha256sum -c - ; then
    wget -c 'https://snap.stanford.edu/data/gplus_combined.txt.gz' -O gplus_combined.txt.gz && gzip -fd gplus_combined.txt.gz
fi

if ! echo '6bc15ae46e4e4e29c4d338366d188583690d8f0db8f9d6ad42f7f0826afeb4fa  twitter_combined.txt' | sha256sum -c - ; then 
    wget -c 'https://snap.stanford.edu/data/twitter_combined.txt.gz' -O twitter_combined.txt.gz && gzip -fd twitter_combined.txt.gz
fi

if ! echo 'f41c026ed8af3cc3359f1ca5573d0605fb09ae0eefa34544b820fd8c6e2ef296  facebook_combined.txt' | sha256sum -c - ; then
    wget -c 'https://snap.stanford.edu/data/facebook_combined.txt.gz' -O facebook_combined.txt.gz && gzip -fd facebook_combined.txt.gz
fi
