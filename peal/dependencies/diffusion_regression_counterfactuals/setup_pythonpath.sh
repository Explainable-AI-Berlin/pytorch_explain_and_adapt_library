#!/bin/bash
if [ ! -d "$PWD/related_work/ACE" ] || [ ! -d "$PWD/related_work/diffae" ]; then
    echo "Seems like this was not run from the root of the project. Please run it from there."
    exit 1
fi

export PYTHONPATH=$PYTHONPATH:$PWD:$PWD/related_work/ACE:$PWD/related_work/diffae
