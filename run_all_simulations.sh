#!/bin/bash

# Run all scripts in the mammals directory
for script in mammals/*.sh; do
    gnome-terminal -- bash -c "cd $(pwd)/mammals; bash $(basename $script); exec bash"
    sleep 2
done

# Run all scripts in the viridiplantae directory
for script in viridiplantae/*.sh; do
    gnome-terminal -- bash -c "cd $(pwd)/viridiplantae; bash $(basename $script); exec bash"
    sleep 2
done