#!/bin/bash

printf "\n Updating local git config files to clear jupyter notebook output on commit..."
git config --local include.path ../env/.gitconfig
printf "Done"

printf "\n Renormalizing repository for changes to take effect...\n"
git add --renormalize .
printf "Done"
