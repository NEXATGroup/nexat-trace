#!/bin/bash

# package version should comply with PEP 440 (https://peps.python.org/pep-0440/)

# use git describe for version string
# will result in X.Y.Z for release tag and X.Y.Zdev12345678 for development commits
version=$(git describe --tags --always)

# remove all alphabetic characters (a-z and A-Z)
version=$(echo "$version" | tr -d 'a-zA-Z')

# replace the first occurrence of '-' with 'dev' and remove all subsequent '-'
version=$(echo "$version" | sed 's/-/dev/;s/-//g')


echo "$version" > ./version
echo $version