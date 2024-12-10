#!/bin/bash

# build the app for distribution
pyinstaller macos.spec

APP_VERSION=$( python -c "from importlib.metadata import version; print(version('letta'))")
BUILD_ARCH=$(uname -m)
# fix for known silent exec fail from https://github.com/pyinstaller/pyinstaller/issues/5154#issuecomment-2508011279
WORKDIR=dist/letta.app/Contents/MacOS
mv $WORKDIR/letta $WORKDIR/letta_cli

cat << EOF > $WORKDIR/letta
#!/bin/bash
# This is the launcher for OSX, this way the app will be opened
# when you double click it from the apps folder
open -n /Applications/letta.app/Contents/MacOS/letta_cli
EOF

chmod +x $WORKDIR/letta

mkdir dist/package
cp -r dist/letta.app dist/package

# Create the .dmg file
create-dmg --volname "Letta Installer" --window-size 800 400 --icon-size 100 --background "assets/installer_background.png" --icon letta.app 200 200 --hide-extension letta.app --app-drop-link 600 185 "Letta-Installer-${APP_VERSION}-${BUILD_ARCH}.dmg" dist/package