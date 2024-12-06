#!/bin/bash

# build the app for distribution
pyinstaller letta.spec

# fix for known silent exec fail from https://github.com/pyinstaller/pyinstaller/issues/5154#issuecomment-2508011279
APP_NAME="letta"
APP_VERSION=$( python -c "from importlib.metadata import version; print(version('letta'))")
BUILD_ARCH=$(uname -m)
WORKDIR=dist/$APP_NAME.app/Contents/MacOS
mv $WORKDIR/$APP_NAME $WORKDIR/${APP_NAME}_cli

cat << EOF > $WORKDIR/$APP_NAME
#!/bin/bash
# This is the launcher for OSX, this way the app will be opened
# when you double click it from the apps folder
open -n /Applications/${APP_NAME}.app/Contents/MacOS/${APP_NAME}_cli
EOF

chmod +x $WORKDIR/$APP_NAME

mkdir dist/package
cp -r dist/$APP_NAME.app dist/package

# Create the .dmg file
create-dmg --volname "Letta Installer" --window-size 800 400 --icon-size 100 --background "assets/installer_background.png" --icon letta.app 200 200 --hide-extension letta.app --app-drop-link 600 185 "Letta-Installer-${APP_VERSION}-${BUILD_ARCH}.dmg" dist/package