#!/bin/bash

# from https://github.com/pyinstaller/pyinstaller/issues/5154#issuecomment-2508011279
APP_NAME="letta"

WORKDIR=dist/$APP_NAME.app/Contents/MacOS
mv $WORKDIR/$APP_NAME $WORKDIR/${APP_NAME}_cli

cat << EOF > $WORKDIR/$APP_NAME
#!/bin/bash
# This is the launcher for OSX, this way the app will be opened
# when you double click it from the apps folder
open -n /Applications/${APP_NAME}.app/Contents/MacOS/${APP_NAME}_cli
EOF

chmod +x $WORKDIR/$APP_NAME