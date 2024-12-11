# About
These certs are used to set up a localhost https connection to the ADE.

## Instructions
1. Install [mkcert](https://github.com/FiloSottile/mkcert)
2. Run `mkcert -install`
3. Run letta with the environment variable `LOCAL_HTTPS=true`
4. Access the app at [https://app.letta.com/development-servers/local/dashboard](https://app.letta.com/development-servers/local/dashboard)
5. Click "Add remote server" and enter `https://localhost:8283` as the URL, leave password blank unless you have secured your ADE with a password.
