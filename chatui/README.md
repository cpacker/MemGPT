# MemGPT Chat UI (alpha build)

Running the Chat UI requires starting both the MemGPT API server, and the MemGPT Chat UI web application.

## Start the MemGPT API server

Before attempting to run the MemGPT API server, make sure you have `memgpt` installed and have been able to run basic `memgpt` commands (e.g. `memgpt run`). See installation instructions here: https://memgpt.readme.io/docs/quickstart.

Once you have MemGPT installed and running correctly, you can start the server with:

```sh
memgpt server
```

## Start the MemGPT Chat UI web application

Once the API server is running, you can launch the Chat UI web app (which will connect to the MemGPT API server).

If you don't already have Node.js installed, follow the installation instructions here: https://nodejs.org/en.

Then, in your terminal navigate to the `chatui` directory in the MemGPT project, and run:

```sh
npx nx serve
```

Then you should be able to navigate to http://localhost:4200 and access the MemGPT Chat UI.
