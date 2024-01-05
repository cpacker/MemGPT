---
title: Using the MemGPT API
excerpt: How to set up a local MemGPT API server
category: 658135e7f596b800715c1cee
---

![memgpt llama](https://raw.githubusercontent.com/cpacker/MemGPT/main/docs/assets/memgpt_server.webp)

> ⚠️ API under active development
>
> The MemGPT API is under **active development** and **breaking changes are being made frequently**. Do not expect any endpoints or API schema to persist until an official `v1.0` of the MemGPT API is released.
>
> For support and to track ongoing developments, please visit [the MemGPT Discord server](https://discord.gg/9GEQrxmVyE) where you can chat with the MemGPT team and other developers about the API.

> 📘 Check Discord for the latest development build
>
> Make sure to check [Discord](https://discord.gg/9GEQrxmVyE) for updates on the latest development branch to use. The API reference viewable on this page may only apply to the latest dev branch, so if you plan to experiment with the API we recommend you [install MemGPT from source](https://memgpt.readme.io/docs/contributing#installing-from-source) for the time being.

## Starting a server process

You can spawn a MemGPT server process using the following command:

```sh
memgpt server
```

Before attempting to launch a server process, make sure that you have already configured MemGPT (using `memgpt configure`) and are able to successfully create and message an agent using `memgpt run`. For more information, see [our quickstart guide](https://memgpt.readme.io/docs/quickstart).
