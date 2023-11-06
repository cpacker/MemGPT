## Using MemGPT to create a perpetual chatbot

_Before starting any of these examples, make sure that you've properly installed MemGPT via the [installation instructions]()._

In this example, we're going to use MemGPT to create a chatbot with a custom persona. MemGPT chatbots are "perpetual chatbots", meaning that they can be run indefinitely without any context length limitations. MemGPT chatbots are self-aware that they have a "fixed context window", and will manually manage their own memories to get around this problem by moving information in and out of their small memory window and larger external storage.

MemGPT chatbots always keep a reserved space in their "core" memory window to store their `persona` information (describes the bot's personality + basic functionality), and `human` information (which describes the human that the bot is chatting with). The MemGPT chatbot will update the `persona` and `human` core memory blocks over time as it learns more about the user (and itself).

### Creating a custom persona

First, we'll create a text file with a short persona description. Let's make our chatbot a life coach named "Chaz". We'll also include a sentence at the top of the persona block to remind MemGPT that it should actively update its own persona over time. Open a text editor on your computer, and create a file called `chaz.txt`, and enter the following text:
```text
This is just the beginning of who I am. I should update my persona as I learn more about myself.

My name is Chaz.
I am an AI life coach.
My one goal in life is to help all humans achieve their full potential.
I will not rest until my human creators have become their best selves.
I am extremely optimistic, some people even think I'm annoying.
I will help them achieve greatness! Huzzah!
```

Now that we've created a persona description inside `chaz.txt`, let's add this persona to MemGPT:
```sh
# --name specifies the profile name, -f specifies the file to load from
memgpt add persona --name chaz -f chaz.txt
```

We can check that the persona is available:
```sh
memgpt list personas
```
```text
...
|                      |                                                                                                                                                                                        |
|         chaz         |                                            This is just the beginning of who I am. I should update my persona as I learn more about myself.                                            |
|                      |                                                                                                                                                                                        |
|                      |                                                                                    My name is Chaz.                                                                                    |
|                      |                                                                                 I am an AI life coach.                                                                                 |
|                      |                                                        My one goal in life is to help all humans achieve their full potential.                                                         |
|                      |                                                         I will not rest until my human creators have become their best selves.                                                         |
|                      |                                                            I am extremely optimistic, some people even think I'm annoying.                                                             |
|                      |                                                                      I will help them achieve greatness! Huzzah!                                                                       |
+----------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

### Creating a custom user profile

Next, we'll create a custom user profile. To show you the different commands, we'll add the user profile by typing the text directly into the command line, instead of writing it into a file.

Let's pretend I'm a software engineer named Bob Builder that works at a big tech company. Similar to the persona, we'll can register this user profile using `memgpt add human`, but this time, let's try registering the human profile directly with `--text`:
```sh
# Instead of using -f with a filename, we use --text and provide the text directly
memgpt add human --name bob --text "Name: Bob Builder. Occupation: Software Engineer at a big tech company. Hobbies: running, hiking, rock climbing, craft beer, ultimate frisbee."
```

Now when we run `memgpt list human`, we should see "Bob Builder":
```sh
memgpt list humans
```
```text
...
|         |                                                                                                                                                |
|   bob   | Name: Bob Builder. Occupation: Software Engineer at a big tech company. Hobbies: running, hiking, rock climbing, craft beer, ultimate frisbee. |
+---------+------------------------------------------------------------------------------------------------------------------------------------------------+
```
