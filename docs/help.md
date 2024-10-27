---
title: "Help"
---

Welcome to the Fast-LLM Help Center! Here you'll find solutions for common hiccups, references to dive deeper, tutorials, and a few pointers for when you need some extra support. Remember, everyone runs into a snag now and then. Let's sort them out together.

---

## Common Issues & Gotchas 🚧

Let's get ahead of those pesky gotchas! Here's a list of common issues you might run into, with some quick fixes:

- **Docker Permission Denied**: If Docker isn't playing nice, make sure your user has the right permissions. You may need to add yourself to the Docker group, or even (temporarily) use `sudo`.
  
- **CUDA Out of Memory**: When the GPU throws a fit, it's usually a batch size problem. Try reducing `batch_size` or freeing up GPU memory from other apps.

- **`torchrun` Errors**: If you see something cryptic from `torchrun`, double-check that `torch` and `torchvision` are up-to-date. Compatibility issues can sometimes cause trouble.

- **NCCL Errors**: NCCL errors can be a pain. Make sure your NCCL version matches the one Fast-LLM expects. If you're using a different version, you might need to tweak the environment variables.

For a deeper dive, keep an eye on our GitHub Issues page, where other users might already have tackled similar issues.

---

## Reference 📚

For the config nerds (you know who you are), we've got a detailed **Reference Guide** covering every configuration option under the sun. Need to tweak your optimizer settings, batch sizes, or distributed training parameters? It's all in the guide.

---

## Tutorials 👨‍🏫

We've got a couple of excellent tutorials lined up:

- **Quick-Start Guide**: Perfect for getting up and running on a single GPU machine. We cover setting up Docker, running your first training job, and basic troubleshooting.

- **In-Action Guides**: Ready to go big? Check out the guides for setting up Fast-LLM with Slurm and Kubernetes to tackle multi-node training. This is where Fast-LLM truly shines.

---

## Still Stuck? Where to Find Help 🙋

Sometimes, you've tried everything, and Fast-LLM still isn't cooperating. Here's what you can do:

1. **GitHub Issues & Discussions**: This is your best friend. Use the search function to see if anyone has faced a similar issue. The community (and our team) is super active, so there's a good chance you'll find an answer or get help quickly.

2. **Email (last resort)**: If all else fails, drop us a line at `fast-llm-team@servicenow.com`. Seriously, only use this in rare cases. We prefere to answer on GitHub where others can benefit from the conversation too.

Fast-LLM is a growing community, and you're part of it now! Your questions help us improve, and who knows, you might just help the next person who runs into the same roadblock.

---

And that's it! We're excited to see what you build with Fast-LLM. Happy training!
