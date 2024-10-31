---
title: "Help"
---

Welcome to the Fast-LLM Help Center! Here, you'll find fixes for common hiccups, links to dig deeper, tutorials, and pointers for when you need some extra support. Remember, everyone hits a snag now and then. Let's sort them out together and get you back to training.

---

## Common Issues & Gotchas 🚧

Let's stay one step ahead of those pesky gotchas. Here's a list of common issues and quick fixes:

-   **CUDA Out of Memory**: When the GPU throws a fit, a few tweaks can help. First, try lowering `micro_batch_size` or `sequence_length` in the configuration to fit within the available memory. Still stuck? Try setting the `mlp_recompute_level` option to `activation` to save memory in the backward pass, or experiment with higher ZeRO stages for reduced memory usage. And if that's not enough, tensor or model parallelism may be your friend. We've got a guide for this, so you're covered.

-   **Python Hash Seed Sync Error**: Encountering an error like

    ```bash
    RuntimeError: Desync detected for barrier train begin (66830148464 != 133042721120)
    ```
  
    points to a hashing inconsistency. To fix it, set `PYTHONHASHSEED=0` in your environment variables. This ensures consistent hashing across processes, keeping them in sync.

-   **`torchrun` Timeout Errors**: If you see timeout errors related to `torchrun` during rendezvous, it could be DNS resolution or a networking issue. Check that all worker nodes are communicating properly with the master node.

-   **NCCL Errors with Timeout Messages**: Oh, the joys of NCCL errors! If you see something like

    ```bash
    Watchdog caught collective operation timeout: WorkNCCL(SeqNum=408951, OpType=_ALLGATHER_BASE, … , Timeout(ms)=600000) ran for 600351 milliseconds before timing out
    ```
  
    appearing across all GPU workers, it usually means one or more hosts failed to complete a NCCL operation, causing others to block. NCCL errors can be frustrating to diagnose since they rarely specify which node or GPU caused the issue. We're working on improving this by surfacing which messages and operations are in progress during these crashes to better identify any problematic hosts or GPUs. Stay tuned!

For more detailed solutions, check out our GitHub Issues page. Odds are someone's already tackled a similar problem, and you might find the exact fix you need.

---

## Reference 📚

If you're the type who loves configurations and tweaking every detail, the [**Configuration Reference**](reference/configuration) is for you. It covers every config option you could imagine. From optimizer settings to batch sizes to distributed training parameters. It's all in there.

---

## Tutorials 👨‍🏫

We've got some excellent tutorials to help you get the most out of Fast-LLM:

-   [**Quick-Start Guide**](/quick-start): Perfect for launching Fast-LLM on a single GPU machine. We walk you through setting up Docker, running your first training job, and handling common issues.

-   [**In-Action Guides**](/in-action/slurm): Ready to go big? These guides cover setting up Fast-LLM with Slurm and Kubernetes for multi-node training. This is where Fast-LLM really shows its power.

---

## Still Stuck? Where to Find Help 🙋

If Fast-LLM still isn't cooperating, here's where to look next:

1.   **GitHub [Issues](https://github.com/ServiceNow/Fast-LLM/issues) & [Discussions](https://github.com/ServiceNow/Fast-LLM/discussions)**: This is your best resource. Use the search function to see if anyone has run into the same issue. The community and our team are pretty active, so you'll likely find a solution or get help quickly.

2.   **Email (last resort)**: As a final option, you can email us at [fast-llm-team@servicenow.com](mailto:fast-llm-team@servicenow.com). This is only for rare cases, though. GitHub is our go-to for answering questions, as it lets others benefit from the conversation too.

Fast-LLM is a growing community, and your questions and contributions help make it better for everyone. Who knows, you might just solve the next person's roadblock!

---

That's it! We're excited to see what you build with Fast-LLM. Happy training!
