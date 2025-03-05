---
title: "Help"
---

Welcome to the Fast-LLM Help Center! Here, you'll find fixes for common hiccups, links to dig deeper, tutorials, and pointers for when you need some extra support. Remember, everyone hits a snag now and then. Let's sort them out together and get you back to training.

---

## Common Issues & Gotchas 🚧

Let's stay one step ahead of those pesky gotchas. Here's a list of common issues and quick fixes:

-   **CUDA Out of Memory**: When the GPU throws a fit, a few tweaks can help. First, try lowering `micro_batch_size` or `sequence_length` in the configuration to fit within the available memory. Still stuck? Try setting the `mlp_recompute_level` option to `activation` or `full` to save memory in the backward pass, or experiment with higher ZeRO stages for reduced memory usage. And if that's not enough, tensor or model parallelism may be your friend.

-   **Python Hash Seed Sync Error**: Encountering an error like

    ```bash
    RuntimeError: Desync detected for barrier train begin (66830148464 != 133042721120)
    ```

    points to a hashing inconsistency. To fix it, set `PYTHONHASHSEED=0` in your environment variables. This ensures that Python's hash seed is consistent across all processes. If these processes have different hash seeds, they'll generate different hash values, leading to desynchronization, as seen in the error message.

-   **`torchrun` Timeout Errors**: If you see timeout errors related to `torchrun` during rendezvous, it could be DNS resolution or a networking issue. Check that all worker nodes are communicating properly with the master node.

-   **NCCL Errors with Timeout Messages**: Oh, the joys of NCCL errors! If you see something like

    ```bash
    Watchdog caught collective operation timeout: WorkNCCL(SeqNum=408951, OpType=_ALLGATHER_BASE, … , Timeout(ms)=600000) ran for 600351 milliseconds before timing out
    ```

    appearing across all GPU workers, it usually means one or more hosts failed to complete a NCCL operation, causing others to block. NCCL errors can be frustrating to diagnose since they rarely specify which node or GPU caused the issue. It is difficult to surface which messages and operations are in progress during these crashes. If the issue happens at a specific moment of training like dataset preparation or model export, the issue might be that this specific procedure took too long and timed out other processes (e.g. when preparing large datasets for long training runs, or saving large models on slow storage). In this case, it can help to increase the timeout `distributed_timeout: 3600`.
    In some other cases, the best we can do is to restart the training job and hope it doesn't happen again. If the issue persists, it might be because of network congestion or a problematic GPU. If the worker that crashed is consistent across multiple runs, it's likely a hardware issue. If you can't resolve it, open an issue on GitHub, and we'll help you troubleshoot.

For more detailed solutions, check out our GitHub Issues page. Odds are someone's already tackled a similar problem, and you might find the exact fix you need.

---

## Reference 📚

If you're the type who loves configurations and tweaking every detail, the [**Configuration Reference**](user_guide/configuration.md) is for you. It covers every config option you could imagine. From optimizer settings to batch sizes to distributed training parameters. It's all in there.

---

## Tutorials 👨‍🏫

We've got some excellent tutorials to help you get the most out of Fast-LLM:

-   [**Quick-Start Guide**](quick-start.md): Perfect for launching Fast-LLM on a single GPU machine. We walk you through running your first training job (either locally or on a cluster), and handling common issues.

-   [**Cookbook**](recipes/train-llama-8b.md): Ready to go big? These recipes cover real-world scenarios like training big models from scratch, continuing training from a checkpoint, and more. This is where Fast-LLM really shows its power.

---

## Still Stuck? Where to Find Help 🙋

If Fast-LLM still isn't cooperating, here's where to look next:

1.  **GitHub [Issues](https://github.com/ServiceNow/Fast-LLM/issues) & [Discussions](https://github.com/ServiceNow/Fast-LLM/discussions)**: This is your best resource. Use the search function to see if anyone has run into the same issue. The community and our team are pretty active, so you'll likely find a solution or get help quickly.

2.  **Email (last resort)**: As a final option, you can email us at [fast-llm-team@servicenow.com](mailto:fast-llm-team@servicenow.com). This is only for rare cases, though. GitHub is our go-to for answering questions, as it lets others benefit from the conversation too.

Fast-LLM is a growing community, and your questions and contributions help make it better for everyone. Who knows, you might just solve the next person's roadblock!

---

That's it! We're excited to see what you build with Fast-LLM. Happy training!
