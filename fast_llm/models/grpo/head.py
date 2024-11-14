import torch
import torch.nn.functional as F
from fast_llm.layers.language_model.head import LanguageModelHead
from fast_llm.layers.language_model.config import LanguageModelLossNames

class GRPOHead(LanguageModelHead):
    def masked_mean(self, values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Calculate mean of values with masks applied"""
        return (values * masks).sum() / (masks.sum() + 1e-8)

    def compute_grpo_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        ref_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        config: GRPOConfig,
    ) -> torch.Tensor:
        masks = labels != -100
        masks = masks[:, 1:]

        new_log_probs = torch.gather(
            F.log_softmax(logits[:, :-1, :], dim=-1),
            dim=2,
            index=labels[:, 1:].unsqueeze(2),
        ).squeeze(2)

        # surrogate loss calculation
        log_ratio_new_old = new_log_probs - old_logprobs
        ratio_new_old = torch.exp(log_ratio_new_old)
        weights = advantages if config.use_advantages else rewards

        surr1 = ratio_new_old * weights
        clamped_ratio = torch.clamp(
            ratio_new_old, 
            1 - config.epsilon, 
            1 + config.epsilon
        )
        surr2 = clamped_ratio * weights
        surrogate_loss = torch.min(surr1, surr2)

        # KL divergence approximation
        log_ratio_ref_new = ref_logprobs - new_log_probs
        approx_kl = torch.exp(log_ratio_ref_new) - log_ratio_ref_new - 1

        # Final loss computation
        loss = -self.masked_mean(
            surrogate_loss - config.kl_coef * approx_kl,
            masks
        )

        # Early stopping based on ratio threshold
        if self.masked_mean(ratio_new_old, masks) > config.ratio_threshold:
            loss = loss * 0

        return loss

    def forward(self, input_: torch.Tensor, kwargs: dict):
        # Regular language model forward pass
        output = super().forward(input_, kwargs)
        
        # If we have GRPO inputs, compute GRPO loss
        if all(k in kwargs for k in ["rewards", "advantages", "ref_logprobs", "old_logprobs"]):
            grpo_loss = self.compute_grpo_loss(
                logits=kwargs["logits"],
                labels=kwargs["labels"],
                rewards=kwargs["rewards"],
                advantages=kwargs["advantages"],
                ref_logprobs=kwargs["ref_logprobs"],
                old_logprobs=kwargs["old_logprobs"],
                config=kwargs["grpo_config"],
            )
            kwargs[LanguageModelLossNames.grpo_loss] = grpo_loss
            
        return output
