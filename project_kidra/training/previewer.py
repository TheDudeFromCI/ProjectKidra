import torch

from project_kidra.moe import MoEModel
from project_kidra.corpus import Tokenizer


class Previewer:
    """Generates sample text from a model so training progress can be eyeballed."""

    def __init__(
        self,
        model: MoEModel,
        tokenizer: Tokenizer,
        num_tokens: int = 50,
        temperature: float = 1.0,
    ) -> None:
        """Create a previewer using an injected model and tokenizer."""

        if num_tokens < 1:
            raise ValueError("num_tokens must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.num_tokens = num_tokens
        self.temperature = temperature

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor) -> list[int]:
        """Sample num_tokens continuation ids for a (1, seq) prompt, restoring train mode after."""

        was_training = self.model.training
        self.model.eval()
        try:
            tokens = prompt.clone()
            window = self.model.config.context_window
            generated: list[int] = []
            for _ in range(self.num_tokens):
                logits, _ = self.model(tokens[:, -window:])
                probs = torch.softmax(logits[:, -1, :] / self.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated.append(int(next_token.item()))
                tokens = torch.cat([tokens, next_token], dim=1)
            return generated
        finally:
            self.model.train(was_training)

    def preview(self, prompt: torch.Tensor) -> str:
        """Return the decoded prompt followed by the generated continuation."""

        generated = self.generate(prompt)
        prompt_text = self.tokenizer.decode([int(t) for t in prompt[0].tolist()])
        return f"{prompt_text} >>> {self.tokenizer.decode(generated)}"
