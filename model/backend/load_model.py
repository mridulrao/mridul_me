from pathlib import Path
from typing import Dict, List, Literal
from pydantic import BaseModel

Backend = Literal["mlx", "cuda", "cpu"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ModelConfig(BaseModel):
    model_id: str = "LiquidAI/LFM2.5-350M"
    backend: Backend
    dtype: str = "bfloat16"
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_k: int = 50
    repetition_penalty: float = 1.1


class ModelRunner:
    def __init__(self, config: ModelConfig | None = None):
        if config is None:
            config = ModelConfig()

        self.config = config
        self.model_source, self.local_files_only = self._resolve_model_source(
            config.model_id
        )

        if config.backend == "mlx":
            self._load_mlx()
        elif config.backend == "cuda":
            self._load_cuda()
        else:
            self._load_cpu()

    def _resolve_model_source(self, model_id: str) -> tuple[str, bool]:
        model_path = Path(model_id).expanduser()
        candidate_paths = []

        if model_path.is_absolute():
            candidate_paths.append(model_path)
        else:
            candidate_paths.append(Path.cwd() / model_path)
            candidate_paths.append(PROJECT_ROOT / model_path)

        for candidate in candidate_paths:
            if candidate.exists():
                return str(candidate.resolve()), True

        if any(sep in model_id for sep in ("/", "\\")) or model_id.startswith((".", "~")):
            searched = ", ".join(str(path.resolve()) for path in candidate_paths)
            raise FileNotFoundError(
                f"Local model path not found: {model_id}. Checked: {searched}"
            )

        return model_id, False

    # will have lazy initialization for model and tokenizer
    def _load_mlx(self):
        from mlx_lm import load

        # MLX-optimized checkpoint for Apple Silicon
        self.model, self.tokenizer = load(self.model_source)

    def _load_cuda(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_source,
            device_map="auto",
            local_files_only=self.local_files_only,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",  # enable if installed
        )
        self.model.eval()

    def _load_cpu(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_source,
            local_files_only=self.local_files_only,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        self.model.eval()

    def chat(self, messages: List[Dict[str, str]]) -> str:
        if self.config.backend == "mlx":
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            sampler = make_sampler(temp=self.config.temperature)

            result = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.config.max_new_tokens,
                sampler=sampler,
            )
            print("=========")
            print(result)
            print("=========")
            return result
            

        else:
            import torch

            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                )

            new_tokens = output[0][inputs["input_ids"].shape[-1]:]
            result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            print("=========")
            print(result)
            print("=========")
            return result
