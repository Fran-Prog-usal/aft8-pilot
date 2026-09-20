"""Instrumentación de Mistral para teacher forcing y gradientes de atención."""
from __future__ import annotations

import torch

MODEL_ID = "mistralai/Mistral-7B-v0.3"
REVISION = "caa1feb0e54d415e2df31207e5f4e273e33509b1"


def validate_attention(attentions):
    """Exige pesos causales finitos, normalizados y conectados en todas las capas."""
    if not attentions:
        raise ValueError("Faltan las matrices de atención; se requiere backend eager.")
    for index, attention in enumerate(attentions):
        values = attention.detach().float()
        if not attention.requires_grad or values.ndim != 4:
            raise ValueError(f"Atención desconectada o forma inválida en capa {index}.")
        if not torch.isfinite(values).all() or values.min() < -1e-4:
            raise ValueError(f"Atención no finita o negativa en capa {index}.")
        if (values.sum(-1) - 1).abs().max() > .05:
            raise ValueError(f"Atención no normalizada en capa {index}.")
        if torch.triu(values, diagonal=1).abs().max() > 1e-5:
            raise ValueError(f"Atención no causal en capa {index}.")


def tokenize_exact(tokenizer, text):
    """Tokeniza sin plantilla ni truncamiento y añade un único BOS explícito."""
    encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded != text:
        raise ValueError("El decode del tokenizador no reconstruye exactamente el texto.")
    if tokenizer.bos_token_id is None:
        raise ValueError("El tokenizador no define BOS.")
    if any(token in tokenizer.all_special_ids for token in ids):
        raise ValueError("El texto produce tokens especiales no autorizados.")
    return {"input_ids": [tokenizer.bos_token_id] + ids,
            "offsets": [(0, 0)] + [tuple(offset) for offset in encoded["offset_mapping"]],
            "special_mask": [True] + [False] * len(ids),
            "tokens": tokenizer.convert_ids_to_tokens([tokenizer.bos_token_id] + ids),
            "decode_exact": True}


class MistralExtractor:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.model.eval()
        self.model.requires_grad_(False)
        if getattr(self.model.config, "sliding_window", None) is not None:
            raise ValueError("Esta especificación exige atención causal sin ventana deslizante.")

    @classmethod
    def load(cls):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("La ejecución científica requiere CUDA y soporte bfloat16.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION,
                                                    dtype=torch.bfloat16, attn_implementation="eager")
        if model.config.model_type != "mistral" or model.config.num_hidden_layers != 32:
            raise ValueError("Arquitectura distinta de la especificación Mistral-7B-v0.3.")
        return cls(model.to("cuda"), tokenizer)

    def forward(self, text):
        encoding = tokenize_exact(self.tokenizer, text)
        ids = torch.tensor([encoding["input_ids"]], device=self.device)
        if ids.shape[1] > self.model.config.max_position_embeddings:
            raise ValueError("El texto excede el contexto; no se permite truncar.")
        with torch.enable_grad():
            embeddings = self.model.get_input_embeddings()(ids).detach().requires_grad_(True)
            output = self.model(inputs_embeds=embeddings, output_hidden_states=True,
                                output_attentions=True, use_cache=False)
            attentions = list(output.attentions) if output.attentions is not None else []
            validate_attention(attentions)
            logits = output.logits[0].float()
        return encoding, ids[0], logits, attentions, output.hidden_states
