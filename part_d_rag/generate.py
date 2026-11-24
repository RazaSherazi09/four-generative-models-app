import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

GENERATOR_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MEDICAL_NOTICE = "For educational use only. Not medical advice."


def load_generator(model_id: str = GENERATOR_MODEL_ID, device: torch.device | None = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    return tokenizer, model, device


def build_rag_prompt(question: str, top_chunks: list[dict]) -> list[dict]:
    """Build chat messages instructing Qwen2.5-1.5B-Instruct to answer strictly from [1]-[5]."""
    passages_block = "\n\n".join(
        f"[{i}] (Abstract ID: {c['abstract_id']}): {c['text']}"
        for i, c in enumerate(top_chunks[:5], start=1)
    )
    system_prompt = (
        "You are a biomedical research assistant. Answer the user's question ONLY using "
        "facts explicitly stated in the provided passages [1] through [5]. "
        "Cite the supporting passage numbers in square brackets (e.g., [1], [2]) after every claim. "
        "Do not use outside knowledge or extrapolate beyond the passages. "
        "If the provided passages do not contain enough information to answer the question, "
        'respond strictly with: "I do not know based on the provided passages."'
    )
    user_prompt = (
        f"Passages:\n{passages_block}\n\n"
        f"Question: {question}\n\n"
        "Answer (cite passages [1]-[5], or state that you do not know):"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


@torch.no_grad()
def generate_answer(
    question: str,
    top_chunks: list[dict],
    tokenizer,
    model,
    device: torch.device,
    max_new_tokens: int = 256,
) -> str:
    """Greedy decoding (do_sample=False, max_new_tokens=256) using the model's chat template."""
    messages = build_rag_prompt(question, top_chunks)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Clear sampling parameters so greedy decoding runs cleanly without warnings
    gen_config = model.generation_config
    gen_config.do_sample = False
    gen_config.temperature = None
    gen_config.top_p = None
    gen_config.top_k = None

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0, input_len:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer