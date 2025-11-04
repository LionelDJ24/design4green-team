import torchfrom transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "EleutherAI/pythia-70m-deduped"def load_model(optimized=False):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if optimized:
        print("⚙️ Chargement du modèle optimisé (INT8 dynamique)...")
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    else:
        print("🔹 Chargement du modèle FP32 (baseline)...")

    return tokenizer, model

def generate_summary(text, tokenizer, model, max_words=15):
    prompt = f"Résume ce texte en {max_words} mots maximum : {text}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # On nettoie un peureturn " ".join(summary.split()[:max_words])