from flask import Flask, request, jsonify, render_templatefrom codecarbon import EmissionsTrackerfrom model_utils import load_model, generate_summaryimport time, psutil

app = Flask(__name__)
# Charger les deux versions du modèle au démarragetokenizer_base, model_base = load_model(optimized=False)
tokenizer_opt, model_opt = load_model(optimized=True)
@app.route("/")def home():
    return render_template("index.html")
@app.route("/summarize", methods=["POST"])def summarize():
    data = request.json
    text = data.get("text", "")
    optimized = data.get("optimized", False)

    tracker = EmissionsTracker(output_file="carbon.json", measure_power_secs=1)
    tracker.start()
    start_time = time.time()

    tokenizer = tokenizer_opt if optimized else tokenizer_base
    model = model_opt if optimized else model_base
    summary = generate_summary(text, tokenizer, model)

    latency = round((time.time() - start_time) * 1000, 2)
    emissions = tracker.stop()
    memory = psutil.Process().memory_info().rss / (1024 * 1024)

    return jsonify({
        "summary": summary,
        "optimized": optimized,
        "latency_ms": latency,
        "energy_wh": round(emissions * 1000, 6),
        "memory_mb": round(memory, 2)
    })
if __name__ == "__main__":
    app.run(debug=True)