# Developer guide

[Back to the project overview](../README.md)

## Optional Gemini setup

The dashboard can run without a Gemini key. To enable generated advisor responses, copy [`.env.example`](../.env.example) to `.env` in the repository root and enter your own key:

```dotenv
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

`api.py` reads the root `.env` at startup, so restart the backend after editing it. Keep `.env` local; it is ignored by Git. Model availability depends on your Gemini account. If generation is unavailable, the advisor has a deterministic fallback.

## Verification commands

Run the Python suite from the repository root with the project environment activated:

```bash
python -m pytest tests/ -q
```

Check the frontend from its own directory:

```bash
cd frontend
npm run build
npm run lint
```

These commands are provided for local verification; this repository does not currently publish a CI pass status. The Python tests cover trading transitions, reward and risk mathematics, market data, evaluation behaviour, API contracts and explanation handling. They do not rerun all historical training experiments.

## Training and evaluation

Inspect the available options first:

```bash
python src/train_agent.py --help
python src/walk_forward_eval.py --help
```

Training is unnecessary for the bundled demo and can be computationally expensive. Run new experiments in a separate clone: training writes model files, and walk-forward evaluation writes `walk_forward_results.csv` in the repository root.

For a new run, retain the commit, input data hashes, date ranges, seeds, parameters and environment versions together with its outputs. Do not describe a new run as a reproduction of the bundled historical tables; see [result provenance](EVALUATION.md).

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Python package cannot be imported | Activate the project environment and run `python -m pip install -r requirements.txt` with that same interpreter. |
| PowerShell blocks environment activation | Use `.\env\Scripts\python.exe` directly in place of `python`, or activate from Command Prompt. No system-wide policy change is needed. |
| Missing model or dataset | Start `python api.py` from the repository root; inspect the backend's model-loading output and the `models/` and `data/` folders. |
| Dashboard loads but API calls fail | Ensure the backend is listening on port 8000. Open `http://localhost:8000/docs` and inspect the backend terminal. |
| Port already in use | Stop the earlier project server in its terminal. The Windows launcher expects ports 8000 and 5173 to be free. |
| Advisor gives a fallback response | Check the optional key, configured model and network access; fallback responses are supported behaviour. |
| Chart uses archived data | Check the source/date labels. Live feeds depend on external services; archived candles are not current quotes. |
| SHAP or simulation requests take time | These requests compute model explanations or run multiple scenarios. Try a smaller simulation count and inspect the backend output. |

The Windows launcher writes its own diagnostic files to `logs/`. If reporting a problem, include the command and relevant error text rather than your `.env` file.
