"""The on-device answer writer: Qwen2.5-0.5B-Instruct, 4-bit, as one ONNX graph (models/manifest.json "answer-lm",
built by scripts/native/quantize_answer_lm.py; 327 MB), run with plain onnxruntime + tokenizers. No key, no network,
no torch.

One model per process, loaded once (answer.warm() at startup, else on first use). Generation is a greedy decode loop
with a KV cache on a worker thread (onnxruntime releases the GIL); text reaches the event loop through
call_soon_threadsafe. Capped at ANSWER_LM_MAX_TOKENS. Cancelling stops the loop before its next token (a
threading.Event checked every step). One generation at a time (a lock): a cancelled draft frees the model at once, an
uncancelled one finishes first.

onnxruntime / tokenizers are imported lazily, so a runtime without them imports this module fine and the answer falls
back to the extractive sentence."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

from . import config

# Qwen2.5's chat format (its tokenizer_config chat_template with a system message), written out so the tokenizers
# package alone can encode it; tests/test_answer_local.py checks it against transformers' apply_chat_template.
CHAT = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

_lock = threading.Lock()          # one load, then one generation at a time
_engine = None
_load_error: str | None = None
_LOAD: dict = {}


def model_dir() -> Path:
    return Path(config.ANSWER_LM_DIR)


def installed() -> bool:
    d = model_dir()
    return (d / "model.onnx").exists() and (d / "tokenizer.json").exists() and (d / "config.json").exists()


class OnnxEngine:
    """Tokenizer + ONNX session + the greedy decode loop (inputs input_ids, attention_mask, position_ids,
    past_key_values.N.key/value; outputs logits [batch, 1, vocab] for the last position, present.N.key/value)."""

    def __init__(self, d: Path, threads: int):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.np = np
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads > 0:
            so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(d / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        cfg = json.loads((d / "config.json").read_text())
        self.layers = cfg["num_hidden_layers"]
        self.kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
        eos = cfg.get("eos_token_id")
        gen = d / "generation_config.json"
        if gen.exists():
            eos = json.loads(gen.read_text()).get("eos_token_id", eos)
        self.eos = set(eos if isinstance(eos, list) else [eos])
        for t in ("<|im_end|>", "<|endoftext|>"):
            i = self.tok.token_to_id(t)
            if i is not None:
                self.eos.add(i)
        self.outputs = [o.name for o in self.sess.get_outputs()]

    def prompt_ids(self, system: str, user: str) -> list[int]:
        return self.tok.encode(CHAT.format(system=system, user=user), add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def generate(self, ids: list[int], max_new: int, on_token, stop) -> None:
        np = self.np
        past = {}
        for i in range(self.layers):
            for kv in ("key", "value"):
                past[f"past_key_values.{i}.{kv}"] = np.zeros((1, self.kv_heads, 0, self.head_dim), np.float32)
        cur = np.array([ids], dtype=np.int64)
        n = len(ids)
        for step in range(max_new):
            if stop():
                return
            feed = {"input_ids": cur, "attention_mask": np.ones((1, n), np.int64),
                    "position_ids": np.arange(n - cur.shape[1], n, dtype=np.int64)[None, :], **past}
            outs = self.sess.run(self.outputs, feed)
            nxt = int(np.argmax(outs[0][0, -1]))
            for name, val in zip(self.outputs[1:], outs[1:]):
                past["past_key_values." + name[len("present."):]] = val
            if nxt in self.eos:
                return
            on_token(nxt)
            cur = np.array([[nxt]], dtype=np.int64)
            n += 1


def load():
    """Load the engine once (thread-safe). Raises when the model is missing or fails to load."""
    global _engine, _load_error
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is not None:
            return _engine
        if not installed():
            _load_error = f"answer model not installed in {model_dir()} (scripts/fetch_models.py fetch answer-lm)"
            raise FileNotFoundError(_load_error)
        t = time.perf_counter()
        _engine = OnnxEngine(model_dir(), config.ANSWER_LM_THREADS)
        _load_error = None
        _LOAD["s"] = round(time.perf_counter() - t, 2)
        return _engine


def loaded() -> bool:
    return _engine is not None


def info() -> dict:
    return {"dir": str(model_dir()), "installed": installed(), "loaded": loaded(), "runtime": "onnxruntime (CPU)",
            "threads": config.ANSWER_LM_THREADS, "load_s": _LOAD.get("s"), "load_error": _load_error}


def unload():
    """Tests."""
    global _engine
    with _lock:
        _engine = None


class Generation:
    """One generation on the worker thread. `queue` gets text pieces, then None (end) or an Exception.
    n_in / n_out: prompt and generated tokens (for usage()); cancel() stops it before the next token."""

    def __init__(self, system: str, user: str, max_new: int, loop: asyncio.AbstractEventLoop):
        self.system, self.user, self.max_new, self.loop = system, user, max_new, loop
        self.queue: asyncio.Queue = asyncio.Queue()
        self.stop = threading.Event()
        self.n_in = 0
        self.n_out = 0
        self.thread = threading.Thread(target=self._work, name="answer-lm", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def cancel(self):
        self.stop.set()

    def _put(self, x):
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, x)
        except RuntimeError:              # the loop closed (server shutdown): nobody is listening
            pass

    def _work(self):
        try:
            eng = load()
            out: list[int] = []
            sent = [""]

            def on_token(tid: int):
                out.append(tid)
                self.n_out += 1
                text = eng.decode(out)
                if text.endswith("�"):           # half of a multi-byte character: wait for the rest
                    return
                if text.startswith(sent[0]) and len(text) > len(sent[0]):
                    self._put(text[len(sent[0]):])
                    sent[0] = text

            with _lock:                   # one generation at a time on the one model
                if not self.stop.is_set():
                    ids = eng.prompt_ids(self.system, self.user)
                    self.n_in = len(ids)
                    eng.generate(ids, self.max_new, on_token, self.stop.is_set)
            self._put(None)
        except BaseException as e:  # noqa: BLE001 - surfaced to the stream, which falls back to extractive
            self._put(e)
