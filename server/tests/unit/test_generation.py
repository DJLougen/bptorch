import pytest
from fastapi.testclient import TestClient

from neural_blueprint.api.main import app
from neural_blueprint.runtime.generation import GenerationEngine
from neural_blueprint.runtime.tokenizer import DEFAULT_CHARS, CharacterTokenizer
from neural_blueprint.templates.architectures import create_arch_1_nanogpt_tiny
from neural_blueprint.tracing.debugger import global_session_manager


def test_tokenizer_roundtrip():
    tok = CharacterTokenizer(vocab_size=len(DEFAULT_CHARS))
    text = "Hi"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text
    assert len(encoded) == 2


def test_generation_engine_tokens():
    project = create_arch_1_nanogpt_tiny()
    session_id = "gen1"
    existing = global_session_manager.get_session(session_id)
    if existing is not None:
        existing.stop()

    session = global_session_manager.create_training_session(session_id, project, "cpu")
    engine = GenerationEngine(session)

    vocab_size = int(project.model.config.get("vocab_size", 32))
    prompt_ids = engine.encode_prompt("Hi")

    tokens = list(
        engine.generate_tokens(
            input_ids=prompt_ids,
            max_new_tokens=4,
            temperature=0.0,
        )
    )

    assert len(tokens) == 4
    for token_id, token_str in tokens:
        assert isinstance(token_id, int)
        assert 0 <= token_id < vocab_size
        assert isinstance(token_str, str)


def test_generate_route_non_stream():
    client = TestClient(app)
    project = create_arch_1_nanogpt_tiny()
    session_id = "gen_api_test"
    existing = global_session_manager.get_session(session_id)
    if existing is not None:
        existing.stop()

    global_session_manager.create_training_session(session_id, project, "cpu")

    response = client.post(
        f"/api/v1/sessions/{session_id}/generate",
        json={"prompt": "Hi", "max_new_tokens": 4, "temperature": 0.0, "stream": False},
    )

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert isinstance(data.get("text"), str)
    assert len(data.get("token_ids", [])) == 4


def test_tokenizer_byte_fallback():
    tok = CharacterTokenizer(vocab_size=len(DEFAULT_CHARS), byte_fallback=True)
    # Unicode characters outside standard ASCII/DEFAULT_CHARS
    text = "Hello 世界 🚀"
    encoded = tok.encode(text)
    assert len(encoded) > len("Hello  ")
    assert all(0 <= t < tok.vocab_size for t in encoded)


def test_prompt_templates():
    from neural_blueprint.runtime.generation import PromptTemplate

    raw = PromptTemplate.apply("Hello", "raw")
    assert raw == "Hello"

    chatml = PromptTemplate.apply("Hello", "chatml")
    assert "<|im_start|>user\nHello<|im_end|>" in chatml
    assert "<|im_start|>assistant" in chatml

    alpaca = PromptTemplate.apply("Hello", "alpaca")
    assert "### Instruction:\nHello" in alpaca

    llama3 = PromptTemplate.apply("Hello", "llama3")
    assert "<|begin_of_text|>" in llama3
    assert "<|start_header_id|>user" in llama3


def test_generation_engine_with_kv_cache_and_template():
    project = create_arch_1_nanogpt_tiny()
    session_id = "gen_kv_cache"
    existing = global_session_manager.get_session(session_id)
    if existing is not None:
        existing.stop()

    session = global_session_manager.create_training_session(session_id, project, "cpu")
    engine = GenerationEngine(session)

    prompt_ids = engine.encode_prompt("Hello", template="chatml")
    tokens = list(
        engine.generate_tokens(
            input_ids=prompt_ids,
            max_new_tokens=4,
            temperature=0.0,
            use_cache=True,
        )
    )
    assert len(tokens) == 4
    assert engine.cache.step == 4


def test_generate_route_with_template_and_cache():
    client = TestClient(app)
    project = create_arch_1_nanogpt_tiny()
    session_id = "gen_api_template"
    existing = global_session_manager.get_session(session_id)
    if existing is not None:
        existing.stop()

    global_session_manager.create_training_session(session_id, project, "cpu")

    response = client.post(
        f"/api/v1/sessions/{session_id}/generate",
        json={
            "prompt": "Hello",
            "template": "chatml",
            "use_cache": True,
            "max_new_tokens": 4,
            "temperature": 0.0,
            "stream": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert len(data.get("token_ids", [])) == 4

def _create_engine(session_id: str, project):
    existing = global_session_manager.get_session(session_id)
    if existing is not None:
        existing.stop()
    session = global_session_manager.create_training_session(session_id, project, "cpu")
    return GenerationEngine(session)


def test_cache_matches_no_cache_nanogpt():
    project = create_arch_1_nanogpt_tiny()
    prompt_ids = _create_engine("gen_cache_parity_a", project).encode_prompt("Hi")

    engine_cached = _create_engine("gen_cache_parity_b", project)
    cached_ids = [
        tid
        for tid, _ in engine_cached.generate_tokens(
            input_ids=prompt_ids,
            max_new_tokens=4,
            temperature=0.0,
            use_cache=True,
        )
    ]

    engine_full = _create_engine("gen_cache_parity_c", project)
    full_ids = [
        tid
        for tid, _ in engine_full.generate_tokens(
            input_ids=prompt_ids,
            max_new_tokens=4,
            temperature=0.0,
            use_cache=False,
        )
    ]

    assert cached_ids == full_ids


def test_decode_cache_kv_sequence_grows():
    project = create_arch_1_nanogpt_tiny()
    engine = _create_engine("gen_cache_growth", project)
    prompt_ids = engine.encode_prompt("Hello")
    prompt_len = prompt_ids.size(1)
    new_tokens = 4

    list(
        engine.generate_tokens(
            input_ids=prompt_ids,
            max_new_tokens=new_tokens,
            temperature=0.0,
            use_cache=True,
        )
    )

    assert engine.decode_cache.k
    first_k = next(iter(engine.decode_cache.k.values()))
    assert first_k.size(-2) >= prompt_len + new_tokens - 1


def test_llama_tiny_cache_matches_no_cache():
    from neural_blueprint.templates.llama import create_llama_tiny_template

    project = create_llama_tiny_template()
    prompt_ids = _create_engine("gen_llama_cache_a", project).encode_prompt("Hi")

    engine_cached = _create_engine("gen_llama_cache_b", project)
    cached_ids = [
        tid
        for tid, _ in engine_cached.generate_tokens(
            input_ids=prompt_ids,
            max_new_tokens=4,
            temperature=0.0,
            use_cache=True,
        )
    ]

    engine_full = _create_engine("gen_llama_cache_c", project)
    full_ids = [
        tid
        for tid, _ in engine_full.generate_tokens(
            input_ids=prompt_ids,
            max_new_tokens=4,
            temperature=0.0,
            use_cache=False,
        )
    ]

    assert len(cached_ids) == 4
    assert cached_ids == full_ids


def test_rope_decode_positions_use_past_len():
    import torch

    from neural_blueprint.registry.primitives.attention import RoPENode
    from neural_blueprint.runtime.kv_context import DecodeCache, decode_cache_scope

    node = RoPENode()
    spec = node.build_runtime({"head_dim": 4, "n_head": 2, "base": 10000.0})
    apply_rope = spec.factory()

    x = torch.randn(1, 1, 8)
    out_zero = apply_rope(x)

    cache = DecodeCache(enabled=True, past_len=5)
    with decode_cache_scope(cache):
        out_offset = apply_rope(x)

    assert not torch.allclose(out_zero, out_offset)
