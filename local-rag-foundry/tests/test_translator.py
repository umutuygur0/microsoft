import torch

from src.translator import LocalTranslator


class _FakeTokenizer:
    """Deterministic stand-in for the NLLB tokenizer: 'translation' is just
    uppercasing, so tests can assert on exact output without downloading a
    real model. Includes a real (tiny) "input_ids" tensor so the
    length-scaled max_length calculation in _translate_batch has something
    real to call .shape on."""

    def __call__(self, texts, return_tensors=None, padding=None, truncation=None, max_length=None, src_lang=None):
        return {"_texts": texts, "input_ids": torch.zeros((len(texts), 4), dtype=torch.long)}

    def batch_decode(self, generated, skip_special_tokens=True):
        return generated  # the fake model already returns plain strings


class _FakeModel:
    def generate(self, **kwargs):
        return [t.upper() for t in kwargs["_texts"]]


def make_translator_with_fake_model() -> LocalTranslator:
    translator = LocalTranslator()
    translator._tokenizer = _FakeTokenizer()
    translator._model = _FakeModel()
    translator.state = "ready"
    return translator


def test_not_ready_returns_none():
    translator = LocalTranslator()
    assert not translator.ready
    assert translator.translate_to_turkish("hello") is None


def test_empty_text_returns_none():
    translator = make_translator_with_fake_model()
    assert translator.translate_to_turkish("   ") is None


def test_translate_simple_sentence():
    translator = make_translator_with_fake_model()
    result = translator.translate_to_turkish("hello world")
    assert result == "HELLO WORLD"


def test_translate_preserves_blank_lines():
    translator = make_translator_with_fake_model()
    result = translator.translate_to_turkish("first line\n\nsecond line")
    assert result == "FIRST LINE\n\nSECOND LINE"


def test_translate_maps_known_headings_deterministically():
    # Headings are looked up, not sent through the (fake, uppercasing) model
    # -- if they were translated by the model they'd come back upper-cased.
    translator = make_translator_with_fake_model()
    text = "Summary\nUse a calibrated detector.\n\nSafety Warnings\nDo not use flames."
    result = translator.translate_to_turkish(text)
    lines = result.split("\n")
    assert lines[0] == "Özet"
    assert lines[1] == "USE A CALIBRATED DETECTOR."
    assert lines[3] == "Güvenlik Uyarıları"


def test_translate_preserves_list_prefixes():
    translator = make_translator_with_fake_model()
    result = translator.translate_to_turkish("1. do the first thing\n- do another thing")
    lines = result.split("\n")
    assert lines[0] == "1. DO THE FIRST THING"
    assert lines[1] == "- DO ANOTHER THING"


def test_translate_returns_none_on_model_failure():
    translator = LocalTranslator()
    translator.state = "ready"

    class _BrokenModel:
        def generate(self, **kwargs):
            raise RuntimeError("boom")

    translator._tokenizer = _FakeTokenizer()
    translator._model = _BrokenModel()
    assert translator.translate_to_turkish("hello") is None
