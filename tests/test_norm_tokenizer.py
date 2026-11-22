"""Unit tests for Python <-> JS normalization parity and Akshara 1,024-token tokenizer."""
import json
import subprocess
from shrota.norm import normalize_text, devanagari_ratio, tag_utterance
from shrota.tokenizer import ShrotaTokenizer, split_aksharas


def test_python_js_norm_parity():
    samples = [
        "नमस्ते\u200d! के छ तपाईंको खबर ?   ",
        "नेपालको राजधानी काठमाडौँ हो .",
        "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम् ॥",
        "यो १,२३४ रूपैयाँ हो — धन्यवाद!",
        "तिमी कहाँ जाँदै छौ??",
    ]
    for keep_p in (True, False):
        py_outs = [normalize_text(s, keep_punct=keep_p) for s in samples]
        js_code = (
            f"import {{ normalizeText }} from './docs/norm.js';\n"
            f"const samples = {json.dumps(samples, ensure_ascii=False)};\n"
            f"console.log(JSON.stringify(samples.map(s => normalizeText(s, {str(keep_p).lower()}))));\n"
        )
        p = subprocess.run(
            ["node", "--input-type=module", "-e", js_code],
            cwd=".",
            text=True,
            capture_output=True,
            check=True,
        )
        js_outs = json.loads(p.stdout.strip())
        assert py_outs == js_outs, f"Mismatch for keep_punct={keep_p}: {py_outs} != {js_outs}"


def test_tokenizer_1024_roundtrip():
    corpus = [
        "नेपालको संविधानले सबै नागरिकहरूलाई समान अधिकार दिएको छ।",
        "तिमी विद्यालय कहिले जान्छौ?",
        "काठमाडौँ उपत्यकामा आज पानी परेको छ।",
        "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्।",
    ]
    tok = ShrotaTokenizer.build_from_texts(corpus, vocab_size=1024)
    assert tok.vocab_size == 1024
    for text in corpus:
        ids = tok.encode(text, lang="ne", punct_mode="punct")
        dec = tok.decode(ids)
        assert dec == normalize_text(text, keep_punct=True)
    aks = split_aksharas("विद्यालय")
    assert len(aks) == 4
    assert devanagari_ratio(corpus[0]) > 0.95
    assert tag_utterance("नमस्ते नेपाल", lang="ne").startswith("<ne> <nopunct>")
    assert tag_utterance("नमस्ते नेपाल।", lang="ne").startswith("<ne> <punct>")
