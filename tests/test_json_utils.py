import pytest

from physics_agent.json_utils import extract_json


def test_extract_json_plain():
    result = extract_json('{"a": 1, "b": "two"}')
    assert result == {"a": 1, "b": "two"}


def test_extract_json_strips_markdown_fences():
    result = extract_json('```json\n{"a": 1}\n```')
    assert result == {"a": 1}


def test_extract_json_extracts_from_surrounding_prose():
    result = extract_json('Sure, here you go: {"a": 1} -- hope that helps!')
    assert result == {"a": 1}


def test_extract_json_raises_when_no_json_object_present():
    with pytest.raises(ValueError):
        extract_json("no json here at all")


# -- backslash sanitization: the real failure from a live run ----------------


def test_extract_json_handles_invalid_single_char_escapes():
    # \l, \m, \p, \c are not valid JSON escapes at all -- previously a hard
    # json.JSONDecodeError crash.
    raw = r'{"a": "$r \le R$ and $\mu_0$ and $\pi$ and $\cdot$"}'
    result = extract_json(raw)
    assert result["a"] == r"$r \le R$ and $\mu_0$ and $\pi$ and $\cdot$"


def test_extract_json_handles_valid_looking_escape_that_is_actually_latex():
    # \t IS a valid JSON escape (tab) -- \text{mm} previously "succeeded"
    # silently as a tab character followed by garbled literal text "ext{mm}".
    raw = r'{"unit": "5.0\text{ mm}"}'
    result = extract_json(raw)
    assert result["unit"] == r"5.0\text{ mm}"  # preserved as literal text, not corrupted


def test_extract_json_handles_the_exact_real_failure_from_a_live_run():
    # Reproduces (trimmed) the actual raw model output that crashed
    # generate_problem_set_cli.py with "Invalid \escape: line 1 column 240".
    raw = (
        r'{"problem_text": "An infinitely long cylindrical wire of radius '
        r'$R = 5.0\text{ mm}$ carries a total current $I = 10.0\text{ A}$. '
        r'The current is distributed with a radial current density '
        r'$J(r) = J_0 (1 - r/R)$ for $r \le R$, where $r$ is the distance '
        r'from the central axis. Calculate the magnitude of the magnetic '
        r'field at a distance of $r = 2.5\text{ mm}$ from the axis. Use '
        r'$\mu_0 = 4\pi \times 10^{-7}\text{ T}\cdot\text{m/A}$.", '
        r'"target_concepts": ["Ampere\'s Law"], "rationale": "test"}'
    )
    result = extract_json(raw)
    assert "cylindrical wire" in result["problem_text"]
    # The sanitizer makes this parseable without crashing; it doesn't try
    # to guess that \' was meant to just be an apostrophe -- it's not a
    # valid JSON escape either way, so the backslash survives literally,
    # same as any other invalid escape sequence.
    assert result["target_concepts"] == ["Ampere\\'s Law"]


def test_extract_json_preserves_genuine_unicode_escape():
    raw = r'{"a": "caf\u00e9"}'
    result = extract_json(raw)
    assert result["a"] == "café"


def test_extract_json_preserves_genuine_backslash_and_quote_escapes():
    raw = r'{"path": "C:\\Users\\test", "quote": "she said \"hi\""}'
    result = extract_json(raw)
    assert result["path"] == "C:\\Users\\test"
    assert result["quote"] == 'she said "hi"'


def test_extract_json_handles_greek_letter_commands_starting_with_valid_escape_letters():
    # \rho, \tau, \nabla, \beta, \frac all start with a letter that's a
    # valid single-char JSON escape (r, t, n, b, f) -- must not be
    # misinterpreted as those escapes just because the letter matches.
    raw = r'{"a": "\rho and \tau and \nabla and \beta and \frac{1}{2}"}'
    result = extract_json(raw)
    assert result["a"] == r"\rho and \tau and \nabla and \beta and \frac{1}{2}"
