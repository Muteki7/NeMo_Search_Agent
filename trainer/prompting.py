"""Single place that renders chat prompts, with the chat template's date pinned.

Why this exists: the Llama 3.x chat template writes a "Cutting Knowledge Date / Today Date" header into every
system block, and unless told otherwise it fills the date from the real clock (`strftime_now`). The same data would
then produce slightly different prompt token ids on different days (1-2 tokens of ~270), so a run or an eval could
not be replayed exactly, and runs made on different days would not see identical inputs.

Every script that turns messages into text or token ids (spike, rollout, eval) must call `apply_chat_template`
from this module instead of `tokenizer.apply_chat_template`, so no render site can forget the pin.

Scope: this only covers prompts WE render. Rendering done elsewhere (for example Gym's own model server during
`gym eval run`) uses its own template call and today's real date.
"""

# The date the Llama 3.1 template used as its default before it switched to the real clock.
PROMPT_DATE = "26 Jul 2024"


def apply_chat_template(tokenizer, messages, **kwargs):
    """`tokenizer.apply_chat_template(messages, **kwargs)` with `date_string` fixed to PROMPT_DATE.

    Raises if the caller passes a different `date_string`, so a pinned date cannot be overridden by accident.
    """
    requested = kwargs.pop("date_string", PROMPT_DATE)
    if requested != PROMPT_DATE:
        raise ValueError(f"date_string must be the pinned {PROMPT_DATE!r}, got {requested!r}")
    return tokenizer.apply_chat_template(messages, date_string=PROMPT_DATE, **kwargs)


def assert_date_pinned(rendered_text: str) -> None:
    """Check on a rendered prompt string: the header appears exactly once and carries the pinned date."""
    assert rendered_text.count("Cutting Knowledge Date") == 1, "system header missing or duplicated"
    assert f"Today Date: {PROMPT_DATE}" in rendered_text, "rendered prompt does not carry the pinned date"
