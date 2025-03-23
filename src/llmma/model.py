import typing as t

from attrs import define, field


@define
class Info:
    prompt_cost: float
    completion_cost: float
    context_limit: int
    output_limit: int | None = None
    limit_per_minute: int | None = None
    image_input_cost: float | None = None
    hf_repo: str | None = None
    chat: bool = True
    local: bool = False
    quirks: dict[str, t.Any] = field(factory=dict)

    def __attrs_post_init__(self):
        if self.output_limit is None:
            self.output_limit = self.context_limit // 2


@define
class Usage:
    prompt_tokens: int
    completion_tokens: int
