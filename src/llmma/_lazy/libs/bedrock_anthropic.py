import os

from anthropic import AnthropicBedrock, AsyncAnthropicBedrock
from attrs import define

from .anthropic import Anthropic


@define
class BedrockAnthropic(Anthropic):
    api_key = ""
    aws_access_key: str | None = None
    aws_secret_key: str | None = None
    aws_region: str | None = None

    def __attrs_post_init__(self):
        aws_access_key = self.aws_access_key or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = self.aws_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")

        self.client = AnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=self.aws_region,
        )

        self.async_client = AsyncAnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=self.aws_region,
        )
