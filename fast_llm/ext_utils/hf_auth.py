# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This is a cut down version of getting an authentication token for HF hub
from https://github.com/huggingface/huggingface_hub/tree/d29290fbd2689fdc125e27ce0f91e8227a2f20de
It does not include getting the token in Google colab environment.
Also, it does not look for the token in the old path as it was marked
for removal in the future versions of the huggingface_hub.

NOTE: It is needed to track changes in huggingface_hub or include it as dependency later.
"""

import os
import pathlib

from typing import Optional

# default cache
default_home = os.path.join(os.path.expanduser("~"), ".cache")
HF_HOME = os.path.expanduser(
    os.getenv(
        "HF_HOME",
        os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "huggingface"),
    )
)

HF_TOKEN_PATH = os.environ.get("HF_TOKEN_PATH", os.path.join(HF_HOME, "token"))


def hf_auth_get_token() -> Optional[str]:
    """
    Get token if user is logged in.

    Token is retrieved in priority from the `HF_TOKEN` environment variable. Otherwise, we read the token file located
    in the Hugging Face home folder. Returns None if user is not logged in. To log in, use [`login`] or
    `huggingface-cli login`.

    Returns:
        `str` or `None`: The token, `None` if it doesn't exist.
    """
    return _get_token_from_environment() or _get_token_from_file()


def _get_token_from_environment() -> Optional[str]:
    # `HF_TOKEN` has priority (keep `HUGGING_FACE_HUB_TOKEN` for backward compatibility)
    return _clean_token(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def _get_token_from_file() -> Optional[str]:
    try:
        return _clean_token(pathlib.Path(HF_TOKEN_PATH).read_text())
    except FileNotFoundError:
        return None


def _clean_token(token: Optional[str]) -> Optional[str]:
    """Clean token by removing trailing and leading spaces and newlines.

    If token is an empty string, return None.
    """
    if token is None:
        return None
    return token.replace("\r", "").replace("\n", "").strip() or None
