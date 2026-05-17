from __future__ import annotations

import importlib.metadata

from graphein import verbose

from dynamicmpnn import utils
from dynamicmpnn.utils import register_custom_omegaconf_resolvers

verbose(False)


try:
    __version__ = importlib.metadata.version("dynamicmpnn")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"


__all__ = ["__version__", "register_custom_omegaconf_resolvers"]
