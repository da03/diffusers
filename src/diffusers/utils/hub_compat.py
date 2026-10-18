"""Download adapters for the legacy pipeline's Hub 0.x and 1.x callers."""
import inspect
import warnings

from huggingface_hub import hf_hub_download as _hf_hub_download
from huggingface_hub import snapshot_download as _snapshot_download


def _download(download, *args, **kwargs):
    if "use_auth_token" in kwargs:
        legacy_token = kwargs.pop("use_auth_token")
        if "token" in kwargs and legacy_token is not None and kwargs["token"] != legacy_token:
            raise ValueError("Specify only one of token and use_auth_token.")
        kwargs.setdefault("token", legacy_token)
    supported = inspect.signature(download).parameters
    if "resume_download" not in supported and "resume_download" in kwargs:
        if kwargs.pop("resume_download"):
            warnings.warn("Hub 1.x resumes downloads automatically; resume_download is no longer needed.", FutureWarning)
    if "proxies" not in supported and "proxies" in kwargs:
        if kwargs.pop("proxies") is not None:
            raise ValueError(
                "Hub 1.x does not accept per-call proxies. Configure HTTP_PROXY/HTTPS_PROXY "
                "or huggingface_hub.set_client_factory instead."
            )
    return download(*args, **kwargs)


def hf_hub_download(*args, **kwargs):
    return _download(_hf_hub_download, *args, **kwargs)


def snapshot_download(*args, **kwargs):
    return _download(_snapshot_download, *args, **kwargs)
