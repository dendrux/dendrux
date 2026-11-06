"""Exercise a checkpoint store without the optional SQL public-page optimization."""

from dendrux.runtime._capabilities import capabilities


class KeysetStore:
    def __init__(self, inner):
        self.inner = inner
        self.pages = 0

    @property
    def capabilities(self):
        return capabilities(self.inner)

    def __getattr__(self, name):
        if name == "get_public_traces":
            raise AttributeError(name)
        return getattr(self.inner, name)

    async def get_traces_after(self, *args, **kwargs):
        self.pages += 1
        return await self.inner.get_traces_after(*args, **kwargs)
