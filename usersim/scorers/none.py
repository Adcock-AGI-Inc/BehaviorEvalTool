class NoScorer:
    def score(self, conv, sc) -> dict:
        return {"verdict": "-", "lines": []}


def create(cfg, key="", out=print, limits=None):
    return NoScorer()
