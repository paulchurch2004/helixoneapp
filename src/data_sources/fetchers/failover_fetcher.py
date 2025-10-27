class FailoverFetcher:
    def __init__(self, fetchers):
        self.fetchers = fetchers

    def fetch(self, *args, **kwargs):
        last_exception = None
        for fetcher in self.fetchers:
            try:
                result = fetcher.fetch(*args, **kwargs)
                if result:
                    return result
            except Exception as e:
                last_exception = e
                continue
        raise RuntimeError(f"Aucune source n'a pu fournir les données : {last_exception}")
