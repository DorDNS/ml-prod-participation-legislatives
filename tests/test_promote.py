import pytest
from mlflow.exceptions import MlflowException

from ml.promote import current_champion, promote, rollback

MODEL = "polling-station-turnout"


class FakeVersion:
    def __init__(self, version, tags=None):
        self.version = version
        self.tags = dict(tags or {})


class FakeClient:
    """In-memory registry: some versions, one alias."""

    def __init__(self, versions, champion=None):
        self.versions = {str(v): FakeVersion(str(v)) for v in versions}
        self.champion = None if champion is None else str(champion)

    def search_model_versions(self, filter_string):
        return list(self.versions.values())

    def get_model_version(self, name, version):
        return self.versions[str(version)]

    def get_model_version_by_alias(self, name, alias):
        if self.champion is None:
            raise MlflowException("no champion")
        return self.versions[self.champion]

    def set_model_version_tag(self, name, version, key, value):
        self.versions[str(version)].tags[key] = value

    def set_registered_model_alias(self, name, alias, version):
        self.champion = str(version)


def test_promote_moves_the_alias_and_marks_both_versions_approved():
    client = FakeClient([1, 2], champion=1)
    result = promote(client, MODEL, "2")
    assert result == {"action": "promote", "previous_version": "1", "new_version": "2"}
    assert client.champion == "2"
    assert client.versions["1"].tags["approved"] == "true"
    assert client.versions["2"].tags["approved"] == "true"


def test_promote_works_without_a_champion():
    client = FakeClient([1])
    result = promote(client, MODEL, "1")
    assert result["previous_version"] is None
    assert current_champion(client, MODEL) == "1"


def test_promote_refuses_an_unknown_version():
    client = FakeClient([1], champion=1)
    with pytest.raises(ValueError):
        promote(client, MODEL, "9")


def test_promote_refuses_the_current_champion():
    client = FakeClient([1, 2], champion=2)
    with pytest.raises(ValueError):
        promote(client, MODEL, "2")


def test_rollback_goes_back_to_an_approved_version():
    client = FakeClient([1, 2], champion=1)
    promote(client, MODEL, "2")
    result = rollback(client, MODEL, "1")
    assert result == {"action": "rollback", "previous_version": "2", "new_version": "1"}
    assert client.champion == "1"


def test_rollback_refuses_a_version_that_was_never_champion():
    client = FakeClient([1, 2, 3], champion=1)
    promote(client, MODEL, "2")
    with pytest.raises(ValueError, match="not approved"):
        rollback(client, MODEL, "3")
    assert client.champion == "2"


def test_rollback_refuses_the_current_champion_and_unknown_versions():
    client = FakeClient([1, 2], champion=2)
    with pytest.raises(ValueError):
        rollback(client, MODEL, "2")
    with pytest.raises(ValueError):
        rollback(client, MODEL, "7")


def test_rollback_needs_a_champion():
    client = FakeClient([1])
    with pytest.raises(ValueError):
        rollback(client, MODEL, "1")
