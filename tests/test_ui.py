"""Tests for the web page served at the root of the API."""

from fastapi.testclient import TestClient

from app.main import app

FIELDS = ["prev_turnout", "prev2_turnout", "registered", "dept_code", "election_year"]


def test_home_serves_an_html_page():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_home_page_has_the_five_fields_and_calls_the_api():
    page = TestClient(app).get("/").text
    for field in FIELDS:
        assert f'name="{field}"' in page
    assert "/predict" in page
    assert "/model-info" in page


def test_home_page_is_not_listed_in_the_api_schema():
    assert "/" not in app.openapi()["paths"]
