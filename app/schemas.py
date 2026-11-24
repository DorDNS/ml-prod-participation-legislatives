"""Request and response schemas of the prediction API."""

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """The five inputs of the turnout model, in the order the model expects."""

    model_config = ConfigDict(str_strip_whitespace=True)

    prev_turnout: float = Field(
        ge=0,
        le=100,
        allow_inf_nan=False,
        description="Turnout of the last first round, in %",
    )
    prev2_turnout: float = Field(
        ge=0,
        le=100,
        allow_inf_nan=False,
        description="Turnout of the first round before that, in %",
    )
    registered: float = Field(
        gt=0,
        allow_inf_nan=False,
        description="Number of registered voters in the polling station",
    )
    dept_code: str = Field(
        min_length=1,
        max_length=3,
        description='Department code, for example "75" or "2A"',
    )
    election_year: int = Field(ge=2000, le=2100, description="Year of the election")


class PredictionResponse(BaseModel):
    """The predicted turnout and the model version that produced it."""

    turnout_pct: float
    model_version: str