from atlas_rtc.adapters.mock_adapter import MockAdapter, MockScenario
from atlas_rtc.contracts.json_schema import JSONSchemaContract
from atlas_rtc.controller.runtime import RuntimeController


def test_runtime_controller_produces_valid_json():
    scenario = MockScenario(
        prompt="",
        planned_tokens=["{", '"name"', ":", '"alice"', ",", '"age"', ":", "30", "}"],
        candidates=[
            {"Hello": 1.8, "{": 1.1},
            {'"name"': 2.0, '"junk"': 1.8},
            {":": 2.4},
            {'"alice"': 2.0},
            {",": 2.0},
            {'"age"': 2.1, '"city"': 2.0},
            {":": 2.5},
            {"30": 2.2},
            {"}": 2.4},
        ],
    )
    controller = RuntimeController(
        adapter=MockAdapter(scenario),
        contract=JSONSchemaContract(required_keys=["name", "age"]),
    )
    text, result, state = controller.run("Return JSON.")
    assert result.valid is True
    assert text.startswith("{")
    assert len(state.intervention_history) >= 1
