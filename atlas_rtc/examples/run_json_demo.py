from atlas_rtc.adapters.mock_adapter import MockAdapter, MockScenario
from atlas_rtc.contracts.json_schema import JSONSchemaContract
from atlas_rtc.controller.runtime import RuntimeController


def build_scenario() -> MockScenario:
    return MockScenario(
        prompt="",
        planned_tokens=["{", '"name"', ":", '"alice"', ",", '"age"', ":", "30", "}"],
        candidates=[
            {"Hello": 2.0, "{": 1.1},
            {'"name"': 2.0, '"junk"': 1.8},
            {":": 2.4, "-": 0.3},
            {'"alice"': 2.0, '"bob"': 1.2},
            {",": 2.0, "}": 0.9},
            {'"age"': 2.1, '"city"': 1.9},
            {":": 2.5},
            {"30": 2.2, '"unknown"': 1.0},
            {"}": 2.4},
        ],
    )


def main() -> None:
    adapter = MockAdapter(build_scenario())
    contract = JSONSchemaContract(required_keys=["name", "age"])
    controller = RuntimeController(adapter=adapter, contract=contract)
    text, result, state = controller.run("Return JSON with name and age.")
    print("Output:", text)
    print("Valid:", result.valid)
    print("Errors:", result.errors)
    print("Interventions:", len(state.intervention_history))


if __name__ == "__main__":
    main()
