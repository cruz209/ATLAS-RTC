from atlas_rtc.contracts.json_schema import JSONSchemaContract


def test_json_contract_validation_success():
    contract = JSONSchemaContract(required_keys=["name", "age"])
    result = contract.validate('{"name": "alice", "age": 30}')
    assert result.valid is True


def test_json_contract_validation_missing_key():
    contract = JSONSchemaContract(required_keys=["name", "age"])
    result = contract.validate('{"name": "alice"}')
    assert result.valid is False
    assert "missing keys" in result.errors[0]
