from infrastructure.models.laya_decision import slave_decision


def test_laya_real_inference():
    # Arrange
    question = "Which category best describes this message?"

    input_text = (
        "My invoice was charged twice. "
        "Please refund the duplicate payment."
    )

    options = ["Billing", "Technical", "Other"]

    # Act
    result = slave_decision(
        question=question,
        input_text=input_text,
        options=options,
    )

    # Assert
    assert result is not None

    assert result["selected_option"] in options

    assert isinstance(result["probabilities"], dict)
    assert isinstance(result["prediction_set"], list)

    assert isinstance(result["abstain"], bool)
    assert "route" in result
    assert "route_reason" in result