from unittest.mock import patch

from infrastructure.models.laya_decision import slave_decision

def test_slave_decision_returns_expected_result():
    # Arrange
    fake_output = {
        "answers": {
            "decision": {
                "choice": "Approved",
                "probabilities": {
                    "Approved": 0.9,
                    "Rejected": 0.1,
                },
                "prediction_set": ["Approved"],
                "abstain": False,
                "unbound": False,
                "route": "auto",
                "route_reason": "test",
            }
        }
    }

    with patch("laya.Agent") as MockAgent:
        MockAgent.return_value.predict.return_value = fake_output

        # Act
        result = slave_decision(
            question="Should this be approved?",
            input_text="Request details",
            options=["Approved", "Rejected"],
        )

        # Assert
        assert result["selected_option"] == "Approved"
        assert result["probabilities"]["Approved"] == 0.9
        assert result["prediction_set"] == ["Approved"]
        assert result["abstain"] is False
        assert result["route"] == "auto"