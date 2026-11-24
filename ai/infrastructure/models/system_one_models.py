import torch
from anyjev import Decider, Question
from anyjev.backends.hf import HFBackend


def create_decider(model_id: str) -> Decider:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = "float16" if device == "mps" else "float32"

    backend = HFBackend(
        model_id,
        device=device,
        dtype=dtype,
        batch_size=1,
    )

    return Decider(backend)


#This function utilizes MedGemma wrapped with Nokia's AnyJev Library
def master_decision(
        decider: Decider,
        question: str,
        input_text: str,
        options: list[str],
):
    decision_question = Question.choice(
        question=question,
        options=options,
        name="decision",
    )

    result = decider.decide(
        input_text,
        [decision_question],
    )["decision"]

    return {
        "selected_option": result.argmax,
        "distribution": result.distribution,
        "level": result.level,
        "diagnostics": result.diagnostics,
    }


#This functon utilizes Laya Model
def slave_decision(
        question: str,
        input_text: str,
        options: list[str],
):
    import laya

    agent = laya.Agent("convaiinnovations/laya")

    questions = {
        "decision": {
            "type": "choice",
            "instructions": question,
            "criteria": {
                option: option for option in options
            },
        }
    }

    output = agent.predict(input_text, questions)
    result = output["answers"]["decision"]

    return {
        "selected_option": result["choice"],
        "probabilities": result["probabilities"],
        "prediction_set": result["prediction_set"],
        "abstain": result["abstain"],
        "unbound": result["unbound"],
        "route": result["route"],
        "route_reason": result["route_reason"],
    }
