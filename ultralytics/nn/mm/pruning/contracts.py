from __future__ import annotations
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Consumer contract registry and multi-output producer specifications for YOLOMM structured pruning.

This module defines the channel semantics and input-output contract rules for all
multi-input consumer modules and multi-output producer modules used in YOLOMM
models.  These contracts guide the pruning strategy by specifying:

- How many inputs a consumer expects (and whether they can be independently pruned).
- How the output channel count relates to the input channel counts.
- Which output slot(s) a downstream consumer is allowed to read from a multi-output
  producer (FCM, MultiHeadCrossAttention).
"""

from dataclasses import dataclass
from typing import Literal

from ultralytics.nn.mm.pruning.graph import PruneNode





ContractMode = Literal[
    "concat_like",
    "detect_head",
    "equal_width_left_output",
    "left_output",
    "declared_output",
]

OutputLayout = Literal[
    "single",
    "tuple_same_width",
]





class UnsupportedMultiInputConsumerError(RuntimeError):


    pass

class UnsupportedMultiOutputProducerError(RuntimeError):


    pass





@dataclass(frozen=True)
class ConsumerContract:














    module_name: str
    min_inputs: int
    mode: ContractMode
    allow_independent_inputs: bool

@dataclass(frozen=True)
class ProducerOutputSpec:











    module_name: str
    output_slots: int
    layout: OutputLayout




















CONSUMER_CONTRACTS: dict[str, ConsumerContract] = {

    "Concat": ConsumerContract(
        module_name="Concat",
        min_inputs=2,
        mode="concat_like",
        allow_independent_inputs=True,
    ),

    "Detect": ConsumerContract(
        module_name="Detect",
        min_inputs=1,
        mode="detect_head",
        allow_independent_inputs=True,
    ),


    "FeatureFusion": ConsumerContract(
        module_name="FeatureFusion",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "FCM": ConsumerContract(
        module_name="FCM",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "FCMFeatureFusion": ConsumerContract(
        module_name="FCMFeatureFusion",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "ConvMixFusion": ConsumerContract(
        module_name="ConvMixFusion",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "ChannelGate": ConsumerContract(
        module_name="ChannelGate",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "CAM": ConsumerContract(
        module_name="CAM",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "SEFN": ConsumerContract(
        module_name="SEFN",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "FusionConvMSAA": ConsumerContract(
        module_name="FusionConvMSAA",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "MSC": ConsumerContract(
        module_name="MSC",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "SpatialDependencyPerception": ConsumerContract(
        module_name="SpatialDependencyPerception",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "FDFEF": ConsumerContract(
        module_name="FDFEF",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "DEA": ConsumerContract(
        module_name="DEA",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "MJRNet": ConsumerContract(
        module_name="MJRNet",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "MSIA": ConsumerContract(
        module_name="MSIA",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "RFF": ConsumerContract(
        module_name="RFF",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),

    "MCFGatedFusion": ConsumerContract(
        module_name="MCFGatedFusion",
        min_inputs=2,
        mode="declared_output",
        allow_independent_inputs=True,
    ),
    "CrossTransformerFusion": ConsumerContract(
        module_name="CrossTransformerFusion",
        min_inputs=2,
        mode="declared_output",
        allow_independent_inputs=False,
    ),
    "MultiHeadCrossAttention": ConsumerContract(
        module_name="MultiHeadCrossAttention",
        min_inputs=2,
        mode="declared_output",
        allow_independent_inputs=False,
    ),
}














PRODUCER_OUTPUT_SPECS: dict[str, ProducerOutputSpec] = {

    "FCM": ProducerOutputSpec(
        module_name="FCM",
        output_slots=2,
        layout="tuple_same_width",
    ),


    "MultiHeadCrossAttention": ProducerOutputSpec(
        module_name="MultiHeadCrossAttention",
        output_slots=2,
        layout="tuple_same_width",
    ),
}





def require_contract(node: PruneNode) -> ConsumerContract:












    contract = CONSUMER_CONTRACTS.get(node.type_name)
    if contract is None:
        raise UnsupportedMultiInputConsumerError(
            f"No consumer contract registered for multi-input module '{node.type_name}' "
            f"(layer {node.idx}).  Add a CONSUMER_CONTRACTS entry or mark the module "
            "as a single-input layer."
        )
    return contract

def require_output_spec(node: PruneNode) -> ProducerOutputSpec:












    spec = PRODUCER_OUTPUT_SPECS.get(node.type_name)
    if spec is None:
        raise UnsupportedMultiOutputProducerError(
            f"No output spec registered for multi-output producer '{node.type_name}' "
            f"(layer {node.idx}).  Add a PRODUCER_OUTPUT_SPECS entry."
        )
    return spec

