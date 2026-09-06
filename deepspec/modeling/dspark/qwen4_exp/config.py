import copy

from deepspec.modeling.dspark.common import validate_target_layer_ids


# Qwen4Exp (Qwen3.8-Flash-Next) is a Hyper-Connection (hc_count=4) model:
# every decoder layer's stream is a 4-way multiplexed hidden state, and the
# native-width (hidden_size) residual only exists momentarily inside each
# layer's *_hyper_connection.forward() contraction. The draft model itself
# is a plain (non-HC) Qwen3-style transformer, so it is built from the
# target's *text_config* the same way Gemma4 DSpark builds from
# target_config.text_config -- see gemma4/config.py for the precedent this
# file mirrors. The draft model reuses Qwen3DSparkModel unchanged (see
# qwen4_exp/__init__.py); only this config-construction step is
# architecture-specific.
TRAIN_ATTN_IMPLEMENTATION = "flex_attention"


def get_qwen4_exp_text_config(target_config):
    assert target_config.model_type == "qwen4_exp", (
        "Qwen4Exp DSpark expects a Qwen4Exp top-level target config, "
        f"got model_type={target_config.model_type!r}."
    )
    text_config = target_config.text_config
    assert text_config.model_type == "qwen4_exp_text", (
        "Qwen4Exp DSpark expects target_config.text_config.model_type to be "
        f"'qwen4_exp_text', got {text_config.model_type!r}."
    )
    return copy.deepcopy(text_config)


def _validate_required_text_fields(text_config) -> None:
    required_fields = (
        "vocab_size",
        "hidden_size",
        "moe_intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_act",
        "initializer_range",
        "max_position_embeddings",
        "rms_norm_eps",
        "rope_parameters",
        "attention_bias",
        "attention_dropout",
    )
    for field in required_fields:
        assert hasattr(text_config, field), (
            f"target_config.text_config.{field} must be provided."
        )


# Qwen4ExpTextConfig has NO plain `intermediate_size`: the target is a
# sparse-MoE model (moe_intermediate_size=512/expert, shared_expert
# intermediate_size=512). The draft model (Qwen3DSparkModel -> Qwen3MLP) is a
# small, independently-initialized DENSE transformer, so its FFN width is a
# free hyperparameter, not something inherited from the target. We follow
# Qwen3-8B's own dense hidden:intermediate ratio (4096:12288 = 1:3) applied to
# the draft's hidden_size, rather than reusing any MoE-specific width, since
# there is no principled 1:1 mapping from expert width to a dense draft FFN.
DRAFT_INTERMEDIATE_SIZE_RATIO = 3


def build_draft_config(target_config, model_args):
    draft_config = get_qwen4_exp_text_config(target_config)
    _validate_required_text_fields(draft_config)

    num_target_layers = int(draft_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    layer_types = ["full_attention"] * num_draft_layers

    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when "
            "confidence_head_alpha > 0."
        )

    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, (
            "markov_head_type must be provided when markov_rank > 0."
        )

    # The draft model is a plain Qwen3-style transformer (Qwen3DSparkModel),
    # NOT a Qwen4Exp/HC model, so architectures/layer_types/attn_implementation
    # below describe the DRAFT model, not the target. num_target_layers /
    # target_layer_ids record how the draft ties back into the target's
    # (HC-contracted) hidden states -- see prepare_target_cache_qwen38_flash_next.py
    # for how those tapped tensors are produced.
    draft_config.architectures = ["Qwen3DSparkModel"]
    draft_config.target_model_type = str(target_config.model_type)
    draft_config.target_text_model_type = str(draft_config.model_type)
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.intermediate_size = int(draft_config.hidden_size) * DRAFT_INTERMEDIATE_SIZE_RATIO
    draft_config.block_size = int(model_args.block_size)
    draft_config.tie_word_embeddings = False
    draft_config.layer_types = layer_types
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
    return draft_config


__all__ = [
    "build_draft_config",
    "get_qwen4_exp_text_config",
]
