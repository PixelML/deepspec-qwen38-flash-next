"""Native vLLM general-plugin entry point, imported on every worker process."""


def register():
    from vllm import ModelRegistry

    # EAGLEConfig(method='dflash') retains the saved Qwen3DSparkModel's
    # identity with a DFlash prefix. Do not replace the generic DSpark model
    # or relabel the checkpoint as Qwen3ForCausalLM.
    ModelRegistry.register_model(
        "DFlashQwen3DSparkModel", "dflash_epoch7:DFlashQwen3DSparkModel"
    )
