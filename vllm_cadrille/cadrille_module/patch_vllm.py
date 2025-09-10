def patch_vllm_proccessor(processor):
    from types import MethodType
    from transformers.models.qwen2_vl import Qwen2VLImageProcessor, Qwen2VLProcessor
    from vllm.model_executor.models.qwen2_vl import Qwen2VLProcessingInfo

    def _patched_get_hf_processor(self, *, min_pixels=None, max_pixels=None) -> Qwen2VLProcessor:
        return processor

    def _patched_get_image_processor(self, *, min_pixels=None, max_pixels=None):
        proc = _patched_get_hf_processor(self, min_pixels=min_pixels, max_pixels=max_pixels)
        image_processor = proc.image_processor
        assert isinstance(image_processor, Qwen2VLImageProcessor)
        return image_processor

    # Bind methods to the class (affects all instances)
    Qwen2VLProcessingInfo.get_hf_processor   = MethodType(_patched_get_hf_processor, Qwen2VLProcessingInfo)
    Qwen2VLProcessingInfo.get_image_processor= MethodType(_patched_get_image_processor, Qwen2VLProcessingInfo)
