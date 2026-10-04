"""A shared MiniLM model with bounded inference memory for small instances."""

from functools import cached_property

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2


class SharedMiniLMEmbedding(ONNXMiniLM_L6_V2):
    def __init__(self):
        super().__init__(preferred_providers=["CPUExecutionProvider"])

    @cached_property
    def model(self):
        options = self.ort.SessionOptions()
        options.log_severity_level = 3
        options.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        path = self.DOWNLOAD_PATH / self.EXTRACTED_FOLDER_NAME / "model.onnx"
        return self.ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"], sess_options=options,
        )
