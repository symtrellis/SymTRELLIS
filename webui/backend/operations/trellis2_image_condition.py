from typing import Any

import torch
from PIL import Image

from inference.trellis2 import preprocess_image

from ..loaders.trellis2 import DEVICE, TRELLIS2Loader
from . import Emit, Operation, OperationContext, OperationInputs, OperationOutput, OperationResult

IMAGE_PNG = "image_png"
IMAGE_CONDITION_512 = "image_condition_512"
IMAGE_CONDITION_1024 = "image_condition_1024"


class Trellis2ImageCondition(Operation):
    operation_id = "trellis2.image_condition"
    execution_kind = "node_run"
    queue_kind = "gpu"
    creates_session = True
    output_roles = (IMAGE_PNG, IMAGE_CONDITION_512, IMAGE_CONDITION_1024)

    def __init__(self, loader: TRELLIS2Loader):
        self.loader = loader

    def resolve_inputs(self, coordinator: Any, request: Any) -> OperationInputs:
        if len(request.input_upload_keys) != 1:
            raise ValueError("trellis2.image_condition expects exactly one uploaded image")

        upload_key = request.input_upload_keys[0]
        upload_record = coordinator.storage.read_upload(upload_key)
        if upload_record is None:
            raise ValueError(f"Upload record not found: {upload_key}")

        return OperationInputs(
            records={"upload": upload_record},
            uploads={"image": coordinator.storage.upload_path(upload_key)},
        )

    def key_parts(self, inputs: OperationInputs, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "upload_content_hash": inputs.records["upload"]["content_hash"],
        }

    async def run(
        self,
        inputs: OperationInputs,
        params: dict[str, Any],
        context: OperationContext,
        emit: Emit,
    ) -> OperationResult:
        upload_record = inputs.records["upload"]

        with Image.open(inputs.uploads["image"]) as image_file:
            source_image = image_file.copy()

        source_png = source_image.convert("RGBA" if "A" in source_image.getbands() else "RGB")

        rembg_model = self.loader.rembg_model
        rembg_model.to(DEVICE)
        processed_image_512 = preprocess_image(source_image.copy(), rembg_model=rembg_model, target_size=512)
        processed_image_1024 = preprocess_image(source_image.copy(), rembg_model=rembg_model, target_size=1024)
        rembg_model.cpu()

        image_path = context.work_dir / "image.png"
        cond_512_path = context.work_dir / "image_condition_512.pt"
        cond_1024_path = context.work_dir / "image_condition_1024.pt"

        source_png.save(image_path)

        image_cond_model = self.loader.image_cond_model
        image_cond_model.to(DEVICE)

        image_cond_model.image_size = 512
        cond_512 = image_cond_model([processed_image_512]).detach().cpu()
        torch.save(cond_512, cond_512_path)

        image_cond_model.image_size = 1024
        cond_1024 = image_cond_model([processed_image_1024]).detach().cpu()
        torch.save(cond_1024, cond_1024_path)

        image_cond_model.cpu()

        return OperationResult(
            outputs=[
                OperationOutput(
                    role=IMAGE_PNG,
                    path=image_path,
                    filename="image.png",
                    metadata={
                        "height": source_png.height,
                        "mode": source_png.mode,
                        "width": source_png.width,
                    },
                ),
                OperationOutput(
                    role=IMAGE_CONDITION_512,
                    path=cond_512_path,
                    filename="image_condition_512.pt",
                    metadata={
                        "dtype": str(cond_512.dtype).replace("torch.", ""),
                        "resolution": 512,
                        "shape": list(cond_512.shape),
                        "sourceImageHeight": processed_image_512.height,
                        "sourceImageWidth": processed_image_512.width,
                    },
                ),
                OperationOutput(
                    role=IMAGE_CONDITION_1024,
                    path=cond_1024_path,
                    filename="image_condition_1024.pt",
                    metadata={
                        "dtype": str(cond_1024.dtype).replace("torch.", ""),
                        "resolution": 1024,
                        "shape": list(cond_1024.shape),
                        "sourceImageHeight": processed_image_1024.height,
                        "sourceImageWidth": processed_image_1024.width,
                    },
                ),
            ],
            metadata={
                "conditionResolutions": [512, 1024],
                "inputImageHeight": source_png.height,
                "inputImageWidth": source_png.width,
                "sourceContentHash": upload_record["content_hash"],
                "sourceFilename": upload_record["filename"],
                "sourceMimeType": upload_record["mime_type"],
            },
        )
