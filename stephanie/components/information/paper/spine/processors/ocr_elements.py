from .base import BaseSpineProcessor, ProcessorResult


class OcrElementsProcessor(BaseSpineProcessor):
    name = "ocr_elements"

    async def run(self, *, arxiv_id, pdf_path, elements, context):
        # OCR now lives in PdfElementsTool
        return elements, ProcessorResult(name=self.name, ran=False)
