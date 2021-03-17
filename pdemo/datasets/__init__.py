from . import _pdfs
from . import _test_images
from . import _pdf_images

pdfs = _pdfs.DownloadProtocol()
preprocessing_toolbox = _pdfs.PdfProcessingToolbox()
test_images = _test_images.Imgs()
streamlit_assets = _pdf_images.Imgs()
