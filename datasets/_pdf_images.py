import PIL.Image as I
import pathlib
import dataclasses
import yaml

_ROOT = pathlib.Path(__file__).parent.absolute()
_CACHE = _ROOT / "pdf_imgs"
# _META_FILE = _ROOT / "_pdfs_metadata.yaml"


@dataclasses.dataclass
class ImgTriplet:
    tag: str
    image: I.Image
    pdf_location: str
    img_location: str


class Imgs:
    def __init__(self):
        files = _CACHE.glob("**/*.png")
        for img in files:
            try:
                self.__dict__[img.stem] = ImgTriplet(
                    tag=img.stem,
                    image=I.open(img),
                    img_location=img,
                    pdf_location=list(_CACHE.glob(str(img.stem) + "*.pdf"))[0],
                )
            except:
                raise
