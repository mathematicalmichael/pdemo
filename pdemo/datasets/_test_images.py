import PIL.Image as I
import pathlib

_ROOT = pathlib.Path(__file__).parent.absolute()
_CACHE = _ROOT / "test_images_cache"


class Imgs():
	def __init__(self):
		files = _CACHE.glob('**/*')
		for img in files:
			try:
				self.__dict__[img.stem] = I.open(img)
			except:
				continue

