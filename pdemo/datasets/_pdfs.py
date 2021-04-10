# -*- coding: utf-8 -*-
"""This module facilitates downloading data and wiping the download cache"""
###############################################################################
# Imports

import pathlib
import yaml
import logging
import dataclasses
import sys
import hashlib
import typing as T
import requests
import mimetypes
import re
import PyPDF2
import numpy as np
import pandas as pd
import copy
import PIL
import io
import itertools as it
from PIL import Image as I
import tensorflow as tf
import sklearn.preprocessing as pp
from collections import defaultdict

###############################################################################
# Get module context
C = sys.modules[__name__.split(".")[0]].context
M = sys.modules[__name__.split(".")[0]]

###############################################################################
# Setup Logging
L = logging.getLogger(__name__)

###############################################################################
# Setup Locations
_ROOT = pathlib.Path(__file__).parent.absolute()
_CACHE = _ROOT / ".cache"
_CACHE_PDFS = _CACHE / "application/pdf"
_CACHE_JPEGS = _CACHE / "image/jpeg"
# _CACHE_MODELS = _CACHE / "models"
_META_FILE = _ROOT / "_pdfs_metadata.yaml"

# create cache if needed
if not _CACHE.exists():
    _CACHE.mkdir(parents=False, exist_ok=True)

###########################################################
# Special Errors


class MetadataMissingKeyInConfig(LookupError):
    """Raised when a key is missing from the metadata file"""

    def __init__(self, key):
        message = "Key missing from yaml config. Key: " f"<{key}>, File: {__META_FILE}"
        super().__init__(message)


class DownloadUrlError(TypeError):
    """Raised for response status codes not equal to 200"""

    def __init__(self, url):
        message = f"Encountered error downloading: <{url}>"
        super().__init__(message)


###########################################################
# Dataclasses
@dataclasses.dataclass
class TextBoxTopology:
    """Simple struct for working with scales and locations from pdfs"""

    scale_x: np.float32
    scale_y: np.float32
    location_x: np.float32
    location_y: np.float32
    scale_z: np.float32 = 1.0
    location_z: np.float32 = 0.0

    @property
    def location(self):
        return np.array([self.location_x, self.location_y, self.location_z]).astype(
            np.float32
        )

    @property
    def scale(self):
        return np.array([self.scale_x, self.scale_y, self.scale_z]).astype(np.float32)


@dataclasses.dataclass
class DownloadProtocol:
    """Nice and simple download protocol."""

    __cache_location: str = _CACHE
    __metadata_location: str = _META_FILE
    _metadata: dict = None
    # metadata yaml keys
    _source_key: str = "sources"
    _urls_key: str = "urls"
    _tag_key: str = "tag"
    _mime_refernece: mimetypes.MimeTypes = dataclasses.field(
        default_factory=lambda: mimetypes.MimeTypes(), repr=False
    )
    # _tag_lookup: defaultdict = dataclasses.field(
    #     default_factory=lambda: defautdict(list), repr=False
    # )
    template_filename: str = "{tag}-{hexdigest}{extension}"

    @property
    def metadata(self):
        if self._metadata is None:
            self.metadata_reload()
        return self._metadata

    @metadata.setter
    def metadata(self, dictionary: dict):
        assert isinstance(dictionary, dict)
        assert self._source_key in dictionary
        assert self._urls_key in dictionary[self._source_key]
        self._metadata = dictionary

    @metadata.deleter
    def metadata(self):
        self._metadata = None

    def metadata_reload(self, check_keys: bool = True) -> None:
        """This sets the self.metadata attribute from some yaml source.

        Functionally this method opens a file, calls yaml.safe_load on
        the __metadata_location which is the file path, then checks for
        some keys to exist (optional).

        Args:
            check_keys (bool): checks for two keys to exist.
                `sources`
                `sources` -> `urls`
        Raises:
            MetadataMissingKeyInConfig if check_keys is True and the keys
                are not found
        """
        _source_key = self._source_key
        _urls_key = self._urls_key
        L.info(f"Loading metadata config from <{self.__metadata_location}>")
        with open(self.__metadata_location, "r") as fin:
            self._metadata = yaml.safe_load(fin)
        if check_keys:
            L.info(f"Checking metadata for `{_source_key}` key...")
            sources = self._metadata.get(_source_key)
            if sources is None:
                raise MetadataMissingKeyInConfig(_source_key)
            L.info(f"Checking metadata for `{_urls_key}` key...")
            urls = sources.get(_urls_key)
            if urls is None:
                raise MetadataMissingKeyInConfig(_urls_key)

    def _remove_cache(self, path=None) -> None:
        """Clears out anything stored in the self.__cache_location"""
        if path is None:
            pth = pathlib.Path(self.__cache_location)
        else:
            pth = path
        for child in pth.glob("*"):
            if child.is_file():
                child.unlink()
            else:
                self._remove_cache(child)
        pth.rmdir()

    def wipe_cache(self) -> None:
        """This function removes everything in the cache and recreates
        the basic directory structure."""
        L.warning(f"Wiping/removing anything in <{self.__cache_location}>")
        self._remove_cache()
        self.__cache_location.mkdir(parents=True, exist_ok=True)

    def download(self, wipe_cache: bool = True) -> None:
        """This function downloads the urls specified in the metadata file

        Args:
            wipe_cache (bool): default True; if true, the containerized cache
                for this module is wiped (everything in ./.cache) and a clean
                download is initiated.
        """
        if wipe_cache:
            self.wipe_cache()
        sources = self.metadata.get(self._source_key)
        if sources is None:
            raise MetadataMissingKeyInConfig(self._source_key)
        urls = sources.get(self._urls_key)
        try:
            with requests.Session() as session:
                for url, value in urls.items():
                    if isinstance(value, dict):
                        tag = value.get(self._tag_key)
                    else:
                        tag = None
                    md5 = hashlib.md5()
                    md5.update(url.encode())
                    hexhash = md5.hexdigest()
                    L.info(f"{'Buffering': <12}: {url}")
                    response = session.get(url)
                    if response.status_code != 200:
                        L.error(
                            f"FAILED DOWNLOAD - http response status {response.status_code} - {url}"
                        )
                        raise DownloadUrlError(url)
                    content_type = response.headers["Content-Type"]
                    # note the assertion type checks for the second spot in the
                    # types map inv tuple. This spot delinates common mime types
                    assert (
                        content_type in self._mime_refernece.types_map_inv[1]
                    ), f"'{content_type}' not in standard list of mime type."
                    filename = self.template_filename.format(
                        tag=tag,
                        hexdigest=hexhash,
                        extension=self._mime_refernece.types_map_inv[1][content_type][
                            0
                        ],  # use first one found
                    )
                    dump_path = self.__cache_location / content_type / filename
                    dump_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(dump_path, "wb") as fout:
                        fout.write(response.content)
                    L.info(f"Buffer Saved: '{dump_path}'")
                    L.debug(f"Buffer Saved: ('{url}', '{dump_path}')")
        except Exception as e:
            L.debug("Encountered an error during the download procedure: {e}")
            self.wipe_cache()
            raise

    @staticmethod
    def get_pdf_paths(sanitize: bool = True) -> T.List[pathlib.Path]:
        """Return a list of whatever pdfs are present in the pdf section
        of the .cache

        Args:
            sanitize (bool): default True; if True, then all of the files are
                checked to have the correct mimetype of 'application/pdf'. If
                a file is found and it is not the correct type, an alert is
                posted and the file is removed from the list. An error of this
                nature will not block execution (No error is raised, only
                passively posted to the logger)
        Returns:
            typing.List[pathlib.Path]
        """
        L.info(f"Searching cache at: {_CACHE_PDFS}")
        proposal = list(_CACHE_PDFS.glob("**/*.pdf"))
        if not proposal:
            L.warning("Cache is empty... did you call download yet?")
            return proposal
        if sanitize:
            L.info(f"Running sanitization subroutine on {len(proposal):03} files")
            for iii, somefile in enumerate(proposal):
                ftype = mimetypes.guess_type(somefile)[0]
                if ftype != "application/pdf":
                    L.warning("Incorrect filetype discovered - running diagnostics")
                    L.error(f"Bad type for file: <{somefile}>; Type: {ftype}")
                    proposal[iii] = None
                    L.warning("Incorrect file removed from proposal list.")
        proposal = [xxx for xxx in proposal if xxx is not None]
        return proposal

    def debug_load_file(
        self, random: bool = True, index: int = None
    ) -> PyPDF2.pdf.PdfFileReader:
        """Wrapper for quickly loading a pdf for debugging purposes.

        Args:
            random (bool): default True; call numpy random choice on the pdfs
            index (int): default None; if supplied, index overrides the random
                behavior and selects the position from the sorted path list.

        Returns:
            PyPDF2.pdf.PdfFileReader
        """
        paths = self.get_pdf_paths()
        paths.sort()
        if random and index is None:
            file_path = np.random.choice(paths)
        else:
            if index is None:
                L.info(
                    f"No index specified out of {len(paths)}, loading the first one..."
                )
                index = 0
            file_path = paths[index]
        L.info(f"Loading: {file_path}")
        fin = open(file_path, "rb")
        pdf = PyPDF2.PdfFileReader(fin)
        pdf.file_path = file_path
        return pdf

    def get_tag_lookup(self):
        tmp = defaultdict(list)
        sources = self.metadata.get("sources")
        if sources is not None:
            urls = sources.get("urls")
            if urls is not None:
                for url, data in urls.items():
                    tag = data.get("tag")
                    if tag is not None:
                        tmp[tag].append(url)
        return dict(tmp)


@dataclasses.dataclass
class PdfProcessingToolbox:
    re_line_tokenizer: re.Pattern = dataclasses.field(
        default_factory=lambda: re.compile(b"\n+|\r+|\n+\r+|\r+\n+")
    )
    re_text_object: re.Pattern = dataclasses.field(
        default_factory=lambda: re.compile(
            b"(?<=[\n+|\r+|\n+\r+|\r+\n+])BT\s*.*?[\n+|\r+|\n+\r+|\r+\n+]ET(?=[\n+|\r+|\n+\r+|\r+\n+])",
            re.MULTILINE + re.DOTALL,
        )
    )
    re_text_transformation_matrix: re.Pattern = dataclasses.field(
        default_factory=lambda: re.compile(
            b"((?:[\d|\.|\+|\-]+\s+){6})Tm(?=[\n+|\r+|\n+\r+|\r+\n+])",
            re.MULTILINE + re.DOTALL,
        )
    )
    re_hexstring: re.Pattern = dataclasses.field(
        default_factory=lambda: re.compile(
            b"(?<!\<)<[^\<].*?>(?!\>)", re.MULTILINE + re.DOTALL
        )
    )

    def extract_topo_from_matrix(
        self,
        transformation_matrix_binary_data: bytes,
        normalize: bool = False,
        preserve_aspect_ratio: bool = True,
        cropbox: PyPDF2.generic.RectangleObject = None,
    ) -> TextBoxTopology:
        """This function mainly serves to extract out coordinates and scale
        of the text matrix.

        Normalization of components to [0,1.0] is optional. This essentially
        takes the userspace coordinates from the cropbox and uses the upper
        right vertex as maximal values. Note: the coordinate system in a pdf
        has origin at the bottom left of a page and the media, art, crop boxes,
        etc are bounding boxes measured from the bottom left.

        Args:
            transformation_matrix_binary_data (bytes): raw data bytes from
                the `Tm` message in the PDF.
                    Message:
                        b`9 0 0 9 36 36.8445 Tm`
                    Data that you should pass:
                        b`9 0 0 9 36 36.8445`
                        E.g. 6 int or float-looking things.
            normalize (bool):
                transform x and y coodinates to normalized space [0,1]
                Transformation is performed based on the cropbox object's
                upper right coords.
            preserve_aspect_ratio (bool):
                defaults to True, if normalize is true then this option comes
                into play by taking whatever the maximum is between the
                two axis are and using that to divide the result.
            cropbox (PyPDF2.generic.RectangleObject):
                for a `PyPDF2.pdf.PageObject` called a `page`, the
                `page['/CropBox']` element. MediaBox could also probably be
                used.
        Returns:
            TextBoxTopology:
                w/ the following set
                    scale_x (float)
                    scale_y (float)
                    location_x (float)
                    location_y (float)
        """
        mat = np.array(transformation_matrix_binary_data.decode().split()).astype(
            np.float32
        )
        mat = np.concatenate(
            [mat.reshape(3, 2), np.array([[0.0], [0.0], [1.0]])], axis=1
        )
        if normalize:
            assert cropbox is not None
            max_x = float(cropbox.getUpperRight_x())
            assert max_x > 1.0
            max_y = float(cropbox.getUpperRight_y())
            assert max_y > 1.0
            if preserve_aspect_ratio:
                max_x = max_y = max(max_x, max_y)
        else:
            max_x = 1.0
            max_y = 1.0
        result = TextBoxTopology(
            scale_x=mat[0, 0],
            scale_y=mat[1, 1],
            location_x=mat[2, 0] / max_x,
            location_y=mat[2, 1] / max_y,
        )
        return result

    def extract_images_from_page(
        self,
        page: PyPDF2.pdf.PageObject,
        minimum_pixel_count: int = 2 ** 11,
        minimum_entropy: float = 6.0,
        soft_fail: bool = True,
        include_img_obj_in_tracking: bool = True,
        tracking_info: dict = None,
    ) -> PIL.Image.Image:
        # TODO refactor this with less branching, if possible
        assert isinstance(page, PyPDF2.pdf.PageObject), "Not a PyPDF2 Page Object"
        if tracking_info is None:
            tracking_info = dict()
        assert isinstance(tracking_info, dict)
        resources = page.get("/Resources")
        L.debug(f"Processing PDF Page; {tracking_info}")
        if resources is not None:
            xobjs = resources.get("/XObject")
            if xobjs is not None:
                counter = -1
                for namekey, obj in xobjs.items():
                    obj = obj.getObject()
                    if obj.get("/Subtype") == "/Image":
                        counter += 1
                        img_capture_info = copy.deepcopy(tracking_info)
                        if img_capture_info.get("note") is None:
                            img_capture_info["note"] = ""
                        if img_capture_info.get("errors") is None:
                            img_capture_info["errors"] = False
                        img_capture_info["image_ordinal_position"] = counter
                        if include_img_obj_in_tracking:
                            img_capture_info["_pdf_img_obj"] = obj.getObject()
                        img_capture_info["height"] = obj.get("/Height")
                        img_capture_info["width"] = obj.get("/Width")
                        img_capture_info["img_filter"] = obj.get("/Filter")
                        img_capture_info["colorspace"] = obj.get("/ColorSpace")
                        if hasattr(img_capture_info["colorspace"], "getObject"):
                            cs_dat = img_capture_info["colorspace"].getObject()
                            if not isinstance(cs_dat, list):
                                cs_dat = [cs_dat]
                            for iii, thing in enumerate(cs_dat):
                                if hasattr(thing, "getObject"):
                                    cs_dat[iii] = thing.getObject()
                            img_capture_info["colorspace"] = tuple(cs_dat)
                        component_bits = obj.get("/BitsPerComponent")
                        L.debug(f"Image Encountered! {img_capture_info}")
                        try:
                            height = int(img_capture_info["height"])
                            width = int(img_capture_info["width"])
                        except:
                            height = 0
                            width = 0
                        pixel_count = height * width
                        try:
                            if pixel_count < minimum_pixel_count:
                                img_capture_info[
                                    "note"
                                ] += f"Minimum Pixel Count Not Achieved: {pixel_count} / {minimum_pixel_count}. "
                                assert pixel_count >= minimum_pixel_count
                            hasher = hashlib.md5()
                            buffer = io.BytesIO()
                            try:
                                binary_dat = obj.getData()
                                img_capture_info[
                                    "note"
                                ] += "Sourced from object.getData. "
                            except:
                                binary_dat = obj._data
                                img_capture_info[
                                    "note"
                                ] += "Sourced from object._data. "
                            buffer.write(binary_dat)
                            hasher.update(binary_dat)
                            img_capture_info["md5sum"] = hasher.hexdigest()
                            buffer.seek(0)
                            try:
                                img = I.open(buffer)
                            except:
                                img = None
                            # try to decode the bitstream manually
                            if img is None:
                                decode = obj.get("/Decode")
                                if decode is None:
                                    img_capture_info[
                                        "note"
                                    ] += "Unknown how to set domain. /Decode seems to be missing. "
                                    domain = None
                                    unsigned = True
                                else:
                                    img_capture_info["domain"] = decode
                                    domain = np.array([float(xxx) for xxx in decode])
                                    # row 0 mins
                                    # row 1 maxes
                                    domain = domain.reshape(2, -1)
                                    if domain.min() >= 0:
                                        unsigned = True
                                    else:
                                        unsigned = False
                                if component_bits == 1:
                                    raise NotImplementedError(
                                        "numpy is used for processing, 1 byte (8 bit) atomic"
                                    )
                                elif component_bits == 2:
                                    raise NotImplementedError(
                                        "numpy is used for processing, 1 byte (8 bit) atomic"
                                    )
                                elif component_bits == 3:
                                    raise NotImplementedError(
                                        "numpy is used for processing, 1 byte (8 bit) atomic"
                                    )
                                elif component_bits == 4:
                                    raise NotImplementedError(
                                        "numpy is used for processing, 1 byte (8 bit) atomic"
                                    )
                                elif component_bits == 8:
                                    if unsigned:
                                        dtype_guess = np.uint8
                                    else:
                                        dtype_guess = np.int8
                                elif component_bits == 12:
                                    raise NotImplementedError(
                                        "numpy is used for processing, 1 byte (8 bit) atomic"
                                    )
                                elif component_bits == 16:
                                    if unsigned:
                                        dtype_guess = np.uint16
                                    else:
                                        dtype_guess = np.int16
                                else:
                                    raise NotImplementedError(
                                        f"BitsPerComponent: {component_bits} outside of PDF spec"
                                    )
                                data = (
                                    np.frombuffer(binary_dat, dtype=dtype_guess)
                                    .reshape(height, width, -1)
                                    .squeeze()
                                )
                                # TODO spot check some of these, a few images appeared to look as if they were a negative or were from a weird color space
                                # translate to zero
                                img_capture_info[
                                    "note"
                                ] += "checkme - processed by parsing raw binary array. "
                                if domain is not None:
                                    data = data - domain[0]
                                    # scale down to 1
                                    data = data / domain[1]
                                try:
                                    img = I.fromarray(data)
                                except Exception as e:
                                    img = None
                                    raise
                            try:
                                img = img.convert("RGB")
                            except Exception as e:
                                img = None
                                img_capture_info["note"] += "RGB conversion error. "
                                raise
                            img_capture_info["image"] = img
                            if img.entropy() < minimum_entropy:
                                img_capture_info[
                                    "note"
                                ] += "checkme - low entropy detected. "
                                assert (
                                    img.entropy() >= minimum_entropy
                                ), "Low image entropy"
                                img_capture_info["low_entropy"] = True
                            yield img_capture_info
                        except Exception as e:
                            if soft_fail is True:
                                page = tracking_info.get("page")
                                pdf_label = tracking_info.get("pdf_label")
                                note = tracking_info.get("note")
                                # L.warning(
                                #     f"Soft Fail Engaged, <{e}> Tracking img_capture_info: {pdf_label}.{page} '{note}'"
                                # )
                                L.debug(
                                    f"Soft Fail Engaged, <{e}> Tracking img_capture_info: {img_capture_info}"
                                )
                                img_capture_info["errors"] = True
                                img_capture_info["note"] += f"{e}. "
                                yield img_capture_info
                                continue
                            else:
                                L.error(
                                    f"Hard Fail Engaged, error processing an image from a page: {e}"
                                )
                                L.error(
                                    f"Hard Fail Engaged, Tracking img_capture_info: {img_capture_info}"
                                )
                                L.debug(
                                    f"Hard Fail Engaged, error processing an image from a page: {e}"
                                )
                                L.debug(
                                    f"Hard Fail Engaged, Tracking img_capture_info: {img_capture_info}"
                                )
                                raise

    def extract_images_from_pdfs(
        self,
        pdf_paths: T.Iterable = None,
        dump: bool = False,
        file_not_tagged_default: str = "unknown_file_not_tagged_default",
        normalize_pdf_tag: bool = True,
        # ) -> T.Iterable[dict]:
    ) -> pd.DataFrame:
        """"""
        imgs = []
        if pdf_paths is None:
            pdf_paths = DownloadProtocol.get_pdf_paths()
        for path in pdf_paths:
            pdf = PyPDF2.PdfFileReader(open(path, "rb"))
            # Autotag pdf_label
            if "-" in path.name:
                label_for_model = path.name.split("-")[0]
                if label_for_model == "":
                    L.warning(
                        f"Autotagger Degraded - Type I: defaulting to {file_not_tagged_default}"
                    )
                    label_for_model = file_not_tagged_default
            else:
                L.warning(
                    f"Autotagger Degraded - Type II: defaulting to {file_not_tagged_default}"
                )
                label_for_model = file_not_tagged_default
            if normalize_pdf_tag:
                label_for_model = label_for_model.lower().strip()
                label_for_model = re.sub("\s+", "_", label_for_model)
            for iii, page in enumerate(pdf.pages):
                # L.info(f"Processing: {label_for_model}.page_{iii:04}")
                imgs.append(
                    self.extract_images_from_page(
                        page,
                        tracking_info={
                            "page": iii,
                            "filepath": str(path),
                            "pdf_label": label_for_model,
                        },
                    )
                )
        findings = list(it.chain.from_iterable(imgs))
        if dump is True:
            L.info(f"Saving images found to `{_CACHE_JPEGS}`")
            if not _CACHE_JPEGS.exists():
                _CACHE_JPEGS.mkdir(parents=True, exist_ok=True)
            for finding in findings:
                im = finding.get("image")
                label_for_model = finding.get("pdf_label")
                if im is None:
                    continue
                if finding.get("error"):
                    continue
                im = im.convert("RGB")
                buff = io.BytesIO()
                im.save(buff, format="jpeg")
                buff.seek(0)
                md5 = hashlib.md5()
                md5.update(buff.read())
                save_loc = _CACHE_JPEGS / label_for_model
                if not save_loc.exists():
                    save_loc.mkdir(parents=True, exist_ok=True)
                fname = save_loc / f"{md5.hexdigest()}.jpg"
                L.info(f"Saving: {fname}")
                im.save(fname)
                L.debug(f"Saving: {fname}")
        return pd.DataFrame(findings)


# def setup_model(
# mod.compile(optimizer='adam',loss='binary_crossentropy')
# target
# target.value_counts()
# targ
# targ.sum(axis=0)
# mlb.classes
# mlb.classes_
# 1/targ.sum(axis=0)
# weights = 1/targ.sum(axis=0)
# np.round(weights,2)
# import numpy as np
# np.round(weights,2)
# np.round(weights,.5)
# np.round(weights,5)
# np.round(weights,4)
# weights
# mod.compile(optimizer='adam',loss='binary_crossentropy',loss_weights=weights)
# tdata.shape
# mod.train?
# mod.fit?
# mod.fit(x=tdata, y=targ, batch_size=64, shuffle=True, epochs=64, validation_split=.2)
#
# @dataclasses.dataclass
# class FontDetails:
#     font_name: str = None
#     font_family: str = None
#     font_size: float = None
#
#     def __iter__(self) -> tuple:
#         return (self.font_family, self.font_name)
#

#
#
# @dataclasses.dataclass
# class RGBColor:
#     red: float
#     green: float
#     blue: float
#
#     def __iter__(self) -> T.Generator:
#         yield self.red
#         yield self.green
#         yield self.blue
#
#
# @dataclasses.dataclass
# class PageParser:
#     pagenum: int
#     fonts: dict = None
#     binary_data: bytes = None
#     images: dict = None
#     texts: list = None
#
#
# # A selection of stateful components which will
# # be useful to know for processing text and image
# # objects out of the bitstream.
#
# trackers = {"Tc", "Tw", "Td", "scn", "Tf", "Tm"}
