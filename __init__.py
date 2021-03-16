# -*- coding: utf-8 -*-
#!/usr/local/bin/python
###############################################################################
# Imports
import logging
import logging.config
import pathlib
import yaml
import uuid

###############################################################################
# Location of the configuration yaml file
_ROOT_PATH = pathlib.Path(__file__).parent
_CONFIG_PATH = (pathlib.Path(__file__).parent / ".config/config.yml").absolute()

# For dumping logs or anything emphermal
_LOGS_PATH = _ROOT_PATH / "logs"
_LOGS_PATH.mkdir(parents=True, exist_ok=True)

###############################################################################
# set the run id, then pass to the format string for the logs
run_id = uuid.uuid4()

###############################################################################
# Load Config
with open(_CONFIG_PATH, "r") as fin:
    _CONFIG = yaml.safe_load(fin)

# Adapt any file locations of the handlers to cross platfomr locs
logging_conf = _CONFIG.get("logging")
if logging_conf:
    handlers = logging_conf.get("handlers")
    if handlers is not None:
        for handler, data in handlers.items():
            if "filename" in data:
                handlers[handler]["filename"] = str(_LOGS_PATH / data["filename"])
        logging_conf["handlers"] = handlers
    formatters = logging_conf.get("formatters")
    if "verbose" in formatters:
        if "format" in formatters["verbose"]:
            stamped_with_runid = formatters["verbose"]["format"].format(run_id=run_id)
            logging_conf["formatters"]["verbose"]["format"] = stamped_with_runid
    _CONFIG["logging"] = logging_conf

del data
del handlers
del handler
del stamped_with_runid
del formatters
###############################################################################
# Setup Logging from configs
logging.config.dictConfig(logging_conf)

###############################################################################
# Module imports
from . import context
from . import datasets
from . import preferences
from . import neural
from . import preprocessing

###############################################################################
# Cleanup
del logging_conf
del fin
del logging
del pathlib
del yaml
del uuid
