import uuid
import logging

L = logging.getLogger(__name__)

run_id = uuid.uuid4()
L.info(f"Run Identifier: '{run_id}'")
L.debug(f"{__name__}._run_id: '{run_id}'")

############################################################
# Load Config runtime settings


############################################################
