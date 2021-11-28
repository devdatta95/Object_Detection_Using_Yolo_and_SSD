from app import app
from config import *
from waitress import serve

import logging
logger = logging.getLogger("main")


logger.debug("[INFO] Running Flask Server at:  {}:{} ".format(SERVER_URL, SERVER_PORT))
serve(app, host=SERVER_URL, port=SERVER_PORT)

