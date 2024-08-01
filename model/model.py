import multiprocessing
import subprocess
from typing import Dict

import httpx
import requests
import logging
import time
import os

TORCHSERVE_ENDPOINT = "http://0.0.0.0:8888/predictions/hallucinator"
TORCHSERVE_HEALTH_ENDPOINT = "http://0.0.0.0:8888/ping"

class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._model = None
        self.torchserve_ready = False

    def start_torchserve(self):
        env = os.environ.copy()
        env['BASETEN_DATA_DIR'] = self._data_dir
        # env['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
        subprocess.run(
            [
                "torchserve",
                "--start",
                "--model-store",
                f"{self._data_dir}/model-store",
                "--models",
                "hallucinator.mar",
                "--no-config-snapshots",
                "--ts-config",
                f"{self._data_dir}/config.properties"
            ],
            check=True,
            env=env
        )

    def load(self):
        process = multiprocessing.Process(target=self.start_torchserve)
        process.start()

        # Need to wait for the torchserve server to start up
        while not self.torchserve_ready:
            try:
                res = requests.get(TORCHSERVE_HEALTH_ENDPOINT)
                if res.status_code == 200:
                    self.torchserve_ready = True
                    logging.info("🔥Torchserve is ready!")
            except Exception as e:
                logging.info("⏳Torchserve is loading...")
                time.sleep(5)

    async def predict(self, request: Dict):
        async with httpx.AsyncClient() as client:
            res = await client.post(
                TORCHSERVE_ENDPOINT, json=request, timeout=600000
            )
        return res.json()
