from ts.torch_handler.base_handler import BaseHandler
from utils.payload import Payload
from utils.timer import RuntimeSummarizer
from fsspec.asyn import AsyncFileSystem
from gcsfs import GCSFileSystem
from urllib.parse import urlparse
from google.oauth2.service_account import Credentials
from pathlib import Path
import logging
import numpy as np
import os
import torch
import portalocker
from typing import List, Tuple
from tenacity import retry, stop_after_attempt, wait_random
import traceback
from safetensors.torch import save as safe_save
import tensorflow.compat.v1 as tf
# from dotenv import find_dotenv, load_dotenv
from utils.tensors_io import read_safetensors_file
from utils.payload import Payload, Task
from utils.hparams import HParams

tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT: int = 120

class HallucinatorHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.init = False
        self.device = None
        self.manifest = None
        self.set_size = None
        self.threshold = None
        self.model = None
        # if dotenv := find_dotenv():
        #     load_dotenv(dotenv)

        self.data_dir = os.environ.get('BASETEN_DATA_DIR', str(Path(__file__).parent))
        logger.info("data_dir is %s", self.data_dir)
        logger.info("MODEL_STORE is %s", os.environ.get('MODEL_STORE', ''))
        self.rs = RuntimeSummarizer()

        self.credentials = Credentials.from_service_account_file(
            str(Path(self.data_dir) / 'service_account.json'),
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        self.project = "camb-ai-82e9a"
        self.fs = self.get_fs()

    def get_fs(self) -> AsyncFileSystem:
        return GCSFileSystem(project=self.project, access="read_write",
                            token=self.credentials, skip_instance_cache=True,
                            requests_timeout=REQUEST_TIMEOUT)

    @staticmethod
    def get_gpu_id(context) -> int:
        return context.system_properties.get("gpu_id") or 0

    @staticmethod
    def load_graph(frozen_graph_filename):
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        if isinstance(frozen_graph_filename, str):
            frozen_graph_filename = str(frozen_graph_filename)
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="hallucinator_v1")
            return graph

    def expand_features_set(self, features: torch.Tensor, num_samples: int, threshold: int, set_size: int) -> torch.Tensor:
        # test
        matching_set = features.cpu().numpy()
        matching_set = matching_set / 10
        matching_size = matching_set.shape[0]
        new_samples = []
        cur_num_samples = 0
        with tf.compat.v1.Session(graph=self.model) as sess:
            while cur_num_samples < num_samples:
                batch = dict()
                if matching_size < threshold:
                    num_new_samples = set_size - matching_size
                    padded_data = np.zeros((num_new_samples, matching_set.shape[1]))
                    batch['b'] = np.concatenate([np.ones_like(matching_set), np.zeros_like(padded_data)], 0)[None, ...]
                    batch['x'] = np.concatenate([matching_set, padded_data], axis=0)[None, ...]
                    batch['m'] = np.ones_like(batch['b'])
                    sample = sess.run(['hallucinator_v1/acset_vae/Reshape_11:0'], {
                        'hallucinator_v1/acset_vae/Placeholder:0':batch['x'],
                        'hallucinator_v1/acset_vae/Placeholder_1:0':batch['b'],
                        'hallucinator_v1/acset_vae/Placeholder_2:0':batch['m']
                    })[0]
                    new_sample = sample[0,matching_size:] * 10
                    cur_num_samples += num_new_samples
                else:
                    num_new_samples = set_size - threshold
                    ind = np.random.choice(matching_size, threshold, replace=False)
                    padded_data = np.zeros((num_new_samples, matching_set.shape[1]))
                    obs_data = matching_set[ind]
                    batch['x'] = np.concatenate([obs_data, padded_data], 0)[None, ...]
                    batch['b'] = np.concatenate([np.ones_like(obs_data), np.zeros_like(padded_data)], 0)[None, ...]
                    batch['m'] = np.ones_like(batch['b'])
                    sample = sess.run(['hallucinator_v1/acset_vae/Reshape_11:0'], {
                        'hallucinator_v1/acset_vae/Placeholder:0':batch['x'],
                        'hallucinator_v1/acset_vae/Placeholder_1:0':batch['b'],
                        'hallucinator_v1/acset_vae/Placeholder_2:0':batch['m']
                    })[0]
                    new_sample = sample[0,num_new_samples:,:] * 10
                    cur_num_samples += num_new_samples
                
                new_samples.append(new_sample)
        new_samples = np.concatenate(new_samples, 0)
        new_samples = new_samples[:num_samples]
        return torch.from_numpy(new_samples).contiguous()


    def initialize(self, context):
        # super().initialize(context)
        self.rs.tag('initialize')
        self.init = True
        self.manifest = context.manifest
        self.device = torch.device("cuda:" + str(self.get_gpu_id(context)) if torch.cuda.is_available() else "cpu")
        self.set_size = self.manifest['model']['set_size'] # 200
        self.threshold = self.manifest['model']['threshold'] # 100

        #NOTE: we don't need the config file for now
        # config_file = self.manifest['model']['config_file']
        # params = HParams(config_file)

        #NOTE: copying from the baseten model cache is not really necessary. disable for now
        # def cache_copy(src: Path, dst: Path):
        #     if src.exists():
        #         dst.parent.mkdir(parents=True, exist_ok=True)
        #         with portalocker.Lock(str(dst), 'ab') as dst_file:
        #             if dst.stat().st_size == 0:
        #                 # dst cached file is still empty, so we copy it from src cache
        #                 with open(src, 'rb') as src_file:
        #                     dst_file.write(src_file.read())
        #                 logger.info(f'copied {src} to {dst}')

        src = Path('/app/model_cache/cambai-ml/deployments/hallucinator_v1')
        # dst = Path(Path.home(), Path(params.dict["exp_dir"]))
        ckpt_files = ["frozen_graph.pb"]
        # for ckpt_file in ckpt_files:
        #     cache_copy(
        #         src / ckpt_file,
        #         dst / ckpt_file
        #     )
        self.model = self.load_graph(src / ckpt_files[0])
        logger.info("Loaded Hallucinator.")
        self.rs.tag(None)
        logger.info('\n======================================================\n' +
                     f'HallucinatorHandler.initialize runtime summary:\n\t{self.rs.summarize_runtime()}')


    def preprocess(self, data: List[dict]) -> List[dict]:
        """
        Args:
            Data (List[dict]): A list of dictionaries containing a batch of payloads with the following fields:
                input_path (str): The path to the input audio file.
                outputs (list): A list of dictionaries representing the output types and their corresponding paths.

        Requires:
            - Request should be made with application/json as header.

        Returns:
            dict: The batch of requests formatted for the inference method.

        """
        # reset the timer at the start of each new job
        self.rs.reset()
        self.rs.tag('preprocessing requests')

        # parse payloads
        payloads: List[Payload] = []
        for d in data:
            try:
                payloads.append(Payload(**(d['body'])))
            except Exception as e:
                logger.warning('failed to parse payload: %s\n\%s\%s', d['body'], e, traceback.format_exc())
                payloads.append(Payload(noop_reason="failed to parse payload"))

        # read headers
        tasks: List[Task] = []
        @retry(stop=stop_after_attempt(5), wait=wait_random(min=1, max=5))
        def read_header(env, path):
            features_path = Path(f'cambai-{env}-bucket', path)
            if not self.fs.exists(features_path):
                return None
            # use of small block size below is intentional
            with self.fs.open(features_path, 'rb', block_size=2**8) as file:
                return read_safetensors_file(file)

        for payload in payloads:
            try:
                tasks.append(Task(
                    **payload.model_dump(),
                    output_path=payload.input_path
                ))
            except Exception as e:
                logger.warning('failed to prepare task for payload: %s\n%s\n%s', payload.model_dump_json(), e, traceback.format_exc())
                tasks.append(Task(**payload.model_dump(), output_path=""))
                tasks[-1].noop_reason = "failed to create task from payload"

            if tasks[-1].noop_reason:
                continue

            # intentionally reading header of the output_path here (which will usually be equivalent to the input, but doesn't have to be). None if file does not exist
            h = read_header(tasks[-1].env, tasks[-1].output_path)
            if h.get("wavlm-l6-h12s") is not None:
                if h["wavlm-l6-h12s"]["shape"][-2] == tasks[-1].num_samples:
                    tasks[-1].noop_reason = f"hallucinations to {tasks[-1].num_samples} samples already exist"
                    continue

        self.rs.tag(None)
        return [t.model_dump() for t in tasks]


    def inference(self, data: List[dict]) -> List[dict]:
        @retry(stop=stop_after_attempt(5), wait=wait_random(min=1, max=5))
        def read_features(env, path):
            # use of small block size below is intentional
            with self.fs.open(Path(f'cambai-{env}-bucket', path), 'rb', block_size=20 * 2**20) as file:
                return read_safetensors_file(file, 'wavlm-l6', length_index=-2)

        @retry(stop=stop_after_attempt(5), wait=wait_random(min=1, max=5))
        def write_features(env, path, features, generated_features):
            # use of small block size below is intentional
            with self.fs.open(Path(f'cambai-{env}-bucket', path), 'wb', block_size=20 * 2**20) as file:
                file.write(safe_save({
                    'wavlm-l6': features.half().contiguous(),
                    'wavlm-l6-h12s': generated_features.half().contiguous()
                }))

        for d in data:
            try:
                task: Task = Task(**d)
                if task.noop_reason is not None:
                    continue
                self.rs.tag('reading features')
                features = read_features(task.env, task.input_path)
                self.rs.tag('generating features')
                generated_features = self.expand_features_set(features, task.num_samples, self.threshold, self.set_size)
                self.rs.tag('writing features')
                write_features(task.env, task.output_path, features, generated_features)
                self.rs.tag(None)
            except Exception as e:
                logger.warning('failed to expand features: %s\n%s\n%s', d, e, traceback.format_exc())
                d['noop_reason'] = 'failed to expand features'

        return data


    def postprocess(self, data: List[dict]) -> List[dict]:
        self.rs.tag('postprocessing requests')
        output_paths = []

        for d in data:
            output_paths.append({"output_path": d['output_path']})
            if d['noop_reason'] is not None:
                output_paths[-1]['noop_reason'] = d['noop_reason']

        self.rs.tag(None)
        logger.info('\n======================================================\n' +
                f'HallucinatorHandler.handle runtime summary:\n\t{self.rs.summarize_runtime()}')

        return output_paths

def main():
    import argparse
    import json

    class MockContext:
        def __init__(self):
            self.manifest = {
                'model': {
                    'set_size': 200,
                    'threshold': 100,
                    'config_file': "./exp/speech_XXL_cond/params.json"
                }
            }
            # self.system_properties = { 'gpu_id': 0 }

    parser = argparse.ArgumentParser(description="CLI Wrapper for TorchServe Handler with pdb debugging support")
    parser.add_argument("input_data", type=str, nargs='+', help="Input payloads for the model")

    args = parser.parse_args()

    data = []
    for json_file in args.input_data:
        with open(json_file, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            data.append({ "body": json_data })

    handler = HallucinatorHandler()
    context = MockContext()
    handler.initialize(context)
    data_preprocess = handler.preprocess(data)
    output = handler.inference(data_preprocess)
    output = handler.postprocess(output)

if __name__ == "__main__":
    main()
