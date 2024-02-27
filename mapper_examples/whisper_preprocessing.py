import runhouse as rh
import numpy as np
from transformers import AutoProcessor
from datasets import Audio, load_dataset
from time import time


class WhisperPreprocessor(rh.Module):
    def __init__(self):
        super().__init__()
        self.processor = None

    def load_split(self, split: str):
        minds_split = load_dataset("PolyAI/minds14", "en-US", split=split)
        return minds_split.cast_column("audio", Audio(sampling_rate=16000))

    def process(self, split: str):
        if not self.processor:
            self.processor = AutoProcessor.from_pretrained("openai/whisper-small")
        audio = self.load_split(split)

        def preprocess(sample):
            return self.processor(sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"], padding=True)
        return audio.map(preprocess, remove_columns=audio.column_names)["input_features"]


if __name__ == "__main__":
    cluster = rh.cluster(name="rh-4x8-cpu", instance_type="CPU:8", num_instances=4)
    audio_preproc_env = rh.env(working_dir="reqs:./mapper_examples", name="audio_preproc", compute={"CPU": 1})
    remote_preproc = WhisperPreprocessor().to(cluster, env=audio_preproc_env, name="whisper_preproc")

    splits = [f"train[{split}%:{split + 4}%]" for split in range(0, 100, 4)]
    mapper_env = rh.env(working_dir="reqs:./mapper_examples", name="mapper_env")
    mapper = rh.mapper(remote_preproc, method="process").to(cluster, env=mapper_env, name="whisper_mapper")
    mapper.add_replicas(len(splits) - mapper.remote.num_replicas)

    start = time()
    splits_batches = mapper.map(splits)
    feats = [item for batch in splits_batches for item in batch]
    print(f"Time to process {len(feats)} audio files: {time() - start}")
    print(np.array(feats[0]).shape)
