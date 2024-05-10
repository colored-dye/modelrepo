#    Copyright 2020 Neal Lathia
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
from functools import partial
from typing import Any, List

from modelstore.metadata import metadata
from modelstore.models.model_manager import ModelManager
from modelstore.storage.storage import CloudStorage
from modelstore.utils.log import logger

# pylint disable=import-outside-toplevel
MODEL_DIRECTORY = "transformers"


class HFManager(ModelManager):

    """
    Model persistence for Transformer models:
    https://huggingface.co/transformers/main_classes/model.html#transformers.TFPreTrainedModel.save_pretrained
    https://github.com/huggingface/transformers/blob/e50a931c118b9f55f77a743bf703f436bf7a7c29/src/transformers/modeling_utils.py#L676
    """

    NAME = "transformers"

    def __init__(self, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)

    def required_dependencies(self) -> list:
        return ["transformers"]

    def optional_dependencies(self) -> list:
        deps = super().optional_dependencies()
        return deps + ["torch", "tensorflow"]

    def _required_kwargs(self):
        return ["model"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        if "config" in kwargs:
            from transformers import PretrainedConfig

            config = kwargs.get("config")
            if not isinstance(config, PretrainedConfig):
                logger.debug("unknown config type: %s", type(config))
                return False

        if "tokenizer" in kwargs:
            from transformers import PreTrainedTokenizerBase

            tokenizer = kwargs.get("tokenizer")
            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                logger.debug("unknown tokenizer type: %s", type(tokenizer))
                return False

        if "processor" in kwargs:
            from transformers import ProcessorMixin
            from transformers.image_processing_utils import BaseImageProcessor

            processor = kwargs.get("processor")
            if not isinstance(processor, (ProcessorMixin, BaseImageProcessor)):
                logger.debug("unknown processor type: %s", type(processor))
                return False

        # The model must be either a PyTorch or TF pretrained model
        try:
            from transformers import TFPreTrainedModel

            if isinstance(kwargs.get("model"), TFPreTrainedModel):
                return True
        except RuntimeError:
            # Cannot import tensorflow things
            pass

        from transformers import PreTrainedModel

        return isinstance(kwargs.get("model"), PreTrainedModel)

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("Model not matched with transformers")
        return [
            partial(
                _save_transformers,
                entities=[
                    kwargs["model"],
                    kwargs.get("tokenizer"),
                    kwargs.get("config"),
                    kwargs.get("processor"),
                ],
            ),
        ]

    def get_params(self, **kwargs) -> dict:
        """
        Returns a dictionary containing the config for the model
        """
        if "config" in kwargs:
            return kwargs["config"].to_dict()
        return {}

    def load(self, model_path: str, meta_data: metadata.Summary, backend: str = "torch") -> Any:
        """
        
        """
        super().load(model_path, meta_data)
        model_dir = _get_model_directory(model_path)
        model_files = set(os.listdir(model_dir))
        logger.debug("Loading from: %s...", model_files)

        # pylint: disable=import-outside-toplevel
        # Infer whether a tokenizer was saved
        tokenizer = None
        if any(x in model_files for x in ["tokenizer.json", "tokenizer_config.json"]):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            logger.debug("Loaded: %s...", type(tokenizer))

        # Infer whether a config was saved
        if "config.json" in model_files:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_dir)
            logger.debug("Loaded: %s...", type(config))

        processor = None
        if "preprocessor_config.json" in model_files:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(model_dir)
            logger.debug("Loaded: %s...", type(processor))

        print(f"Backend: {backend}")

        if backend == "torch":
            from transformers import AutoModel

            logger.debug("Loading with AutoModel...")
            model = AutoModel.from_pretrained(model_dir)
        elif backend == "tf":
            from transformers import TFAutoModel

            logger.debug("Loading with TFAutoModel...")
            model = TFAutoModel.from_pretrained(model_dir)
        elif backend == "sklearn":
            import joblib
            logger.debug("Loading with joblib...")
            model_files = os.listdir(model_dir)
            model_files = [f for f in model_files if f.endswith(".joblib")]
            if len(model_files) == 0:
                raise ValueError("No available model!")
            elif len(model_files) > 1:
                raise ValueError(f"Multiple models: {model_files}")
            model_file = os.path.join(model_dir, model_files[0])
            model = joblib.load(model_file)

        if tokenizer is not None:
            return model, tokenizer, config
        if processor is not None:
            return model, processor, config
        return model, config


def _get_model_directory(parent_dir: str) -> str:
    return os.path.join(parent_dir, MODEL_DIRECTORY)


def _save_transformers(
    tmp_dir: str,
    entities: List,
) -> str:
    model_dir = _get_model_directory(tmp_dir)
    os.makedirs(model_dir)
    for entity in entities:
        if entity is not None:
            entity.save_pretrained(model_dir)
    return model_dir
