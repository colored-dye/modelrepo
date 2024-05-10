import tempfile
import os

from HFManager import HFManager
from modelstore.storage.local import FileSystemStorage
from modelstore import ModelStore


def load(model_store: ModelStore, domain: str, model_id: str, backend: str = "torch"):
        """Loads a model into memory
        
        backend: torch, tf, sklearn
        """
        storage = model_store.storage
        meta_data = storage.get_meta_data(domain, model_id)
        manager = HFManager(storage)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_files = model_store.download(tmp_dir, domain, model_id)
            return manager.load(model_files, meta_data, backend)
