import os
import json
import joblib
from datetime import datetime

class LocalModelRegistry:
    def __init__(self, registry_dir="models"):
        self.registry_dir = registry_dir
        self.metadata_path = os.path.join(registry_dir, "registry_metadata.json")
        os.makedirs(registry_dir, exist_ok=True)
        self._init_metadata()

    def _init_metadata(self):
        if not os.path.exists(self.metadata_path):
            with open(self.metadata_path, "w") as f:
                json.dump({"models": {}}, f, indent=4)

    def _read_metadata(self):
        with open(self.metadata_path, "r") as f:
            return json.load(f)

    def _write_metadata(self, metadata):
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def register_model(self, model, name, metrics, params, features, tag="production"):
        """
        Saves a trained model file and logs its metadata to the registry.
        """
        metadata = self._read_metadata()
        
        # Initialize dictionary for model name if it doesn't exist
        if name not in metadata["models"]:
            metadata["models"][name] = []

        # Determine version number (increment major/minor or just simple counter)
        version_num = len(metadata["models"][name]) + 1
        version_str = f"v{version_num}.0.0"
        
        # Save model object
        model_filename = f"{name}_{version_str}.joblib"
        model_filepath = os.path.join(self.registry_dir, model_filename)
        joblib.dump(model, model_filepath)
        
        # Compile run metadata
        run_metadata = {
            "version": version_str,
            "filename": model_filename,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
            "params": params,
            "features": features,
            "tags": [tag]
        }
        
        # If new model is set as production, clear the production tag from previous versions
        if tag == "production":
            for existing_run in metadata["models"][name]:
                if "production" in existing_run["tags"]:
                    existing_run["tags"].remove("production")
                    
        metadata["models"][name].append(run_metadata)
        self._write_metadata(metadata)
        print(f"[OK] Registered model '{name}' version '{version_str}' marked as '{tag}'")
        return version_str

    def load_model(self, name, version_or_tag="production"):
        """
        Loads a registered model based on its version string (e.g. 'v1.0.0') or tag (e.g. 'production').
        Returns:
            model_object, run_metadata
        """
        metadata = self._read_metadata()
        if name not in metadata["models"] or not metadata["models"][name]:
            raise FileNotFoundError(f"No registered model found with name: '{name}'")
            
        target_run = None
        # Check if version_or_tag matches tag (e.g., 'production') or version (e.g., 'v1.0.0')
        for run in metadata["models"][name]:
            if version_or_tag in run["tags"] or run["version"] == version_or_tag:
                target_run = run
                break
                
        if target_run is None:
            # Default to latest if tag not found
            target_run = metadata["models"][name][-1]
            print(f"Warning: Tag/version '{version_or_tag}' not found for '{name}'. Defaulting to latest.")
            
        model_filepath = os.path.join(self.registry_dir, target_run["filename"])
        if not os.path.exists(model_filepath):
            raise FileNotFoundError(f"Model file '{model_filepath}' not found in registry folder.")
            
        model = joblib.load(model_filepath)
        return model, target_run

    def get_latest_metadata(self, name):
        """Returns metadata for the latest registered version of a model"""
        metadata = self._read_metadata()
        if name not in metadata["models"] or not metadata["models"][name]:
            return None
        return metadata["models"][name][-1]
