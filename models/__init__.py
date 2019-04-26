import os
import importlib

from .registry import MODEL_REGISTRY

# List all the modules in the models subdirectory
modules = []
# r=root, d=directories, f = files
for root, _, files in os.walk('models'):
    for file in files:
        # Filter init files, registry, the chache and possibly any other file which should not be there anyway
        if "__init__" not in file and "registry" not in file and ".py" in file and "__pycache__" not in root:
            # Make the path and filename match the string needed for importlib
            modules.append(os.path.join(root, file).replace("/", ".").replace(".py", ""))

# Importing all the models which are annotated with @Model
for module in modules:
    importlib.import_module(module)

# Expose all the models
for m in MODEL_REGISTRY:
    globals()[m] = MODEL_REGISTRY[m]
