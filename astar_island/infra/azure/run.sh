#!/bin/bash
# Azure ML job script — download data from Azure Blob, calibrate, predict, and submit.
# Observations are fetched by GitHub Actions and uploaded to blob storage.
set -euo pipefail

echo "=== AINM Azure ML Job ==="
echo "  Python: $(python --version)"
echo "  Working dir: $(pwd)"

# Fetch AINM token from Azure Key Vault
if [ -z "${AINM_ACCESS_TOKEN:-}" ]; then
    echo "Fetching AINM token from Key Vault..."
    pip install -q azure-identity azure-keyvault-secrets azure-storage-blob
    AINM_ACCESS_TOKEN=$(python -c "
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
client = SecretClient(vault_url='https://nmaiexperis7070804657.vault.azure.net', credential=DefaultAzureCredential())
print(client.get_secret('AINM-ACCESS-TOKEN').value)
")
    export AINM_ACCESS_TOKEN
    echo "Token retrieved."
fi

# Download round data from Azure Blob Storage (uploaded by GitHub Actions)
echo "Downloading data from Azure Blob Storage..."
mkdir -p data/rounds
python -c "
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient

container = ContainerClient(
    account_url='https://nmaiexperis3807107646.blob.core.windows.net',
    container_name='ainm-data',
    credential=DefaultAzureCredential(),
)

blobs = container.list_blobs(name_starts_with='rounds/')
for blob in blobs:
    local_path = f'data/{blob.name}'
    import os
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, 'wb') as f:
        f.write(container.download_blob(blob).readall())
    print(f'  Downloaded: {local_path}')

print('Blob download complete.')
"

# Install project dependencies with uv
uv sync --no-dev

# Calibrate and submit using data already downloaded from blob (skip API fetch)
uv run python main.py --calibrate --submit --no-fetch

echo "=== Done ==="
