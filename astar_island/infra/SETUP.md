# GCE Spot Instance Setup

One-time setup to let GitHub Actions spin up GCE spot VMs for fast calibration.

## 1. Set your project

```bash
export GCP_PROJECT=your-project-id
gcloud config set project $GCP_PROJECT
```

## 2. Enable required APIs

```bash
gcloud services enable compute.googleapis.com storage.googleapis.com iamcredentials.googleapis.com
```

## 3. Create a service account

```bash
gcloud iam service-accounts create ainm-runner \
  --display-name="AINM GCE Runner"

# Grant permissions: create VMs + read/write GCS
gcloud projects add-iam-policy-binding $GCP_PROJECT \
  --member="serviceAccount:ainm-runner@${GCP_PROJECT}.iam.gserviceaccount.com" \
  --role="roles/compute.instanceAdmin.v1"

gcloud projects add-iam-policy-binding $GCP_PROJECT \
  --member="serviceAccount:ainm-runner@${GCP_PROJECT}.iam.gserviceaccount.com" \
  --role="roles/storage.admin"
```

## 4. Set up Workload Identity Federation (keyless auth from GitHub Actions)

```bash
# Create a workload identity pool
gcloud iam workload-identity-pools create github-pool \
  --location="global" \
  --display-name="GitHub Actions Pool"

# Create a provider for your GitHub repo
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# Allow the GitHub repo to impersonate the service account
gcloud iam service-accounts add-iam-policy-binding \
  ainm-runner@${GCP_PROJECT}.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/$(gcloud projects describe $GCP_PROJECT --format='value(projectNumber)')/locations/global/workloadIdentityPools/github-pool/attribute.repository/tomdgr/ainm_main_ml"
```

## 5. Add GitHub secrets

Go to **Settings > Secrets and variables > Actions** in your GitHub repo and add:

| Secret | Value |
|--------|-------|
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | `projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/providers/github-provider` |
| `GCP_SERVICE_ACCOUNT` | `ainm-runner@YOUR_PROJECT.iam.gserviceaccount.com` |
| `AINM_ACCESS_TOKEN` | *(already set)* |

Get your project number:
```bash
gcloud projects describe $GCP_PROJECT --format='value(projectNumber)'
```

## 6. Test it

Trigger the workflow manually from GitHub Actions:
```bash
gh workflow run gce_submit.yml
```

## Machine types & cost (spot pricing, europe-west1)

| Machine | vCPUs | Spot $/hr | Calibration est. |
|---------|-------|-----------|-------------------|
| e2-standard-8 | 8 | ~$0.07 | ~5 min |
| e2-standard-16 | 16 | ~$0.14 | ~3 min |
| c3-highcpu-22 | 22 | ~$0.19 | ~2 min |
| c3-highcpu-44 | 44 | ~$0.38 | ~1 min |

Default is `c3-highcpu-22`. Override via workflow dispatch input.
