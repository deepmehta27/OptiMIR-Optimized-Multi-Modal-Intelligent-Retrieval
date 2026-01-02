# Artifact Registry (Docker) repo - Cloud Run cannot pull images from your local machine — it needs a registry
resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = var.artifact_repo_id
  description   = "Docker images for Optimir"
  format        = "DOCKER"
}

# Service account that Cloud Run will run as - Principle of least privilege (real production practice)
resource "google_service_account" "run_sa" {
  account_id   = "${var.service_name}-sa"
  display_name = "Cloud Run runtime SA for ${var.service_name}"
}

# Create Secret Manager secrets (containers only; do NOT store secret values in TF) - Keeps API keys out of Terraform state
resource "google_secret_manager_secret" "secrets" {
  for_each  = toset(var.secrets)
  secret_id = each.value

  replication {
    auto {}
  }
}

# Allow the Cloud Run service account to access secrets at runtime - This turns our Dockerized FastAPI app into a live cloud service.
resource "google_secret_manager_secret_iam_member" "secret_access" {
  for_each = google_secret_manager_secret.secrets

  secret_id = each.value.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.run_sa.email}"
}

# Cloud Run v2 service
resource "google_cloud_run_v2_service" "api" {
  name     = var.service_name
  location = var.region
  deletion_protection = false

  template {
    service_account = google_service_account.run_sa.email

    containers {
      image = var.image

      # Example: make the secret available as an env var (no secret value in Terraform)
      # Repeat env blocks for each secret you want injected.
      dynamic "env" {
        for_each = toset(var.secrets)
        content {
          name = env.value
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.secrets[env.value].secret_id
              version = "latest"
            }
          }
        }
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }
  }

  depends_on = [google_secret_manager_secret_iam_member.secret_access]
}

# IAM: only your Google account can invoke
resource "google_cloud_run_v2_service_iam_member" "invoker" {
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "user:${var.invoker_user_email}"
}
