# Terraform, talk to Google Cloud using my logged-in account.
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  # Uses ADC (you already ran gcloud auth application-default login)
}
