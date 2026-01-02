# These are the parameters of my cloud setup.
variable "project_id" {
  type        = string
  description = "GCP project id"
  default     = "resounding-sled-437604-v0"
}

variable "region" {
  type        = string
  description = "GCP region"
  default     = "us-central1"
}

variable "service_name" {
  type        = string
  description = "Cloud Run service name"
  default     = "optimir-api"
}

variable "invoker_user_email" {
  type        = string
  description = "Google account email allowed to invoke the service"
  default     = "deepmehta827@gmail.com"
}

variable "artifact_repo_id" {
  type        = string
  description = "Artifact Registry repository id"
  default     = "optimir"
}

variable "image" {
  type        = string
  description = "Full container image URL (Artifact Registry). Example: us-central1-docker.pkg.dev/<project>/<repo>/<image>:<tag>"
}

variable "secrets" {
  type        = list(string)
  description = "Secret Manager secret IDs to create (values added out-of-band)"
  default     = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
}
