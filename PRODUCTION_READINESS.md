# Production readiness

## Code controls now implemented

- `APP_ENV=production` fails startup unless Cognito identity, an allowed Cognito group, protected metrics, a hosted HTTPS Qdrant endpoint and key, Postgres with `sslmode=verify-full` and a mounted CA bundle, Redis over `rediss://`, and the selected model-provider credential are configured.
- Cognito ID tokens are validated against the configured issuer, audience, signature, expiration, and token use. The API requires membership in `COGNITO_REQUIRED_GROUP`. Conversation checkpoint IDs and rate-limit keys are scoped to the authenticated subject.
- The Streamlit container no longer inherits the API's complete `.env`; it receives only its URL and auth settings. User prompts are no longer attached to exported Logfire spans.
- Memory expiry can run independently of traffic with `python scripts/cleanup_expired_memory.py`; schedule it as a recurring task in production.
- Backup uploads request SSE-KMS. Production backup mode requires an explicit customer-managed KMS key; remote Qdrant backups require HTTPS. The Docker image includes `pg_dump` for Postgres backups.
- Portkey use requires the saved config ID; the client no longer falls back to inline configuration.
- `.gitignore` and `.dockerignore` exclude credentials, source corpora, raw evaluation responses, source-linked holdouts, and backup artifacts. The public repository contains only aggregate evaluation results.

## AWS deployment work still required

1. **Network and edge:** deploy UI and API tasks in private subnets; expose only the UI through an HTTPS load balancer with an ACM certificate. Keep database, Redis, and API ingress restricted to their required security groups. Provide private egress to Groq/Portkey and hosted Qdrant.
2. **Identity:** configure the Cognito app client callback for `https://<application-domain>/oauth2callback`, create the `rag-users` group, and inject Streamlit OIDC secrets through AWS Secrets Manager. Do not place those values in `.env` or image layers.
3. **Data services:** provision encrypted Postgres and Redis with backups, private networking, TLS, and rotation. Mount the current RDS CA bundle and use a database URL with `sslmode=verify-full&sslrootcert=<mounted-path>`. Use an HTTPS Qdrant Cloud endpoint and store its API key as a secret.
4. **Scheduled operations:** run `scripts/cleanup_expired_memory.py` on a schedule. Run `scripts/backup.py --s3-uri s3://<private-bucket>/<prefix> --s3-kms-key-id <kms-key-id>` on a separate schedule. Configure S3 Block Public Access, bucket versioning, lifecycle retention, KMS key policy, and alerts for failed jobs.
5. **Monitoring and recovery:** scrape `/metrics` with its bearer token from a private monitoring path; route container logs to CloudWatch without prompts or document bodies; configure alerts for unhealthy targets, elevated 5xx/429s, task restarts, database/cache failures, and backup/cleanup failures. Perform a restore drill before launch.
6. **Authorization scope:** every member of `COGNITO_REQUIRED_GROUP` can query the same indexed corpus. Document-level and tenant-level ACL filters are not implemented. Do not place restricted, tenant-specific sources in this shared collection until those ACLs are added and verified.
7. **Quality release gate:** the post-fix evaluation and separate-model review are complete, but a non-author still needs to review the authored questions and references. Add a frozen, independently sampled holdout before treating the metrics as general quality evidence.

## Current boundary

The local Compose stack is for private development. The AWS resources above have not been created or deployed. Deployment needs an AWS account/role, target region, application domain and certificate, Cognito configuration, Qdrant endpoint, and a cost/availability choice for managed databases and NAT egress.
