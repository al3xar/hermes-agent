{{- define "hades.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- define "hades.fullname" -}}
{{- if .Values.fullnameOverride }}{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}{{- $name := include "hades.name" . }}{{- if contains $name .Release.Name }}{{- .Release.Name | trunc 63 | trimSuffix "-" }}{{- else }}{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}{{- end }}{{- end }}
{{- end }}
{{- define "hades.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
app.kubernetes.io/name: {{ include "hades.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: cyber-range-red
{{- end }}
{{- define "hades.secretName" -}}
{{- .Values.secrets.existingSecret | default (printf "%s-secrets" (include "hades.fullname" .)) }}
{{- end }}
{{- define "hades.langfuse.secretName" -}}
{{- printf "%s-langfuse-secrets" (include "hades.fullname" .) }}
{{- end }}
{{- /*
  Env común (no sensible) compartido por langfuse-web y langfuse-worker.
  Espejo del docker-compose: apunta a los Services internos langfuse-* y a la
  whitelist del vLLM de Athena. Los secretos (DATABASE_URL, claves S3, etc.)
  llegan aparte vía envFrom del Secret de Langfuse.
*/ -}}
{{- define "hades.langfuse.commonEnv" -}}
- name: TELEMETRY_ENABLED
  value: "false"
- name: CLICKHOUSE_MIGRATION_URL
  value: "clickhouse://langfuse-clickhouse:9000"
- name: CLICKHOUSE_URL
  value: "http://langfuse-clickhouse:8123"
- name: CLICKHOUSE_CLUSTER_ENABLED
  value: "false"
- name: REDIS_HOST
  value: "langfuse-redis"
- name: REDIS_PORT
  value: "6379"
- name: LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT
  value: "http://langfuse-minio:9000"
- name: LANGFUSE_S3_MEDIA_UPLOAD_BUCKET
  value: {{ .Values.langfuse.minio.bucket | quote }}
- name: LANGFUSE_S3_MEDIA_UPLOAD_FORCE_PATH_STYLE
  value: "true"
- name: LANGFUSE_S3_MEDIA_UPLOAD_REGION
  value: "us-east-1"
- name: LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT
  value: "http://langfuse-minio:9000"
- name: LANGFUSE_S3_EVENT_UPLOAD_BUCKET
  value: {{ .Values.langfuse.minio.bucket | quote }}
- name: LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE
  value: "true"
- name: LANGFUSE_S3_EVENT_UPLOAD_REGION
  value: "us-east-1"
- name: LANGFUSE_LLM_CONNECTION_WHITELISTED_HOST
  value: {{ .Values.langfuse.llmWhitelist.host | quote }}
- name: LANGFUSE_LLM_CONNECTION_WHITELISTED_IPS
  value: {{ .Values.langfuse.llmWhitelist.ips | quote }}
- name: LANGFUSE_LLM_CONNECTION_WHITELISTED_IP_SEGMENTS
  value: {{ .Values.langfuse.llmWhitelist.ipSegments | quote }}
{{- end }}
