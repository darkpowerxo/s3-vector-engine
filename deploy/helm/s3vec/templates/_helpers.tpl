{{/*
Common labels
*/}}
{{- define "s3vec.labels" -}}
app.kubernetes.io/part-of: s3-vector-engine
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{/*
Selector labels for a component
*/}}
{{- define "s3vec.selectorLabels" -}}
app.kubernetes.io/name: {{ . }}
{{- end }}
