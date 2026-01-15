# Helper script PowerShell pour exécuter la CLI de prédiction via Docker
# Usage: .\docker-predict.ps1 --input sample.json

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

docker-compose run --rm cli $Arguments

