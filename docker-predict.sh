#!/bin/bash
# Helper script pour exécuter la CLI de prédiction via Docker
# Usage: ./docker-predict.sh --input sample.json

docker-compose run --rm cli "$@"

