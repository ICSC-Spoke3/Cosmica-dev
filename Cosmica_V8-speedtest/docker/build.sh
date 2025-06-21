docker build -t sdegno-dev -f dev.Dockerfile .
docker build -t sdegno-prod -f prod.Dockerfile.prod --build-arg SSH_KEY="$(cat ./github_deploy_key)" .