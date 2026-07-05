with open("docker-compose.yml", "r") as f:
    content = f.read()
content = content.replace("WORKER_POLL_INTERVAL: \"5.0\"", "WORKER_POLL_INTERVAL: \"5.0\"\n      API_KEY: \"real-test-token\"")
with open("docker-compose.yml", "w") as f:
    f.write(content)
