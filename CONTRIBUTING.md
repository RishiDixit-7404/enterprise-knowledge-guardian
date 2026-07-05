# Contributor's Guide

Welcome to the Enterprise Knowledge Guardian (EKG) project! 

The EKG system enforces strict enterprise standards. We value correctness, reproducibility, offline capability, and zero-cost local iteration.

## 1. Development Workflow

1. **Clone & Setup**: Use Python 3.12+ and create a virtual environment (`.venv`).
2. **Deterministic Dependencies**: Do not run `pip install <package>`. Add your dependency to `requirements.in` and run:
   ```bash
   pip-compile requirements.in
   pip-sync requirements.txt
   ```
3. **Local Services**: Always run `docker-compose up -d` before executing tests to ensure PostgreSQL/pgvector and Neo4j are available.
4. **Mock-First**: When building new connectors or services, build the Mock class *first* and make it the default in `settings.py` so that no API keys or paid accounts are required for other developers.

## 2. Testing & TDD Expectations

* Test-Driven Development (TDD) is strongly encouraged.
* Write unit tests for your mock implementations first.
* All tests must pass offline. Tests making real outbound HTTP requests to paid APIs will be rejected. 
* To run the test suite locally without third-party telemetry, use:
  ```bash
  DEEPEVAL_TELEMETRY_OPT_OUT=1 pytest -v
  ```
* 100% test pass rate is mandatory before opening a PR.

## 3. Pull Request Workflow

1. Branch off `main` using the format `feature/<name>` or `fix/<name>`.
2. Commit your changes locally. Ensure commit messages are descriptive.
3. Open a Pull Request against `main`.
4. Ensure the GitHub Actions CI (if enabled) passes.
5. Provide a summary of the implementation, including any new environment variables required if real services are configured.

## 4. Coding Standards

* **No fabricated metrics**: Never claim a performance metric (e.g. latency, precision, recall) in comments or logs unless it was directly measured by the evaluation harness.
* **$0 Cost Defaults**: Ensure all added features are accessible for free developers. Use fake models by default.
* **Typing**: Use standard Python type hinting across all functions and classes.
* **Docstrings**: Document the inputs, outputs, and potential side effects of complex agent nodes.

## 5. Branch Strategy

* `main`: The stable, production-ready branch.
* `feature/*`: For active development.

Thank you for adhering to these guidelines and helping us maintain a reliable enterprise evaluation framework!
