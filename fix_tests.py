import os
import re

def fix_conftest():
    with open('tests/conftest.py', 'r') as f:
        content = f.read()
    content = content.replace('@pytest.fixture(scope="module")\ndef client():', '@pytest.fixture(scope="function")\ndef client():')
    with open('tests/conftest.py', 'w') as f:
        f.write(content)

def fix_test_file(path, tests_needing_client=None):
    if not os.path.exists(path): return
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Remove module-level `client = TestClient(app)`
    new_lines = []
    for line in lines:
        if line.strip() == 'client = TestClient(app)':
            continue
        new_lines.append(line)
    content = "".join(new_lines)
    
    # Add client to test signatures if needed
    if tests_needing_client:
        for t in tests_needing_client:
            # e.g., def test_eval_run_endpoint(populated_db_and_graph): -> def test_eval_run_endpoint(populated_db_and_graph, client):
            # def test_eval_run_endpoint(): -> def test_eval_run_endpoint(client):
            match = re.search(r'def ' + t + r'\((.*?)\):', content)
            if match:
                args = match.group(1).strip()
                new_args = f"{args}, client" if args else "client"
                content = content[:match.start()] + f"def {t}({new_args}):" + content[match.end():]
    
    with open(path, 'w') as f:
        f.write(content)

fix_conftest()

fix_test_file('tests/test_agents.py', ['test_query_endpoint_langgraph_flow'])
fix_test_file('tests/test_eval.py', ['test_eval_run_endpoint', 'test_eval_records_persisted', 'test_metrics_endpoint'])
fix_test_file('tests/test_graph.py', ['test_api_graph_endpoint', 'test_api_graph_endpoint_not_found'])
fix_test_file('tests/test_e2e_pipeline.py', ['test_e2e_pipeline_generates_runlog'])
fix_test_file('tests/test_query.py', [])

# Fix test_auth.py
with open('tests/test_auth.py', 'r') as f:
    auth_content = f.read()

# Replace test_auth_fail_closed_wrong_token to include monkeypatch
old_wrong_token = """def test_auth_fail_closed_wrong_token():
    \"\"\"Verify that a wrong token is rejected.\"\"\"
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides.clear()
    
    with TestClient(app) as client:
        response = client.get("/health", headers={"Authorization": "Bearer wrong-token"})
        assert response.status_code == 401
        assert "Invalid or missing" in response.json()["detail"]
        
    app.dependency_overrides = original_overrides"""

new_wrong_token = """def test_auth_fail_closed_wrong_token(monkeypatch):
    \"\"\"Verify that a wrong token is rejected when API_KEY is configured.\"\"\"
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides.clear()
    
    monkeypatch.setattr(settings, "API_KEY", "real-test-token")
    
    with TestClient(app) as client:
        response = client.get("/health", headers={"Authorization": "Bearer wrong-token"})
        assert response.status_code == 401
        assert "Invalid or missing" in response.json()["detail"]
        
    app.dependency_overrides = original_overrides"""

auth_content = auth_content.replace(old_wrong_token, new_wrong_token)

with open('tests/test_auth.py', 'w') as f:
    f.write(auth_content)
