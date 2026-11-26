import json

from sis.services.model_preferences import ModelPreferences


def test_model_preference_survives_reload_without_storing_credentials(tmp_path):
    path = tmp_path / "settings" / "preferences.json"
    preferences = ModelPreferences(path)
    assert preferences.read()["preferred_model"] == "opencode-go/deepseek-v4.1-flash"

    saved = preferences.write({
        "preferred_model": "opencode-go/deepseek-v4.1-flash",
        "policy": "specific",
        "allow_subscription_models": True,
        "allow_metered_models": True,
        "external_permissions": {"ordinary_project_metadata": True, "not_a_supported_category": True},
    })
    assert saved["allow_metered_models"] is False
    assert saved["external_permissions"] == {
        "ordinary_project_metadata": True,
        "repository_locations": False,
        "publication_status": False,
        "approved_project_documentation": False,
    }
    assert "api_key" not in path.read_text(encoding="utf-8")
    assert ModelPreferences(path).read() == saved


def test_route_observation_distinguishes_worked_and_blocked(tmp_path):
    preferences = ModelPreferences(tmp_path / "preferences.json")
    preferences.observe_model("opencode-go/deepseek-v4.1-flash", "worked", "completed a turn")
    preferences.observe_model("opencode/deepseek-v4.1-flash-free", "blocked", "permission_denied")

    readback = ModelPreferences(preferences.path).read()["model_observations"]
    assert readback["opencode-go/deepseek-v4.1-flash"]["status"] == "worked"
    assert readback["opencode/deepseek-v4.1-flash-free"]["status"] == "blocked"


def test_project_grants_are_bound_to_project_route_and_supported_categories(tmp_path):
    preferences = ModelPreferences(tmp_path / "preferences.json")
    saved = preferences.write({"project_grants": [
        {"project_id": "jev", "provider": "opencode-go/deepseek-v4.1-flash",
         "categories": ["publication_status", "source_code", "ordinary_project_metadata"]},
        {"project_id": "../secret", "provider": "x", "categories": ["repository_locations"]},
    ]})
    assert saved["project_grants"] == [{
        "project_id": "jev", "provider": "opencode-go/deepseek-v4.1-flash",
        "categories": ["ordinary_project_metadata", "publication_status"],
    }]
    assert ModelPreferences(preferences.path).read()["project_grants"] == saved["project_grants"]
