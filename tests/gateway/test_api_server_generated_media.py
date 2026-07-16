"""Regression coverage for generated media returned through the API server."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.mark.asyncio
async def test_run_agent_appends_generated_image_when_model_omits_path() -> None:
    adapter = APIServerAdapter(PlatformConfig())
    image_path = "/tmp/generated/office.jpeg"
    tool_call_id = "image-call-1"
    mock_agent = MagicMock()
    mock_agent.run_conversation.return_value = {
        "final_response": "[Image generated: blue office building]",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "function": {"name": "image_generate", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({"success": True, "image": image_path}),
            },
            {"role": "assistant", "content": "[Image generated: blue office building]"},
        ],
    }

    with patch.object(adapter, "_create_agent", return_value=mock_agent):
        result, _usage = await adapter._run_agent(
            user_message="Generate a blue office building",
            conversation_history=[],
            session_id="generated-media-regression",
        )

    assert result["final_response"] == (
        "[Image generated: blue office building]\n"
        f"MEDIA:{image_path}"
    )


@pytest.mark.asyncio
async def test_responses_register_document_for_authenticated_delivery(tmp_path) -> None:
    document = tmp_path / "broker-package.pdf"
    document.write_bytes(b"%PDF-1.7\nSYMCRG")
    adapter = APIServerAdapter(PlatformConfig())

    attachments = adapter._register_response_media(
        f"Your package is ready.\nMEDIA:{document}"
    )

    assert len(attachments) == 1
    assert attachments[0]["filename"] == "broker-package.pdf"
    assert attachments[0]["content_type"] == "application/pdf"
    request = MagicMock()
    request.match_info = {"media_id": attachments[0]["id"]}
    request.headers = {}
    response = await adapter._handle_response_media(request)
    assert response.status == 200
    assert response._path == document


def test_responses_refuse_unsafe_media_paths() -> None:
    adapter = APIServerAdapter(PlatformConfig())

    attachments = adapter._register_response_media(
        "MEDIA:/home/marquise/.hermes/profiles/symcrg/.env"
    )

    assert attachments == []


def test_responses_refuse_stale_host_files(tmp_path) -> None:
    stale_file = tmp_path / "existing-notes.txt"
    stale_file.write_text("private host notes", encoding="utf-8")
    os.utime(stale_file, (1, 1))
    adapter = APIServerAdapter(PlatformConfig())

    attachments = adapter._register_response_media(f"MEDIA:{stale_file}")

    assert attachments == []
