from plugins.image_gen.gemini.client import request_payload


def test_interactions_payload_requests_supported_jpeg_format() -> None:
    payload = request_payload(
        prompt="SYMCRG QA",
        model="gemini-3.1-flash-image",
        aspect="square",
    )

    assert payload["response_format"]["mime_type"] == "image/jpeg"
