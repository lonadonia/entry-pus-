"""
Error normalisation — the full matrix.

Every shape asserted here was captured live from api.apidot.ai on 2026-09-18
(see the module docstring of bot/ai_errors.py), which is why the tests pin exact
bodies rather than paraphrases: the classification must survive the provider's
real inconsistencies, not an idealised version of them.
"""

import pytest
import requests

from bot import ai_errors as err


def classify(body, status=400, provider="apidot", model="apidot:gpt-5.5",
             headers=None):
    return err.classify_response(provider=provider, model=model,
                                 http_status=status, body=body, headers=headers)


# ── Billing: the reported incident, in all its shapes ──────────────────────────

def test_insufficient_credits_http_400_flat_shape():
    """The exact body the running app received (note: type says invalid_request)."""
    error = classify({"error": {"message": "Insufficient credits",
                                "type": "invalid_request_error",
                                "code": "http_error"}}, status=400)
    assert error.category == err.BILLING
    assert error.retryable is False
    assert error.scope == err.ACCOUNT
    assert error.stops_provider is True
    assert error.allows_model_failover is False


def test_insufficient_credits_http_400_wrapped_shape():
    """The /v1/messages variant: {"code":400,"error":{…,"type":"http_error"}}."""
    error = classify({"code": 400, "error": {"message": "Insufficient credits",
                                             "type": "http_error"}}, status=400)
    assert error.category == err.BILLING
    assert error.retryable is False


def test_insufficient_credits_http_402_documented_shape():
    """The status the docs imply for billing — must classify identically."""
    error = classify({"error": {"message": "Payment required",
                                "type": "insufficient_credits"}}, status=402)
    assert error.category == err.BILLING
    assert error.retryable is False


@pytest.mark.parametrize("message", [
    "Insufficient credits", "insufficient_credits", "Not enough credits",
    "You are out of credits", "credit limit reached", "balance is too low",
    "Your balance is insufficient", "Payment required",
])
def test_billing_synonyms_all_normalise(message):
    error = classify({"error": {"message": message,
                                "type": "invalid_request_error"}}, status=400)
    assert error.category == err.BILLING, message


def test_billing_is_never_reported_as_a_generic_failure():
    error = classify({"error": {"message": "Insufficient credits",
                                "type": "invalid_request_error"}}, status=400)
    text = error.operator_message()
    assert "insufficient credits" in text.lower()
    assert "LLM unavailable" not in text
    assert "returned None" not in text


# ── Authentication ─────────────────────────────────────────────────────────────

def test_invalid_api_key_401():
    error = classify({"code": 401,
                      "error": {"message": "Invalid API key format (length: 17)",
                                "type": "authentication_error"}}, status=401)
    assert error.category == err.AUTHENTICATION
    assert error.retryable is False
    assert error.scope == err.ACCOUNT
    assert error.stops_provider is True


def test_balance_endpoint_auth_error_shape():
    """The documented {"code":401,"name":"authentication_error"} shape."""
    error = classify({"code": 401, "name": "authentication_error"}, status=401)
    assert error.category == err.AUTHENTICATION


def test_403_is_authentication_scoped_to_the_account():
    error = classify({"error": {"message": "Forbidden"}}, status=403)
    assert error.category == err.AUTHENTICATION
    assert error.scope == err.ACCOUNT


# ── Model / request / rate limit / upstream ────────────────────────────────────

def test_model_not_found_allows_sibling_failover():
    error = classify({"error": {"message": "Model not found: gpt-9 for vendor 10000001",
                                "type": "invalid_request_error",
                                "code": "model_not_found"}}, status=404)
    assert error.category == err.MODEL_UNAVAILABLE
    assert error.scope == err.MODEL
    assert error.allows_model_failover is True
    assert error.retryable is False        # this model won't appear by retrying


def test_validation_error_delivered_with_http_200():
    """
    APIDot answers a missing `messages` with HTTP 200 and an error body.
    Trusting the transport status would treat a permanent bug as a success.
    """
    body = {"code": 400,
            "error": {"message": "Invalid request parameters: messages: Field required",
                      "type": "validation_error",
                      "details": {"validation_errors": [{"type": "missing"}]}}}
    error = classify(body, status=200)
    assert error is not None, "an error body with HTTP 200 must not read as success"
    assert error.category == err.INVALID_REQUEST
    assert error.retryable is False
    assert error.http_status == 400        # body.code outranks the HTTP status
    assert error.allows_model_failover is False


def test_rate_limit_is_distinct_from_billing():
    error = classify({"error": {"message": "Rate limit exceeded",
                                "type": "rate_limit_error"}}, status=429)
    assert error.category == err.RATE_LIMIT
    assert error.retryable is True
    assert error.scope == err.PROVIDER
    assert error.stops_provider is False
    assert "rate limit" in error.operator_message().lower()


def test_rate_limit_honours_retry_after():
    error = classify({"error": {"message": "Too many requests"}}, status=429,
                     headers={"Retry-After": "12"})
    assert error.retry_after == 12.0


def test_retry_after_http_date_is_ignored_not_crashed():
    error = classify({"error": {"message": "Too many requests"}}, status=429,
                     headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
    assert error.retry_after is None
    assert error.retryable is True


def test_retry_after_is_capped():
    error = classify({"error": {"message": "slow down"}}, status=429,
                     headers={"Retry-After": "99999"})
    assert error.retry_after == 300.0


def test_upstream_5xx_is_retryable_and_route_specific():
    error = classify({"error": {"message": "Server exception, please try again later",
                                "type": "api_error"}}, status=500)
    assert error.category == err.UPSTREAM
    assert error.retryable is True
    assert error.allows_model_failover is True


def test_request_id_is_captured_when_present():
    error = classify({"error": {"message": "boom"}}, status=500,
                     headers={"x-request-id": "req_abc123"})
    assert error.request_id == "req_abc123"
    assert "req_abc123" in error.log_line()


# ── Success detection ──────────────────────────────────────────────────────────

def test_successful_body_classifies_as_no_error():
    assert classify({"choices": [{"message": {"content": "hi"}}]}, status=200) is None


def test_kie_success_envelope_is_not_an_error():
    assert classify({"code": 200, "msg": "success", "data": {"taskId": "abc"}},
                    status=200, provider="kie") is None


def test_kie_error_envelope_is_classified():
    error = classify({"code": 500, "msg": "Server exception"}, status=200,
                     provider="kie")
    assert error is not None
    assert error.category == err.UPSTREAM
    assert error.retryable is True


# ── Transport-level failures ───────────────────────────────────────────────────

def test_timeout_is_retryable():
    error = err.classify_exception(requests.exceptions.ReadTimeout("timed out"),
                                   provider="apidot")
    assert error.category == err.TIMEOUT
    assert error.retryable is True


def test_connection_error_is_retryable():
    error = err.classify_exception(requests.exceptions.ConnectionError("reset"),
                                   provider="apidot")
    assert error.category == err.NETWORK
    assert error.retryable is True


def test_tls_error_is_not_retried():
    """A TLS failure is an environment problem; hammering it never helps."""
    error = err.classify_exception(requests.exceptions.SSLError("bad handshake"),
                                   provider="apidot")
    assert error.category == err.NETWORK
    assert error.retryable is False


def test_malformed_json_is_a_parse_error_not_a_retry():
    error = err.classify_exception(ValueError("Expecting value: line 1 column 1"),
                                   provider="apidot")
    assert error.category == err.PARSING
    assert error.retryable is False


def test_parse_error_is_explicit_and_named():
    error = err.parsing_error("apidot", "apidot:gpt-5.5", "no choices in response")
    assert "APIDOT_RESPONSE_PARSE_ERROR" in error.message
    assert error.category == err.PARSING
    assert error.retryable is False


# ── Policy tables ──────────────────────────────────────────────────────────────

def test_only_account_scoped_categories_stop_the_provider():
    def stops(category):
        return err.ProviderError(provider="apidot", model=None, category=category,
                                 message="", retryable=False,
                                 scope=err.ACCOUNT).stops_provider
    assert stops(err.BILLING) and stops(err.AUTHENTICATION)
    for category in (err.RATE_LIMIT, err.UPSTREAM, err.TIMEOUT, err.NETWORK,
                     err.INVALID_REQUEST, err.MODEL_UNAVAILABLE, err.PARSING):
        assert not stops(category), category


def test_billing_and_auth_are_excluded_from_model_failover():
    assert err.BILLING not in err.MODEL_FAILOVER_CATEGORIES
    assert err.AUTHENTICATION not in err.MODEL_FAILOVER_CATEGORIES
    assert err.RATE_LIMIT not in err.MODEL_FAILOVER_CATEGORIES
    assert err.INVALID_REQUEST not in err.MODEL_FAILOVER_CATEGORIES


def test_log_line_carries_diagnostics_and_no_secrets():
    error = classify({"code": 401, "error": {"message": "Invalid API key format",
                                             "type": "authentication_error"}},
                     status=401)
    line = error.log_line()
    assert "category=AUTHENTICATION" in line
    assert "retryable=False" in line
    assert "sk-" not in line and "Bearer" not in line
