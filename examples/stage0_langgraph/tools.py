# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The three customer-support tools, as LangChain tools over an HTTP backend.

Two of these — lookup_order and process_return — move to an Amazon Bedrock
AgentCore Gateway in stage 1. search_faq stays a local function through every
stage, which is what a real partial migration looks like.

The backend is whatever ORDERS_API_BASE points at. Run
examples/stage0_langgraph/local_api.py to serve it locally; the default
https://api.example.com does not resolve, on purpose, because pretending it does
is how you end up shipping an agent whose tools have never returned a payload.

Tool bodies return an {"error": ...} payload rather than raising. A raised
exception inside a ToolNode kills the graph run; an error payload is something
the model can read and react to.
"""

import os

import requests
from langchain_core.tools import tool

TIMEOUT_SECONDS = 30


def _base_url() -> str:
    # Read at call time, not import time, so a caller can point the tools at a
    # local stub after this module is imported.
    return os.environ.get("ORDERS_API_BASE", "https://api.example.com").rstrip("/")


def _request(method: str, path: str, **kwargs) -> dict:
    try:
        response = requests.request(
            method, f"{_base_url()}{path}", timeout=TIMEOUT_SECONDS, **kwargs
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"error": f"{method} {path} failed: {exc}"}


@tool
def lookup_order(order_id: str) -> dict:
    """Look up the status, carrier and tracking number of an order by its ID."""
    return _request("GET", f"/orders/{order_id}")


@tool
def process_return(order_id: str, reason: str) -> dict:
    """Initiate a return for an order, given the order ID and the reason."""
    return _request("POST", "/returns", json={"order_id": order_id, "reason": reason})


@tool
def search_faq(query: str) -> dict:
    """Search the support knowledge base for a policy or FAQ answer."""
    return _request("GET", "/faq/search", params={"q": query})


SUPPORT_TOOLS = [lookup_order, process_return, search_faq]
