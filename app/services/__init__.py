"""Service package: business logic, independent of transport.

Services drive infrastructure adapters and own the rules (defaults,
metrics, logging, error translation). They never import FastAPI.
"""
