"""API package: HTTP routes and request/response schemas.

Routes are thin: they validate input (via Pydantic schemas), delegate
to the service layer, and translate domain errors into HTTP responses.
No business logic lives here.
"""
