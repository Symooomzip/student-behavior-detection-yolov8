"""Launch the local React + FastAPI workspace."""


def launch(cfg: dict, weights=None, port=7860, device=None):
    import uvicorn
    from .web import create_app

    uvicorn.run(create_app(cfg, weights=weights, device=device), host="127.0.0.1", port=port)
