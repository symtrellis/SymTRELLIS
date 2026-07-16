from backend.app import server, server_lifespan

server.launch(
    show_error=True,
    app_kwargs={"lifespan": server_lifespan},
)
