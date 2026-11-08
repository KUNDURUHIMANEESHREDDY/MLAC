"""MLAC command-line interface."""
import argparse
import ipaddress


def is_loopback_host(host: str) -> bool:
    """Return whether a bind host is loopback-only."""
    if host.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main():
    """Parse CLI commands and start the API with proxy-header trust disabled."""
    parser = argparse.ArgumentParser(prog="mlac")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version")
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "version":
        from mlac import __version__
        print(__version__)
    elif args.command == "serve":
        if not is_loopback_host(args.host):
            raise SystemExit(
                "MLAC API is localhost-only: non-loopback binds remain disabled until "
                "durable tenant-scoped storage, distributed rate limiting, TLS deployment, "
                "and end-to-end security acceptance are complete."
            )
        try:
            import uvicorn
        except ImportError as exc:
            raise SystemExit("Install API dependencies: pip install 'mlac[api]'") from exc
        uvicorn.run(
            "mlac.api:app",
            host=args.host,
            port=args.port,
            reload=False,
            proxy_headers=False,
        )


if __name__ == "__main__":
    main()
