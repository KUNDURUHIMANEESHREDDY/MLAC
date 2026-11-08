"""MLAC command-line interface."""
import argparse

def main():
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
        try:
            import uvicorn
        except ImportError as exc:
            raise SystemExit("Install API dependencies: pip install 'mlac[api]'") from exc
        uvicorn.run("mlac.api:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
