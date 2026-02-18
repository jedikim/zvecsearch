"""Allow running as python -m zvecsearch."""


def main():
    try:
        from zvecsearch.cli import cli
        cli()
    except ImportError:
        print("CLI not yet implemented")


if __name__ == "__main__":
    main()
