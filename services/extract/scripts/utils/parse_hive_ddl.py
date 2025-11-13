import argparse
import json
import sys
from simple_ddl_parser import DDLParser


def parse_hive_ddl(ddl: str):
    """Return structured metadata for the supplied Hive DDL."""
    return DDLParser(ddl).run(group_by_type=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddl", help="DDL content as string")
    parser.add_argument("--ddl-file", help="Path to DDL file")
    args = parser.parse_args()

    ddl_content = None
    
    # Read from file if provided
    if args.ddl_file:
        with open(args.ddl_file, 'r') as f:
            ddl_content = f.read()
    # Read from argument if provided
    elif args.ddl:
        # Check if it's a file path (starts with / or contains /)
        if args.ddl.startswith('/') or '/' in args.ddl:
            try:
                with open(args.ddl, 'r') as f:
                    ddl_content = f.read()
            except (IOError, OSError):
                # Not a valid file, treat as DDL content
                ddl_content = args.ddl
        else:
            ddl_content = args.ddl
    # Read from stdin if no arguments
    else:
        ddl_content = sys.stdin.read()

    if not ddl_content:
        parser.error("DDL content is required (use --ddl, --ddl-file, or stdin)")

    # group_by_type=True causes the parser to organize tables/columns cleanly
    parsed_info = parse_hive_ddl(ddl_content)
    print(json.dumps(parsed_info, indent=4))


if __name__ == "__main__":
    main()
