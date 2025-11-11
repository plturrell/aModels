import argparse
import json
from simple_ddl_parser import DDLParser


def parse_hive_ddl(ddl: str):
    """Return structured metadata for the supplied Hive DDL."""
    return DDLParser(ddl).run(group_by_type=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddl", required=True)
    args = parser.parse_args()

    # group_by_type=True causes the parser to organize tables/columns cleanly
    parsed_info = parse_hive_ddl(args.ddl)
    print(json.dumps(parsed_info, indent=4))


if __name__ == "__main__":
    main()
