import argparse


def main():
  parser = argparse.ArgumentParser(
    prog="xarray-kat", description="Command-line tool for xarray-kat package tasks"
  )

  subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")
  p = subparsers.add_parser(
    "cmp-vs-katdal", help="Compare the MSv4 DataTree values vs a katdal dataset"
  )
  p.add_argument("url")
  p.add_argument("--applycal", default="all")
  args = parser.parse_args()

  if args.command == "cmp-vs-katdal":
    from xarray_kat.scripts.katdal_compare import compare_vs_katdal

    compare_vs_katdal(args)
  else:
    parser.print_help()


if __name__ == "__main__":
  main()
