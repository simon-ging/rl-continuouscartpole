import glob
from pathlib import Path


if __name__ == '__main__':
    data = glob.glob(
        "*/*/config.json")  # (Path("src") / "tests" / "list.txt").open("rt").read().splitlines()
    print(len(data), "files")
    for f in data:
        print(f)
        file_content = Path(f).open("rt").read()
        file_content = file_content.replace(
            "env_state_transform", "env_observ_transform")
        # Path(f).open("wt").write(file_content)
