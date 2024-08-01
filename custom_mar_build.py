import argparse, sys, os, shutil, json
from datetime import datetime
from pathlib import Path

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="hallucinator")
    parser.add_argument("--handler", type=str, default="hallucinator_handler.py")
    return parser.parse_args()


def copy_extra_files(extra_files, tmp_dir):
    """Copy extra files to tmp_dir. work with both folder and file"""
    for file in extra_files:
        if Path(file).is_dir():
            shutil.copytree(file, tmp_dir.joinpath(file))
        else:
            shutil.copy(file, tmp_dir.joinpath(file))


def main():
    args = parse()
    model_name = args.model_name
    handler = args.handler

    tmp_dir = Path("./dummy_temp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    extra_files = [handler, "models", "utils", "exp", "data"]
    copy_extra_files(extra_files, tmp_dir)

    manifest = {
        "createdOn": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "runtime": "python",
        "model": {
            "modelName": model_name,
            "serializedFile": "",
            "set_size": 200,
            "threshold": 100,
            "config_file": "./exp/speech_XXL_cond/params.json",
            "handler": handler,
            "modelVersion": "1.0"
        },
        "archiverVersion": "1.0"
    }

    manifest_folder = tmp_dir.joinpath("MAR-INF")
    manifest_folder.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_folder.joinpath("MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
    print("making archive...")
    zipfile = shutil.make_archive(f"./data/model-store/{model_name}", 'zip', tmp_dir)
    print(f"Created zip file: {zipfile}, renaming now...")
    zippath = Path(zipfile)
    marpath = zippath.rename(zippath.with_suffix(".mar"))
    print(f"Created MAR file: {marpath}")

    # clean up
    shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == '__main__':
    main()
