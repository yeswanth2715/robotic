from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
archive = ROOT / "data" / "MovingCables_sample.tar"
destination = ROOT / "data" / "movingcables_sample"
destination.mkdir(parents=True, exist_ok=True)

with tarfile.open(archive, "r") as tar:
    tar.extractall(destination, filter="data")

print(f"extracted={destination}")
