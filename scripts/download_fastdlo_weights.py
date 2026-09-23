from pathlib import Path
from mega import Mega

root=Path(__file__).resolve().parents[1]
target=root/'code'/'fastdlo_src'/'fastdlo-master'/'weights'
target.mkdir(parents=True,exist_ok=True)
url='https://mega.nz/file/YNsmnYwa#y9DiZEly-MQ_s8vHifSCDZLghaOe89pd4tZKQ5IOEME'
print(Mega().download_url(url, str(target)))
