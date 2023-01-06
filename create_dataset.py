import json
import wget
import base64
from PIL import Image
from io import BytesIO
from pathlib import Path
from tqdm.auto import tqdm
from bs4 import BeautifulSoup


def resize_and_convert_to_rgb(image, emojis_size):
    resized_image = image.resize((emojis_size, emojis_size))
    # https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
    resized_image.load()  # required for png.split()

    rgb_image = Image.new("RGB", resized_image.size, (255, 255, 255))
    rgb_image.paste(
        resized_image, mask=resized_image.split()[3]
    )  # 3 is the alpha channel
    return rgb_image


def extract_and_convert_to_pill(soup):
    try:
        base64_data = soup.img["src"].replace("data:image/png;base64,", "")
        byte_data = base64.urlsafe_b64decode(base64_data)
        image_data = BytesIO(byte_data)
        image = Image.open(image_data)
        image = image.convert("RGBA")
        return image
    except:
        return None


def download_emoji_pages(dataset_path, emojis_size):
    urls = [
        "https://unicode.org/emoji/charts/full-emoji-list.html",
        "https://unicode.org/emoji/charts/full-emoji-modifiers.html",
    ]
    class_names = {}

    images_path = dataset_path / Path(f"{emojis_size}x{emojis_size}_emojis")

    for url in urls:
        file_name = Path(url).name
        output_path = dataset_path / file_name
        if not output_path.is_file():
            print(f"Did not find {file_name} directory, Lets Download it...")
            wget.download(url=url, out=str(output_path))

        print(f"Opening {file_name} and converting it to  BeautifulSoup object")
        with open(output_path) as fp:
            soup = BeautifulSoup(fp, "html.parser")
        rows = soup.find_all("tr")
        print(f"Extracting the emojis form {file_name} file")
        for i, row in enumerate(tqdm(rows[:])):
            columns = row.find_all("td")
            name = 0
            if columns != []:
                for j, column in enumerate(columns[3:9]):
                    image = extract_and_convert_to_pill(column)
                    if image != None and j != 3:
                        name += 1
                        class_path = images_path / f"{file_name}_{columns[0].text}"
                        class_path.mkdir(parents=True, exist_ok=True)
                        image_path = class_path / f"{name}.jpg"
                        resized_image = resize_and_convert_to_rgb(image, emojis_size)
                        resized_image.save(image_path, "JPEG", quality=100)
                    class_names[i] = columns[-1].text.strip()
    with open(images_path / "class_names.json", "w") as outfile:
        json.dump(class_names, outfile)

    print("Done!")


def create_dataset(emojis_size):

    base_path = Path().absolute()
    dataset_path = base_path / Path("data")

    # If the dataset folder doesn't exist, make it!
    if dataset_path.is_dir():
        print(f"Dataset directory exists: {dataset_path}")
    else:
        print(f"Did not find {dataset_path} directory, creating one...")
        dataset_path.mkdir(parents=True, exist_ok=True)

    download_emoji_pages(dataset_path, emojis_size)


if __name__ == "__main__":
    create_dataset(emojis_size=10)
