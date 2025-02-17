import requests
import os
from dotenv import load_dotenv

class API():
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("API_KEY")
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.MOVIE_URL = "https://api.themoviedb.org/3/movie"
        self.IMAGE_URL = "https://image.tmdb.org/t/p/original"
        self.IMAGE_FOLDER = "images"
    def get_movie_image_paths(self, movie_id):
        Movie_URL = f"https://api.themoviedb.org/3/movie/{movie_id}/images?include_image_language=en%2Cko"
        response = requests.get(Movie_URL, headers=self.headers)
        json_response = response.json()
        return json_response

    def parse_movie_image_paths(self, json_response):
        english_posters = []
        korean_posters = []
        for poster in json_response["posters"]:
            if poster["iso_639_1"] == "en":
                english_posters.append(poster["file_path"])
            elif poster["iso_639_1"] == "ko":
                korean_posters.append(poster["file_path"])
        return {"en": english_posters, "ko": korean_posters}

    def download_images(self,movie_id, image_paths, language):
        for image_path in image_paths:
            image_url = f"{self.IMAGE_URL}{image_path}"
            response = requests.get(image_url)

            # check directory exists else create
            if not os.path.exists(self.IMAGE_FOLDER):
                os.makedirs(self.IMAGE_FOLDER)
            if not os.path.exists(f"{self.IMAGE_FOLDER}/{movie_id}"):
                os.makedirs(f"{self.IMAGE_FOLDER}/{movie_id}")
            if not os.path.exists(f"{self.IMAGE_FOLDER}/{movie_id}/{language}"):
                os.makedirs(f"{self.IMAGE_FOLDER}/{movie_id}/{language}")

            with open(f"{self.IMAGE_FOLDER}/{movie_id}/{language}/{image_path[1:]}", "wb") as file:
                file.write(response.content)
                print(f"Downloaded {image_path}")


movie_id = 496243

api = API()
json_response = api.get_movie_image_paths(movie_id)
poster_language_dict = api.parse_movie_image_paths(json_response)
for language, image_paths in poster_language_dict.items():
    api.download_images(movie_id, image_paths, language)

