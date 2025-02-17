import requests
import os
from dotenv import load_dotenv
import ratelimit

# rate limit 20 requests per second
@ratelimit.sleep_and_retry
@ratelimit.limits(calls=20, period=1)
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

        # if korean has none or english has none, raise error
        if not english_posters:
            raise Exception(f"No English Posters Found For {movie_id}")
        if not korean_posters:
            raise Exception(f"No Korean Posters Found For {movie_id}")

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

    def get_movie_high_rated_movie(self, page=1):
        MOVIE_URL = f"https://api.themoviedb.org/3/movie/top_rated?api_key={self.api_key}&language=en-US&page={page}"
        response = requests.get(MOVIE_URL, headers=self.headers)
        json_response = response.json()
        return json_response



# movie_id = 496243
#
api = API()

top_rated_movies = api.get_movie_high_rated_movie()
page = top_rated_movies["page"]
results = top_rated_movies["results"]
print(f"Page: {page}")
for movie in results:
    # ensure it was originally in english
    if movie["original_language"] == "en":
        print(f"Movie candidate found: {movie['title']}")
        movie_id = movie["id"]
        # if movie exists in folder, skip
        if os.path.exists(f"images/{movie_id}"):
            print(f"Skipping {movie['title']}")
            continue
        json_response = api.get_movie_image_paths(movie_id)
        poster_language_dict = api.parse_movie_image_paths(json_response)
        for language, image_paths in poster_language_dict.items():
            api.download_images(movie_id, image_paths, language)

# json_response = api.get_movie_image_paths(movie_id)
# poster_language_dict = api.parse_movie_image_paths(json_response)
# for language, image_paths in poster_language_dict.items():
#     api.download_images(movie_id, image_paths, language)

