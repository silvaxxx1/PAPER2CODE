import requests

# Function to download the file
def get_wieghts(url, destination):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"File downloaded successfully: {destination}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

