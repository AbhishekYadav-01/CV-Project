import gdown

def download_assets():
    # URL for the .glb file (3D object)
    url_glb = 'https://drive.google.com/uc?id=1tGyOA98PIxVTP7scK4RfyfpSyzsU0Q-5'
    output_glb = 'object.glb'

    # URL for the .jpeg file (scene image)
    url_jpeg = 'https://drive.google.com/uc?id=1G7ZxnwVP7aem7Tv-wGJaI_dASYTxfE6H'
    output_jpeg = 'scene_image.jpeg'

    # Download the files
    gdown.download(url_glb, output_glb, quiet=False)
    gdown.download(url_jpeg, output_jpeg, quiet=False)

if __name__ == "__main__":
    download_assets()
