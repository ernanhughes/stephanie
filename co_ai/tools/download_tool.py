# Import libraries
import requests


def download_pdf(url: str):
    response = requests.get(url)
    # Write content in pdf file
    pdf = open("pdf" + str(i) + ".pdf", "wb")
    pdf.write(response.content)
    pdf.close()
    print("File ", i, " downloaded")


