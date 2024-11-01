import requests
from urllib.parse import urljoin
#from markdown2pdf import markdown2pdf
import markdown2pdf
import os

def download_image(url, filename):
  """Downloads an image from the given URL and saves it with the specified filename."""
  try:
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise exception for bad status codes
    with open(filename, 'wb') as f:
      for chunk in response.iter_content(1024):
        f.write(chunk)
    print(f"Downloaded image: {url}")
  except requests.exceptions.RequestException as e:
    print(f"Error downloading image: {url} - {e}")

def convert_mdx_to_pdf(mdx_file, pdf_file, base_url):
  """Converts an MDX file to PDF, downloading referenced images and updating paths."""
  with open(mdx_file, 'r') as f:
    mdx_content = f.read()

  # Update image references with downloaded image paths
  updated_content = ""
  for line in mdx_content.splitlines():
    if line.startswith("<img"):  # Check for image reference line
      image_url = line.split(" ")[1][5:-1]  # Extract image URL
      print(image_url)
      filename = f"./images/{image_url.split('/')[-1]}"  # Generate filename
      #download_image(image_url, filename)  # Download image
      updated_content += f"![xxx]({filename})\n"  # Update reference with local path
    else:
      updated_content += line + "\n"

  md_file_name = "{}.md".format(mdx_file[:-4])
  # Open the file in write mode ('w')
  with open(md_file_name, "w") as file:
    # Write content to the file
    file.write(updated_content)


  # Convert updated Markdown content to PDF
  markdown2pdf.convert_md_2_pdf(md_file_name)
  print(f"Converted MDX to PDF: {mdx_file}")

# Replace with your base URL where images are located (if applicable)
base_url = "https://yourwebsite.com/images/"  # Modify as needed

# Loop through your MDX files
for mdx_file in [f for f in os.listdir('./') if f.endswith('mdx')]:  # Replace with your filenames
  pdf_file = mdx_file.replace(".mdx", ".pdf")  # Generate output PDF filename
  convert_mdx_to_pdf(mdx_file, pdf_file, base_url)

print("All MDX files processed!")
