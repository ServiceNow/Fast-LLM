from PIL import Image
import io

def convert_queries_and_annotations_to_messages(queries, annotations):
    messages = []
    # Add each query and annotation as a user-assistant pair
    for i, (q, a) in enumerate(zip(queries, annotations)):
        if i == 0:
            # Prepend "<|image|>" to the first query
            q = f"<|image|>{q}"
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    return messages

def image_loading_function(images):
    """
    Load an image from a file path
    """
    assert images is not None
    if not isinstance(images, list):
        images = [images]
    image_pils = []
    for image in images:
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, Image.Image):
            pass
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        image_pils.append(image)
    return image_pils