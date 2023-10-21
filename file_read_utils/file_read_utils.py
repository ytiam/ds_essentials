def read_multijson_in_newline_format(m: str) -> list():
    '''
    Read file which contains multiple dictionaries but in every elemnt is in new line format
    
    Arguments:
    m - file path
    
    Output:
    list of all the dictionaries which the file contains
    '''
    try:
        with open(m,'r', encoding='utf8') as f:
            temp_meta = f.readlines()
    except:
        with open(m,'r', encoding='utf16') as f:
            temp_meta = f.readlines()
    
    all_dicts = []
    for i, item in enumerate(temp_meta):
        if item == "{\n":
            new_dict = {}
            continue
        elif item == "}\n":
            all_dicts.append(new_dict)
        else:
            item = item.strip()
            item = item.rstrip(',\n')
            k, v = item.split(' : ')
            k, v = k.lstrip('"'), v.lstrip('"')
            k, v = k.rstrip('"'), v.rstrip('"')
            new_dict[k]=v
    return all_dicts


def read_image(img_path: str, resize_dim: (int,int) = None):
    '''
    Read Image from a given path and resize the image height and width if required
    
    Arguments:
    img_path - Image file path
    resize_dim - Default is None. If not None, the Image will be resized to (w1,h1) size
    
    Output:
    Image file original/resized
    '''
    image_data = Image.open(img_path)
    if resize_dim:
        image_data = image_data.resize(resize_dim)
    return image_data