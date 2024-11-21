def query_imgs_constructor_func(query_item, icl_items_list):
    query = ""
    images = []
    for item in icl_items_list:
        answer = item['keypoint_chosen'].replace("_", " ")
        query += f"""
            source image: <image> support image: <image> Question: What is the corresponding keypoint name in source image at the red dot location in support image? Output: {answer} 
            """
        images.append(item['image'])
        images.append(item['support_image'])
    query += f"""
            source image: <image> support image: <image> Question: What is the corresponding keypoint name in source image at the red dot location in support image? Output:  
            """
    images.append(query_item['image'])
    images.append(query_item['support_image'])
    return query, images

def gt_post_processor_func(op):
    return op.replace("_", " ")

def query_imgs_constructor_func2(query_item, icl_items_list):
    query = ""
    images = []
    options = "A. left eye B. right eye C. nose D. left ear E. right ear"
    option_mapper = {
        "left_eye": "A. left eye",
        "right_eye": "B. right eye",
        "nose": "C. nose",
        "left_ear": "D. left ear",
        "right_ear": "E. right ear"
    }
    for item in icl_items_list:
        answer = option_mapper[item['keypoint_chosen']][0]
        query += f"""
            source image: <image> support image: <image> Question: What is the corresponding keypoint name in source image at the red dot location in support image? {options} Output: {answer} 
            """
        images.append(item['image'])
        images.append(item['support_image'])
    query += f"""
            source image: <image> support image: <image> Question: What is the corresponding keypoint name in source image at the red dot location in support image? {options} Output:  
            """
    images.append(query_item['image'])
    images.append(query_item['support_image'])
    return query, images

def gt_post_processor_func2(op):
    option_mapper = {
        "left_eye": "A. left eye",
        "right_eye": "B. right eye",
        "nose": "C. nose",
        "left_ear": "D. left ear",
        "right_ear": "E. right ear"
    }
    return option_mapper[op][0]

def get_ic_pp_func(style):
    if style=="vqa_style":
        return query_imgs_constructor_func, gt_post_processor_func
    else:
        return query_imgs_constructor_func2, gt_post_processor_func2