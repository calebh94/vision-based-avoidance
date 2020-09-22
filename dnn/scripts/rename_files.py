import os

def main():

    for filename in os.listdir("data/testdata/"):
        if filename.__contains__('.jpg') == True:
            new_filename = filename
            new_filename = new_filename.replace("_jpg","")
            new_filename = new_filename.replace(".jpg", "")
            new_filename = new_filename.replace(".", "_")
            # new_filename = new_filename.replace("_", ".")

            # new_filename = new_filename.replace("output_image", "")
            # new_filename = new_filename.replace("world_name_obstacle")
            new_filename = new_filename + ".jpg"
            src = "data/testdata/" + filename
            dst = "data/testdata/" + new_filename
            os.rename(src,dst)


if __name__ == "__main__":
    main()

