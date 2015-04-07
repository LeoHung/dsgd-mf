import os
import os.path
import re 


def gen(input, output_f):
    f = open(input)

    movie_id = int(f.readline().split(":")[0])
    for line in f:
        tmp = line.split(",")
        user_id, rating = int(tmp[0]), int(tmp[1])
        print >> output_f, "%d,%d,%d" %(user_id, movie_id, rating)

def main(input_dir, output):
    output_f = open(output, "w")

    for file in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, file)
        gen(input_file_path, output_f)        

    output_f.close()

if __name__ == "__main__":
    from sys import argv
    tmp_file = argv[2] +".tmp"
    main(argv[1], tmp_file) 
    os.system("sort %s > %s" %(tmp_file, argv[2]))
    os.system("rm %s" %(tmp_file))
