from os.path import isfile
from eval_acc import LoadSparseMatrix, LoadMatrix, CalculateError

def main(path, v_file):
    epoch = 0
    V, select = LoadSparseMatrix(v_file)
    print "epoch,error"
    while True:
        w_file = "%sW_%d.csv" %(path, epoch)
        h_file = "%sH_%d.csv" %(path, epoch)
        if(not (isfile(w_file) and isfile(h_file))):
            break
        else:
            W = LoadMatrix(w_file)
            H = LoadMatrix(h_file)
            error = CalculateError(V, W, H, select)
            print "%d,%f" %(epoch, error)

        epoch += 1


def main_2(v_file, w_file, h_file):
    W = LoadMatrix(w_file)
    H = LoadMatrix(h_file)
    V, select = LoadSparseMatrix(v_file)

    print "error: %f" %(CalculateError(V, W, H, select))

if __name__ == "__main__":
    from sys import argv
    main_2(argv[1], argv[2], argv[3])
