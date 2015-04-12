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
    
if __name__ == "__main__":
    from sys import argv
    main('/tmp/sanchuah.',argv[1])
