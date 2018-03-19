# Lisa Pegram

# Assume that a file containing a series of integers is named numbers.txt
# and exists on the computer's disk. Write a program that calulates the
# average of all the numbers stored in the file.

# for the sake of testing you can use exercise 7 to generate
# a file that will provide numbers to average

def main():
    # assign values to 0 count can never be float
    total = 0.0
    count = 0
    # open file with numbers to be averaged
    pathnameIn = r'C:\Users\Lisa.Pegram\Desktop\SQL'
    filenameIn = 'numbers.txt'
    pathIn = pathnameIn + "/" + filenameIn

    pathnameOut = r'C:\Users\Lisa.Pegram\Desktop\SQL'
    filenameOut = "average_output.txt"
    pathOut = pathnameOut + "/" + filenameOut

    file_ = open(pathIn, 'r')
    #fileOut = open(pathOut, 'w')

    # iterate for loop for all lines in 'file'
    for line in file_:
        number = float(line)
        count += 1
        total += number
    average = str(total / count)
    print("The average of all the numbers found in 'numbers.txt' is:", average)
          #format(average, '.2f'))
    #file_.write(str(average))
    fileOut = open(pathOut, 'w')
    fileOut.write(average)
    file_.close


main()
